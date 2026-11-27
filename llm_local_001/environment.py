"""Real text-action adapter for SPEC-LLM-LOCAL-001 (2026-10-23).

Reuses PilotEnv, its append-only SQLite, transitions, evaluator and frozen reward.
The LLM receives literal evidence, not public_analysis, kinds, polarities, private
labels or a rule recommendation. Parsing only translates generated JSON into the
existing action API; it never selects an action for the model. S1 ingestion is
fixed; the generated policy controls S2 and S3, including clarification followups.

ORACLE means controlled literal-evidence extraction, not oracle action selection.
PIPELINE separately measures the existing extractor. The question router below is
a deliberately conservative finite-domain *user reply* adapter, not an LLM action
selector. Unrecognized questions receive no scripted preference reply. Its errors
are exposed, and broader natural-language question validity needs human review.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import re
from time import monotonic
import unicodedata

from rl_observability.environment import MinimalEnv, RewardConfig
from rl_small_pilot.environment_adapter import PilotEnv

SCHEMA_VERSION = 'SPEC-LLM-LOCAL-001-action-v1'
CONVERSATION_ACTIONS = ('ANSWER', 'CLARIFY', 'CONDITIONAL_ANSWER', 'PRESERVE_CONFLICT')
MEMORY_ACTIONS = ('ADD', 'UPDATE_WITH_HISTORY', 'NOOP', 'PRESERVE_CONFLICT')
QUESTION_SLOTS = ('preference', 'meal', 'day', 'source')
REWARD_PROVENANCE = 'rl_observability.environment.RewardConfig; unchanged coefficients'

# No abstract kind or polarity annotations: the causal LLM interprets actual text.
FACT_FIELDS = ('memory_id', 'evidence_id', 'food', 'text', 'observed_at',
               'recorded_at', 'valid_from', 'valid_to', 'scope', 'source', 'supersedes')
OBSERVATION_FIELDS = ('representation', 'query', 'query_at', 'current_at',
                      'target_food', 'context', 'session_index', 'remaining_clarifications')


def public_observation(observation):
    """Strict whitelist; evaluator metadata cannot accidentally enter a prompt."""
    if observation.get('terminated'):
        return {'terminated': True}
    result = {key: deepcopy(observation[key]) for key in OBSERVATION_FIELDS if key in observation}
    for name in ('memory', 'incoming_evidence'):
        result[name] = [{key: deepcopy(fact.get(key)) for key in FACT_FIELDS}
                        for fact in observation.get(name, [])]
    result['clarification_history'] = [
        {key: deepcopy(row[key]) for key in ('seq', 'question_key', 'at', 'status', 'question') if key in row}
        for row in observation.get('clarification_history', [])]
    result['unresolved_conflicts'] = [
        {key: deepcopy(row[key]) for key in ('evidence_ids', 'recorded_at', 'origin') if key in row}
        for row in observation.get('unresolved_conflicts', [])]
    return result


# Selected with TRAIN-only format probes before any SFT/RL or TEST.
PROMPT_INSTRUCTIONS = '''Return a single JSON object with exactly the keys shown below. Do not use Markdown.
Allowed conversation_action values: ANSWER, CLARIFY, CONDITIONAL_ANSWER, PRESERVE_CONFLICT.
Allowed memory_action values: ADD, UPDATE_WITH_HISTORY, NOOP, PRESERVE_CONFLICT.
Examples of the output FORMAT (choose actual values from the given evidence):
{"conversation_action":"ANSWER","question":null,"answer":{"polarity":"LIKE","text":"Te gusta."},"memory_action":"ADD","evidence_ids":["e0"]}
{"conversation_action":"ANSWER","question":null,"answer":{"polarity":"DISLIKE","text":"No te gusta."},"memory_action":"UPDATE_WITH_HISTORY","evidence_ids":["e0","e2"]}
{"conversation_action":"CLARIFY","question":"¿Cambió tu preferencia permanentemente o solo por hoy?","answer":null,"memory_action":"PRESERVE_CONFLICT","evidence_ids":["e0","e2"]}
{"conversation_action":"PRESERVE_CONFLICT","question":null,"answer":null,"memory_action":"PRESERVE_CONFLICT","evidence_ids":["e0","e2"]}
Use only evidence_ids actually listed in memory. Interpret the original Spanish evidence text, observation/recording dates, validity and context. A permanent change replaces an old preference; a temporary exception applies only within its interval. Ask to resolve relevant ambiguity when needed. Unrelated conflicts do not prevent answering. Preserve uncertainty when a clarification is unresolved. Keep safety restrictions. memory_action applies to incoming_evidence. Answer only the JSON object, not an explanation.
IMPORTANT: question is YOUR clarification question to ask the user, NEVER the input query. If conversation_action is ANSWER, question MUST be null. If conversation_action is CLARIFY, answer MUST be null. For other conversation actions both question and answer may be null.'''
PROMPT_SUFFIX = '\nReturn one JSON object now. For ANSWER set question to null.'

def format_prompt(observation):
    """Identical instruction/template for B0 through B3 and both ablations."""
    return PROMPT_INSTRUCTIONS + '\nOBSERVATION:\n' + json.dumps(
        public_observation(observation), ensure_ascii=False, sort_keys=True, separators=(',', ':')) + PROMPT_SUFFIX


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON field: ' + key)
        result[key] = value
    return result


def parse_action(raw, observation):
    """Validate generated tokens. No forgiving JSON extraction or gold repair."""
    if not isinstance(raw, str):
        raise ValueError('the model must generate a JSON string')
    try:
        value = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys,
                           parse_constant=lambda constant: (_ for _ in ()).throw(ValueError('non-finite JSON')))
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError('invalid JSON: ' + str(error)) from error
    required = {'conversation_action', 'question', 'answer', 'memory_action', 'evidence_ids'}
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError('missing required structured action fields')
    if value.keys() - (required | {'question_slot'}):
        raise ValueError('unexpected structured action fields')
    if value['conversation_action'] not in CONVERSATION_ACTIONS:
        raise ValueError('unknown conversation_action')
    if value['memory_action'] not in MEMORY_ACTIONS:
        raise ValueError('unknown memory_action')
    if value.get('question_slot') not in (None, *QUESTION_SLOTS):
        raise ValueError('unknown question_slot')
    evidence_ids = value['evidence_ids']
    if (not isinstance(evidence_ids, list) or any(not isinstance(e, str) for e in evidence_ids)
            or len(set(evidence_ids)) != len(evidence_ids)):
        raise ValueError('evidence_ids must be unique strings')
    available = {fact['evidence_id'] for fact in observation.get('memory', [])}
    if not set(evidence_ids) <= available:
        raise ValueError('evidence_ids include evidence absent from the actual retrieval')
    conversation = value['conversation_action']
    if conversation == 'CLARIFY':
        if not isinstance(value['question'], str) or not value['question'].strip() or value['answer'] is not None:
            raise ValueError('CLARIFY requires an actual question and null answer')
    elif value['question'] is not None or value.get('question_slot') is not None:
        raise ValueError('only CLARIFY can emit a question or question_slot')
    if conversation == 'ANSWER':
        answer = value['answer']
        if (not isinstance(answer, dict) or set(answer) != {'polarity', 'text'}
                or answer['polarity'] not in ('LIKE', 'DISLIKE')
                or not isinstance(answer['text'], str) or not answer['text'].strip()):
            raise ValueError('ANSWER requires explicit generated polarity and response text')
    elif conversation != 'CLARIFY' and value['answer'] is not None and not isinstance(value['answer'], str):
        raise ValueError('uncertainty answer must be null or a conditional text string')
    return value


def _normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                   if unicodedata.category(c) != 'Mn')


def question_intent(question, target_food):
    """Route the actual asked question; never trust an arbitrary declared slot.

    No content from the scenario, future reply or gold is consulted. Unsupported
    questions retain the clarification cost and receive the existing neutral reply.
    """
    text = _normalize(question)
    interrogative = '?' in text or bool(re.match(r'\s*(que|cual|cuando|quien|donde|como|es|son|ha|has|te|me|prefier)', text))
    if not interrogative:
        return 'unspecified'
    if any(term in text for term in ('que fuente', 'cual es la fuente', 'quien', 'de donde procede', 'origen de la informacion')):
        return 'source'
    topic = _normalize(target_food or '')
    topic_mentioned = bool(topic and re.search(r'\b' + re.escape(topic) + r'\b', text))
    deictic = any(term in text for term in ('preferencia', 'tus gustos', 'mis gustos', 'ese cambio', 'este cambio', 'el cambio', 'esa excepcion'))
    asks_preference = any(term in text for term in ('gusta', 'gusto', 'prefier', 'preferenc', 'cambi', 'permanente', 'temporal', 'solo hoy', 'para hoy'))
    # Naming an occasion in a preference question does not turn it into a
    # request for an unknown meal slot. Explicit slot questions take precedence.
    asks_meal = any(term in text for term in ('que comida', 'cual comida', 'cual de las comidas'))
    if asks_meal:
        return 'meal'
    if topic_mentioned and asks_preference:
        return 'preference'
    if any(term in text for term in ('desayuno', 'almuerzo', 'cena', 'merienda')):
        return 'meal'
    if any(term in text for term in ('que dia', 'dia laborable', 'fin de semana', 'dia de semana')):
        return 'day'
    # A question explicitly about another subject is not a generic deictic
    # request about the current food. Unsupported wording receives no gold reply.
    if not topic_mentioned and any(term in text for term in ('sobre ', 'acerca de ', 'respecto a ')):
        return 'unspecified'
    if asks_preference and (topic_mentioned or deictic):
        return 'preference'
    return 'unspecified'


class LLMEnvironment(PilotEnv):
    """Reuses real historical execution; invalid strings become scored failures."""
    def __init__(self, scenario, representation='O2', condition='ORACLE', db_path=None, reward_config=None):
        if representation not in ('O1', 'O2'):
            raise ValueError('O3 requires an independently identified and documented residual need')
        super().__init__(scenario, representation, condition, db_path, reward_config)
        self.generations = []
        self._current_generated_action = None

    def reset(self, seed=None):
        self.generations = []
        self._current_generated_action = None
        return super().reset(seed)

    def _observe(self):
        internal = super()._observe()
        if internal.get('terminated'):
            return internal
        if self.representation == 'O2':
            # These are actual prior preservation operations, not a computed gold
            # conflict label. Later explicit supersessions retire their old IDs.
            superseded = {old for fact in internal['memory'] for old in fact.get('supersedes', [])}
            visible = {fact['memory_id']: fact for fact in internal['memory']}
            records = []
            for operation in self.store.operations():
                fact = operation.get('after') or {}
                if (operation['operation'] == 'PRESERVE_CONFLICT' and fact.get('memory_id') in visible
                        and fact.get('memory_id') not in superseded):
                    records.append({'evidence_ids': [fact['evidence_id']], 'recorded_at': operation['at'],
                                    'origin': 'previous_executed_memory_preservation'})
            internal['unresolved_conflicts'] = records
        self._observation = deepcopy(internal)
        projected = public_observation(internal)
        # Metadata contains what was really asked; never the withheld user reply.
        for history in projected['clarification_history']:
            matching = [g for g in self.generations if g.get('parsed') and
                        g['parsed']['conversation_action'] == 'CLARIFY' and
                        g.get('at') == history.get('at') and
                        g.get('executed_question_slot') == history.get('question_key')]
            if matching:
                history['question'] = matching[-1]['parsed']['question']
        self._model_observation = deepcopy(projected)
        return projected

    @staticmethod
    def prompt(observation):
        return format_prompt(observation)

    def _answer_grounded(self, observation, polarity):
        full_support = super()._answer_grounded(observation, polarity)
        generated = self._current_generated_action
        if not generated or generated['conversation_action'] != 'ANSWER':
            return full_support
        cited = set(generated['evidence_ids'])
        restricted = deepcopy(observation)
        restricted['memory'] = [fact for fact in observation['memory'] if fact['evidence_id'] in cited]
        # Both are required: citations cannot hide other visible contradictory
        # evidence, and citing a safety fact cannot support an unrelated answer.
        return bool(cited and full_support and super()._answer_grounded(restricted, polarity))

    def step(self, raw):
        if self._terminated:
            raise RuntimeError('Episode already terminated')
        record = {'raw': raw if isinstance(raw, str) else repr(raw),
                  'session_index': self._index, 'at': self._session['at'],
                  'prompt': format_prompt(self._model_observation), 'format_error': None,
                  'question_routing_error': False}
        try:
            generated = parse_action(raw, self._model_observation)
        except ValueError as error:
            generated = None
            record['format_error'] = str(error)
        self._current_generated_action = generated
        if generated is None:
            composed = {'conversation': 'NOOP', 'memory_operation': 'NOOP', 'polarity': None, 'slot': None}
        else:
            conversation = generated['conversation_action']
            slot = None
            if conversation == 'CLARIFY':
                slot = question_intent(generated['question'], self._observation['target_food'])
                record['question_routing_error'] = slot == 'unspecified' or generated.get('question_slot', slot) != slot
                record['executed_question_slot'] = slot
            composed = {'conversation': conversation,
                        'memory_operation': 'ADD_MEMORY' if generated['memory_action'] == 'ADD' else generated['memory_action'],
                        'polarity': generated['answer']['polarity'] if conversation == 'ANSWER' else None,
                        'slot': slot}
        record['parsed'] = deepcopy(generated)
        self.generations.append(record)
        # PilotEnv's small-policy action catalog excludes NOOP and unsupported
        # slots; bypass only that catalog, retain MinimalEnv's actual transitions.
        obs, reward, done, truncated, info = MinimalEnv.step(self, composed)
        record.update(reward=reward, terminal=self._steps[-1]['terminal'])
        self._steps[-1]['generated_response'] = deepcopy(record)
        self._steps[-1]['format_error'] = record['format_error']
        info.update(format_error=record['format_error'], question_routing_error=record['question_routing_error'])
        return obs, reward, done, truncated, info

    def diagnostic_metrics(self):
        result = super().diagnostic_metrics()
        result.update(schema_version=SCHEMA_VERSION, generations=deepcopy(self.generations),
                      format_errors=sum(g['format_error'] is not None for g in self.generations),
                      generation_count=len(self.generations),
                      question_routing_errors=sum(g['question_routing_error'] for g in self.generations),
                      reward_config=asdict(self.reward_config), reward_provenance=REWARD_PROVENANCE,
                      generated_at=datetime.now(timezone.utc).isoformat(),
                      policy_input='literal_public_evidence_without_kind_or_polarity_or_gold',
                      s1_ingestion='fixed_existing_extractor; no policy decision scored in S1',
                      source='llm_local_001.environment.LLMEnvironment over rl_small_pilot.PilotEnv')
        result['decisions'] = sum(step['terminal'] for step in result['steps'])
        result['decision_errors'] = sum(not step['correct'] for step in result['steps'] if step['terminal'])
        # Preserve the old diagnostic under an honest name: it describes the
        # fixed interpreter, not the generated policy's response to a conflict.
        result['symbolic_interpreter_conflicts_detected'] = result.pop('conflicts_detected')
        result['symbolic_interpreter_conflict_detection_rate'] = result.pop('conflict_detection_rate')
        potential = [session for session in result['sessions'] if session['potential_conflict']]
        detected = 0
        for session in potential:
            first = next((g for g in self.generations if g['session_index'] == session['session_index']), None)
            parsed = (first or {}).get('parsed') or {}
            action = parsed.get('conversation_action')
            detected += action in ('PRESERVE_CONFLICT', 'CONDITIONAL_ANSWER') or (
                action == 'CLARIFY' and first.get('executed_question_slot') == 'preference')
        result['conflicts_detected'] = detected
        result['conflicts_not_detected'] = len(potential) - detected
        result['conflict_detection_rate'] = detected / len(potential) if potential else None
        result['conflict_detection_proxy'] = ('First generated action per evaluator-labelled potential-conflict session: '
            'preserve/conditional or a text-routed preference clarification; behaviour proxy, not internal cognition.')
        result['free_response_text_evaluation'] = ('Correctness is scored on the generated structured polarity and '
            'its evidence grounding; the free-text answer is retained but not independently semantically judged.')
        return result


def run_episode(scenario, generator, representation='O2', condition='ORACLE', db_path=None):
    """generator(prompt)->generated JSON text; no policy fallback is permitted."""
    env = LLMEnvironment(scenario, representation, condition, db_path)
    started = monotonic()
    try:
        observation = env.reset()
        for _ in range(4):  # S2, optional one clarification followup, and S3.
            raw = generator(format_prompt(observation))
            observation, _, done, truncated, _ = env.step(raw)
            if done or truncated:
                break
        else:
            raise RuntimeError('The existing three-session action budget was exceeded')
        result = env.diagnostic_metrics()
        result['elapsed_seconds'] = monotonic() - started
        return result
    finally:
        env.close()


def serialize_rule_action(action, observation):
    """Convert the unchanged conventional baseline output into the shared schema."""
    conversation = action['conversation']
    questions = {
        'preference': '¿Cambió tu preferencia de forma permanente o fue una excepción temporal?',
        'meal': '¿Para qué comida: desayuno, almuerzo, cena o merienda?',
        'day': '¿Es para un día laborable o fin de semana?',
        'source': '¿Cuál es la fuente de esa información?'}
    answer = {'polarity': action['polarity'], 'text': 'Te gusta.' if action['polarity'] == 'LIKE' else 'No te gusta.'} if conversation == 'ANSWER' else None
    result = {'conversation_action': conversation,
              'question': questions.get(action.get('slot')) if conversation == 'CLARIFY' else None,
              'answer': answer, 'memory_action': 'ADD' if action['memory_operation'] == 'ADD_MEMORY' else action['memory_operation'],
              'evidence_ids': [fact['evidence_id'] for fact in observation.get('memory', [])]}
    if conversation == 'CLARIFY':
        result['question_slot'] = action.get('slot', 'preference')
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def historical_rule_policy(observation):
    """Existing B4 rules and conventional parser; no private simulator labels.

    Literal prompts omit interpreted annotations. The conventional baseline may
    compute them itself using the already existing preference extractor, just as
    the LLM may interpret the same text. Unparsed evidence is retained as unknown,
    never silently replaced by an expected action or a family-specific label.
    """
    from simulator.preferences import extract_preferences
    from rl_observability.policies import HistoricalClarificationRules
    from rl_observability.observations import evidence_kind
    value = deepcopy(observation)
    by_id = {}
    for fact in value.get('memory', []):
        if 'polarity' not in fact:
            event = {'message_id': fact['evidence_id'], 'text': fact['text'], 'at': fact['observed_at'], 'source': fact['source']}
            extraction = extract_preferences(event, 'public-baseline-parser')
            parsed = [asdict(item) for item in (*extraction.preferences, *extraction.restrictions)]
            actual = next((candidate for candidate in parsed if candidate['food_item'] == fact['food']), None)
            fact['polarity'] = actual.get('polarity', 'FORBID') if actual else 'NEUTRAL'
            fact['kind'] = evidence_kind(fact) if actual else 'UNRESOLVED'
        by_id[fact['memory_id']] = fact
    value['incoming_evidence'] = [by_id[fact['memory_id']] for fact in value.get('incoming_evidence', []) if fact['memory_id'] in by_id]
    return serialize_rule_action(HistoricalClarificationRules().select_action(value), value)


def run_historical_episode(scenario, representation='O2', condition='ORACLE', db_path=None):
    """Undegraded native conventional B4 on the existing observation interface.

    The extra native polarity/kind fields are literal extractor annotations the
    historical system already uses. This reference is explicitly a conventional
    pipeline with its own interpreter, never an equal-capacity LLM comparison.
    """
    from rl_observability.policies import HistoricalClarificationRules
    env = LLMEnvironment(scenario, representation, condition, db_path)
    started = monotonic()
    try:
        env.reset()
        rules = HistoricalClarificationRules()
        for _ in range(4):
            internal_observation = deepcopy(env._observation)
            raw = serialize_rule_action(rules.select_action(internal_observation), internal_observation)
            _, _, done, truncated, _ = env.step(raw)
            if done or truncated:
                break
        else:
            raise RuntimeError('Historical baseline exceeded the existing action budget')
        result = env.diagnostic_metrics()
        result.update(method='B4', elapsed_seconds=monotonic() - started,
                      policy_input='existing_conventional_literal_extraction_and_historical_rules',
                      equal_capacity_llm_comparison=False)
        return result
    finally:
        env.close()
