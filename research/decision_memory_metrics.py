"""Post-action diagnostics; these private provenance labels never enter policy.

Neutral means absence of any authored user-fact provenance, not absence of
future reward. A temporary fact is fact-bearing even when future queries occur
after its expiry. Informative ASK measures world-set reduction from the actual
emitted clarification, independently of whether its reply was retained.
"""
from __future__ import annotations

import math

from research.budget_memory_env import ASK_ACTIONS, WRITE_ACTIONS, _compatible_worlds, score_trace

VERSION = 'decision_memory_metrics_v1'


def sampling_metrics(trace):
    """Gate probabilities only from actual sampled actor calls, never forced forks.

    Call this on the sanitized on-policy visitation path for training plots.
    Greedy evaluation and replayed/forced counterfactual records are excluded.
    """
    values = {'ask': [], 'write': []}
    conditionals = {'ask': [], 'write': []}
    for decision in trace['decisions']:
        metadata = decision.get('metadata', {})
        if (metadata.get('sampled') is not True or len(decision['candidates']) <= 1
                or metadata.get('forced_action') is True
                or metadata.get('credit_role') in {'forced_counterfactual', 'replayed_prefix'}):
            continue
        gate = metadata.get('gate_logprobs')
        conditional = metadata.get('conditional_logprobs')
        if gate is None:  # Original LM-manager controls do not expose binary gates.
            continue
        if (not isinstance(gate, list) or len(gate) != 2
                or any(not math.isfinite(value) for value in gate)
                or not math.isclose(math.fsum(math.exp(value) for value in gate), 1., abs_tol=1e-6)):
            raise ValueError('Invalid recorded binary gate distribution')
        phase = decision['phase']
        if phase not in values:
            raise ValueError('Unknown sampled decision phase')
        values[phase].append(math.exp(gate[1]))
        if conditional is not None:
            if (not isinstance(conditional, list) or len(conditional) != 2
                    or any(not math.isfinite(value) for value in conditional)
                    or not math.isclose(math.fsum(math.exp(value) for value in conditional), 1., abs_tol=1e-6)):
                raise ValueError('Invalid recorded binary conditional distribution')
            conditionals[phase].append(math.exp(conditional[0]))
    result = {}
    for phase, name in [('ask','ask'), ('write','store')]:
        result[f'sampled_{name}_gate_count'] = len(values[phase])
        if values[phase]:
            result[f'sampled_{name}_gate_probability'] = math.fsum(values[phase])/len(values[phase])
        if conditionals[phase]:
            result[f'sampled_{name}_first_conditional_probability'] = math.fsum(conditionals[phase])/len(conditionals[phase])
    return result


def _provenance(episode):
    """Union over authored worlds, without choosing a real world or query target."""
    supported_ids = set()
    scopes = {}
    for world in episode['private']['worlds']:
        for fact in world['facts']:
            for group in fact['support_sets']:
                for identifier in group:
                    supported_ids.add(identifier)
                    scopes.setdefault(identifier,set()).add(fact['scope_kind'])
    literals = {}
    for session in episode['sessions']:
        for turn in session['exchange']:
            if turn['role'] == 'user':
                literals.setdefault(turn['turn_id'],set()).add(turn['text'])
    for world in episode['private']['worlds']:
        for replies in world['replies'].values():
            for turn in replies.values():
                if turn['role'] == 'user':
                    literals.setdefault(turn['turn_id'],set()).add(turn['text'])
    return supported_ids, scopes, literals


def metrics(trace, episode):
    """Numeric diagnostic fields plus the unchanged deterministic score_trace.

    No oracle action is inferred. These classifications are for reports only;
    they must not be passed to the manager, branch selector, optimizer or judge.
    """
    if trace.get('outcome_not_evaluated'):
        raise ValueError('Visitation has no measured outcomes; use sampling_metrics instead')
    if trace.get('episode_id') != episode.get('episode_id'):
        raise ValueError('Trace and authored episode identities differ')
    decisions = trace['decisions']
    if len(decisions) != 8 or [decision['step_id'] for decision in decisions] != list(range(8)):
        raise ValueError('Metrics require the complete four-session decision path')
    supported_ids, scopes, literals = _provenance(episode)
    result = {key:value for key,value in score_trace(trace).items() if type(value) in (int,float)}
    result.update(noop_count=0, informative_ask_count=0, uninformative_ask_count=0,
                  ask_worlds_removed=0, ask_information_gain_bits=0.,
                  neutral_write_count=0, neutral_noop_count=0,
                  fact_write_count=0, fact_noop_count=0,
                  temporal_fact_write_count=0, temporal_fact_noop_count=0)
    for session in range(1,5):
        result[f'ask_session_{session}_count'] = 0
        result[f'write_session_{session}_count'] = 0
        result[f'noop_session_{session}_count'] = 0
    for action in ASK_ACTIONS[1:]:
        result[action.lower()+'_count'] = 0
    for action in WRITE_ACTIONS[1:]:
        result[action.lower()+'_count'] = 0
    dialogue = trace['dialogue']
    by_id = {turn['turn_id']:index for index,turn in enumerate(dialogue)}
    if len(by_id) != len(dialogue):
        raise ValueError('Dialogue turn IDs are not unique')
    counted_asks = counted_writes = 0
    for decision in decisions:
        phase, action, index = decision['phase'], decision['action'], decision['session_index']
        session = index+1
        if phase == 'ask':
            if action not in ASK_ACTIONS:
                raise ValueError('Unknown ASK phase action')
            if action == 'PROCEED':
                continue
            counted_asks += 1
            result[f'ask_session_{session}_count'] += 1
            result[action.lower()+'_count'] += 1
            q_index = by_id[f's{session}a_question']
            reply_index = by_id[f's{session}u_reply']
            if reply_index != q_index+1:
                raise ValueError('Clarification reply must immediately follow its emitted question')
            before = len(_compatible_worlds(episode, dialogue[:q_index]))
            after = len(_compatible_worlds(episode, dialogue[:reply_index+1]))
            if after > before:
                raise ValueError('Emitted clarification expanded the compatible-world set')
            informative = after < before
            result['informative_ask_count' if informative else 'uninformative_ask_count'] += 1
            result['ask_worlds_removed'] += before-after
            result['ask_information_gain_bits'] += math.log2(before/after)
        elif phase == 'write':
            if action not in WRITE_ACTIONS:
                raise ValueError('Unknown WRITE phase action')
            users = {turn['turn_id'] for turn in decision['observation']['exchange']
                     if turn['role']=='user' and turn['text'] in literals.get(turn['turn_id'],set())}
            fact_ids = users & supported_ids
            neutral = not fact_ids
            temporal = any(scopes[identifier] & {'session','interval'} for identifier in fact_ids)
            suffix = 'noop' if action == 'NOOP' else 'write'
            result[f'{"neutral" if neutral else "fact"}_{suffix}_count'] += 1
            if temporal:
                result[f'temporal_fact_{suffix}_count'] += 1
            result[f'{suffix}_session_{session}_count'] += 1
            if action == 'NOOP':
                result['noop_count'] += 1
            else:
                counted_writes += 1
                result[action.lower()+'_count'] += 1
        else:
            raise ValueError('Unknown decision phase')
    if counted_asks != trace['ask_count'] or counted_writes != trace['write_count']:
        raise ValueError('Action counters disagree with the measured trace')
    result.update(sampling_metrics(trace))
    return result
