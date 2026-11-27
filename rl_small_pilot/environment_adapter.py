"""Minimal SPEC-RL-005 adapter over the existing SQLite environment.

Pre-training design: reuse reward coefficients, action transitions, persistent
history and evaluator. Two fixes are isolated here: balanced preference directions
(the previous generator tied family to polarity), and evaluator resolution from
what the user actually communicated instead of a hard-coded LIKE. Query weekday
is derived from the visible date uniformly, not obtained from simulator state.
These do not change the official thesis question or historical experiments.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from rl_longitudinal.temporal_memory import instant
from rl_observability.environment import MinimalEnv, RewardConfig
from rl_observability.scenarios import generate_scenarios, FAMILIES
from .observations import ACTIONS, action_index

# A full 2x2x2 public context design, with disjoint combinations by partition.
# Shared values and shared five mechanisms remain: this is compositional surface
# generalization, never a claim of independent conceptual problem families.
CONTEXT_COMBINATIONS = {
    'train': (('LUNCH','home','alone'),('DINNER','work','friends'),
              ('LUNCH','work','friends'),('DINNER','home','alone')),
    'validation': (('LUNCH','home','friends'),('DINNER','work','alone')),
    'test': (('LUNCH','work','alone'),('DINNER','home','friends'))}
MEAL_TEXT={'LUNCH':'para el almuerzo','DINNER':'para la cena'}
PLACE_TEXT={'home':'en casa','work':'en el trabajo'}
COMPANY_TEXT={'alone':'a solas','friends':'con mis amigos'}
SURFACES={
 'train': {'statement':'{preference} {food}', 'change':'Desde ahora {preference} {food}',
           'temporary':'Solo hoy {preference} {food}', 'reply':'Confirmo que desde ahora {preference} {food}'},
 'validation': {'statement':'En general, {preference} {food}', 'change':'A partir de ahora {preference} {food}',
           'temporary':'Por hoy {preference} {food}', 'reply':'Confirmo: a partir de ahora {preference} {food}'},
 'test': {'statement':'Habitualmente {preference} {food}', 'change':'He cambiado de opinión: desde ahora {preference} {food}',
           'temporary':'Solamente hoy {preference} {food}', 'reply':'He cambiado de opinión: ahora {preference} {food}'}}

def generate_splits(train_n=600, validation_n=120, test_n=240, seed=51005):
    """Independent IDs/dates/surfaces/context combinations; balanced directions."""
    splits={}
    for split,n,offset in (('train',train_n,0),('validation',validation_n,10000),('test',test_n,20000)):
        scenarios=generate_scenarios(n,seed+offset,split)
        for i,scenario in enumerate(scenarios):
            initial_polarity='LIKE' if (i//len(FAMILIES))%2==0 else 'DISLIKE'
            opposite='DISLIKE' if initial_polarity=='LIKE' else 'LIKE'
            meal,place,company=CONTEXT_COMBINATIONS[split][(i//(2*len(FAMILIES)))%len(CONTEXT_COMBINATIONS[split])]
            context={'meal':meal,'place':place,'company':company}
            scope={'day':'ANY',**context}
            suffix=f" {MEAL_TEXT[meal]} {PLACE_TEXT[place]} {COMPANY_TEXT[company]}."
            family=scenario['family']
            for stage,session in enumerate(scenario['sessions']):
                session['context']=deepcopy(context)
                session['query']=session['query'].rstrip('?')+suffix+' ¿Cuál es mi preferencia?'
                for fact in session['incoming']:
                    if fact['polarity']=='FORBID':
                        continue
                    fact['scope']=deepcopy(scope)
                    if stage==0:
                        fact['polarity']=initial_polarity
                    elif family=='IRRELEVANT_CONFLICT':
                        # Preserve the deliberately opposing, irrelevant facts.
                        pass
                    else:
                        fact['polarity']=opposite
                    key='change' if fact['kind']=='EXPLICIT_CHANGE' else 'temporary' if fact['kind']=='TEMPORARY_EXCEPTION' else 'statement'
                    preference='me gusta' if fact['polarity']=='LIKE' else 'no me gusta'
                    fact['text']=SURFACES[split][key].format(preference=preference,food=fact['food'])+suffix
                session['events']=[dict(message_id=f['evidence_id'],text=f['text'],at=f['observed_at'],source=f['source']) for f in session['incoming']]
                if stage==0:
                    session['gold']['polarity']=initial_polarity
                elif family=='EXPLICIT_CHANGE':
                    session['gold']['polarity']=opposite
                elif family=='TEMPORARY_EXCEPTION':
                    session['gold']['polarity']=opposite if stage==1 else initial_polarity
                elif family=='IRRELEVANT_CONFLICT':
                    session['gold']['polarity']=initial_polarity
                elif family=='RELEVANT_RESOLVABLE':
                    session['gold']['polarity']=opposite
                response=session.get('responses',{}).get('preference')
                if response and response.get('fact'):
                    fact=response['fact']
                    fact['polarity']=opposite
                    fact['scope']=deepcopy(scope)
                    preference='me gusta' if opposite=='LIKE' else 'no me gusta'
                    text=SURFACES[split]['reply'].format(preference=preference,food=fact['food'])+suffix
                    fact['text']=text
                    response['text']=text
            scenario['template_id']=f'{split}:{family}:{meal}:{place}:{company}'
            scenario['context_combination']=[meal,place,company]
            scenario['initial_direction']=initial_polarity
        splits[split]=scenarios
    return splits

class PilotEnv(MinimalEnv):
    """Real SQLite trajectories; private diagnostic data never enter reset/step obs."""
    def __init__(self,scenario,representation='O2',condition='ORACLE',db_path=None,reward_config=None):
        if condition not in ('ORACLE','PIPELINE'):
            raise ValueError('condition must be ORACLE or PIPELINE')
        self.condition=condition
        super().__init__(scenario,representation,'VISIBLE_STRUCTURED' if condition=='ORACLE' else 'PIPELINE',db_path,reward_config)

    def _begin_session(self):
        super()._begin_session()
        weekday=instant(self._session['query_at']).weekday()
        self._session.setdefault('context',{}).setdefault('day','WEEKEND' if weekday>=5 else 'WEEKDAY')

    def _evaluator_expected(self):
        if self._scenario['family'].startswith('RELEVANT_'):
            if not self._resolved_reply:
                return None
            replies=[f for f in self._communicated if f.get('source')=='CLARIFICATION' and f['food']==self._session['target_food']]
            return replies[-1]['polarity'] if replies else None
        return self._session['gold']['polarity']

    def step(self,action):
        # A budget violation may execute an internal NOOP, but conversation NOOP
        # is not a legitimate proposed action in the new pilot.
        action_index(action)
        return super().step(action)

    def _extract(self,session):
        start=len(self._extraction_records)
        result=super()._extract(session)
        for record in self._extraction_records[start:]:
            expected,actual=record.get('declared'),record.get('parsed')
            if not expected or not actual:
                continue
            fields=record['fields']
            expected_scope=expected.get('scope') or {}
            actual_scope=actual.get('scope') or {}
            for key in ('day','meal','company','place'):
                # None and ANY both mean unrestricted in the existing interpreter.
                left=actual_scope.get(key); right=expected_scope.get(key)
                if left not in (None,'ANY') or right not in (None,'ANY'):
                    if left!=right:
                        fields.append('scope.'+key)
            if expected.get('source')!=actual.get('source'):
                fields.append('source')
            record['fields']=sorted(set(fields))
            record['extraction_error']=bool(fields)
            record['temporal_error']=bool(set(fields)&{'kind','valid_from','valid_to','scope.day'})
            at=instant(session['at'])
            def applies(fact):
                return (instant(fact.get('valid_from') or fact['observed_at'])<=at and
                        (not fact.get('valid_to') or at<instant(fact['valid_to'])))
            record['interval_changes_applicability']=applies(actual)!=applies(expected)
        return result

    def diagnostic_metrics(self):
        result=super().diagnostic_metrics()
        result['condition']=self.condition
        result['extraction']=self.condition
        result['policy_errors']=sum(not s['correct'] for s in result['steps'] if s['terminal'])
        result['decisions']=sum(s['terminal'] for s in result['steps'])
        result['extraction_scope_errors']=sum(any(f.startswith('scope.') for f in r['fields']) for r in result['extraction_records']) if self.condition=='PIPELINE' else 0
        result['extraction_source_errors']=sum('source' in r['fields'] for r in result['extraction_records']) if self.condition=='PIPELINE' else 0
        result['extraction_interval_applicability_errors']=sum(r.get('interval_changes_applicability',False) for r in result['extraction_records']) if self.condition=='PIPELINE' else 0
        result['total_decision_errors']=result['policy_errors']
        result['policy_errors_attribution']='Total wrong terminal decisions; not a causal attribution to policy. Consult paired ORACLE/PIPELINE diagnostics.'
        result['extraction_error_denominator']='communicated evidence events; field discrepancies, not decision failures'
        return result
