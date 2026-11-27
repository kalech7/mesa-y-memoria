"""Pre-training reward checks over validation users, never used as policy input."""
from copy import deepcopy
from statistics import mean
from rl_observability.policies import HistoricalClarificationRules
from .evaluate import evaluate


class FixedPolicy:
    def __init__(self,conversation,polarity=None):
        self.name=conversation+(polarity or '')
        self.conversation,self.polarity=conversation,polarity

    def select_action(self,obs):
        operation='ADD_MEMORY' if obs.get('incoming_evidence') else 'NOOP'
        return dict(conversation=self.conversation,polarity=self.polarity,
                    slot=None,memory_operation=operation)


class AlwaysAsk:
    name='ALWAYS_ASK_WHEN_BUDGET_THEN_STRONG_RULES'
    def select_action(self,obs):
        if obs.get('remaining_clarifications',0):
            return dict(conversation='CLARIFY',slot='preference',polarity=None,memory_operation='NOOP')
        return HistoricalClarificationRules().select_action(obs)


def audit_rewards(scenarios,representation='O2'):
    policies=[HistoricalClarificationRules(),FixedPolicy('PRESERVE_CONFLICT'),
              FixedPolicy('CONDITIONAL_ANSWER'),FixedPolicy('ANSWER','LIKE'),
              FixedPolicy('ANSWER','DISLIKE'),AlwaysAsk()]
    results={}
    for policy in policies:
        rows=evaluate(policy,deepcopy(scenarios),representation)
        results[policy.name]={'users':len(rows),'return':mean(r['return'] for r in rows),
            'error_rate':mean(r['error_rate'] for r in rows),
            'questions':sum(r['questions'] for r in rows),
            'unnecessary_questions':sum(r['unnecessary_asks'] for r in rows)}
    baseline=results[policies[0].name]['return']
    passed=all(result['return']<=baseline+1e-9 for result in results.values())
    return {'passed':passed,'condition':'ORACLE','representation':representation,
        'checks':'Constant preserve, conditional, answer and always-ask do not beat strong rules on balanced validation cases.',
        'results':results,'reward_changed':False,'no_optimality_claim':True}
