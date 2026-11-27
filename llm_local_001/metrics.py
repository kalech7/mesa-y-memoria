"""Complete-user bootstrap, paired seed aggregation, and observable errors."""
from collections import defaultdict
from statistics import fmean

from rl_feasibility.statistics import bootstrap_mean


def value(row, field):
    if field == 'errors':
        return sum(not step['correct'] for step in row['steps'] if step['terminal'])
    if field == 'decisions':
        # Each actual generated action, including malformed and clarification.
        return len(row['steps'])
    if field == 'terminal_decisions':
        return sum(bool(step['terminal']) for step in row['steps'])
    return row.get(field, 0)


def summarize(rows, repetitions=1000):
    if not rows:
        raise ValueError('Cannot summarize an unevaluated configuration')
    # A user may have multiple seeds/trajectories: average within user first.
    groups = defaultdict(list)
    for row in rows:
        groups[row['user_id']].append(row)
    fields = ('return', 'future_accuracy', 'error_rate', 'format_errors', 'questions',
              'necessary_asks', 'unnecessary_asks', 'incorrect_updates', 'extraction_errors',
              'temporal_decision_errors', 'unsupported_answers', 'question_routing_errors',
              'conflicts_not_detected', 'elapsed_seconds', 'prompt_tokens', 'completion_tokens',
              'generation_truncations')
    result = {field: bootstrap_mean([fmean(value(r, field) for r in group)
                                    for group in groups.values()], repetitions)
              for field in fields}
    counts = {field: sum(value(row, field) for row in rows)
              for field in ('decisions', 'terminal_decisions', 'errors', 'format_errors',
                            'necessary_asks', 'unnecessary_asks', 'incorrect_updates', 'extraction_errors',
                            'temporal_decision_errors', 'conflicts_not_detected')}
    return {'users': len(groups), 'episodes_including_seeds': len(rows), 'counts': counts,
            'metrics': result, 'confidence_unit': 'complete user, seeds averaged within user',
            'error_rate_denominator': 'all terminal S2/S3 decisions, including malformed actions',
            'decision_count_includes_clarifications': True}


def paired(left, right, repetitions=1000):
    def index(rows):
        indexed = defaultdict(list)
        for row in rows:
            indexed[row['user_id']].append(row)
        return indexed
    a, b = index(left), index(right)
    if not a or a.keys() != b.keys():
        raise ValueError('Matched complete users required for paired comparison')
    return {field: bootstrap_mean([fmean(value(r, field) for r in a[uid]) -
                                  fmean(value(r, field) for r in b[uid])
                                  for uid in sorted(a)], repetitions)
            for field in ('return', 'error_rate', 'future_accuracy')}


def error_rows(rows):
    """Conservative candidate causes; do not attribute fixed parser errors to LLM."""
    output = []
    for row in rows:
        for step in row['steps']:
            if step.get('correct') is not False:
                continue
            causes = []
            if step.get('format_error') or step.get('generated_response', {}).get('format_error'):
                causes.append('response_format')
            if row['representation'] == 'O1':
                causes.append('restricted_historical_exposure_possible')
            if step.get('incorrect_updates'):
                causes.append('memory_operation')
            if row.get('condition') == 'PIPELINE' and row.get('extraction_errors'):
                causes.append('extraction_discrepancy_possible')
            if step.get('unsupported_answer'):
                causes.append('unsupported_answer_or_temporal_interpretation')
            output.append({'user_id': row['user_id'], 'family': row['family'],
                           'seed': row.get('seed'),
                           'session_index': step['session_index'],
                           'causes': causes or ['clarification_or_response_decision'],
                           'causal_status': 'candidate causes; human review pending',
                           'reward': step['reward'],
                           'proposed_action': step['proposed_action']})
    return output
