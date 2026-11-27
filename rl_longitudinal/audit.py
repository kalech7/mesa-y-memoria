"""Dataset and artifact audits; no output from here is sent to a policy."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _time(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def semantic_signature(episode, *, temporal_ranks=False):
    """Exclude identity/wording; optionally ignore exact inter-event spacing.

    Rank-normalized signatures detect repeated *scripts* despite new calendar
    years and gaps. Food names are canonicalized by first communicated role.
    The private future is examined solely by this offline contamination audit.
    """
    source = deepcopy(episode)
    times = set()
    foods = []
    for session in source['sessions']:
        times.update([session['at'], session['query_at']])
        for fact in session.get('incoming', []):
            for key in ('observed_at', 'recorded_at', 'valid_from', 'valid_to'):
                if fact.get(key):
                    times.add(fact[key])
            if fact['food'] not in foods:
                foods.append(fact['food'])
        for reply in session.get('responses', {}).values():
            fact = reply.get('fact')
            if fact:
                for key in ('observed_at', 'recorded_at', 'valid_from', 'valid_to'):
                    if fact.get(key):
                        times.add(fact[key])
                if fact['food'] not in foods:
                    foods.append(fact['food'])
    instants = sorted({_time(t) for t in times})
    origin = instants[0]
    time_map = {t: instants.index(_time(t)) if temporal_ranks else (_time(t) - origin).total_seconds()
                for t in times}
    identity = {}
    for session in source['sessions']:
        for fact in session.get('incoming', []):
            identity.setdefault(fact['memory_id'], len(identity))
    def food(value):
        return foods.index(value) if value in foods else value
    def fact_record(fact):
        if not fact:
            return None
        return {key: (food(value) if key == 'food' else time_map.get(value) if key in
                      ('observed_at', 'recorded_at', 'valid_from', 'valid_to') and value else
                      [identity.get(i, 'clarification') for i in value] if key == 'supersedes' else value)
                for key, value in fact.items() if key not in ('memory_id', 'user_id', 'evidence_id', 'text')}
    sessions = []
    for session in source['sessions']:
        replies = []
        for key, reply in session.get('responses', {}).items():
            slot, _, item = key.partition(':')
            replies.append([slot, food(item), reply.get('status'), reply.get('value'), fact_record(reply.get('fact'))])
        sessions.append({'at':time_map[session['at']], 'query_at':time_map[session['query_at']],
                         'context':session.get('context', {}), 'target_food':food(session.get('target_food')),
                         'incoming':[fact_record(f) for f in session.get('incoming', [])],
                         'responses':replies,
                         'acceptable_foods':[food(f) for f in session['gold'].get('foods', [])],
                         'acceptable_conversations':session['gold'].get('acceptable_conversations'),
                         'available':session.get('user_available', True)})
    return digest(sessions)


def audit_partitions(partitions):
    summary = {}
    for name, episodes in partitions.items():
        exact = {semantic_signature(e) for e in episodes}
        abstract = {semantic_signature(e, temporal_ranks=True) for e in episodes}
        summary[name] = {'users':len(episodes), 'unique_user_ids':len({e['user_id'] for e in episodes}),
                         'families':dict(Counter(e['family'] for e in episodes)),
                         'session_counts':dict(Counter(len(e['sessions']) for e in episodes)),
                         'unique_native_signatures':len(exact), 'unique_abstract_scripts':len(abstract)}
    overlaps = {}
    names = list(partitions)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            a, b = partitions[left], partitions[right]
            overlaps[f'{left}/{right}'] = {
                'users':len({e['user_id'] for e in a} & {e['user_id'] for e in b}),
                'native_states':len({semantic_signature(e) for e in a} & {semantic_signature(e) for e in b}),
                'abstract_scripts':len({semantic_signature(e, temporal_ranks=True) for e in a} &
                                       {semantic_signature(e, temporal_ranks=True) for e in b}),
                'structural_family_overlap':sorted({e['family'] for e in a} & {e['family'] for e in b})}
    return {'partitions':summary, 'overlaps':overlaps,
            'interpretation':'New IDs/dates do not establish new reasoning families. Abstract overlap is reported, not hidden.'}


def verify_historical_manifest(path):
    expected = json.loads(Path(path).read_text())
    differences = []
    for name, expected_hash in expected.items():
        file = Path(name)
        actual = hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None
        if actual != expected_hash:
            differences.append({'file':name, 'expected':expected_hash, 'actual':actual})
    return {'checked':len(expected), 'unchanged':not differences, 'differences':differences}
