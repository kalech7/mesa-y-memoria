"""Repair exposure-count annotation only; preserve the registered evaluator.

No policy, prompt, reward, observation, seed, checkpoint or metric is changed.
The original source and prospective protocol stay byte-for-byte unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def annotate(rows):
    """Read the real frozen prompt delimiter and allow its fixed suffix."""
    for row in rows:
        counts = []
        for generation in row.get('generations', []):
            text = generation['prompt'].split('\nOBSERVATION:\n', 1)[1]
            public, end = json.JSONDecoder().raw_decode(text)
            if text[end:] != '\nReturn one JSON object now. For ANSWER set question to null.':
                raise ValueError('Unexpected frozen prompt suffix; inspect rather than guessing')
            memory = public.get('memory', [])
            counts.append({'session_index': generation['session_index'],
                           'retrieved_evidence_count': len(memory),
                           'evidence_ids': [fact['evidence_id'] for fact in memory],
                           'incoming_evidence_count': len(public.get('incoming_evidence', []))})
        row['supplement_actual_exposure_counts'] = counts
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    original = repo / 'scripts/evaluate_locked_observability.py'
    spec = importlib.util.spec_from_file_location('registered_observability', original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, protocol, out = args.pilot.resolve(), args.protocol.resolve(), args.output.resolve()
    module.verify_completed(root, protocol)
    out.mkdir(parents=True, exist_ok=False)
    amendment = {'created_at': datetime.now(timezone.utc).isoformat(),
                 'source': str(Path(__file__).resolve()),
                 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'registered_source': str(original),
                 'registered_source_sha256': module.sha256(original),
                 'prospective_protocol': str(protocol), 'protocol_sha256': module.sha256(protocol),
                 'primary_pilot': str(root), 'change': 'Exposure-count metadata annotation only: '
                    'actual OBSERVATION delimiter instead of obsolete OBSERVACION; '
                    'raw_decode the JSON before the fixed frozen suffix.',
                 'registered_source_and_protocol_unchanged': True,
                 'policy_generation_environment_reward_or_metrics_changed': False,
                 'old_partial_run_preserved': str(root / 'supplementary_observability_raw'),
                 'incomplete_base_only_generation_repeated_without_selection': True,
                 'primary_test_results_already_known': True,
                 'all_registered_B0_and_three_B1_configurations_still_mandatory': True}
    module.write_json(out / 'technical_metadata_recovery.json', amendment)
    # This single reporting function is replaced; original.execute and every
    # generation/evaluation/selection gate remain the registered implementation.
    module.annotate = annotate
    result = module.execute(root, protocol, out)
    result['provenance']['technical_metadata_recovery'] = amendment
    module.write_json(out / 'supplementary_observability.json', result)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
