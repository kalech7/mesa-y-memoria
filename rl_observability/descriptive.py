"""Post-inference descriptive action aggregation; never changes a checkpoint."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil


def summarize_actions(run):
    run=Path(run)
    original=run/'test'/'metrics.json'
    metrics=deepcopy(json.loads(original.read_text()))
    seeds=json.loads((run/'config.json').read_text())['seeds']
    for representation,methods in metrics.items():
        for prefix in ('B3','B4'):
            counts=Counter()
            for seed in seeds:
                for action,count in methods[f'{prefix}_seed{seed}']['actions_proposed'].items():
                    counts[action]+=count/len(seeds)
            mean=methods[f'{prefix}_seed_mean']
            mean['actions_proposed']=dict(counts)
            mean['actions_proposed_aggregation']='mean number of actions across training seeds, same users'
    out=run/'descriptive_analysis'
    out.mkdir(exist_ok=True)
    (out/'test_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    shutil.copyfile(Path(__file__),out/'source_snapshot.py')
    (out/'manifest.json').write_text(json.dumps({
        'input_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'purpose':'Seed-mean action counts use all seeds, not the reference trajectory retained in an averaged numeric row.',
        'numeric_metrics_changed':False,'policies_changed':False,'inference_reexecuted':False,
        'historical_and_raw_results_retained':True},indent=2)+'\n')
    return metrics
