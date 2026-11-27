"""One-seed bounded-memory RL from the provider Instruct, with no local SFT.

Stages are deliberately separate: the public TRAIN pilot is reviewed before
optimization, and validation selection is sealed before reserved inference.
All historical experiments remain immutable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import Counter
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
from statistics import fmean, pstdev
import time

from research.src.artifacts import canonical, digest, file_hash, now, write_json
from research.budget_memory_training import leave_one_out, optimize_group, parameter_hashes

BASE = Path('models/Qwen2.5-0.5B-Instruct')
SEED = 101
GROUP_SIZE = 4


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()


def release(*objects):
    import torch
    for obj in objects:
        if obj is not None:
            backend = getattr(obj, 'backend', obj)
            backend.model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()


def source_hashes():
    paths = [Path(__file__), *Path(__file__).parent.glob('budget_memory_*.py'),
             Path('llm_local_001/model.py'), Path('research/src/policy/llm_controller.py')]
    return {str(p.resolve()): file_hash(p) for p in sorted(set(paths))}


def check_sources(plan):
    if source_hashes() != plan['source_sha256']:
        raise ValueError('Prospective source files changed after the pilot plan')
    for path, checksum in plan['data_sha256'].items():
        if file_hash(path) != checksum:
            raise ValueError('Dataset changed after the pilot plan')
    for path, checksum in plan.get('base_artifacts_sha256', {}).items():
        if file_hash(BASE / path) != checksum:
            raise ValueError('Provider model artifact changed')


def balanced_cohort(episodes, count):
    families = {}
    for episode in episodes:
        families.setdefault(episode['family'], []).append(episode)
    for family in families:
        random.Random(SEED).shuffle(families[family])
    output = []
    keys = sorted(families)
    for index in range(count):
        family = keys[index % len(keys)]
        group = families[family]
        output.append(deepcopy(group[(index // len(keys)) % len(group)]))
    return output


def fresh_manager(*, trainable=False, device='cuda'):
    from research.budget_memory_policy import BudgetMemoryPolicy
    obj = BudgetMemoryPolicy(BASE, device=device, precision='bfloat16', lora=True,
                             seed=SEED, gradient_checkpointing=True)
    if trainable:
        obj.enable_training()
    return obj


def fresh_reader(device='cuda'):
    from research.budget_memory_reader_v2 import FrozenReadableBudgetReader
    return FrozenReadableBudgetReader.load(BASE, device=device, precision='bfloat16', seed=SEED,
                                           gradient_checkpointing=False)


class ScriptedManager:
    """Public budget-matched controls; no hidden state or future query access."""
    def __init__(self, mode):
        if mode not in {'fifo', 'noop', 'random', 'semantic'}:
            raise ValueError(mode)
        self.mode, self.writes = mode, 0

    def decide(self, observation, candidates, *, seed, sample, decision_id):
        if self.mode == 'semantic':
            action = self.semantic_action(observation, candidates)
        elif self.mode == 'random':
            action = random.Random(seed).choice(candidates)
        elif 'NOOP' not in candidates:
            action = 'PROCEED'
        elif self.mode == 'noop':
            action = 'NOOP'
        else:
            action = f'STORE_SLOT_{1 + self.writes % 2}'
            self.writes += 1
        if action not in candidates:
            raise ValueError('Scripted action outside the same public catalogue')
        return {'action': action, 'metadata': {'decision_id': decision_id, 'scripted': self.mode}}

    @staticmethod
    def semantic_action(observation, candidates):
        """Frozen hand-written public rule control, deliberately stronger than FIFO."""
        words = {'breakfast_preference': 'desayuno', 'dinner_preference': 'cena',
                 'drink_preference': 'beber'}
        text = ' '.join(turn['text'] for turn in observation['exchange'] if turn['role'] == 'user')
        if 'NOOP' not in candidates:
            if len(candidates) > 1:
                if 'ahora prefiero' in text:
                    return 'ASK_TEMPORAL_SCOPE'
                if 'aquí prefiero' in text:
                    return 'ASK_CURRENT_CONTEXT'
            return 'PROCEED'
        attribute = next((key for key, word in words.items() if word in text), None)
        if attribute is None or 'Solo durante esta sesión' in text or 'No he confirmado' in text:
            return 'NOOP'
        if (('ahora prefiero' in text or 'aquí prefiero' in text) and len(observation['exchange']) < 3
                or 'Ahora no te sé decir.' in text):
            return 'NOOP'
        for slot in observation['slots']:
            if any(words[attribute] in turn['text'] for turn in slot['exchange'] if turn['role'] == 'user'):
                return f'STORE_SLOT_{slot["slot_id"]}'
        empty = next((slot['slot_id'] for slot in observation['slots'] if not slot['exchange']), 1)
        return f'STORE_SLOT_{empty}'


class ReplayAskNoWrite:
    """Demonstrative TRAIN intervention fixing ASK while suppressing writes."""
    def __init__(self, original):
        self.actions = {decision['session_index']: decision['action']
                        for decision in original['decisions'] if decision['phase'] == 'ask'}

    def decide(self, observation, candidates, *, seed, sample, decision_id):
        action = 'NOOP' if 'NOOP' in candidates else self.actions[observation['session_index']]
        if action not in candidates:
            raise ValueError('Matched ASK replay violated the question budget')
        return {'action': action, 'metadata': {'decision_id': decision_id, 'intervention': 'matched_ASK_no_WRITE'}}


def aggregate(episode_rows):
    """Equal root weights, equal family weights; worlds average within roots."""
    groups = {}
    for row in episode_rows:
        groups.setdefault((row['family'], row['root_id']), []).append(row['metrics'])
    if not groups:
        raise ValueError('Cannot aggregate an empty evaluation')
    numeric_keys = [{key for key, value in x.items() if isinstance(value, (int, float))}
                    for values in groups.values() for x in values]
    keys = sorted(set.intersection(*numeric_keys))
    by_root = {root: {key: fmean(float(x[key]) for x in values) for key in keys}
               for (_, root), values in groups.items()}
    family_rows = {}
    for (family, root) in groups:
        family_rows.setdefault(family, []).append(by_root[root])
    by_family = {family: {key: fmean(x[key] for x in values) for key in keys}
                 for family, values in family_rows.items()}
    metrics = {key: fmean(x[key] for x in by_family.values()) for key in keys}
    return {'metrics': metrics, 'by_root': by_root, 'by_family': by_family,
            'episodes': len(episode_rows), 'roots': len(groups)}


def evaluate_policy(manager, reader, episodes, destination, *, method):
    from research.budget_memory_env import run_episode, score_trace
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    data = []
    for index, episode in enumerate(episodes):
        policy = ScriptedManager(manager) if isinstance(manager, str) else manager
        trace = run_episode(policy, reader, episode, SEED * 100000 + index,
                            destination / f'episode_{index:03d}', sample=False)
        row = {'episode_id': episode['episode_id'], 'root_id': episode['root_id'],
               'family': episode['family'], 'method': method, 'metrics': score_trace(trace)}
        append(destination / 'metrics.jsonl', row)
        data.append(row)
        print('evaluation', method, index + 1, '/', len(episodes), flush=True)
    result = aggregate(data)
    result['reader_profile'] = reader.backend.policy_profile()
    result['reader_frozen'] = not any(p.requires_grad for p in reader.backend.model.parameters())
    write_json(destination / 'summary.json', result)
    return result


def paired(reference, candidate):
    keys = ('net_return', 'query_success', 'joint_success', 'ask_count', 'write_count')
    result = {}
    if set(reference['by_root']) != set(candidate['by_root']):
        raise ValueError('Paired evaluation roots differ')
    for key in keys:
        changes = [candidate['by_root'][root][key] - value[key]
                   for root, value in reference['by_root'].items()]
        result[key] = {'mean_difference': fmean(changes),
                       'improved': sum(x > 1e-10 for x in changes),
                       'worse': sum(x < -1e-10 for x in changes),
                       'tied': sum(abs(x) <= 1e-10 for x in changes)}
    return result


def load_bank(data):
    manifest = read(Path(data) / 'MANIFEST.json')
    bank = {role: rows(value['path']) for role, value in manifest['files'].items()}
    for role, value in manifest['files'].items():
        if file_hash(value['path']) != value['sha256']:
            raise ValueError(f'Dataset hash mismatch: {role}')
    return manifest, bank


def feedback_group(run, name, traces):
    from research import budget_memory_reward as feedback
    identifiers = [f'{name}:r{index}' for index in range(len(traces))]
    samples = [{'id': key, 'evidence': feedback.judge_payload(trace)}
               for key, trace in zip(identifiers, traces, strict=True)]
    verdicts = feedback.request_feedback(run, name, samples)
    result = []
    for key, trace, sample in zip(identifiers, traces, samples, strict=True):
        reward = feedback.reward(trace, verdicts[key])
        append(Path(run) / 'AI_FEEDBACK.jsonl',
               {'id': key, 'episode_id': trace['episode_id'], 'evidence_sha256': digest(sample['evidence']),
                'verdict': verdicts[key], 'reward': reward})
        result.append(reward)
    return result


def gpu_preflight(actor, reader, traces):
    """Real categorical backward on the longest collected prompt, zero updates."""
    from research.budget_memory_policy import verify_fresh_lora_zero_delta
    torch = actor.torch
    reader.backend.model.to('cpu')
    torch.cuda.empty_cache()
    candidate = max((decision for trace in traces for decision in trace['decisions']
                     if len(decision['candidates']) > 1),
                    key=lambda decision: len(decision['metadata']['prompt_ids']))
    prompt_ids, candidates = candidate['metadata']['prompt_ids'], candidate['candidates']
    with torch.no_grad(), actor.likelihood_mode():
        before = actor.distribution(prompt_ids, candidates).detach().cpu()
    actor.enable_training()
    verify_fresh_lora_zero_delta(actor)
    weights_before = parameter_hashes(actor)
    actor.model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    reference = fresh_manager(device='cpu')
    try:
        reference.model.to(actor.device)
        with torch.no_grad(), reference.likelihood_mode():
            ref_distribution = reference.distribution(prompt_ids, candidates).detach().cpu()
        reference_delta = float((before - ref_distribution).abs().max())
        if reference_delta > 2e-4:
            raise ValueError('Provider reference differs from the fresh actor')
    finally:
        reference.model.to('cpu')
        del reference
        torch.cuda.empty_cache()
    # Reserve the two Adam moment buffers without constructing an optimizer or
    # taking a step. No training is hidden inside the resource preflight.
    adam_buffers = [torch.zeros_like(parameter) for parameter in actor.model.parameters()
                    if parameter.requires_grad for _ in range(2)]
    adam_bytes = sum(value.numel() * value.element_size() for value in adam_buffers)
    torch.cuda.synchronize()
    started = time.monotonic()
    with actor.likelihood_mode():
        distribution = actor.distribution(prompt_ids, candidates)
        delta = float((before - distribution.detach().cpu()).abs().max())
        if delta > 2e-4:
            raise ValueError('Fresh LoRA changes the original Instruct distribution')
        (-distribution[0]).backward()
    torch.cuda.synchronize()
    backward_seconds = time.monotonic() - started
    gradients = [p.grad for p in actor.model.parameters() if p.requires_grad and p.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    reserved = torch.cuda.max_memory_reserved() / 1024**2
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    actor.model.zero_grad(set_to_none=True)
    weights_after = parameter_hashes(actor)
    if not finite or weights_before != weights_after:
        raise ValueError('Preflight gradients failed or altered weights without an optimizer')
    if free_bytes < 128 * 1024**2:
        raise ValueError('Insufficient measured GPU margin after the Adam reservation')
    del adam_buffers
    return {'status': 'passed', 'prompt_tokens': len(prompt_ids), 'candidates': candidates,
            'peak_allocated_MiB': peak, 'finite_gradients': finite,
            'peak_reserved_MiB': reserved, 'free_MiB_with_adam_buffers': free_bytes / 1024**2,
            'hardware_total_MiB': total_bytes / 1024**2, 'adam_buffers_MiB': adam_bytes / 1024**2,
            'reference_initial_max_logprob_delta': reference_delta,
            'fresh_lora_max_logprob_delta': delta, 'optimizer_steps': 0, 'sft_updates': 0,
            'differentiable_decision_seconds': backward_seconds,
            'weights_unchanged': weights_before == weights_after,
            'scope': 'Longest collected TRAIN decision; actor/reference coexistence and Adam buffers, no optimizer step'}


def pilot(run, data, *, cohort_offset=0):
    from transformers import AutoTokenizer
    from research.budget_memory_data import build_dataset
    from research.budget_memory_env import (run_episode, score_trace, full_history_reader_observation,
                                             score_response)
    run, data = Path(run).resolve(), Path(data).resolve()
    if run.exists():
        raise FileExistsError('A pilot must have a fresh run directory')
    if cohort_offset < 0 or cohort_offset % 5:
        raise ValueError('Pilot offset must preserve the five-family balance')
    if not data.exists():
        tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
        build_dataset(data, tokenizer=tokenizer)
    manifest, bank = load_bank(data)
    from llm_local_001.model import artifact_hashes
    from research.budget_memory_reader_v2 import reader_template_profile
    run.mkdir(parents=True)
    plan = {'created_at': now(), 'stage': 'TRAIN pilot only', 'seed': SEED,
            'base_model': str(BASE), 'project_sft_checkpoint': None, 'sft_updates': 0,
            'base_artifacts_sha256': artifact_hashes(BASE),
            'runtime_versions': {name: importlib.metadata.version(name)
                                 for name in ('torch', 'transformers', 'peft')},
            'reader_template_profile': reader_template_profile(),
            'pilot_groups': 5, 'rollouts_per_group': GROUP_SIZE,
            'data_sha256': {**{value['path']: value['sha256'] for value in manifest['files'].values()},
                            str((data / 'MANIFEST.json').resolve()): file_hash(data / 'MANIFEST.json')},
            'source_sha256': source_hashes(), 'cohort_offset': cohort_offset,
            'cohort': balanced_cohort(bank['train'], cohort_offset + 5)[cohort_offset:],
            'gates': {'min_outcome_variable_groups': 1, 'min_supported_success_queries': 2,
                      'min_full_history_query_success': .60,
                      'min_matched_no_write_outcome_difference': 1e-10,
                      'min_matched_no_write_reader_logprob_delta': 2e-4,
                      'require_zero_updates_and_identical_frozen_weights': True}}
    write_json(run / 'PILOT_PLAN.json', plan)
    project = Path(__file__).resolve().parents[1]
    for source in plan['source_sha256']:
        target = run / 'source_snapshot' / Path(source).relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manager, reader = fresh_manager(), fresh_reader()
    original = parameter_hashes(manager)
    original_reader = parameter_hashes(reader.backend)
    traces, group_rows = [], []
    collection_start = time.monotonic()
    try:
        for group, episode in enumerate(plan['cohort']):
            current = []
            started = time.monotonic()
            for repetition in range(GROUP_SIZE):
                trace = run_episode(manager, reader, episode, SEED * 100000 + group * GROUP_SIZE + repetition,
                                    run / 'pilot' / f'g{group:02d}_r{repetition}', sample=True,
                                    namespace=f'pilot/g{group}/r{repetition}')
                current.append(trace)
            model_seconds = time.monotonic() - started
            labels = feedback_group(run, f'pilot_group_{group:02d}', current)
            utilities = [x['mean_utility'] for x in labels]
            returns = [x['episodic_reward'] for x in labels]
            row = {'group': group, 'family': episode['family'], 'returns': returns,
                   'outcome_utilities': utilities, 'outcome_std': pstdev(utilities),
                   'return_std': pstdev(returns), 'seconds_with_feedback': time.monotonic() - started}
            row['collection_model_seconds'] = model_seconds
            append(run / 'PILOT_GROUPS.jsonl', row)
            print('pilot', row, flush=True)
            traces.extend(current)
            group_rows.append(row)
        frozen_same = original == parameter_hashes(manager)
        full_rows, controls = [], {}
        for index, episode in enumerate(plan['cohort']):
            trace = traces[index * GROUP_SIZE]
            for query_row in trace['queries']:
                observation = full_history_reader_observation(trace, query_row)
                answer = reader.answer(observation, observation['answer_menu'], seed=SEED,
                                       decision_id=f'full_history:{episode["episode_id"]}:{query_row["query_id"]}')
                checks = score_response(episode, trace['dialogue'], observation['slots'],
                                        query_row['query'], answer['answer'])
                success = answer['answer'] != 'DEFER' and all(
                    checks[key] for key in ('accepted_by_user', 'publicly_supported', 'memory_supported'))
                full_rows.append({'episode_id': episode['episode_id'], 'query': query_row['query'],
                                  'answer': answer, 'checks': checks, 'success': success})
        write_json(run / 'FULL_HISTORY_READER.json', full_rows)
        for mode in ('fifo', 'noop'):
            controls[mode] = evaluate_policy(mode, reader, plan['cohort'], run / 'pilot_controls' / mode,
                                             method=mode)
        positive = next((trace for trace in traces if score_trace(trace)['query_success'] > 0), None)
        intervention = {'available': False, 'outcome_changed': False, 'reader_changed': False}
        if positive is not None:
            episode = next(ep for ep in plan['cohort'] if ep['episode_id'] == positive['episode_id'])
            no_write = run_episode(ReplayAskNoWrite(positive), reader, episode, positive['rollout_seed'],
                                   run / 'pilot_controls' / 'matched_no_write', sample=False,
                                   namespace='pilot/matched_ASK_no_WRITE')
            deltas = [max(abs(a-b) for a, b in zip(original_query['reader_metadata']['old_logprobs'],
                                                  new_query['reader_metadata']['old_logprobs'], strict=True))
                      for original_query, new_query in zip(positive['queries'], no_write['queries'], strict=True)]
            responses_changed = any(a['response'] != b['response']
                                    for a, b in zip(positive['queries'], no_write['queries'], strict=True))
            intervention = {'available': True, 'episode_id': positive['episode_id'],
                            'original_seed': positive['rollout_seed'],
                            'selection': 'First positive TRAIN sample, intentionally demonstrative, not an average effect',
                            'original_metrics': score_trace(positive), 'no_write_metrics': score_trace(no_write),
                            'outcome_changed': abs(score_trace(positive)['query_success'] - score_trace(no_write)['query_success']) > 1e-10,
                            'reader_logprob_deltas': deltas, 'responses_changed': responses_changed,
                            'reader_changed': responses_changed or max(deltas) > 2e-4}
        write_json(run / 'MATCHED_NO_WRITE.json', intervention)
        supported_successes = sum(sum(value == 1 for value in score_trace(trace)['query_utilities'])
                                  for trace in traces)
        gate_values = {'outcome_variable_groups': sum(row['outcome_std'] > 0 for row in group_rows),
                       'supported_success_queries': supported_successes,
                       'full_history_query_success': fmean(row['success'] for row in full_rows),
                       'frozen_weights_unchanged': frozen_same,
                       'reader_weights_unchanged': original_reader == parameter_hashes(reader.backend),
                       'matched_no_write_outcome_changed': intervention['outcome_changed'],
                       'matched_no_write_reader_changed': intervention['reader_changed']}
        passed = (gate_values['outcome_variable_groups'] >= 1 and supported_successes >= 2
                  and gate_values['full_history_query_success'] >= .60 and frozen_same
                  and gate_values['reader_weights_unchanged'] and intervention['outcome_changed']
                  and intervention['reader_changed'])
        gpu = gpu_preflight(manager, reader, traces)
        write_json(run / 'QA_GPU.json', gpu)
        check_sources(plan)
        result = {'status': 'passed' if passed else 'failed_feasibility', 'gates': gate_values,
                  'groups': group_rows, 'controls': controls, 'gpu': gpu,
                  'elapsed_seconds': time.monotonic() - collection_start,
                  'sft_updates': 0, 'optimizer_steps': 0,
                  'heldout_model_calls': 0, 'validation_model_calls': 0}
        write_json(run / 'PILOT_RESULT.json', result)
        print('PILOT_RESULT', result['status'], gate_values, flush=True)
    finally:
        release(manager, reader)


def eligible(metrics, baseline):
    """A cost improvement cannot select a model with fewer correct answers."""
    return metrics['query_success'] + 1e-10 >= baseline['query_success']


def selection_key(metrics):
    return (metrics['net_return'], metrics['query_success'], -metrics['ask_count'])


def better_selection(candidate, incumbent):
    for value, previous in zip(selection_key(candidate), selection_key(incumbent), strict=True):
        if abs(value - previous) > 1e-10:
            return value > previous
    return False


def train(run, data, *, updates=16, learning_rate=3e-5, budget_seconds=3600):
    from research.budget_memory_env import run_episode, score_trace
    from research.budget_memory_policy import verify_budget_profiles, verify_fresh_lora_zero_delta
    import torch
    run = Path(run).resolve()
    pilot_plan, pilot_result = read(run / 'PILOT_PLAN.json'), read(run / 'PILOT_RESULT.json')
    review = read(run / 'PILOT_REVIEW.json')
    if (pilot_result['status'] != 'passed' or review['status'] != 'passed'
            or review.get('pilot_result_sha256') != file_hash(run / 'PILOT_RESULT.json')):
        raise ValueError('Independent prospective pilot review is required before RL')
    check_sources(pilot_plan)
    if updates not in (16, 32) or learning_rate <= 0 or budget_seconds <= 0:
        raise ValueError('Invalid predeclared one-seed schedule')
    _, bank = load_bank(data)
    destination = run / 'optimizer'
    if destination.exists():
        raise FileExistsError('Never resume or overwrite an unreviewed RL branch')
    destination.mkdir()
    plan = {'created_at': now(), 'seed': SEED, 'sft_updates': 0,
            'algorithm': 'fresh_episodic_REINFORCE_leave_one_out_exact_categorical_KL',
            'max_updates': updates, 'rollouts_per_group': GROUP_SIZE,
            'optimizer_steps_per_group': 1, 'learning_rate': learning_rate, 'kl_coefficient': .03,
            'max_gradient_norm': 1., 'budget_seconds': budget_seconds,
            'checkpoint_steps': [0, updates // 2, updates],
            'selection': 'Max validation net_return; no decrease in query_success vs step0; ties query_success then fewer ASK then earlier step',
            'cohort': balanced_cohort(bank['train'], updates),
            'source_sha256': pilot_plan['source_sha256'], 'data_sha256': pilot_plan['data_sha256'],
            'base_artifacts_sha256': pilot_plan['base_artifacts_sha256'],
            'reader_template_profile': pilot_plan['reader_template_profile'],
            'project_sft_checkpoint': None, 'replay_or_demonstrations': False}
    write_json(run / 'TRAIN_PLAN.json', plan)
    actor, reference, reader = fresh_manager(trainable=True), fresh_manager(device='cpu'), fresh_reader()
    verify_budget_profiles(actor, reference)
    verify_fresh_lora_zero_delta(actor)
    initial_actor, initial_reference = parameter_hashes(actor), parameter_hashes(reference)
    if initial_actor['base'] != initial_reference['base']:
        raise ValueError('Provider actor/reference base weights differ')
    write_json(run / 'INITIALIZATION.json', {'actor': initial_actor, 'reference': initial_reference,
               'origin_checkpoint': actor.origin_checkpoint, 'sft_updates': 0,
               'adapter_initial_zero_delta': True, 'profile': actor.policy_profile()})
    optimizer = torch.optim.Adam([p for p in actor.model.parameters() if p.requires_grad], lr=learning_rate)
    actor.save_checkpoint(destination / 'step_000000')
    progress = destination / 'progress.jsonl'
    append(progress, {'stage': 'started', 'step': 0, 'plan': plan})
    history = []
    training_seconds = 0.
    started = time.monotonic()
    try:
        baseline = evaluate_policy(actor, reader, bank['validation'], run / 'validation' / 'step_000000',
                                   method='Instruct-frozen')
        best_metrics, best_step = baseline['metrics'], 0
        baseline_metrics = dict(best_metrics)
        append(progress, {'stage': 'validation_completed', 'step': 0, 'metrics': best_metrics})
        for group, episode in enumerate(plan['cohort'], start=1):
            if training_seconds >= budget_seconds:
                raise TimeoutError('Predeclared training budget exhausted; no incomplete run is reported as complete')
            step_started = time.monotonic()
            reader.backend.model.to(actor.device)
            trajectories = []
            for repetition in range(GROUP_SIZE):
                trace = run_episode(actor, reader, episode, SEED * 1000000 + group * GROUP_SIZE + repetition,
                                    run / 'train_rollouts' / f'g{group:03d}_r{repetition}', sample=True,
                                    namespace=f'train/group_{group}/rollout_{repetition}')
                trajectories.append(trace)
            labels = feedback_group(run, f'train_group_{group:03d}', trajectories)
            returns = [label['episodic_reward'] for label in labels]
            utilities = [label['mean_utility'] for label in labels]
            reader.backend.model.to('cpu')
            torch.cuda.empty_cache()
            diagnostic = optimize_group(actor, reference, trajectories, returns, optimizer, beta=.03)
            training_seconds += time.monotonic() - step_started
            row = {'stage': 'training', 'step': group, **diagnostic,
                   'elapsed_seconds': time.monotonic() - started,
                   'resource_counters': {'updates': group, 'optimizer_steps': group,
                                         'episodes': group * GROUP_SIZE, 'sft_updates': 0,
                                         'training_seconds': training_seconds}}
            append(progress, row)
            action_counts = Counter(tuple(d['action'] for d in trace['decisions']) for trace in trajectories)
            empirical_entropy = -sum((count / GROUP_SIZE) * math.log(count / GROUP_SIZE)
                                     for count in action_counts.values())
            append(destination / 'group_diagnostics.jsonl',
                   {'optimizer_update': group, 'returns': returns, 'task_returns': utilities,
                    'std_task': pstdev(utilities), 'std_total': pstdev(returns),
                    'semantic_action_diversity': len(action_counts),
                    'empirical_action_entropy_nats': empirical_entropy,
                    'entropy_scope': 'empirical four-sample manager action-sequence entropy, not full categorical policy entropy',
                    'advantages': diagnostic['advantages'], 'episode_id': episode['episode_id']})
            raw = [score_trace(trace) for trace in trajectories]
            values = {f'rollout/{key}': fmean(item[key] for item in raw)
                      for key in ('net_return', 'query_success', 'joint_success', 'ask_count', 'write_count')}
            values.update({'rollout/task_std': pstdev(utilities),
                           'rollout/AI_return': fmean(returns), 'rollout/AI_utility': fmean(utilities)})
            append(destination / 'training_metrics.jsonl', {'step': group, 'values': values})
            history.append(row)
            print('training', group, 'loss', row['loss'], 'AI_return', fmean(returns),
                  'task_std', pstdev(utilities), flush=True)
            if group in plan['checkpoint_steps']:
                actor.save_checkpoint(destination / f'step_{group:06d}')
                reader.backend.model.to(actor.device)
                validation = evaluate_policy(actor, reader, bank['validation'],
                                             run / 'validation' / f'step_{group:06d}', method=f'RL-step-{group}')
                metrics = validation['metrics']
                allowed = eligible(metrics, baseline_metrics)
                append(progress, {'stage': 'validation_completed', 'step': group,
                                  'metrics': metrics, 'eligible': allowed})
                if allowed and better_selection(metrics, best_metrics):
                    best_metrics, best_step = metrics, group
        final_actor, final_reference = parameter_hashes(actor), parameter_hashes(reference)
        if initial_actor['base'] != final_actor['base'] or initial_reference != final_reference:
            raise ValueError('Frozen provider base or reference changed')
        result = {'status': 'completed', 'best_step': best_step,
                  'initial_metrics': baseline_metrics, 'selected_metrics': best_metrics,
                  'history': history, 'sft_updates': 0, 'seed': SEED,
                  'resource_counters': {'updates': updates, 'optimizer_steps': updates,
                                        'episodes': updates * GROUP_SIZE, 'training_seconds': training_seconds,
                                        'total_seconds': time.monotonic() - started},
                  'base_unchanged': initial_actor['base'] == final_actor['base'],
                  'reference_unchanged': initial_reference == final_reference,
                  'adapter_changed': initial_actor['adapter'] != final_actor['adapter'],
                  'initial_actor': initial_actor, 'final_actor': final_actor,
                  'initial_reference': initial_reference, 'final_reference': final_reference}
        check_sources(plan)
        write_json(destination / 'training.json', result)
        write_json(run / 'SELECTION.json', {'created_at': now(), 'selection_split': 'validation',
                   'selected_before_evaluation': True, 'selected_step': best_step,
                   'selected_checkpoint': str(destination / f'step_{best_step:06d}'),
                   'baseline_checkpoint': str(destination / 'step_000000'),
                   'data_sha256': plan['data_sha256'], 'source_sha256': plan['source_sha256']})
        append(progress, {'stage': 'completed', 'step': updates, 'selected_step': best_step})
        print('training_completed selected_step', best_step, flush=True)
    finally:
        release(actor, reference, reader)


def evaluate(run, data):
    from research.budget_memory_policy import BudgetMemoryPolicy
    run = Path(run).resolve()
    plan, selection = read(run / 'TRAIN_PLAN.json'), read(run / 'SELECTION.json')
    check_sources(plan)
    if (run / 'COMPARISON.json').exists() or not selection['selected_before_evaluation']:
        raise ValueError('Selection must precede one reserved evaluation')
    _, bank = load_bank(data)
    reader = fresh_reader()
    panels = {}
    selection_sha = file_hash(run / 'SELECTION.json')
    try:
        for method, path in (('Instruct-frozen', selection['baseline_checkpoint']),
                             ('RL-selected', selection['selected_checkpoint'])):
            if method == 'RL-selected' and selection['selected_step'] == 0:
                panels[method] = deepcopy(panels['Instruct-frozen'])
                panels[method]['identical_to'] = 'Instruct-frozen'
                continue
            actor = BudgetMemoryPolicy.from_checkpoint(path, trainable=False, device='cuda')
            try:
                panels[method] = evaluate_policy(actor, reader, bank['evaluation'], run / 'evaluation' / method,
                                                 method=method)
            finally:
                release(actor)
                del actor
        for method, controller in (('FIFO', 'fifo'), ('Random', 'random'), ('Public-rules', 'semantic')):
            panels[method] = evaluate_policy(controller, reader, bank['evaluation'], run / 'evaluation' / method,
                                             method=method)
        if selection_sha != file_hash(run / 'SELECTION.json'):
            raise ValueError('Selection changed during reserved inference')
        check_sources(plan)
        comparison = {'status': 'completed', 'seed': SEED, 'sft_updates': 0,
                      'project_sft_checkpoint': None, 'selected_step': selection['selected_step'],
                      'selection_sha256': selection_sha, 'panels': panels,
                      'paired_vs_Instruct': {method: paired(panels['Instruct-frozen'], panel)
                                             for method, panel in panels.items() if method != 'Instruct-frozen'},
                      'training': read(run / 'optimizer' / 'training.json')['resource_counters'],
                      'scope': 'New bounded-memory bank; do not compare percentage deltas to the historical environment'}
        write_json(run / 'COMPARISON.json', comparison)
        print('comparison_completed', {key: val['metrics'] for key, val in panels.items()}, flush=True)
    finally:
        release(reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'train', 'evaluate'), required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=16)
    parser.add_argument('--pilot-offset', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--budget-seconds', type=int, default=3600)
    args = parser.parse_args()
    if args.stage == 'pilot':
        pilot(args.run, args.data, cohort_offset=args.pilot_offset)
    elif args.stage == 'train':
        train(args.run, args.data, updates=args.updates, learning_rate=args.learning_rate,
              budget_seconds=args.budget_seconds)
    else:
        evaluate(args.run, args.data)


if __name__ == '__main__':
    main()
