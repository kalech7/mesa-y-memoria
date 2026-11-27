"""Frozen-model development contrasts and no-update GRPO signal probes.

This is not a training command. AI-authored prompt assistance is a separate arm.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
import tempfile
import time

from llm_local_001.model import Generation, artifact_hashes
from .src.artifacts import ROOT, canonical, digest, file_hash, now, write_json
from .src.closure_protocol import source_hashes
from .src.env.episode import EpisodeEnvironment
from .src.evaluation.runner import run_episode
from .src.policy.llm_controller import LLMController, PHASE_LIMITS, SpecLocalPolicy
from .src.policy.rule_controller import evidence_model
from .src.training.episodic_grpo import EpisodeRollout, PhaseSample, group_telemetry
from .src.training.sft import parameter_hash
from .signal_diagnostic import (ACTIONS, GreedyCache, action_id, enumerate_observations,
                                load_versioned_development, options)


ASSISTANCE = {
    "select": "Comprueba el atributo consultado y si falta alcance temporal o contexto. Una preferencia explícita con fecha no requiere preguntar su duración otra vez. Una declaración ambigua puede requerir TEMPORAL_SCOPE para una consulta posterior. No preguntes sobre la cena para resolver una consulta de bebida ya sustentada. Decide solo con esta observación.",
    "respond": "Aplica conjuntamente atributo, contexto, sesión y fechas de cada regla. Respeta las versiones SUPERSEDE. El fin exclusivo no pertenece al intervalo. Una excepción vigente específica conserva la regla general fuera de su alcance. Cita los IDs reales que sustentan la respuesta; si no hay evidencia suficiente conserva la incertidumbre.",
    "write": "Antes de emitir JSON, verifica la cita literal y su turn_id. Conserva contexto y vigencia conjuntamente. Hasta una fecha inclusive significa fin exclusivo al día siguiente. No confundas la fecha actual con una fecha de fin. Una excepción ADD conserva la regla general; una preferencia habitual nueva puede SUPERSEDE las versiones pertinentes. Si la duración sigue desconocida usa MARK_UNRESOLVED, sin inventar una fecha. Una declaración de cena no cambia la bebida. Respeta el máximo de dos operaciones.",
}
METRICS = ("C2", "FJS", "J_primary", "R_long", "R_short", "questions", "invalid_outputs", "unsupported_operations")


def metrics(row):
    return {**{k: row["metrics"][k] for k in ("C2", "FJS", "questions", "invalid_outputs", "unsupported_operations")},
            "J_primary": row["metrics"]["C2"] * row["metrics"]["FJS"],
            **{k: row["rewards_by_component"][k] for k in ("R_long", "R_short")}}


def summarize(rows, episodes):
    """World/realization variants are not extra independent users."""
    by_id = {(e["root_id"], e["latent_world_id"], e["realization_id"]): e for e in episodes}
    grouped = defaultdict(lambda: defaultdict(list))
    strata = {}
    for row in rows:
        key = tuple(row[k] for k in ("root_id", "latent_world_id", "realization_id"))
        episode = by_id[key]
        strata[key[0]] = episode["authoring"]["stratum"]
        grouped[key[0]][key[1]].append(metrics(row))
    roots = {rid: {m: mean(mean(r[m] for r in variants) for variants in worlds.values()) for m in METRICS}
             for rid, worlds in grouped.items()}
    return {"episodes": len(rows), "roots": len(roots),
            "counts": {m: sum(metrics(r)[m] for r in rows) for m in ("C2", "FJS", "J_primary", "questions", "invalid_outputs", "unsupported_operations")},
            "root_weighted": {m: mean(r[m] for r in roots.values()) for m in METRICS},
            "strata": {s: {m: mean(v[m] for rid, v in roots.items() if strata[rid] == s) for m in METRICS}
                        for s in sorted(set(strata.values()))}, "by_root": roots}


class BudgetBackend:
    def __init__(self, backend, allowed, limit):
        self.original, self.allowed, self.limit, self.calls = backend, allowed, limit, 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def generate(self, prompt, sample=False):
        request = digest({"prompt": prompt, "max_new_tokens": self.config.max_new_tokens})
        if request not in self.allowed:
            raise ValueError("Unplanned model observation")
        if self.calls >= self.limit:
            raise RuntimeError("Predeclared new-generation budget exhausted")
        self.calls += 1
        return self.original.generate(prompt, sample=sample)


def planned_prefixes(cases, assisted_root, encoder=None):
    builders = [LLMController(object()), LLMController(object(), prompt_root=assisted_root)]
    allowed = set()
    def add(builder, phase, obs):
        prompt = builder.prompt(phase, obs)
        if encoder is not None:
            from .src.policy.llm_controller import MAX_INPUT_TOKENS
            if len(encoder(prompt)) > MAX_INPUT_TOKENS:
                raise ValueError("Planned input exceeds the unchanged context limit")
        allowed.add(digest({"prompt": prompt, "max_new_tokens": PHASE_LIMITS[phase]}))
    with tempfile.TemporaryDirectory(prefix="dev_review_prefixes_") as work:
        for i, case in enumerate(cases):
            for condition in ("O1", "O2"):
                env = EpisodeEnvironment(case["episode"], Path(work) / f"{i}_{condition}_select.sqlite")
                try:
                    for builder in builders if condition == "O2" else builders[:1]:
                        add(builder, "select", env.observe("select", condition))
                finally:
                    env.dispose()
                for j, action in enumerate(ACTIONS):
                    env = EpisodeEnvironment(case["episode"], Path(work) / f"{i}_{condition}_{j}.sqlite")
                    try:
                        if action.startswith("ASK:"):
                            env.ask(action.split(":", 1)[1])
                        for phase in ("respond", "write"):
                            for builder in builders if condition == "O2" else builders[:1]:
                                add(builder, phase, env.observe(phase, condition, "I2" if action.startswith("ASK:") else None))
                        if condition == "O2":
                            add(builders[0], "write", env.observe("write", condition, "I1"))
                    finally:
                        env.dispose()
    return allowed


def probe_episodes(episodes):
    groups = defaultdict(set)
    for e in episodes:
        groups[e["authoring"]["stratum"]].add(e["root_id"])
    selected = {rid for roots in groups.values() for rid in sorted(roots)[:2]}
    return sorted([e for e in episodes if e["root_id"] in selected and e["realization_id"] == "v0"],
                  key=lambda e: (e["root_id"], e["latent_world_id"]))


def run(directory, device="cuda"):
    directory = Path(directory).resolve()
    study = json.loads((directory / "study_plan.json").read_text())
    baseline_dir = directory / "paired_b1"
    completion = json.loads((baseline_dir / "completion.json").read_text())
    if completion.get("status") != "executed" or completion.get("training") is not False:
        raise ValueError("Complete the frozen B1 action matrix first")
    for name, expected in completion["artifacts_sha256"].items():
        if file_hash(baseline_dir / name) != expected:
            raise ValueError("Baseline artifact changed: " + name)
    baseline_plan = json.loads((baseline_dir / "plan.json").read_text())
    episodes, bank = load_versioned_development(ROOT.parent / study["bank"])
    if file_hash(bank) != study["bank_sha256"]:
        raise ValueError("Study bank changed")
    output = directory / "contrasts"
    output.mkdir()
    started = time.monotonic()
    prompt_root = output / "assisted_prompts"
    prompt_root.mkdir()
    for phase in ("select", "respond", "write", "reader"):
        text = (ROOT / "prompts" / f"{phase}.txt").read_text()
        (prompt_root / f"{phase}.txt").write_text(text.rstrip() + "\n" + ASSISTANCE.get(phase, "") + "\n")
    write_json(output / "assistance.json", {"source": "AI-authored generic checklist", "instructions": ASSISTANCE,
        "not_RL": True, "hidden_information": False, "human_ratings": False})
    sources = {**source_hashes(), "research/signal_diagnostic.py": file_hash(Path(__file__).with_name("signal_diagnostic.py")),
               "research/development_review.py": file_hash(__file__)}
    write_json(output / "sources.json", sources)
    cases = enumerate_observations(episodes)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT.parent / study["checkpoint"], local_files_only=True)
    lengths = []
    def encode(prompt):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
        lengths.append(len(ids))
        return ids
    allowed = planned_prefixes(cases, prompt_root, encode)
    sampled_episodes = probe_episodes(episodes)
    assisted_hashes = {p.name: file_hash(p) for p in prompt_root.iterdir()}
    write_json(output / "plan.json", {"created_at": now(), "study_plan_sha256": file_hash(directory / "study_plan.json"),
        "training": False, "optimizer_steps": 0, "episodes": len(episodes), "observable_states": len({c["state_id"] for c in cases}),
        "greedy_prefixes_allowed": len(allowed), "maximum_input_tokens": max(lengths), "sampled_group_cases": len(sampled_episodes),
        "sampled_roots": sorted({e["root_id"] for e in sampled_episodes}), "temperatures": study["selector_temperatures"],
        "draws_per_state": study["selector_draws_per_state"], "assistance_sha256": digest(ASSISTANCE),
        "assisted_prompt_hashes": assisted_hashes})
    baseline_rows = [json.loads(line) for line in (baseline_dir / "diagnostico_50.jsonl").open()]
    natural = {r["diagnostic_case_id"]: r for r in baseline_rows if r["action_id"] == "NATURAL"}
    forced = {(r["diagnostic_case_id"], r["action_id"]): r for r in baseline_rows if r["writer"] == "B1" and r["action_id"] in ACTIONS}
    references = {(r["diagnostic_case_id"], r["action_id"]): r for r in baseline_rows if r["writer"] == "reference"}
    analysis = json.loads((baseline_dir / "analisis.json").read_text())
    values = {s["observable_state_id"]: s for s in analysis["states"]}
    checkpoint = ROOT.parent / study["checkpoint"]
    backend = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=False, device=device)
    if any(p.requires_grad for p in backend.model.parameters()):
        raise ValueError("The model must remain frozen")
    before = parameter_hash(backend)
    budget = BudgetBackend(backend, allowed, study["maximum_new_generations"] - completion["calls"]["actual_generations"])
    namespace = digest({"checkpoint": artifact_hashes(checkpoint), "generation": baseline_plan["generation"]})
    cached = GreedyCache(budget, namespace, output / "generation_cache", allowed)
    for path in (baseline_dir / "generation_cache").glob("*.json"):
        entry = json.loads(path.read_text())
        if entry["namespace"] != namespace:
            raise ValueError("Cache model/distribution mismatch")
        cached.cache[entry["key"]] = Generation(**entry["generation"])
    states = output / "states"
    states.mkdir()
    arms = defaultdict(list)
    pair_checks = []
    with (output / "trajectories.jsonl").open("x") as stream:
        for i, case in enumerate(cases):
            episode, cid = case["episode"], case["case_id"]
            for arm, kwargs, controller in (
                ("B1_assisted_prompt", {}, LLMController(cached, prompt_root=prompt_root)),
                ("B1_O1", {"condition": "O1"}, LLMController(cached)),
                ("B1_I1", {"intervention": "I1", "forced_slot": evidence_model(case["select_observation"])["missing_slot"] or "TEMPORAL_SCOPE"}, LLMController(cached)),
            ):
                row = run_episode(episode, controller, rollout_seed=101, method=arm,
                                  db_path=states / f"{cid}_{arm}.sqlite", **kwargs)
                row["review_arm"], row["diagnostic_case_id"] = arm, cid
                stream.write(canonical(row) + "\n")
                stream.flush()
                arms[arm].append(row)
                if arm == "B1_I1":
                    action = "ASK:" + kwargs["forced_slot"]
                    i2 = forced[(cid, action)]
                    assert row["trace"]["response"] == i2["trace"]["response"]
                    for part in ("initial_memory", "user_reply"):
                        assert row["trace"][part] == i2["trace"][part]
                    assert row["trace"]["observations"]["respond"] == i2["trace"]["observations"]["respond"]
                    assert row["trace"]["observations"]["write"]["user_messages"] == episode["user_messages"]
                    arms["B1_I2_paired"].append(i2)
                    pair_checks.append({"case_id": cid, "slot": kwargs["forced_slot"], "same_response": True,
                        "I1": metrics(row), "I2": metrics(i2)})
            write_json(output / "progress.json", {"stage": "controlled_contrasts", "cases": i+1,
                "total_cases": len(cases), "new_calls": budget.calls, "cache_hits": cached.hits})
            print(canonical({"stage": "contrasts", "case": i+1, "total": len(cases), "new_calls": budget.calls}), flush=True)
    arms["B1_O2"] = list(natural.values())
    # An invalid SELECT is retained in the comparator denominator unchanged;
    # replacing WRITE cannot retroactively repair an invalid earlier action.
    arms["B1_reference_writer_natural_action"] = [references[(cid, r["natural_action_id"])]
        if r["natural_action_id"] in ACTIONS else r for cid, r in natural.items()]
    summaries = {arm: summarize(rows, episodes) for arm, rows in arms.items()}
    write_json(output / "controlled_results.json", summaries)
    write_json(output / "PI2_pair_checks.json", pair_checks)
    selector_rows = []
    unique_states = {c["state_id"]: c for c in cases}
    with (output / "selector_samples.jsonl").open("x") as stream:
        for ti, temperature in enumerate(study["selector_temperatures"]):
            backend.config.temperature = temperature
            for si, (state, case) in enumerate(sorted(unique_states.items())):
                for draw in range(study["selector_draws_per_state"]):
                    seed = 85100000 + ti * 100000 + si * 10 + draw
                    backend.seed(seed)
                    controller = LLMController(budget, sample=True)
                    error = None
                    try:
                        selected = controller.select(deepcopy(case["select_observation"]))
                    except ValueError as exc:
                        selected, error = {}, str(exc)
                    action = action_id(selected)
                    av = values[state]["action_values"]
                    actual = av.get(action)
                    baseline_value = av.get(values[state]["B1_action"])
                    row = {"temperature": temperature, "observable_state_id": state, "draw": draw, "seed": seed,
                           "action": action, "error": error, "phase_records": controller.serialized_records(),
                           "R_long_greedy_continuation": actual["R_long"] if actual else None,
                           "better_than_B1_greedy": bool(actual and baseline_value and actual["R_long"] > baseline_value["R_long"] + 1e-9),
                           "best_action_visited": bool(actual and actual["R_long"] >= max(v["R_long"] for v in av.values()) - 1e-9),
                           "regret": max(v["R_long"] for v in av.values()) - actual["R_long"] if actual else None}
                    selector_rows.append(row)
                    stream.write(canonical(row) + "\n")
                    stream.flush()
                write_json(output / "progress.json", {"stage": "selector_sampling", "temperature": temperature,
                    "states": si+1, "total_states": len(unique_states), "new_calls": budget.calls})
            print(canonical({"stage": "selector_temperature_complete", "temperature": temperature, "new_calls": budget.calls}), flush=True)
    selector_summary = {}
    for temperature in study["selector_temperatures"]:
        rows = [r for r in selector_rows if r["temperature"] == temperature]
        selector_summary[str(temperature)] = {"samples": len(rows), "invalid": sum(r["action"] == "INVALID" for r in rows),
            "histogram": dict(Counter(r["action"] for r in rows)),
            "states_visiting_improvement": len({r["observable_state_id"] for r in rows if r["better_than_B1_greedy"]}),
            "states_visiting_optimal": len({r["observable_state_id"] for r in rows if r["best_action_visited"]}),
            "mean_regret_valid": mean(r["regret"] for r in rows if r["regret"] is not None),
            "scope": "Sampled SELECT only; action values average hidden-compatible worlds with greedy frozen continuations, not an on-policy full-return estimate."}
    write_json(output / "selector_results.json", selector_summary)
    group_rows = []
    with (output / "sampled_trajectories.jsonl").open("x") as stream:
        for ti, temperature in enumerate(study["selector_temperatures"]):
            backend.config.temperature = temperature
            for ei, episode in enumerate(sampled_episodes):
                group = []
                for draw in range(4):
                    seed = 95100000 + ti * 100000 + ei * 10 + draw
                    db = states / f"sampled_{ti}_{ei}_{draw}.sqlite"
                    row = run_episode(episode, LLMController(budget, sample=True), rollout_seed=seed,
                                      method="frozen_GRPO_signal_probe", db_path=db)
                    row["temperature"] = temperature
                    stream.write(canonical(row) + "\n")
                    stream.flush()
                    phases = [PhaseSample(p["phase"], Generation(**p["generation"])) for p in row["trace"]["phase_records"]]
                    assert [p.phase for p in phases] == ["select", "respond", "write"]
                    components = row["rewards_by_component"]["components"]
                    group.append(EpisodeRollout(episode["root_id"], episode["latent_world_id"], seed, phases,
                        components["current"], components["future"], row["metrics"]["questions"],
                        row["metrics"]["unsupported_operations"], row["metrics"]["invalid_outputs"], str(db), episode["split"]))
                assert len({tuple(r.phases[0].generation.prompt_ids) for r in group}) == 1
                assert len({r.memory_path for r in group}) == 4 and len({r.rollout_seed for r in group}) == 4
                group_rows.append({"temperature": temperature, "stratum": episode["authoring"]["stratum"],
                    "long": group_telemetry(group, "long"), "short": group_telemetry(group, "short")})
                write_json(output / "progress.json", {"stage": "full_sampling", "temperature": temperature,
                    "groups": len(group_rows), "new_calls": budget.calls})
                print(canonical({"stage": "full_sampled_group", "temperature": temperature,
                    "group": ei+1, "total": len(sampled_episodes), "new_calls": budget.calls}), flush=True)
    write_json(output / "group_diagnostics.json", group_rows)
    group_summary = {str(t): {"groups": sum(r["temperature"] == t for r in group_rows),
        "task_signal_groups": sum(r["temperature"] == t and r["long"]["std_task"] > 0 for r in group_rows),
        "total_signal_groups": sum(r["temperature"] == t and r["long"]["std_total"] > 0 for r in group_rows),
        "semantic_diversity_groups": sum(r["temperature"] == t and r["long"]["semantic_action_diversity"] > 1 for r in group_rows),
        "scope": "Frozen on-policy complete trajectories; no gradient or optimizer step, development-only descriptive sample."}
        for t in study["selector_temperatures"]}
    write_json(output / "group_results.json", group_summary)
    after = parameter_hash(backend)
    assert after == before and not any(p.requires_grad for p in backend.model.parameters())
    assert sources == {**source_hashes(), "research/signal_diagnostic.py": file_hash(Path(__file__).with_name("signal_diagnostic.py")),
                       "research/development_review.py": file_hash(__file__)}
    assert file_hash(bank) == study["bank_sha256"]
    assert assisted_hashes == {p.name: file_hash(p) for p in prompt_root.iterdir()}
    result = {"status": "executed", "training": False, "optimizer_steps": 0, "weights_before": before,
        "weights_after": after, "new_calls": budget.calls, "greedy_cache_hits": cached.hits,
        "combined_new_calls": budget.calls + completion["calls"]["actual_generations"],
        "maximum_new_calls": study["maximum_new_generations"], "PI2_pairs_verified": len(pair_checks),
        "selector_samples": len(selector_rows), "full_sampled_groups": len(group_rows),
        "assistance_is_RL": False, "elapsed_seconds": time.monotonic() - started,
        "artifacts_sha256": {str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file() and p.name not in {"completion.json", "progress.json"}}}
    write_json(output / "completion.json", result)
    write_json(output / "progress.json", {"status": "executed", "new_calls": budget.calls})
    print(canonical({k:v for k,v in result.items() if k != "artifacts_sha256"}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run(args.directory, args.device)


if __name__ == "__main__":
    main()
