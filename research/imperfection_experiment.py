"""Separate imperfect-demonstration and observation-fault experiments.

Preparation and the CPU smoke never train weights. Model comparisons use the
existing sealed competence campaign; robustness uses fixed policies and the
unchanged evaluator. All outputs require a new destination.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
from statistics import mean

from .src.artifacts import canonical, digest, file_hash, write_json
from .src.env.imperfections import ImperfectionConfig, normalize_config

VERSION = "imperfection_experiment_v1"
PROFILES = {
    "clean": {},
    "unavailable20": {"unavailable_reply_rate": .2},
    "partial30": {"partial_reply_rate": .3},
    "memorydrop20": {"memory_omission_rate": .2},
    "combined": {"unavailable_reply_rate": .2, "partial_reply_rate": .3, "memory_omission_rate": .2},
}
METRICS = ("C2", "FJS", "J", "questions", "unsupported_write_episode", "invalid_outputs")


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _new_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Preserve previous results; choose a new output directory")
    path.mkdir(parents=True)
    return path


def prepare(data, output, *, rates=(0., .1, .3), seed=7331):
    """Bind audited disjoint partitions to three independent SFT comparisons."""
    from .competence_data import load_and_check
    from .imperfect_sft import prepare_bundle, _configuration
    data = Path(data).resolve()
    splits, _ = load_and_check(data)
    rates = [_configuration(rate, seed)[0] for rate in rates]
    if not rates or len(set(rates)) != len(rates):
        raise ValueError("Provide distinct imperfect-demonstration rates")
    output = _new_output(output)
    arms = []
    for index, rate in enumerate(rates):
        bundle = output / f"arm_{index:02d}"
        manifest = prepare_bundle(splits["train"], bundle, rate=rate, seed=seed,
            heldout_paths={role: data / (role + ".jsonl") for role in ("validation", "evaluation")})
        arms.append({"rate": rate, "rate_unit": manifest["rate_unit"], "bundle_path": str(bundle),
            "manifest_sha256": file_hash(bundle / "manifest.json"), "selected_root_count": manifest["selected_root_count"],
            "changed_target_count": manifest["changed_target_count"],
            "campaign_output": str(output / "campaigns" / f"arm_{index:02d}"),
            "branches": ["B1c_no_continuation", "B3_more_same_SFT", "B2-I", "B2-L"]})
    plan = {"version": VERSION, "status": "prepared_not_trained", "research_status": "exploratory",
        "data": str(data), "data_sha256": {role: digest(rows) for role, rows in splits.items()},
        "arms": arms, "training_seeds": [101, 202, 303], "faults_during_training": None,
        "comparison": "Each arm's continuations start from its own validation-selected B1c; all arms share initial parents",
        "more_sft_demonstrations": "same frozen arm bundle as initial SFT",
        "RL_prerequisites": "Existing capability and 40-group task-signal gates; never relax to force RL",
        "selection": "validation only; seal all arms' selected checkpoints before inspecting any evaluation result",
        "robustness": {"separate_from_demonstration_comparison": True,
            "profiles": {name: ImperfectionConfig(seed=seed, **config).to_dict() for name, config in PROFILES.items()},
            "scope": "S2 observation delivery; S3 reads the complete actual persisted store"},
        "training_steps": 0, "model_calls": 0, "human_review": False,
        "limitations": ["Controlled synthetic mistakes are not measured frequencies of real users.",
            "Configured root corruption proportion does not prescribe the resulting model accuracy.",
            "Free paraphrases and new human conversations need independent semantic validation."]}
    from .src.closure_protocol import source_files
    paths = [Path(__file__), Path(__file__).with_name("imperfect_sft.py"), *source_files(),
             *Path(__file__).parent.glob("competence_*.py")]
    plan["source_sha256"] = {str(p.resolve()): file_hash(p) for p in paths}
    plan["sha256"] = digest(plan)
    write_json(output / "EXPERIMENT.json", plan)
    return plan


def _load_experiment(path):
    path = Path(path).resolve()
    plan = json.loads((path / "EXPERIMENT.json").read_text())
    if (plan.get("version") != VERSION
            or digest({k: v for k, v in plan.items() if k != "sha256"}) != plan.get("sha256")):
        raise ValueError("Experiment design changed")
    if not plan.get("source_sha256") or any(file_hash(p) != h for p, h in plan["source_sha256"].items()):
        raise ValueError("Frozen experiment source changed")
    from .imperfect_sft import load_examples
    train = _rows(Path(plan["data"]) / "train.jsonl")
    for role, expected in plan["data_sha256"].items():
        if digest(_rows(Path(plan["data"]) / (role + ".jsonl"))) != expected:
            raise ValueError("Frozen experiment partition changed: " + role)
    for arm in plan["arms"]:
        if file_hash(Path(arm["bundle_path"]) / "manifest.json") != arm["manifest_sha256"]:
            raise ValueError("Frozen demonstration arm manifest changed")
        load_examples(arm["bundle_path"], train=train, expected_rate=arm["rate"])
    return path, plan


def prepare_campaigns(experiment, *, tests_report, technical_report):
    """Seal existing training plans for every arm without loading a model."""
    from . import competence_campaign as campaign
    path, plan = _load_experiment(experiment)
    if (path / "CAMPAIGNS.json").exists() or any(Path(a["campaign_output"]).exists() for a in plan["arms"]):
        raise FileExistsError("Campaigns already prepared or partially created; preserve their evidence")
    campaigns = []
    for arm in plan["arms"]:
        value = campaign.prepare(plan["data"], arm["campaign_output"], tests_report=tests_report,
            technical_report=technical_report, imperfect_sft_bundle=arm["bundle_path"])
        campaigns.append({"rate": arm["rate"], "run": arm["campaign_output"], "sha256": value["sha256"]})
    result = {"status": "prepared_not_trained", "experiment_sha256": plan["sha256"], "campaigns": campaigns}
    write_json(path / "CAMPAIGNS.json", result)
    return result


def execute(experiment):
    """Run sealed SFT/continuation comparisons; finish all selection before evaluation."""
    from . import competence_campaign as campaign
    from .competence_training import checkpoint_evidence
    path, plan = _load_experiment(experiment)
    campaigns = json.loads((path / "CAMPAIGNS.json").read_text())
    expected = [(a["rate"], a["campaign_output"]) for a in plan["arms"]]
    if (campaigns.get("experiment_sha256") != plan["sha256"]
            or [(a["rate"], a["run"]) for a in campaigns["campaigns"]] != expected):
        raise ValueError("Prepared campaigns differ from frozen experiment arms")
    seal = path / "SELECTED_ALL_ARMS.json"
    if seal.exists():
        raise FileExistsError("Selection already sealed; do not automatically reopen evaluation")
    for arm in campaigns["campaigns"]:
        if campaign.verify_plan(arm["run"])["sha256"] != arm["sha256"]:
            raise ValueError("Campaign preregistration changed")
        if any((Path(arm["run"]) / name).exists() for name in (
                "SELECTED_CHECKPOINTS.json", "EVALUATION_COMPLETION.json", "evaluation")):
            raise ValueError("An arm was already evaluated before the common selection barrier")
    lock = path / "execution.lock"
    # Exclusive creation prevents two processes from updating the same new run.
    with lock.open("x") as stream:
        stream.write("imperfection comparison in progress\n")
    try:
        selected, selection_evidence = [], {}
        for arm in campaigns["campaigns"]:
            run = Path(arm["run"])
            campaign.adapt(run)
            campaign.signal(run)
            continuation = campaign.continuations(run)
            adaptation = json.loads((run / "ADAPTATION_COMPLETION.json").read_text())
            for name in ("ADAPTATION_COMPLETION.json", "CONTINUATIONS_COMPLETION.json"):
                selection_evidence[str(run / name)] = file_hash(run / name)
            for seed, result in adaptation["results"].items():
                selected.append({"rate": arm["rate"], "method": "B1c", "seed": int(seed),
                                 "checkpoint": checkpoint_evidence(result["selected_checkpoint"])})
                for method, extra in continuation["results"][seed].items():
                    selected.append({"rate": arm["rate"], "method": method, "seed": int(seed),
                                     "checkpoint": checkpoint_evidence(extra["selected_checkpoint"])})
        write_json(seal, {"experiment_sha256": plan["sha256"], "selected_before_any_evaluation": True,
                          "selection_evidence_sha256": selection_evidence, "rows": selected})
        sealed_hash = file_hash(seal)
        outcomes = []
        for arm in campaigns["campaigns"]:
            if any(file_hash(p) != h for p, h in selection_evidence.items()):
                raise ValueError("Arm selection records changed after the common barrier")
            for row in selected:
                if row["rate"] == arm["rate"] and checkpoint_evidence(row["checkpoint"]["path"]) != row["checkpoint"]:
                    raise ValueError("Selected checkpoint changed after the common barrier")
            campaign.evaluate(arm["run"])
            outcome = campaign.report(arm["run"])
            outcomes.append({"rate": arm["rate"], "run": arm["run"], "outcome": outcome})
        if file_hash(seal) != sealed_hash or any(file_hash(p) != h for p, h in selection_evidence.items()):
            raise ValueError("Common selection evidence changed")
        result = {"status": "completed", "research_status": "exploratory", "confirmatory": False,
            "experiment_sha256": plan["sha256"], "selection_sha256": sealed_hash,
            "training_was_executed": True, "outcomes": outcomes,
            "interpretation": "Report observed effects per corruption arm; no forced advantage for RL"}
        write_json(path / "COMPARISON.json", result)
        return result
    except BaseException as error:
        write_json(path / "FAILURE.json", {"status": "failed", "error": repr(error), "automatic_resume": False})
        raise
    finally:
        lock.unlink()


def _aggregate(rows):
    """Average worlds within roots, then roots within equally weighted families."""
    roots = defaultdict(list)
    for row in rows:
        roots[row["root_id"]].append(row)
    root_values = {}
    families = defaultdict(list)
    for rid, group in roots.items():
        worlds = defaultdict(list)
        for row in group:
            worlds[row["latent_world_id"]].append(row)
        root_values[rid] = {metric: mean(mean(r["metrics"][metric] for r in world) for world in worlds.values())
                            for metric in METRICS}
        families[group[0]["scenario_family"]].append(rid)
    by_family = {family: {metric: mean(root_values[rid][metric] for rid in ids) for metric in METRICS}
                 for family, ids in families.items()}
    deliveries = Counter(row["trace"].get("user_response_status", "not_asked") for row in rows)
    fallbacks = sum(row["trace"].get("imperfections", {}).get("reply", {}).get("reason") == "no_safe_partial_fragment"
                    for row in rows if row["trace"].get("imperfections", {}).get("reply") is not None)
    retrievals = [row["trace"].get("imperfections", {}).get("retrieval", {}).get("select:O2") or {
        "input_ids": [record["memory_id"] for record in row["trace"]["observations"]["select"]["memory"]],
        "omitted_ids": []} for row in rows]
    return {"episodes": len(rows), "roots": len(roots),
        "metrics": {metric: mean(v[metric] for v in by_family.values()) for metric in METRICS},
        "by_family": by_family, "by_root": root_values, "reply_delivery_counts": dict(deliveries),
        "partial_fallback_unavailable": fallbacks,
        "select_memory_records": {"offered": sum(len(r.get("input_ids", [])) for r in retrievals),
                                  "omitted": sum(len(r.get("omitted_ids", [])) for r in retrievals)},
        "aggregation": "realizations_then_worlds_then_roots_then_equal_families"}


def _diagnostic_episodes(episodes):
    episodes = deepcopy(list(episodes))
    if not episodes or any(e.get("split") not in {"train", "validation", "dev_smoke"} for e in episodes):
        raise ValueError("Robustness diagnostic accepts development/validation only; never a closed test")
    identities = [(e["root_id"], e["latent_world_id"], e["realization_id"]) for e in episodes]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate episode identity in robustness diagnostic")
    return episodes


def evaluate_profiles(episodes, output, *, profiles=None, seed=7331, policy_factories=None, policy_evidence=None):
    """Run actual reset/ask/respond/write/reopen with shared exogenous faults.

    Accept development/validation only: this diagnostic does not open a sealed
    evaluation/test bank or select checkpoints. Supply factories for fixed LLMs
    through the API; the CLI defaults to explicitly labelled symbolic controls.
    """
    from .src.evaluation.runner import run_episode
    from .src.policy.rule_controller import RuleController
    episodes = _diagnostic_episodes(episodes)
    profiles = PROFILES if profiles is None else profiles
    if not profiles or "clean" not in profiles:
        raise ValueError("A clean reference profile is required")
    configurations = {name: normalize_config({"seed": seed, **config} if isinstance(config, dict) else config)
                      for name, config in profiles.items()}
    if configurations["clean"].active or any(c.seed != seed for c in configurations.values()):
        raise ValueError("Use the same fault seed in all profiles and zero rates for clean")
    factories = ({name: RuleController for name in ("B4", "C-NEVER", "C-ALWAYS")}
                 if policy_factories is None else policy_factories)
    if not factories:
        raise ValueError("At least one fixed policy is required")
    output = _new_output(output)
    original = digest(episodes)
    from .src.closure_protocol import source_files
    source_paths = [Path(__file__), *source_files()]
    sources = {str(p.resolve()): file_hash(p) for p in source_paths}
    declaration = {"version": VERSION, "status": "declared_before_execution", "episodes_sha256": original,
        "profiles": {name: c.to_dict() for name, c in configurations.items()}, "policies": list(factories),
        "policy_evidence": deepcopy(policy_evidence or {"scope": "symbolic_controls" if policy_factories is None else "caller_supplied_fixed_policies"}),
        "fault_seed": seed, "training_steps": 0, "checkpoint_selection": False, "confirmatory": False,
        "source_sha256": sources}
    write_json(output / "PREDECLARATION.json", declaration)
    results, rows = {}, []
    try:
        with (output / "trajectories.jsonl").open("x") as stream:
            for profile, config in configurations.items():
                results[profile] = {}
                for method, factory in factories.items():
                    group = []
                    for index, episode in enumerate(episodes):
                        # Hashed storage components keep user-supplied labels out of paths.
                        storage = output / "sqlite" / digest([profile, method, index])[:24]
                        storage.parent.mkdir(exist_ok=True)
                        row = run_episode(episode, factory(), method=method, imperfections=config,
                                          db_path=storage.with_suffix(".sqlite"))
                        row["profile"] = profile
                        row["metrics"]["J"] = row["metrics"]["C2"] * row["metrics"]["FJS"]
                        group.append(row)
                        rows.append(row)
                        stream.write(canonical(row) + "\n")
                        stream.flush()
                    results[profile][method] = _aggregate(group)
        if digest(episodes) != original or any(file_hash(p) != h for p, h in sources.items()):
            raise RuntimeError("Episodes or frozen source changed while executing robustness diagnostic")
        for profile, methods in results.items():
            for method, result in methods.items():
                clean = results["clean"][method]
                result["delta_vs_clean"] = {metric: result["metrics"][metric] - clean["metrics"][metric] for metric in METRICS}
        summary = {**declaration, "status": "completed", "results": results,
            "trajectory_count": len(rows), "episodes_unchanged": True,
            "limitations": ["Scripted controls do not establish learned-model performance or human realism.",
                "Partial delivery falls back to explicit unavailability without a supported safe fragment; actual counts are reported.",
                "Omissions mask S2 retrieval, not SQLite persistence or S3 access.",
                "No statistical superiority claim is made from this diagnostic."]}
        write_json(output / "summary.json", summary)
        with (output / "metrics.csv").open("x") as stream:
            writer = csv.DictWriter(stream, fieldnames=["profile", "method", *METRICS, "partial", "unavailable"])
            writer.writeheader()
            for profile, methods in results.items():
                for method, result in methods.items():
                    delivery = result["reply_delivery_counts"]
                    writer.writerow({"profile": profile, "method": method, **result["metrics"],
                                     "partial": delivery.get("partial", 0), "unavailable": delivery.get("unavailable", 0)})
        return summary
    except BaseException as error:
        write_json(output / "FAILURE.json", {"status": "failed", "error": repr(error), "preserve_partials": True})
        raise


def smoke(output, *, roots=20, seed=7331):
    """Exercise all rates/profiles on synthetic fixtures, with no model calls."""
    from .src.data.build_bank import build_bank
    from .imperfect_sft import prepare_bundle, load_examples
    if type(roots) is not int or roots < 5 or roots % 5:
        raise ValueError("Smoke roots must be a positive multiple of five, at least five")
    episodes = build_bank("dev_smoke", roots=roots, seed=seed)
    # A supported extra public context fact exercises actual partial delivery.
    # No private gold is read or changed to construct this diagnostic fixture.
    for episode in episodes:
        for reply in episode.get("replies", {}).values():
            if isinstance(reply, dict) and reply["text"].startswith("Como preferencia habitual"):
                reply["text"] += " Estoy en casa."
    output = _new_output(output)
    train = deepcopy(episodes)
    for episode in train:
        episode["split"] = "train"
    arms = []
    for rate in (0., .1, .3):
        arm = output / "sft" / f"rate_{int(rate * 100):02d}"
        manifest = prepare_bundle(train, arm, rate=rate, seed=seed)
        _, audit = load_examples(arm, train=train, expected_rate=rate, expected_seed=seed)
        arms.append({**manifest, "mutation_counts": audit["mutation_counts"], "bundle": str(arm)})
    result = evaluate_profiles(episodes, output / "environment", seed=seed)
    report = {"version": VERSION, "status": "completed", "kind": "CPU_contract_smoke_only",
              "training_steps": 0, "model_calls": 0, "human_review": False,
              "sft_arms": arms, "environment_summary": str(output / "environment/summary.json"),
              "trajectory_count": result["trajectory_count"]}
    write_json(output / "SMOKE.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", required=True, help="New audited competence corpus, prepared with current sources")
    p.add_argument("--rates", type=float, nargs="+", default=[0., .1, .3])
    p = sub.add_parser("smoke")
    p.add_argument("--roots", type=int, default=20)
    p = sub.add_parser("evaluate")
    p.add_argument("--bank", required=True, help="Development or validation JSONL; no closed test")
    p.add_argument("--checkpoint", help="Optional already selected frozen local LLM checkpoint")
    p.add_argument("--device", default="cuda")
    for p in sub.choices.values():
        p.add_argument("--output", required=True)
        p.add_argument("--seed", type=int, default=7331)
    p = sub.add_parser("prepare-campaigns")
    p.add_argument("--experiment", required=True)
    p.add_argument("--tests-report", required=True)
    p.add_argument("--technical-report", required=True)
    p = sub.add_parser("execute")
    p.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.data, args.output, rates=args.rates, seed=args.seed)
    elif args.command == "smoke":
        value = smoke(args.output, roots=args.roots, seed=args.seed)
    elif args.command == "evaluate":
        episodes = _diagnostic_episodes(_rows(args.bank))
        if Path(args.output).exists():
            raise FileExistsError("Preserve previous robustness results")
        if args.checkpoint is None:
            value = evaluate_profiles(episodes, args.output, seed=args.seed)
        else:
            from .competence_training import checkpoint_evidence, _release, PROMPTS
            from .src.policy.llm_controller import SpecLocalPolicy, LLMController
            from .src.policy.rule_controller import RuleController
            evidence = checkpoint_evidence(args.checkpoint)
            policy = SpecLocalPolicy.from_checkpoint(args.checkpoint, trainable=False, device=args.device)
            try:
                factories = {name: RuleController for name in ("B4", "C-NEVER", "C-ALWAYS")}
                factories["selected_LLM"] = lambda: LLMController(policy, prompt_root=PROMPTS, sample=False)
                value = evaluate_profiles(episodes, args.output, seed=args.seed,
                    policy_factories=factories, policy_evidence={"selected_LLM": evidence})
            finally:
                _release(policy)
    elif args.command == "prepare-campaigns":
        value = prepare_campaigns(args.experiment, tests_report=args.tests_report, technical_report=args.technical_report)
    else:
        value = execute(args.experiment)
    status = {"status": value["status"], "output": str(Path(getattr(args, "output", None) or args.experiment).resolve())}
    if args.command != "execute":
        status["training_steps"] = 0
    print(canonical(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
