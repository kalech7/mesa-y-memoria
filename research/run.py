"""Run from repository root: python -m research.run --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .src.artifacts import ROOT, canonical, digest, write_json, file_hash, now
from .src.protocol import gate_status, require_gate, GateError


def smoke(destination):
    from .src.data.build_bank import build_bank, write_bank
    from .src.data.audit_splits import audit_splits
    from .src.policy.rule_controller import RuleController
    from .src.evaluation.runner import run_episode
    from .src.evaluation.persistence_interventions import run_interventions
    from .src.reporting import export_results
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("Preserve prior evidence: choose an empty output directory")
    episodes = build_bank()
    write_bank(episodes, path / "dev_smoke.jsonl")
    write_json(path / "data_audit.json", audit_splits({"dev_smoke": episodes}))
    traces = []
    with (path / "TRAZAS.jsonl").open("x") as stream:
        for episode in episodes:
            row = run_episode(episode, RuleController(), method="mechanism_reference")
            traces.append(row)
            stream.write(canonical(row) + "\n")
    interventions = []
    for episode in episodes:
        if episode.get("opportunity", {}).get("eligible", episode["scenario_family"] == "F4"):
            interventions.append(run_interventions(episode, RuleController, forced_slot="TEMPORAL_SCOPE", method="mechanism_reference"))
    with (path / "INTERVENCIONES.jsonl").open("x") as stream:
        for result in interventions:
            stream.write(canonical(result) + "\n")
    counts = export_results(path / "TRAZAS.jsonl", path / "RESULTADOS.csv")
    summary = {"status": "executed", "purpose": "development_mechanism_validation_only",
               "not_confirmatory": True, "created_at": now(), "independent_roots": len({e["root_id"] for e in episodes}),
               **counts, "invalid_phase_outputs": sum(r["metrics"]["invalid_outputs"] for r in traces),
               "intervention_invariants_passed": all(all(r["invariants"].values()) for r in interventions),
               "traces_sha256": file_hash(path / "TRAZAS.jsonl")}
    write_json(path / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="SPEC-TESIS-CLARIFICACION-MEMORIA-001: local isolated implementation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Show gates without opening test or loading a model")
    sub.add_parser("inventory", help="Read-only repository/hardware/model inventory")
    g0 = sub.add_parser("g0", help="Record technical lock and run approved model smoke when selected")
    g0.add_argument("--model-path")
    g0.add_argument("--device", default=None)
    sm = sub.add_parser("smoke", help="E0 mechanism validation only, at most 40 development roots")
    sm.add_argument("--output", default=str(ROOT / "reports/mechanism_smoke_001"))
    for name, gate in (("build-bank", "G1"), ("train-sft", "G2"), ("train-continuations", "G3"), ("closed-test", "G4")):
        command = sub.add_parser(name, help=f"Execute a frozen campaign after required gate {gate}")
        command.set_defaults(required_gate=gate)
        command.add_argument("--output")
        if name != "build-bank":
            command.add_argument("--campaign", help="JSON with pinned model/data/checkpoint paths")
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = gate_status()
        elif args.command == "inventory":
            from .src.preflight import inventory
            result = inventory()
            write_json(ROOT / "reports/inventory.json", result)
        elif args.command == "g0":
            from .src.preflight import run_g0
            result = run_g0(model_path=args.model_path, device=args.device)
        elif args.command == "smoke":
            result = smoke(args.output)
        else:
            require_gate(args.required_gate)
            if not args.output:
                parser.error("--output is required for a final campaign")
            if args.command == "build-bank":
                from .src.data.build_bank import build_full_bank, write_bank
                from .src.data.audit_splits import audit_splits
                banks = build_full_bank()
                output = Path(args.output)
                output.mkdir(parents=True, exist_ok=True)
                for split, bank in banks.items():
                    write_bank(bank, output / (split + ".jsonl"))
                result = audit_splits(banks)
                write_json(output / "audit.json", result)
            else:
                if not args.campaign:
                    parser.error("--campaign is required for a final campaign")
                from .src.artifacts import load_config
                campaign = load_config(args.campaign)
                if args.command == "train-sft":
                    from .src.training.campaign import run_sft_campaign
                    result = run_sft_campaign(campaign["model_path"], campaign["train_bank"], campaign["validation_bank"],
                                              args.output, device=campaign.get("device", "cuda"))
                elif args.command == "train-continuations":
                    from .src.training.campaign import run_continuations
                    result = run_continuations(campaign["sft_campaign"], campaign["train_bank"], campaign["validation_bank"],
                                               args.output, device=campaign.get("device", "cuda"))
                else:
                    from .src.evaluation.campaign import closed_campaign
                    result = closed_campaign(campaign, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") != "blocked" else 2
    except GateError as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
