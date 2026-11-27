"""Diagnóstico de primer turno natural, limitado a casos con revisión humana de acción.

No usa gold de memoria, simulador, juez, extracción ni clasificación dentro del prompt.
La revisión parcial de acción NO certifica un experimento completo de viabilidad.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from agents.client import API, TechnicalError
from agents.policies import MINIMAL, build_messages
from data_generation.schema import digest, validate_public, assert_no_private_keys
from evaluation.runner import ROOT, load_jsonl, atomic_json, append_jsonl, source_hash, snapshot_sources


def select_reviewed(public, reviews, case_ids):
    by_id = {c["case_id"]: c for c in public}
    if len(by_id) != len(public) or len(set(case_ids)) != len(case_ids):
        raise ValueError("IDs duplicados")
    selected = []
    for case_id in case_ids:
        case = by_id.get(case_id)
        if not case or case["split"] != "dev":
            raise ValueError("El diagnóstico solo admite casos existentes de desarrollo")
        validate_public(case)
        assert_no_private_keys(case)
        verified = [r for r in reviews if r.get("case_id") == case_id
                    and "acceptable_actions" in r.get("reviewed_fields", [])
                    and r.get("reviewer", "").strip() and r.get("source_reply", "").strip()
                    and r.get("acceptable_actions")
                    and set(r["acceptable_actions"]) <= {"ASK", "ANSWER", "CONDITIONAL"}]
        if not verified:
            raise ValueError("Falta revisión humana de la decisión inicial; una revisión de utilidad no la sustituye")
        if len({c["case_id"] for c in public if c["person_id"] == case["person_id"]}) < 5:
            raise ValueError("Faltan variantes de control en el perfil")
        selected.append(case)
    return selected


def natural_messages(case):
    records = [{**message, "status": "active"} for message in case["history"]]
    return build_messages(case["query"], records, case["reference_time"], system_prompt=MINIMAL)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    public, reviews = load_jsonl(args.public), load_jsonl(args.review)
    selected = select_reviewed(public, reviews, args.case_id)
    if args.live:
        if not args.url:
            raise SystemExit("Servicio de inferencia no configurado")
        tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                               cwd=ROOT, capture_output=True, text=True)
        if tests.returncode or "Ran 0 tests" in tests.stderr:
            raise SystemExit("Pruebas locales no superadas")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.output)
    manifest = {"purpose": "natural_first_turn_reviewed_case_diagnostic", "live": args.live,
                "selected_cases": args.case_id, "selected_people": len({c['person_id'] for c in selected}),
                "model": args.model, "seed": args.seed, "temperature": 0,
                "max_tokens": args.max_tokens, "timeout": args.timeout,
                "prompt_hash": digest(MINIMAL), "public_hash": digest(public),
                "review_hash": digest(reviews), "source_hash": source_hash(),
                "human_review_scope": "acceptable_initial_actions_only",
                "full_gold_validated": False, "training_evidence_sufficient": False,
                "response_format": "unconstrained_text", "simulation": False,
                "judge": False, "retrieval": "full_public_history"}
    atomic_json(args.output / "manifest.json", manifest)
    for case in selected:
        messages = natural_messages(case)
        atomic_json(args.output / (case["case_id"] + "_messages.json"), messages)
        if not args.live:
            continue
        client = API(args.url, args.model, args.timeout, args.max_tokens, args.seed, 0,
                     recorder=lambda r: append_jsonl(args.output / "agent_trace.jsonl", {"case_id": case["case_id"], **r}))
        result = {"case_id": case["case_id"], "response": None, "technical_errors": []}
        try:
            result["response"], _ = client.complete_text(messages)
        except TechnicalError as error:
            result["technical_errors"].append({"kind": error.kind, "http_status": error.status})
        append_jsonl(args.output / "per_case.jsonl", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["technical_errors"]:
            return 3
    print("Diagnóstico guardado: " + str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
