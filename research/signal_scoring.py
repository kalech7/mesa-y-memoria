"""Versioned, symmetric correction of S3 conditional memory-ID provenance.

The actor, simulator, reader and reward formula are unchanged. Original traces
and reports are archived; only scoring of faithful conditional responses changes.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from .src.artifacts import canonical, digest, file_hash, write_json, now
from .src.policy.rule_controller import compatible_worlds
from .src.readers.symbolic import applicability
from .src.scoring.evidence import Evaluator, EVALUATOR_VERSION
from .src.scoring.reward import rewards

VERSION = EVALUATOR_VERSION


class DiagnosticEvaluator(Evaluator):
    """Compatibility name; there is one production implementation, not a fork."""


def rescore(output):
    from .signal_diagnostic import compact, analyze, write_reports
    output = Path(output).resolve()
    if (output / "correccion_evaluador.json").exists():
        raise FileExistsError("Scoring revision already exists")
    original_completion = json.loads((output / "completion.json").read_text())
    for name, expected in original_completion["artifacts_sha256"].items():
        if file_hash(output / name) != expected:
            raise ValueError("Original artifact changed: " + name)
    evaluator, compact_rows, changes = DiagnosticEvaluator(), [], []
    staged = output / "diagnostico_50.rescored.tmp"
    with (output / "diagnostico_50.jsonl").open() as source, staged.open("x") as target:
        for line_number, text in enumerate(source, 1):
            row = json.loads(text)
            previous = deepcopy(row["scores"]["future"])
            previous_reward = deepcopy(row["rewards_by_component"])
            future = evaluator.score_future(row["trace"])
            row["scoring_original"] = {"future_score": previous, "rewards_by_component": previous_reward,
                                       "source_path": "diagnostico_50.original.jsonl", "source_line": line_number}
            row["scores"]["future"] = future
            row["future_evaluator_version"] = VERSION
            row["original_evaluator_hash"] = row.get("evaluator_hash")
            row["evaluator_hash"] = digest({"original": row["original_evaluator_hash"],
                "conditional_fix_sha256": file_hash(__file__), "version": VERSION})
            metrics = row["metrics"]
            updated = rewards(row["scores"]["current"]["utility"], future["utility"], metrics["questions"],
                              metrics["unsupported_operations"], metrics["invalid_outputs"])
            assert updated["R_short"] == previous_reward["R_short"]
            assert future["success"] == metrics["FJS"]
            assert row["J_primary"] == metrics["C2"] * metrics["FJS"]
            row["rewards_by_component"] = updated
            row["question_cost_sensitivity"] = {str(c): rewards(row["scores"]["current"]["utility"],
                future["utility"], metrics["questions"], metrics["unsupported_operations"], metrics["invalid_outputs"], question_cost=c)
                for c in (0., .05, .20)}
            reader_trace = {**row["trace"], "future_response": row["reader_diagnostic"]["reference_response"]}
            row["reader_diagnostic"]["reference_score"] = evaluator.score_future(reader_trace)
            if "B1_response_source" in row:
                row["B1_response_source"]["path"] = "diagnostico_50.original.jsonl"
            if future["utility"] != previous["utility"]:
                assert row["trace"]["future_response"]["mode"] == "CONDITIONALIZE"
                changes.append({"line": line_number, "case_id": row["diagnostic_case_id"],
                    "action": row["action_id"], "writer": row["writer"], "old_future_utility": previous["utility"],
                    "new_future_utility": future["utility"], "C2": metrics["C2"], "FJS": metrics["FJS"]})
            payload = canonical(row)
            target.write(payload + "\n")
            row["trace_line"] = line_number
            row["trace_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
            flat = compact(row)
            flat["evaluator_version"] = VERSION
            compact_rows.append(flat)
    revised = analyze(compact_rows)
    original_analysis = json.loads((output / "analisis.json").read_text())
    revised.update(pilot_audit=original_analysis["pilot_audit"], calls=original_analysis["calls"],
                   weights=original_analysis["weights"], future_evaluator_version=VERSION,
                   original_actor_execution_code_sha256=json.loads((output / "plan.json").read_text())["source_files_sha256"])
    # Preserve every original artifact before replacing any requested output.
    names = ["diagnostico_50.jsonl", "comparacion_acciones.csv", "analisis.json", "fallos_clasificados.md", "decision_rl.md", "completion.json"]
    for name in names:
        path = Path(name)
        shutil.copyfile(output / name, output / (path.stem + ".original" + path.suffix))
    staged.replace(output / "diagnostico_50.jsonl")
    with (output / "comparacion_acciones.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(compact_rows[0]))
        writer.writeheader()
        writer.writerows(compact_rows)
    write_json(output / "analisis.json", revised)
    # Report needs only visible text/IDs for examples; read from fixed development.
    from .signal_diagnostic import load_development
    episodes, _ = load_development(json.loads((output / "plan.json").read_text()).get("closure", Path(__file__).parent / "runs/closure_v4_20261024_003"))
    cases = [{"case_id": digest([e[k] for k in ("root_id", "latent_world_id", "realization_id")])[:24], "episode": e} for e in episodes]
    write_reports(output, revised, cases, revised["calls"])
    correction = {"status": "passed", "version": VERSION, "created_at": now(),
        "reason": "Future conditional citations use persisted memory IDs; old scorer validated them against pre-write message/initial-memory IDs",
        "contract": "Faithful conditional: utility 0, FJS 0. Unsupported conditional: utility -1. Reward formula and all coefficients unchanged.",
        "changed_trajectories": len(changes), "changes": changes, "C2_FJS_J_unchanged": True,
        "R_short_unchanged": True, "new_model_calls": 0, "source_sha256": file_hash(__file__),
        "original_scoring_source_sha256": file_hash(Path(__file__).parent / "src/scoring/evidence.py"),
        "original_artifacts": {Path(n).stem + ".original" + Path(n).suffix: file_hash(output / (Path(n).stem + ".original" + Path(n).suffix)) for n in names}}
    write_json(output / "correccion_evaluador.json", correction)
    notice = ("\n## Corrección versionada del evaluador\n\n"
        f"Se conservaron los archivos `*.original.*` y se recalcularon simétricamente {len(compact_rows)} trayectorias. "
        f"En {len(changes)} cambió únicamente la utilidad de una respuesta condicional sustentada, de −1 a 0 conforme al contrato existente. "
        "C2, FJS, éxito conjunto, R_short, acciones, respuestas, memorias y pesos no cambiaron. "
        "La fórmula de recompensa conserva sus términos y coeficientes. No hubo nuevas generaciones. "
        "La auditoría inicial no cubría este cruce entre namespaces de IDs; el caso y las pruebas se documentan en `correccion_evaluador.json` y `pruebas_evaluador_v2.json`.\n")
    for name in ("decision_rl.md", "fallos_clasificados.md"):
        with (output / name).open("a") as target:
            target.write(notice)
    shutil.copyfile(__file__, output / "source_snapshot/signal_scoring.py")
    write_json(output / "completion.json", {**original_completion, "scoring_revision": VERSION,
        "rescored_at": now(), "scoring_model_calls": 0,
        "artifacts_sha256": {name: file_hash(output / name) for name in [*names[:-1], "plan.json", "correccion_evaluador.json"]}})
    return correction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = rescore(args.output)
    print(canonical({"status": result["status"], "version": VERSION, "changed_trajectories": result["changed_trajectories"], "new_model_calls": 0}))


if __name__ == "__main__":
    main()
