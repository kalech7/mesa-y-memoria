"""Regenerate CSV from full traces. Pending experiments never get invented scores."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from .artifacts import now, write_json
from .protocol import gate_status


REPORTS = {
    "INFORME_OBSERVABILIDAD.md": "PI1: E1, contexto y errores por familia",
    "INFORME_VALOR_PERSISTENCIA.md": "PI2: I0/I1/I2 y efectos de adquisición y persistencia",
    "INFORME_SFT_RL.md": "PI3: B1/B2-L/B3/B2-I, checkpoints, semillas y presupuestos",
    "INFORME_GENERALIZACION.md": "E4/E5/E6: ID, OOD, externo, extracción, lector LLM y no respuesta",
}


def write_pending_reports(root):
    root = Path(root)
    for filename, topic in REPORTS.items():
        path = root / "reports" / filename
        if not path.exists():
            path.write_text(f"# {topic}\n\nEstado: `not_executed`.\n\n"
                            "La ejecución comparativa exige superar las puertas del protocolo. "
                            "Las pruebas de mecanismo de dev_smoke no son resultados confirmatorios. "
                            "No existen estimaciones, curvas ni intervalos experimentales de esta etapa.\n")
    write_json(root / "reports/status.json", {"created_at": now(), "gates": gate_status(root),
               "research_status": "not_executed", "human_audit": "pending_human_audit"})


def export_results(traces_path, destination):
    rows = [json.loads(line) for line in Path(traces_path).read_text().splitlines() if line]
    fields = ["method", "seed", "root_id", "latent_world_id", "realization_id", "scenario_family", "split",
              "observation_condition", "extraction_track", "intervention", "reader_id", "technical_status",
              "retry_count", "C2", "FJS", "questions", "unsupported_operations", "write_precision",
              "unsupported_current", "unsupported_future", "invalid_outputs", "future_invalid", "R_long", "R_short"]
    with Path(destination).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {**row, **row.get("metrics", {}), **row.get("rewards_by_component", {})}
            writer.writerow({key: flat.get(key) for key in fields})
    return {"scheduled_episodes": len(rows), "technical_failures": sum(r["technical_status"] == "failed" for r in rows),
            "attempts": sum(len(r.get("attempts", [{}])) for r in rows)}
