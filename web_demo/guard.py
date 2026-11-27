"""Recorded interventions that isolate the memory writer."""
from __future__ import annotations

import json
from pathlib import Path

from research.src.artifacts import file_hash

from .cases import ROOT, expected, question_turn, show_steps


REPORTS = ROOT / "reports/tesis_verificador_memoria_v1"
RESERVED = ROOT / "reports/tesis_verificador_reservado_full_v1"
RESERVED_CASES = (
    ("reservado-temporal", "Excepción con fecha", "La consulta posterior todavía cae dentro del intervalo.", 1),
    ("reservado-permanente", "Cambio habitual", "La preferencia anterior debe dar paso al cambio explícito.", 0),
    ("reservado-sesion", "Cambio temporal aclarado", "La aclaración permite conservar la preferencia habitual para después.", 2),
    ("reservado-duradero", "Cambio habitual aclarado", "Mismo diálogo inicial que el caso anterior, con otra réplica.", 48),
    ("reservado-sin-aclarar", "Duración sin aclarar", "La reparación dispone de la evidencia que obtuvo el selector.", 69),
)
CASES = (
    ("temporal", "Excepción con fecha", "La escritura debe respetar el intervalo indicado.", "sft_adicional", "0006"),
    ("permanente", "Cambio habitual", "Una preferencia explícita debe seguir disponible en otra sesión.", "grpo_credito", "0009"),
    ("contextual", "Preferencia por contexto", "La preferencia general y la excepción pueden coexistir.", "grpo_credito", "0003"),
    ("ambigua", "Duración aclarada", "La respuesta a la pregunta determina el alcance que se puede guardar.", "sft_adicional", "0007"),
    ("veto", "Excepción que termina hoy", "Bloquear un cambio fuera de alcance conserva la preferencia anterior.", "sft_adicional", "0010"),
)
ARMS = (
    ("original", "Escritura original"), ("veto", "Con veto"), ("repair", "Con reparación"),
)
METRICS = ("future_supported", "strict_joint", "unaffected_memory_damage")
SOURCE_LABELS = {"sft_adicional": "SFT adicional", "grpo_credito": "GRPO con crédito por fase"}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _case_rows(folder):
    rows = {arm: _read(folder / f"{arm}.json") for arm, _ in ARMS}
    original = rows["original"]
    phases = ("select", "respond", "write")
    for arm, row in rows.items():
        if row["method"] != arm or row["trace"]["episode"] != original["trace"]["episode"]:
            raise ValueError("Las trayectorias no pertenecen al mismo caso.")
        if any(row["phases"][phase]["observation_sha256"] != original["phases"][phase]["observation_sha256"]
               for phase in phases) or any(row["phases"][phase]["action"] != original["phases"][phase]["action"]
                                            for phase in phases[:2]):
            raise ValueError("Las variantes deben compartir decisión, respuesta y evidencia de escritura.")
    return rows


def _project_case(case_id, title, description, folder, report, source, dataset):
    rows = _case_rows(folder)
    original = rows["original"]
    episode = original["trace"]["episode"]
    scenario = {"id": case_id, "title": title, "description": description, "dataset": dataset,
                "family": episode["scenario_family"], "attribute": episode["current_query"]["attribute"],
                "question": question_turn(episode)["text"], "current_date": episode["current_query"]["date"],
                "future_date": episode["future_query"]["date"], "seed": original["training_seed"]}
    panels = []
    for arm, label in ARMS:
        row = rows[arm]
        panels.append({"id": arm, "label": label, "steps": show_steps(row), "metrics": row["metrics"],
                       "verification": row["guard_decision"],
                       "artifact": str((folder / f"{arm}.json").relative_to(ROOT))})
    return {"scenario": scenario, "expected": expected(episode), "mode": "guard-replay", "models": panels,
            "study": {"source": source, "roots": report["scope"]["independent_roots"],
                      "episodes": report["scope"]["complete_paired_cells"], "seeds": report["scope"]["training_seeds"],
                      "metrics": {arm: {key: report["methods"][arm][key]["all"] for key in METRICS}
                                  for arm, _ in ARMS},
                      "artifact": str((folder.parents[1] / "RESULT.json").relative_to(ROOT))}}


def load_reserved_guard():
    index = RESERVED / "RESULT.json"
    if not index.is_file():
        return {}, None
    report = _read(index)
    comparison_path = ROOT / "research/runs/tesis_comparison_full_v1/RESULTS.json"
    if file_hash(comparison_path) != report["source"]["sha256"]:
        raise ValueError("El informe del verificador no coincide con la comparación reservada.")
    comparison = _read(comparison_path)
    methods = {}
    for name, info in report["methods"].items():
        path = RESERVED / info["result"]
        if file_hash(path) != info["result_sha256"]:
            raise ValueError("El informe por selector del verificador cambió.")
        methods[name] = _read(path)
    sft = methods["SFT"]
    if any(sft["scope"][key] != value for key, value in report["scope"].items()):
        raise ValueError("El panel SFT no coincide con la población de la comparación.")
    cases = {case_id: _project_case(case_id, title, desc, RESERVED / "SFT" / "cases" / f"{number:04d}",
                                    sft, "SFT congelado", "Banco reservado")
             for case_id, title, desc, number in RESERVED_CASES}
    summary = {"scope": report["scope"],
               "methods": {name: {"C2": m["C2"], "FJS": m["FJS"], "strict_joint": m["strict_joint"],
                                   "guard_fjs": {arm: methods[name]["methods"][arm]["future_supported"]["all"]["macro"]
                                                 for arm, _ in ARMS}}
                           for name, m in comparison["means"].items()},
               "grpo_selected_steps": {seed: methods_for_seed["GRPO-SELECT"]["selection"]["step"]
                                       for seed, methods_for_seed in comparison["training"].items()}}
    return cases, summary


def load_guard_cases():
    reports = {}
    cases = {}
    for case_id, title, description, source, index in CASES:
        path = REPORTS / source / "RESULT.json"
        if not path.is_file():
            continue
        if source not in reports:
            report = _read(path)
            original = Path(report["source"]["path"])
            if not original.is_file() and "research" in original.parts:
                original = ROOT.joinpath(*original.parts[original.parts.index("research"):])
            if not original.is_file() or file_hash(original) != report["source"]["sha256"]:
                raise ValueError("El archivo de origen de la verificación de memoria cambió.")
            reports[source] = report
        report = reports[source]
        folder = REPORTS / source / "cases" / index
        cases[case_id] = _project_case(case_id, title, description, folder, report, SOURCE_LABELS[source], "Pilotos previos")
    return cases
