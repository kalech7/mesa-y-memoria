"""Read the reserved multi-turn selector results and project replay examples."""
from __future__ import annotations

import json
from statistics import mean

from research.src.artifacts import digest, file_hash

from .cases import ROOT


RUN = ROOT / "research/runs/tesis_interactive_v1"
BRANCH_COMPARISON = RUN.parent / "tesis_grpo_ramificado_v1"
GUARD_AUDIT = ROOT / "reports/tesis_interactive_bank/VERIFICADOR.json"
METHODS = ("GRPO-longitudinal", "GRPO-inmediato", "SFT-adicional", "SFT",
           "GRPO-longitudinal-terminal", "GRPO-inmediato-terminal")
FAMILIES = {"F0": "Preferencia estable", "F1": "Cambio habitual", "F2": "Excepción temporal",
            "F3": "Preferencia por contexto", "F4": "Duración ambigua"}
EXAMPLE_ROOT = "it32c5ba834e22af00de03c4c1"
METRICS = ("current_supported", "future_supported", "strict_joint", "questions", "unsupported_writes")


def _read(path):
    return json.loads(path.read_bytes())


def _checked_result(run):
    result = _read(run / "RESULTS.json")
    plan = _read(run / "PLAN.json")
    selection_path = run / "SELECTION.json"
    selection = _read(selection_path)
    if (result["status"] != "completed_exploratory" or file_hash(selection_path) != result["selection_sha256"]
            or selection["plan_sha256"] != file_hash(run / "PLAN.json")
            or selection["split"] != "validation" or selection["selected_before_evaluation"] is not True
            or selection["evaluation_sha256"] != plan["split_sha256"]["evaluation"]):
        raise ValueError("La evaluación de varios turnos requiere una selección verificada de validación.")
    return result, selection


def _rows(run, panel):
    path = run / panel["traces"]
    if file_hash(path) != panel["trace_sha256"]:
        raise ValueError("La traza de varios turnos cambió desde la evaluación.")
    entries = list(map(json.loads, path.read_text(encoding="utf-8").splitlines()))
    rows = {(row["root_id"], row["world_id"]): row for row in entries}
    if len(rows) != len(entries) or any(row["split"] != "evaluation" for row in entries):
        raise ValueError("Se requieren trazas reservadas sin episodios duplicados.")
    return rows


def _branch_comparison():
    run = BRANCH_COMPARISON
    if not (run / "AUDIT.json").is_file():
        return None
    plan, selection, result, audit = (_read(run / f"{name}.json") for name in
                                      ("PLAN", "SELECTION", "RESULTS", "AUDIT"))
    selection_hash = file_hash(run / "SELECTION.json")
    if (audit["status"] != "verified" or audit["selection_sha256"] != selection_hash
            or audit["results_sha256"] != file_hash(run / "RESULTS.json")
            or result["selection_sha256"] != selection_hash
            or selection["plan_sha256"] != file_hash(run / "PLAN.json")):
        raise ValueError("La comparación de GRPO por ramas requiere una selección auditada.")
    primary = plan["primary"]
    return {"means": result["means"], "primary_success": result["primary_success"],
            "seeds": plan["seeds"], "contrast": result["contrasts"][primary["comparator"]][primary["metric"]]}


def load_interactive(run=RUN):
    if not (run / "RESULTS.json").is_file():
        return None, {}
    result, selection = _checked_result(run)
    seeds = result["seeds"]
    method_rows = {seed: {method: _rows(run, result["panels"][str(seed)][method]) for method in METHODS}
                   for seed in seeds}
    episodes = list(map(json.loads, (run / "bank/evaluation.jsonl").read_text(encoding="utf-8").splitlines()))
    if digest(episodes) != selection["evaluation_sha256"]:
        raise ValueError("Cambió el banco reservado de la evaluación de varios turnos.")
    stories = {(story["root_id"], story["world_id"]): story for story in episodes}
    examples = {}
    for seed, panels in method_rows.items():
        reference = panels[METHODS[0]]
        if reference.keys() != stories.keys() or any(rows.keys() != reference.keys() for rows in panels.values()):
            raise ValueError("Las políticas de varios turnos deben compartir el conjunto de historias.")
        for family, label in FAMILIES.items():
            # Choose a stable identity, independently of performance or reward.
            identity = min(key for key, trace in reference.items()
                           if trace.get("family", trace.get("scenario_family")) == family)
            case_id = f"{seed}-{family}"
            examples[case_id] = {"id": case_id, "title": label, "seed": seed, "family": family,
                                 "methods": {name: rows[identity] for name, rows in panels.items()},
                                 "story": stories[identity]}
    for index, identity in enumerate(sorted(key for key in stories if key[0] == EXAMPLE_ROOT), 1):
        case_id = f"aclaracion-{index}"
        examples[case_id] = {"id": case_id, "title": f"Ejemplo · Dos aclaraciones ({index})", "seed": seeds[0],
                             "family": "F4", "methods": {method: rows[identity]
                                                          for method, rows in method_rows[seeds[0]].items()},
                             "story": stories[identity]}
    means = {name: {key: values[key] for key in METRICS} for name, values in result["means"].items()}
    for name in ("GRPO-longitudinal-terminal", "GRPO-inmediato-terminal"):
        means[name] = {key: mean(result["panels"][str(seed)][name]["macro"][key] for seed in seeds)
                       for key in METRICS}
    sensitivity = None
    guard = None
    alternate = run.parent / "tesis_interactive_low_supervision_v2"
    if run == RUN and (alternate / "RESULTS.json").is_file():
        reduced, _ = _checked_result(alternate)
        sensitivity = {name: reduced["means"][name]["future_supported"] for name in METHODS[:4]}
    if run == RUN and GUARD_AUDIT.is_file():
        audit = _read(GUARD_AUDIT)
        if audit["results_sha256"] != file_hash(run / "RESULTS.json"):
            raise ValueError("La auditoría de escritura pertenece a otros resultados.")
        guard = audit["counts"]
    return {"roots": result["evaluation_roots"], "episodes": result["evaluation_episodes_per_seed"],
            "seeds": seeds, "means": means, "sensitivity": sensitivity, "guard": guard,
            "branch_comparison": _branch_comparison() if run == RUN else None,
            "contrasts": result["contrasts"], "primary_criteria": result["primary_criteria"],
            "terminal_contrasts": result["terminal_contrasts_longitudinal_minus_immediate"],
            "cases": [{key: item[key] for key in ("id", "title", "seed", "family")} for item in examples.values()]}, examples


def simulate(story, actions):
    """Replay a user's own sequence in a fresh store; reveal only the current observation."""
    from tempfile import TemporaryDirectory
    from pathlib import Path

    from research.tesis_interactive_env import InteractiveEpisode

    with TemporaryDirectory(prefix="tesis-web-interactive-") as directory:
        environment = InteractiveEpisode(story, Path(directory) / "memory.sqlite")
        try:
            observation = environment.observe()
            done = False
            for action in actions:
                if done or action not in environment.legal_actions:
                    raise ValueError("La acción no está disponible en ese turno de la conversación.")
                observation, done = environment.step(action)
            if done:
                trace = environment.summary()
                return {"done": True, "visits": trace["visits"], "future": trace["future"]}
            return {"done": False, "observation": observation, "legal_actions": list(environment.legal_actions)}
        finally:
            environment.close()
