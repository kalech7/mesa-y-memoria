"""Isolate the effect of verifying WRITE with recorded decisions and native SQLite."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from research.competence_evaluation import _run_trajectory
from research.src.artifacts import canonical, digest, file_hash, write_json
from research.src.policy.llm_controller import InvalidOutput
from research.src.scoring.evidence import audit_operation
from research.tesis_memory_guard import operation_issue, plan_write, write_effect
from research.tesis_metrics import _comparison_inputs, compare_trajectories


MODES = ("original", "veto", "repair")
REPORTED = (
    ("current_supported", "Acierto actual sustentado"),
    ("future_supported", "Acierto futuro sustentado"),
    ("strict_joint", "Éxito conjunto estricto"),
    ("unsupported_write", "Episodio con escritura sin respaldo"),
    ("invalid_output", "Episodio con salida inválida"),
    ("missed_authorized_permanent_write", "Cambios permanentes autorizados omitidos"),
    ("temporary_stored_without_expiry", "Excepciones guardadas sin vencimiento"),
    ("unaffected_memory_damage", "Daño a alguna consulta protegida"),
)
WRITE_CONTRASTS = (
    ("invalid_proposals_rejected", "Operaciones malformadas rechazadas"),
    ("unsupported_proposals_rejected", "Operaciones propuestas sin respaldo que se rechazaron"),
    ("supported_proposals_blocked", "Operaciones propuestas con respaldo que se bloquearon"),
    ("unsupported_effects_avoided", "Escrituras sin respaldo del original que se evitaron"),
    ("supported_effects_lost", "Escrituras respaldadas del original que se perdieron"),
)


class ReplayWriter:
    def __init__(self, archived, mode):
        self.archived = archived
        self.mode = mode
        self.decision = None

    def select(self, observation):
        return self.recorded_action("select")

    def recorded_action(self, phase):
        original = self.archived["phases"][phase]
        if original["error"] and not original["action"]:
            raise InvalidOutput(original["error"]["error"])
        return deepcopy(original["action"])

    def write(self, observation):
        if observation != self.archived["trace"]["observations"]["write"]:
            raise ValueError("WRITE did not receive the archived public observation")
        if self.mode == "original":
            return self.recorded_action("write")
        self.decision = plan_write(observation, deepcopy(self.archived["phases"]["write"]["action"]) or None,
                                   mode=self.mode)
        return {"operations": self.decision["operations"]}


def replay_case(archived, directory):
    if (archived["route"], archived["condition"], archived["reader_id"], archived["technical_status"]) != (
            "natural", "O2", "fixed_symbolic", "completed"):
        raise ValueError("Replay requires completed natural O2 traces with the fixed symbolic reader")
    directory.mkdir()
    rows = {}
    for mode in MODES:
        controller = ReplayWriter(archived, mode)
        row = _run_trajectory(controller, archived["trace"]["episode"], directory / f"{mode}.sqlite",
                              fixed_response=archived["phases"]["respond"], method=mode,
                              training_seed=archived["training_seed"])
        for phase in ("select", "respond", "write"):
            if row["phases"][phase]["observation_sha256"] != archived["phases"][phase]["observation_sha256"]:
                raise RuntimeError(f"The archived {phase} observation changed")
        for phase in ("select", "respond"):
            for key in ("action", "error"):
                if row["phases"][phase][key] != archived["phases"][phase][key]:
                    raise RuntimeError(f"Archived {phase} {key} changed")
        for key in ("asked_slot", "user_reply", "user_response_status", "initial_memory"):
            if row["trace"].get(key) != archived["trace"].get(key):
                raise RuntimeError(f"Archived {key} changed")
        if row["metrics"]["C2"] != archived["metrics"]["C2"]:
            raise RuntimeError("Current supported answer changed")
        if mode == "original":
            for key in ("final_memory", "executed_operations", "write_result", "future_response"):
                if row["trace"][key] != archived["trace"][key]:
                    raise RuntimeError(f"The control failed to reproduce archived {key}")
            for key in ("FJS", "unsupported_write_episode", "any_invalid_output"):
                if row["metrics"][key] != archived["metrics"][key]:
                    raise RuntimeError(f"The control failed to reproduce archived {key}")
        row["guard_decision"] = controller.decision
        write_json(directory / f"{mode}.json", row)
        rows[mode] = row
    return rows


def display(entry):
    fraction = f"{entry['numerator']}/{entry['denominator']}"
    return fraction if entry["macro"] is None else f"{100 * entry['macro']:.1f} % ({fraction})"


def _operation_contrasts(guarded, proposal_issues, baseline_effects):
    decision = guarded["guard_decision"]
    accepted = set(decision["accepted_indices"])
    guarded_effects = set(_executed_effects(guarded["trace"]))
    indices = {
        "invalid_proposals_rejected": {index for index, issue in proposal_issues.items()
                                       if issue and issue.startswith("invalid_structure:")},
        "unsupported_proposals_rejected": {index for index, issue in proposal_issues.items()
                                           if issue == "unsupported_evidence"},
        "supported_proposals_blocked": {index for index, issue in proposal_issues.items() if issue is None},
    }
    effects = {
        "unsupported_effects_avoided": [item for item in baseline_effects if not item[1]],
        "supported_effects_lost": [item for item in baseline_effects if item[1]],
    }
    return {**{key: (len(values - accepted), len(values)) for key, values in indices.items()},
            **{key: (sum(item not in guarded_effects for item in values), len(values))
               for key, values in effects.items()}}


def _executed_effects(trace):
    observation = trace["observations"]["write"]
    return [(write_effect(operation), audit_operation(operation, observation))
            for operation in trace["executed_operations"] if operation["op"] != "NOOP"]


def _contrast_inputs(original):
    """Classify proposals and effects in the baseline independently of the guard's choices."""
    trace = original["trace"]
    observation = trace["observations"]["write"]
    proposal = original["phases"]["write"]["action"]
    operations = proposal.get("operations", [])
    issues = {index: operation_issue(observation, operation) for index, operation in enumerate(operations)
              if operation != {"op": "NOOP"}}
    return issues, _executed_effects(trace)


def _contrast_table(result):
    lines = ["| Comparación contra original | Veto | Reparación |", "| --- | ---: | ---: |"]
    for metric, label in WRITE_CONTRASTS:
        counts = [result["write_contrasts"][mode]["all"][metric] for mode in MODES[1:]]
        lines.append(f"| {label} | " + " | ".join(_count(row) for row in counts) + " |")
    return lines


def _count(entry):
    return f"{entry['numerator']}/{entry['denominator']}"


def _empty_contrast_counts():
    return {metric: Counter(numerator=0, denominator=0) for metric, _ in WRITE_CONTRASTS}


def render(result):
    source_detail = ("del conjunto reservado de la comparación" if result["source"].get("comparison_method")
                     else "archivados")
    lines = ["# Verificación de escrituras en memoria", "",
             f"Se reprodujeron {result['scope']['complete_paired_cells']} pares de semilla y episodio {source_detail}. "
             "En cada episodio "
             "se mantuvieron la decisión de preguntar, la réplica del usuario y la respuesta actual. "
             "Cada variante ejecutó su escritura y reabrió SQLite para la consulta posterior.", "",
             "| Indicador | Original | Veto de escritura sin respaldo | Reparación con evidencia |",
             "| --- | ---: | ---: | ---: |"]
    for metric, name in REPORTED:
        cells = [display(result["methods"][mode][metric]["all"]) for mode in MODES]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "Los porcentajes son medias por familia. Las fracciones muestran eventos y oportunidades brutas.", "",
              *_contrast_table(result), "",
              "`RESULT.json` conserva estos conteos por familia y semilla. Una fracción de 0/0 indica que no hubo "
              "oportunidades observadas para medir ese efecto. El bloqueo de una propuesta se mide por su posición "
              "en el lote; una reparación puede recuperar el mismo efecto, definido por atributo, valor, alcance, "
              "estado y registros reemplazados. Si el mismo efecto reaparece con una cita válida, se cuenta como "
              "reparación de la escritura sin respaldo. Una escritura que el original no llegó a ejecutar puede contar como "
              "propuesta, pero queda fuera del denominador de efectos. Las salidas que fallaron el parseo se cuentan "
              "por episodio entre las salidas inválidas.", ""]
    for mode in MODES[1:]:
        changes = result["outcomes"][mode]
        lines.append(f"- **{mode}**. El acierto futuro mejoró en {changes.get('improved', 0)} episodios, "
                     f"empeoró en {changes.get('worse', 0)} y se mantuvo en {changes.get('tied', 0)}.")
    lines += ["", "| Familia | Original, acierto futuro | Veto, acierto futuro | Reparación, acierto futuro |",
              "| --- | ---: | ---: | ---: |"]
    by_family = result["methods"]["original"]["future_supported"]
    for family in sorted(set(by_family) - {"all"}):
        cells = [display(result["methods"][mode]["future_supported"][family]) for mode in MODES]
        lines.append(f"| {family} | {' | '.join(cells)} |")
    if result["examples"]:
        lines += ["", "### Ejemplos de cambios en la sesión posterior", ""]
        for example in result["examples"]:
            before, after = example["original"], example["repair"]
            lines.append(f"- {example['family']}, {example['change']}. El resultado pasó de "
                         f"{before['mode']} ({before['value'] or 'sin valor'}) a {after['mode']} "
                         f"({after['value'] or 'sin valor'}). [Original]({example['directory']}/original.json) · "
                         f"[Reparación]({example['directory']}/repair.json).")
    lines += ["", "El verificador opera sobre la evidencia visible para WRITE. El planificador y el evaluador "
              "comparten la gramática de evidencia del entorno. Los resultados describen estas conversaciones "
              "simuladas.", ""]
    return "\n".join(lines)


def _replay_panel(archived, output, source):
    if not archived:
        raise ValueError("Empty archived trajectories")
    output.mkdir(parents=True)
    cases = output / "cases"
    cases.mkdir()
    panels = {mode: [] for mode in MODES}
    outcomes = {mode: Counter() for mode in MODES[1:]}
    reasons = {mode: Counter() for mode in MODES[1:]}
    contrasts = {mode: defaultdict(_empty_contrast_counts) for mode in MODES[1:]}
    contrasts_by_seed = defaultdict(lambda: {mode: defaultdict(_empty_contrast_counts) for mode in MODES[1:]})
    examples = []
    for index, item in enumerate(archived):
        directory = cases / f"{index:04d}"
        rows = replay_case(item, directory)
        original = rows["original"]
        baseline = original["metrics"]["FJS"]
        proposal_issues, baseline_effects = _contrast_inputs(original)
        for mode, row in rows.items():
            panels[mode].append(row)
            if mode == "original":
                continue
            decision = row["guard_decision"]
            reasons[mode][decision["status"]] += 1
            for metric, (count, opportunities) in _operation_contrasts(row, proposal_issues, baseline_effects).items():
                for group in ("all", row["scenario_family"]):
                    for tally in (contrasts, contrasts_by_seed[row["training_seed"]]):
                        tally[mode][group][metric].update(numerator=count, denominator=opportunities)
            score = row["metrics"]["FJS"]
            change = "improved" if score > baseline else "worse" if score < baseline else "tied"
            outcomes[mode][change] += 1
            if mode == "repair" and change != "tied":
                examples.append({"family": row["scenario_family"], "change": "mejoró" if score > baseline else "empeoró",
                                 "directory": str(directory.relative_to(output)),
                                 **{arm: {key: rows[arm]["trace"]["future_response"][key] for key in ("mode", "value")}
                                    for arm in ("original", "repair")}})
    result = compare_trajectories(panels, reference="original")
    from research import tesis_memory_guard, tesis_metrics

    result.update(artifact_version="tesis_memory_replay_v2",
                  source=source,
                  source_sha256={Path(path).name: file_hash(path) for path in
                                 (__file__, tesis_memory_guard.__file__, tesis_metrics.__file__)},
                  outcomes=outcomes, guard_decisions=reasons, write_contrasts=contrasts,
                  write_contrasts_by_seed=contrasts_by_seed, examples=examples[:8])
    (output / "INFORME.md").write_text(render(result), encoding="utf-8")
    for mode, rows in panels.items():
        (output / f"{mode}.jsonl").write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
    write_json(output / "RESULT.json", result)
    return result


def _read_trajectories(path, expected_hash=None):
    path = Path(path).resolve()
    content = path.read_bytes()
    checksum = sha256(content).hexdigest()
    if expected_hash is not None and checksum != expected_hash:
        raise ValueError(f"Trajectory hash mismatch: {path}")
    return [json.loads(line) for line in content.splitlines() if line.strip()], {"path": str(path), "sha256": checksum}


def run(input_path, output):
    archived, source = _read_trajectories(input_path)
    return _replay_panel(archived, Path(output).resolve(), source)


def _reserved_inputs(result_path):
    from research.tesis_comparison import METHODS

    comparison = json.loads(result_path.read_bytes())
    selection_path = result_path.with_name("SELECTION.json")
    if file_hash(selection_path) != comparison["selection_sha256"]:
        raise ValueError("Reserved comparison selection hash mismatch")
    selection = json.loads(selection_path.read_bytes())
    if (selection.get("selection_complete") is not True or selection.get("selection_split") != "validation"
            or selection.get("selected_before_evaluation") is not True):
        raise ValueError("Expected selection locked before reserved evaluation")
    entries = _comparison_inputs(result_path)
    if set(entries) != set(METHODS):
        raise ValueError("Comparison methods differ from the selector comparison protocol")
    panels = {}
    loaded = {}
    expected_seeds = sorted(comparison["seeds"])
    for method in METHODS:
        if sorted(seed for _, _, seed in entries[method]) != expected_seeds:
            raise ValueError(f"Missing or repeated comparison seeds for {method}")
        rows, sources = [], []
        for path, checksum, seed in entries[method]:
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError(f"Missing trajectory checksum: {path}")
            key = (path.resolve(), checksum, seed)
            if key not in loaded:
                data, source = _read_trajectories(path, checksum)
                episodes = [row["trace"]["episode"] for row in data]
                if (not data or any(row["training_seed"] != seed or row["evaluation_role"] != "evaluation"
                                    or row["split"] != "evaluation" or row["technical_status"] != "completed"
                                    for row in data)
                        or digest(episodes) != selection["heldback_episodes_sha256"]):
                    raise ValueError(f"Trajectories differ from the reserved bank or seed: {path}")
                loaded[key] = (data, {**source, "training_seed": seed})
            data, source = loaded[key]
            rows.extend(data)
            sources.append(source)
        panels[method] = rows, sources
    return panels, selection


def run_comparison(result_path, output):
    result_path, output = Path(result_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    panels, selection = _reserved_inputs(result_path)
    source = {"path": str(result_path), "sha256": file_hash(result_path),
              "heldback_episodes_sha256": selection["heldback_episodes_sha256"]}
    results = {}
    for method, (rows, trajectories) in panels.items():
        results[method] = _replay_panel(rows, output / method, {**source, "comparison_method": method,
                                                               "trajectories": trajectories})
    scope = next(iter(results.values()))["scope"]
    lines = ["# Verificador de escritura en el conjunto reservado", "",
             f"Se reprodujeron {scope['complete_paired_cells']} pares de semilla y episodio por método, "
             f"correspondientes a {scope['independent_roots']} historias raíz y "
             f"{len(scope['training_seeds'])} semillas. Los métodos pueden utilizar los mismos pesos seleccionados. "
             "Cada contraste conserva la pregunta, la réplica y la respuesta actual.", "",
             "| Selector | Acierto futuro original | Veto | Reparación | Escrituras respaldadas perdidas con veto | Con reparación |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, result in results.items():
        outcomes = [display(result["methods"][mode]["future_supported"]["all"]) for mode in MODES]
        lost = [result["write_contrasts"][mode]["all"]["supported_effects_lost"] for mode in MODES[1:]]
        counts = [_count(metric) for metric in lost]
        lines.append(f"| {method} | " + " | ".join([*outcomes, *counts]) + " |")
    lines += ["", "Los porcentajes son medias por familia. Una fracción de 0/0 indica que el original "
              "no ejecutó ninguna escritura respaldada. Los informes por selector detallan las escrituras "
              "rechazadas, las omisiones y las familias. Los resultados se limitan al banco sintético y a los "
              "checkpoints seleccionados.", ""]
    lines.extend(f"- [{method}]({method}/INFORME.md)" for method in results)
    (output / "INFORME.md").write_text("\n".join([*lines, ""]), encoding="utf-8")
    combined = {"artifact_version": "tesis_memory_replay_comparison_v1", "scope": scope, "source": source,
                "methods": {method: {"report": f"{method}/INFORME.md", "result": f"{method}/RESULT.json",
                                     "result_sha256": file_hash(output / method / "RESULT.json"),
                                     "write_contrasts": result["write_contrasts"]}
                            for method, result in results.items()}}
    write_json(output / "RESULT.json", combined)
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectories", type=Path)
    source.add_argument("--comparison-results", type=Path, help="Completed selector comparison RESULTS.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (run_comparison(args.comparison_results, args.output) if args.comparison_results
              else run(args.trajectories, args.output))
    print(f"{result['scope']['complete_paired_cells']} episodios · {args.output / 'INFORME.md'}")


if __name__ == "__main__":
    main()
