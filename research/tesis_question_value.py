"""Measure the value of asking with every legal branch on development data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from research.competence_evaluation import _phase, _run_trajectory
from research.grpo_select_pilot import MODEL_PATH, SplitController, policy_return, scores
from research.legal_selection_policy import ACTION_CATALOG
from research.src.artifacts import canonical, digest, file_hash, write_json
from research.src.env.episode import EpisodeEnvironment
from research.src.policy.rule_controller import RuleController


ACTIONS = tuple(json.loads(action) for action in ACTION_CATALOG)
PROCEED = 0
TEMPORAL = next(i for i, action in enumerate(ACTIONS) if action["slot"] == "TEMPORAL_SCOPE")
COSTS = (0, .10, .25, .50, 1, 2, 3.5)


def action_name(index):
    return ACTIONS[index]["slot"] or ACTIONS[index]["action"]


class ForcedSelector:
    def __init__(self, downstream, index):
        self.downstream = downstream
        self.index = index

    def select(self, observation):
        return ACTIONS[self.index].copy()

    def respond(self, observation):
        return self.downstream.respond(observation)

    def write(self, observation):
        return self.downstream.write(observation)


def make_controller(policy, index=None):
    if policy is not None:
        return SplitController(policy, policy, forced_action=index)
    return RuleController() if index is None else ForcedSelector(RuleController(), index)


def run_branch(episode, folder, controller, index, *, fixed_response=None):
    mediation = fixed_response is not None
    name = "TEMPORAL_SCOPE_I1" if mediation else action_name(index)
    trace = _run_trajectory(controller, episode, folder / f"{name}.sqlite",
                            route="I1" if mediation else "natural", fixed_response=fixed_response,
                            method="forced_" + name)
    trace_path = folder / f"{name}.json"
    write_json(trace_path, trace)
    metrics = scores(trace)
    task, penalized = policy_return(metrics)
    return trace, {**metrics, "task": task, "return_before_question_cost": penalized + .10 * metrics["questions"],
                   "trace_path": str(trace_path), "trace_sha256": file_hash(trace_path)}


def inspect_case(episode, folder, policy):
    folder.mkdir(parents=True)
    with TemporaryDirectory(prefix="question-value-prefix-") as temporary:
        environment = EpisodeEnvironment(episode, Path(temporary) / "prefix.sqlite")
        try:
            observation = environment.observe("select", "O2")
        finally:
            environment.dispose()
    prefix = digest(observation)
    decision = _phase(make_controller(policy), "select", observation)
    if decision["error"] or decision["action"] not in ACTIONS:
        raise ValueError("The comparison policy must select a legal action")
    natural = ACTIONS.index(decision["action"])
    traces, branches = {}, []
    for index in range(len(ACTIONS)):
        trace, result = run_branch(episode, folder, make_controller(policy, index), index)
        if trace["phases"]["select"]["observation_sha256"] != prefix:
            raise RuntimeError("A forced action did not start from the common SELECT observation")
        traces[index] = trace
        branches.append(result)
    mediation = None
    temporal = traces[TEMPORAL]
    if branches[TEMPORAL]["informative_questions"]:
        hidden, hidden_score = run_branch(episode, folder, make_controller(policy, TEMPORAL), TEMPORAL,
                                          fixed_response=temporal["phases"]["respond"])
        if hidden["trace"]["user_reply"] != temporal["trace"]["user_reply"]:
            raise RuntimeError("The writer intervention changed the reply")
        visible_obs = temporal["trace"]["observations"]["write"]
        hidden_obs = hidden["trace"]["observations"]["write"]
        reply_id = temporal["trace"]["user_reply"]["turn_id"]
        without_reply = {**visible_obs, "user_messages": [m for m in visible_obs["user_messages"]
                                                            if m["turn_id"] != reply_id]}
        if hidden_obs != without_reply or hidden_score["C2"] != branches[TEMPORAL]["C2"]:
            raise RuntimeError("The writer intervention changed more than the availability of the reply")
        mediation = {"reply_available": branches[TEMPORAL]["FJS"], "reply_hidden": hidden_score["FJS"],
                     "delta_FJS": branches[TEMPORAL]["FJS"] - hidden_score["FJS"],
                     "hidden_trace": hidden_score["trace_path"], "hidden_sha256": hidden_score["trace_sha256"]}
    return {**{key: episode[key] for key in ("root_id", "latent_world_id", "realization_id", "scenario_family")},
            "prefix": prefix, "natural_action": natural, "branches": branches, "writer_intervention": mediation}


def return_at(row, index, cost):
    branch = row["branches"][index]
    return branch["return_before_question_cost"] - cost * branch["questions"]


def best_action(values):
    return max(range(len(ACTIONS)), key=lambda index: (round(values[index], 10), -index))


def prefix_summary(rows):
    """Every decision here is shared across worlds with identical observations."""
    worlds = defaultdict(list)
    for row in rows:
        worlds[row["latent_world_id"]].append(row)

    def avg(f):
        return mean(mean(f(row) for row in group) for group in worlds.values())

    natural, = {row["natural_action"] for row in rows}
    values = [avg(lambda row: return_at(row, index, 0)) for index in range(len(ACTIONS))]
    break_even = max(values[1:]) - values[PROCEED]
    best_question = max(range(1, len(ACTIONS)), key=lambda i: values[i])
    result = {"worlds": len(worlds), "realizations": len(rows), "natural_action": action_name(natural),
              "best_question": action_name(best_question),
              "question_break_even_cost": break_even,
              "branches_at_zero_cost": {action_name(i): value for i, value in enumerate(values)}, "frontier": []}
    for cost in COSTS:
        with_cost = [avg(lambda row: return_at(row, index, cost)) for index in range(len(ACTIONS))]
        best = best_action(with_cost)
        hindsight = mean(max(mean(return_at(row, index, cost) for row in group)
                             for index in range(len(ACTIONS))) for group in worlds.values())
        result["frontier"].append({"cost": cost, "action": action_name(best),
            "natural_return": with_cost[natural], "proceed_return": with_cost[PROCEED],
            "best_shared_return": with_cost[best], "per_world_upper_bound": hindsight,
            "selector_regret": max(0, with_cost[best] - with_cost[natural]),
            "perfect_information_gap": max(0, hindsight - with_cost[best]),
            "delta_current_vs_proceed": avg(lambda row: row["branches"][best]["C2"] - row["branches"][0]["C2"]),
            "delta_future_vs_proceed": avg(lambda row: row["branches"][best]["FJS"] - row["branches"][0]["FJS"]),
            "delta_strict_joint_vs_proceed": avg(lambda row: row["branches"][best]["strict_joint"] - row["branches"][0]["strict_joint"])})
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["root_id"], row["prefix"])].append(row)
    prefixes = []
    for (root, prefix), group in sorted(groups.items()):
        prefixes.append({"root_id": root, "prefix": prefix, "family": group[0]["scenario_family"],
                         **prefix_summary(group)})
    family_groups = defaultdict(lambda: defaultdict(list))
    for panel in prefixes:
        family_groups[panel["family"]][panel["root_id"]].append(panel)
    families = {}
    for family, roots in sorted(family_groups.items()):
        families[family] = {"roots": len(roots), "prefixes": sum(map(len, roots.values())), "frontier": []}
        for index, cost in enumerate(COSTS):
            panel = {"cost": cost}
            for field in ("selector_regret", "perfect_information_gap", "delta_current_vs_proceed",
                          "delta_future_vs_proceed", "delta_strict_joint_vs_proceed"):
                panel[field] = mean(mean(row["frontier"][index][field] for row in group) for group in roots.values())
            panel["actions_by_prefix"] = dict(Counter(row["frontier"][index]["action"]
                                              for group in roots.values() for row in group))
            families[family]["frontier"].append(panel)
    return prefixes, families


def render(result):
    lines = ["# Valor de preguntar antes de actualizar la memoria", "",
             f"Auditoría exploratoria sobre {result['roots']} historias de validación y "
             f"{result['legal_branches']} ramas legales. Cada rama se ejecutó en su propia base SQLite.", "",
             "Para cada diálogo visible se compararon cuatro preguntas y continuar. La mejor acción "
             "se calculó promediando los mundos compatibles con el mismo diálogo. La ventaja y el coste "
             "se expresan en unidades de la recompensa registrada en `RESULT.json`.", "",
             "| Familia | Historias | Prefijos con pregunta óptima al coste 0,10 | Δ acierto actual | Δ acierto futuro |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, family in result["families"].items():
        row = next(row for row in family["frontier"] if row["cost"] == .10)
        ask = family["prefixes"] - row["actions_by_prefix"].get("PROCEED", 0)
        lines.append(f"| {name} | {family['roots']} | {ask}/{family['prefixes']} | "
                     f"{100 * row['delta_current_vs_proceed']:+.1f} pp | "
                     f"{100 * row['delta_future_vs_proceed']:+.1f} pp |")
    outcome = result["writer_intervention"]
    lines.extend(["", f"En {outcome['improved']} de {outcome['denominator']} ramas temporales informativas, "
                  "el acierto futuro mejoró al permitir que la aclaración llegara al escritor. "
                  f"Empeoró en {outcome['worse']} ramas. La pregunta, la réplica y la respuesta actual fueron idénticas.", "",
                  "`COSTE_POR_FAMILIA.csv` contiene la sensibilidad al coste y `RESULT.json` el detalle por prefijo. "
                  "`cases.jsonl` enlaza cada resultado con su traza y el estado persistido.", "",
                  "Los máximos estimados dependen del simulador, del escritor, del lector y de la recompensa fijada. "
                  "Son referencias retrospectivas de desarrollo y la evaluación se limita a conversaciones simuladas.", ""])
    return "\n".join(lines)


def run(episodes, output, *, provenance, policy=None):
    if not episodes or any(row["split"] != "validation" for row in episodes):
        raise ValueError("Branch analysis accepts validation episodes only")
    identities = [(row["root_id"], row["latent_world_id"], row["realization_id"]) for row in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError("Repeated validation identity")
    output = Path(output).resolve()
    output.mkdir(parents=True)
    rows = []
    for index, episode in enumerate(episodes, 1):
        rows.append(inspect_case(episode, output / "branches" / f"{index:04d}", policy))
    prefixes, families = summarize(rows)
    interventions = [row["writer_intervention"] for row in rows if row["writer_intervention"] is not None]
    result = {"version": "tesis_question_value_v1", "scope": "exploratory_validation", "provenance": provenance,
              "reward": "2*strict_joint + .5*C2 + .5*FJS + .5*causal_write_delta - .50*unsupported_write - .20*invalid_output - cost*ASK",
              "episodes": len(rows), "roots": len({key[0] for key in identities}), "legal_branches": len(ACTIONS) * len(rows),
              "writer_intervention": {"denominator": len(interventions),
                 "scope": "informative TEMPORAL_SCOPE branches; same reply and current response, hidden only from WRITE",
                 "improved": sum(row["delta_FJS"] > 0 for row in interventions),
                 "worse": sum(row["delta_FJS"] < 0 for row in interventions)},
              "families": families, "prefixes": prefixes}
    (output / "cases.jsonl").write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
    with (output / "COSTE_POR_FAMILIA.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("family", "roots", "prefixes", "cost", "ask_prefixes", "selector_regret", "perfect_information_gap",
                  "delta_current_vs_proceed", "delta_future_vs_proceed", "delta_strict_joint_vs_proceed")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, family in families.items():
            for row in family["frontier"]:
                counts = row["actions_by_prefix"]
                writer.writerow({"family": name, "roots": family["roots"], "prefixes": family["prefixes"],
                                 "ask_prefixes": family["prefixes"] - counts.get("PROCEED", 0),
                                 **{field: row[field] for field in row if field != "actions_by_prefix"}})
    write_json(output / "RESULT.json", result)
    (output / "INFORME.md").write_text(render(result), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional frozen LoRA selector, responder and writer")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--family", action="append", choices=("F0", "F1", "F2", "F3", "F4"))
    parser.add_argument("--roots-per-family", type=int)
    args = parser.parse_args(argv)
    if args.roots_per_family is not None and args.roots_per_family < 1:
        parser.error("--roots-per-family must be positive")
    from research.tesis_bank import load_and_check
    splits, _ = load_and_check(args.bank)
    families = defaultdict(set)
    for row in splits["validation"]:
        if not args.family or row["scenario_family"] in args.family:
            families[row["scenario_family"]].add(row["root_id"])
    roots = {root for group in families.values() for root in sorted(group)[:args.roots_per_family]}
    episodes = [row for row in splits["validation"] if row["root_id"] in roots]
    provenance = {"validation_subset_sha256": digest(episodes), "selected_roots": sorted(roots),
                  "bank_manifest_sha256": file_hash(args.bank / "manifest.json"),
                  "downstream": "public_rules", "code_sha256": file_hash(Path(__file__))}
    policy = None
    if args.checkpoint:
        from research.grpo_select_checkpoint import load_selector
        from research.weak_sft_experiment import selection_record
        provenance["downstream"] = selection_record("frozen_LoRA", args.checkpoint, 0)
        provenance["device"] = args.device
        policy = load_selector(args.checkpoint, device=args.device, model_path=args.model_path)
    result = run(episodes, args.output, provenance=provenance, policy=policy)
    print(f"{result['roots']} historias · {result['legal_branches']} ramas · {args.output / 'RESULT.json'}")


if __name__ == "__main__":
    main()
