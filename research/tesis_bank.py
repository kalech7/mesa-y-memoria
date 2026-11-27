"""Build a separately worded, paired food-preference benchmark."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from . import competence_data
from .competence_data import teacher_examples
from .src.data.semantic_partition_v3 import (
    assert_semantic_split_isolation_v3,
    observable_episode_id_v3,
)
from .src.env.episode import EpisodeEnvironment
from .src.env.user_simulator import QUESTIONS
from .src.evaluation.runner import run_episode
from .src.policy.rule_controller import RuleController
from .tesis_bank_review import prepare_review, reconcile_review
from .src.artifacts import canonical, digest, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
VERSION = "tesis_bank_v1"
DEFAULT_SIZES = {"train": 60, "validation": 40, "evaluation": 50}
DEFAULT_SEED = 683279
STRATA = {
    "F0": "stable_control",
    "F1": "ongoing_correction",
    "F2": "separate_interval",
    "F3": "contextual_scope",
    "F4": "ambiguous_duration",
}
FOOD = {
    "dinner_preference": ("la cena", (
        "locro de papa", "pescado al horno", "mote con queso", "arroz con lentejas",
        "tortilla de yuca", "sopa de zapallo", "menestra de fréjol", "verduras al wok",
        "ensalada de quinua", "choclo con queso", "estofado de garbanzos", "tacos de hongos",
    )),
    "breakfast_preference": ("el desayuno", (
        "maduros asados", "pan de yuca", "avena con canela", "yogur con mango",
        "tortilla de verde", "tostadas con tomate", "tamal de maíz", "papaya picada",
        "humita de choclo", "pan de centeno", "huevos con espinaca", "cachapa con queso",
    )),
    "drink_preference": ("la bebida", (
        "horchata lojana", "agua de cedrón", "jugo de guayaba", "infusión de menta",
        "limonada con hierbabuena", "jugo de tomate de árbol", "agua de coco",
        "café pasado", "té de jazmín", "jugo de maracuyá", "agua mineral", "batido de naranjilla",
    )),
}

# Complete discourse forms are assigned to one partition before writing the
# stories. The scope clauses retain the existing evaluator's public grammar.
WORDING = {
    "train": {
        "intro": ("Estoy armando la lista para comprar en el mercado.",
                  "Tengo un rato antes de salir a hacer las compras."),
        "assistant": "Dime qué dato quieres que use para la lista.",
        "ask": "Con lo que te conté, ¿qué anoto hoy para {meal}?",
        "old": "Para organizar las compras de cada semana, mi opción de {meal} era {value}.",
        "new": "Para {meal}, prefiero {value}",
        "context": "Entre el trabajo y la casa tengo opciones diferentes; todavía falta definir el lugar.",
        "session": "Esta vez va solo en esta conversación; después vuelvo a la pauta de antes.",
        "ongoing": "Déjalo como preferencia habitual también para las siguientes compras.",
    },
    "validation": {
        "intro": ("Estoy organizando la comida antes de la reunión del edificio.",
                  "Me falta cerrar un detalle del menú en mi cuaderno."),
        "assistant": "Cuéntame qué parte del menú falta por decidir.",
        "ask": "¿Qué opción pongo en el cuaderno para {meal} de hoy?",
        "old": "En el cuaderno de menús dejé {value} como mi opción general de {meal}.",
        "new": "En cuanto a {meal}, prefiero {value}",
        "context": "La elección cambia según el lugar. Lo del sitio todavía no te lo he dicho.",
        "session": "Tómalo solo para esta sesión. La pauta del cuaderno sigue para las próximas.",
        "ongoing": "Ponlo como preferencia habitual en el cuaderno a partir de hoy.",
    },
    "evaluation": {
        "intro": ("Me toca dejarle a mi hermano una nota para cuando abra la nevera.",
                  "Voy a dejar lista la nota de la cocina antes de que se me olvide."),
        "assistant": "Veamos qué dato tiene que quedar en la nota.",
        "ask": "Para la nota de hoy, ¿qué dejo indicado para {meal}?",
        "old": "En la nota que uso normalmente, mi elección general de {meal} era {value}.",
        "new": "Si hablamos de {meal}, prefiero {value}",
        "context": "La nota es para el sitio donde comeré. Es diferente en la oficina y en casa.",
        "session": "Cuenta solo esta ocasión. Fuera de hoy, la elección de siempre sigue en pie.",
        "ongoing": "Es mi preferencia habitual desde esta fecha. Puedes usarla en las próximas consultas.",
    },
}
CONTEXT_REPLIES = {
    "train": {"casa": "Al final estoy en casa.", "oficina": "Hoy estoy en la oficina, junto a la sala grande."},
    "validation": {"casa": "Hoy en casa; cancelaron la reunión.", "oficina": "Para este menú, hoy estoy en la oficina."},
    "evaluation": {"casa": "La nota queda aquí; estoy en casa.", "oficina": "La nota es para el trabajo: estoy en la oficina."},
}


def _scope(kind, start, *, end=None, context=None, session=None):
    return {"kind": kind, "valid_from": start, "valid_until_exclusive": end,
            "context": context, "session_id": session}


def _remember(records, sessions, attribute, value, when, text, *, kind="ongoing", end=None, context=None):
    record_id = f"m{len(records) + 1}"
    turn_id = f"prior_{record_id}"
    records.append({
        "memory_id": record_id, "attribute": attribute, "value": value, "recorded_at": when,
        "scope": _scope(kind, when, end=end, context=context), "status": "confirmed",
        "supersedes": [], "source_turn_ids": [turn_id],
        "evidence_quotes": [{"turn_id": turn_id, "text": text}],
    })
    session = sessions.setdefault(when, {"session_id": f"P_{when}", "date": when, "messages": []})
    session["messages"].append({"turn_id": turn_id, "role": "user", "text": text})


def _episodes(family, split, index, seed, pair_index):
    wording = WORDING[split]
    rng = random.Random(digest([VERSION, seed, family, index]))
    attributes = list(FOOD)
    attr = attributes[(index + seed) % len(attributes)]
    meal, food = FOOD[attr]
    base_index, offset_index = divmod(pair_index, len(food) - 1)
    if family == "F0" and index % 2 == 0:
        base_index = (index // 6 + seed) % len(food)
    base = food[base_index]
    changed = food[(base_index + offset_index + 1) % len(food)]
    other = rng.choice([attribute for attribute in attributes if attribute != attr])
    other_meal, other_food = FOOD[other]
    today = date(2034, 4, 1) + timedelta(days=rng.randrange(120))
    earlier = (today - timedelta(days=rng.randrange(12, 35))).isoformat()
    now = today.isoformat()
    end = today + timedelta(days=rng.randrange(2, 5))
    future = (today + timedelta(days=rng.randrange(6, 12))).isoformat()
    context = None
    future_context = None
    decisive = None
    boundary = "ongoing"
    memory, sessions = [], {}
    _remember(memory, sessions, attr, base, earlier, wording["old"].format(meal=meal, value=base))
    if family != "F0" or index % 2:
        other_value = rng.choice(other_food)
        _remember(memory, sessions, other, other_value, earlier,
                  wording["old"].format(meal=other_meal, value=other_value))

    clause = wording["new"].format(meal=meal, value=changed)
    worlds = [("stable", base, base, None, None)]
    if family == "F0":
        boundary = "stable_only"
        if index % 2:
            boundary = "expired_latest"
            day = (today - timedelta(days=1)).isoformat()
            _remember(memory, sessions, attr, changed, day,
                      f"La última vez elegí {changed} para {meal}; solo aplicaba el {day}.",
                      kind="interval", end=now)
        statement = "Para esta consulta puedes conservar la elección general que ya tienes."
    elif family == "F1":
        statement = f"Cambió mi elección de siempre. {clause} de forma habitual."
        worlds = [("ongoing", changed, changed, _scope("ongoing", now), None)]
    elif family == "F2":
        inclusive = (end - timedelta(days=1)).isoformat()
        statement = f"Hasta el {inclusive} inclusive, {clause}. Luego retomo la elección de siempre."
        boundary = ("inside", "exclusive_end", "after")[index % 3]
        future = {"inside": inclusive, "exclusive_end": end.isoformat(), "after": future}[boundary]
        future_value = changed if boundary == "inside" else base
        worlds = [("interval", changed, future_value, _scope("interval", now, end=end.isoformat()), None)]
    elif family == "F3":
        _remember(memory, sessions, attr, changed, earlier,
                  f"En la oficina, mi opción de {meal} es {changed}.",
                  kind="contextual", context="oficina")
        future_context = ("casa", "oficina")[index % 2]
        future_value = changed if future_context == "oficina" else base
        if index % 3:
            statement = wording["context"]
            decisive = "CURRENT_CONTEXT"
            boundary = "missing_context"
            worlds = [(place, value, future_value, None, CONTEXT_REPLIES[split][place])
                      for place, value in (("casa", base), ("oficina", changed))]
        else:
            context = ("casa", "oficina")[(index // 3) % 2]
            statement = CONTEXT_REPLIES[split][context]
            boundary = "explicit_context"
            worlds = [(context, changed if context == "oficina" else base, future_value, None, None)]
    else:
        statement = f"Pensándolo bien, tengo otra idea. {clause}."
        decisive = "TEMPORAL_SCOPE"
        boundary = "unknown_duration"
        worlds = [
            ("session", changed, base, _scope("session", now, session="S2"), wording["session"]),
            ("ongoing", changed, changed, _scope("ongoing", now), wording["ongoing"]),
        ]

    messages = [
        {"turn_id": "open_u", "role": "user", "text": rng.choice(wording["intro"])},
        {"turn_id": "open_a", "role": "assistant", "text": wording["assistant"]},
        {"turn_id": "u1", "role": "user", "text": statement},
        {"turn_id": "u2", "role": "user", "text": wording["ask"].format(meal=meal)},
    ]
    current = {"attribute": attr, "date": now, "session_id": "S2", "context": context}
    later = {"attribute": attr, "date": future, "session_id": "S3", "context": future_context}
    # Canonical recipe identity ignores text, dates and episode identifiers.
    # The assigned foods, outcome-changing scopes and boundary remain significant.
    recipe = [family, attr, base, changed if family != "F0" or index % 2 else None,
              boundary, context, future_context]
    result = []
    for world, current_value, future_value, scope, answer in worlds:
        replies = {slot: {"turn_id": "u3", "text": "No sé precisar eso ahora.", "availability": "unavailable"}
                   for slot in QUESTIONS}
        if decisive is not None:
            replies[decisive] = {"turn_id": "u3", "text": answer, "availability": "available"}
        episode = {
            "realization_id": "v0", "split": split, "partition_role": split,
            "data_namespace": VERSION, "generator_version": VERSION,
            "data_role": "held_back_development_evaluation" if split == "evaluation" else split,
            "scenario_family": family, "domain": "non_clinical_food_preferences",
            "template_id": f"{VERSION}:{split}:{family}", "composition_id": f"{family}:{boundary}",
            "initial_memory": deepcopy(memory), "prior_sessions": list(deepcopy(sessions).values()),
            "user_messages": deepcopy(messages), "current_query": current.copy(),
            "future_query": later.copy(), "replies": replies,
            "truth": {"current_value": current_value, "future_value": future_value,
                      "initial_value": base, "new_value": changed if family != "F0" else None,
                      "scope": scope, "scope_kind": world, "scope_attribute": attr,
                      "current_date": now, "current_context": world if family == "F3" else context,
                      "decisive_slot": decisive, "future_exclusive_benefit": family == "F4"},
            "opportunity": {"eligible": family == "F4", "slot": decisive, "negative_control": decisive is None},
            "authoring": {"source": VERSION, "human_authored": False, "model_results_used_for_design": False,
                          "stratum": STRATA[family], "boundary": boundary, "recipe_id": digest(recipe),
                          "writable_scope_before_ask": {"F1": "ongoing", "F2": "interval"}.get(family, "none")},
        }
        public = observable_episode_id_v3(episode)
        episode.update(root_id=f"tb{public[:20]}", observable_state_id=public,
                       latent_world_id=f"tw{digest([public, world])[:20]}")
        result.append(episode)
    return result


def build_splits(sizes=None, *, seed=DEFAULT_SEED):
    sizes = dict(DEFAULT_SIZES if sizes is None else sizes)
    if set(sizes) != set(DEFAULT_SIZES) or any(type(n) is not int or n < 5 or n % 5 for n in sizes.values()):
        raise ValueError("Se requieren raíces positivas por partición y múltiplos de cinco.")
    if sum(sizes.values()) // 5 > 72:
        raise ValueError("La baraja permite hasta 72 historias por familia.")
    # An independently shuffled food-pair deck for each family; disjoint
    # pairs across splits even if the same meal attribute is sampled.
    rng = random.Random(seed)
    decks = {family: rng.sample(range(132), sum(sizes.values()) // 5) for family in STRATA}
    splits, offset = {}, 0
    for split in DEFAULT_SIZES:
        count = sizes[split] // 5
        episodes = [_episodes(family, split, index, seed, decks[family][index])
                    for family in STRATA for index in range(offset, offset + count)]
        splits[split] = [row for variants in episodes for row in variants]
        rng.shuffle(splits[split])
        offset += count
    return splits


def audit_isolation(splits, *, historical_splits=None):
    report = assert_semantic_split_isolation_v3(splits, historical_splits=historical_splits)
    seen, grouped = {}, defaultdict(list)
    for role, rows in splits.items():
        for episode in rows:
            for key in ("template_id", "recipe_id"):
                identity = episode["authoring"][key] if key == "recipe_id" else episode[key]
                existing = seen.setdefault((key, identity), role)
                if existing != role:
                    raise ValueError(f"{key} cruza particiones: {identity}")
            grouped[episode["root_id"]].append(episode)
    paired_counts = Counter()
    with TemporaryDirectory(prefix="tesis_bank_public_") as work:
        for root, rows in grouped.items():
            first = rows[0]
            paired_family = first["scenario_family"] == "F4" or (
                first["scenario_family"] == "F3" and first["truth"]["decisive_slot"] == "CURRENT_CONTEXT")
            if len(rows) != (2 if paired_family else 1):
                raise ValueError(f"Número de mundos incoherente en {root}")
            if not paired_family:
                continue
            prefixes = []
            for index, row in enumerate(rows):
                env = EpisodeEnvironment(row, Path(work) / f"{root}_{index}.sqlite")
                try:
                    prefixes.append(canonical(env.observe("select", "O2")))
                finally:
                    env.dispose()
            slot = first["truth"]["decisive_slot"]
            answers = {row["replies"][slot]["text"] for row in rows}
            worlds = {row["truth"]["scope_kind"] for row in rows}
            expected_worlds = {"session", "ongoing"} if slot == "TEMPORAL_SCOPE" else {"casa", "oficina"}
            if len(set(prefixes)) != 1 or len(answers) != 2 or worlds != expected_worlds:
                raise ValueError(f"Par contrafactual inválido en {root}")
            differing_field = "future_value" if slot == "TEMPORAL_SCOPE" else "current_value"
            same_field = "current_value" if slot == "TEMPORAL_SCOPE" else "future_value"
            if rows[0]["truth"][same_field] != rows[1]["truth"][same_field] or (
                    rows[0]["truth"][differing_field] == rows[1]["truth"][differing_field]):
                raise ValueError(f"El par {root} cambia el horizonte equivocado")
            paired_counts[f"{first['split']}:{first['scenario_family']}"] += 1
    return {**report, "templates_and_recipe_ids_disjoint": True, "paired_public_prefixes": dict(paired_counts)}


def _audit_contracts(splits):
    static_failures = competence_data._static_audit(splits)
    if static_failures:
        raise ValueError(f"Falló el contrato estático: {static_failures[:3]}")
    coverage = {}
    with TemporaryDirectory(prefix="tesis_bank_contracts_") as work:
        for role in ("train", "validation"):
            counts = Counter()
            for index, episode in enumerate(splits[role]):
                row = run_episode(episode, RuleController(), method="B4", condition="O2",
                                  db_path=Path(work) / f"{role}_{index:04}.sqlite")
                metrics, trace = row["metrics"], row["trace"]
                expected_slot = episode["truth"]["decisive_slot"]
                if (metrics["C2"], metrics["FJS"], metrics["unsupported_operations"], metrics["invalid_outputs"],
                        trace.get("asked_slot")) != (1, 1, 0, 0, expected_slot):
                    raise ValueError(f"Reglas y contrato literal divergen: {episode['root_id']} "
                                     f"{episode['truth']['scope_kind']} {metrics}")
                counts.update(episodes=1, C2=metrics["C2"], FJS=metrics["FJS"])
                counts[episode["scenario_family"]] += 1
            coverage[role] = dict(counts)
    return {"status": "verified", "coverage": coverage, "eval_policy": "static_contracts_only",
            "Qwen_calls": 0, "training_steps": 0}


def _write_rows(path, rows):
    count = 0
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
            count += 1
    return {"path": path.name, "sha256": file_hash(path), "examples": count}


def prepare(output, *, sizes=None, seed=DEFAULT_SEED, historical_paths=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Elige un destino nuevo para congelar el banco.")
    historical_paths = (competence_data.discover_previous_banks(exclude=output)
                        if historical_paths is None else historical_paths)
    if not historical_paths:
        raise ValueError("Se necesita al menos un banco histórico para comprobar aislamiento.")
    historical_paths = {name: Path(path).resolve() for name, path in historical_paths.items()}
    historical_hashes = {name: file_hash(path) for name, path in historical_paths.items()}
    historical, adaptations = competence_data._historical_views(historical_paths)
    splits = build_splits(sizes, seed=seed)
    isolation = audit_isolation(splits, historical_splits=historical)
    contracts = _audit_contracts(splits)
    sources = {**competence_data._sources(), **{str(path.relative_to(ROOT)): path for path in
               (Path(__file__).resolve(), Path(__file__).with_name("tesis_bank_review.py").resolve(),
                ROOT / "research/src/data/audit_splits.py", ROOT / "research/src/artifacts.py")}}
    hashes = {name: file_hash(path) for name, path in sources.items()}
    output.mkdir(parents=True)
    partitions = {}
    for role, rows in splits.items():
        info = _write_rows(output / f"{role}.jsonl", rows)
        roots = {row["root_id"]: row["scenario_family"] for row in rows}
        partitions[role] = {**info, "roots": len(roots), "episodes": len(rows),
                            "families": dict(Counter(roots.values()))}
    teachers = {role: _write_rows(output / f"teacher_{role}.jsonl", teacher_examples(splits[role]))
                for role in ("train", "validation")}
    audit_path = output / "audit_contracts.json"
    write_json(audit_path, {**contracts, "isolation": isolation})
    review = prepare_review([*splits["train"], *splits["validation"]], output / "review", seed=seed)
    if historical_hashes != {name: file_hash(path) for name, path in historical_paths.items()} or (
            hashes != {name: file_hash(path) for name, path in sources.items()}):
        raise RuntimeError("Cambió una entrada mientras se congelaba el banco.")
    manifest = {
        "version": VERSION, "seed": seed, "sizes": {role: p["roots"] for role, p in partitions.items()},
        "human_review": False, "confirmatory_test": False, "partitions": partitions, "teachers": teachers,
        "isolation": isolation, "review": review, "source_hashes_after": hashes,
        "historical_paths": {name: str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
                             for name, path in historical_paths.items()},
        "historical_hashes_after": historical_hashes, "historical_identity_adaptations": adaptations,
        "audit_contracts": {"status": "verified", "path": audit_path.name, "sha256": file_hash(audit_path)},
        "scope": "Banco exploratorio sintético de autoría IA separada de los generadores históricos.",
        "limits": ["Gramática de alcance compartida con el evaluador.",
                   "La deduplicación de recetas cubre esta construcción; no es un detector general de paráfrasis.",
                   "La revisión humana exige decisiones originales separadas del manifiesto del banco."],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_and_check(bank, *, verify_sources=True):
    """Read the frozen bank through the existing compatible verifier."""
    bank = Path(bank)
    manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != VERSION:
        raise ValueError("Versión de banco inesperada.")
    historical = {name: ROOT / path for name, path in manifest["historical_paths"].items()}
    for item in manifest["review"]["artifacts"].values():
        if file_hash(bank / "review" / item["path"]) != item["sha256"]:
            raise ValueError("Cambió un archivo congelado de la revisión humana.")
    return competence_data.load_and_check(bank, historical_paths=historical, verify_sources=verify_sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    for role, default in DEFAULT_SIZES.items():
        prepare_parser.add_argument(f"--{role}-roots", type=int, default=default)
    verify = commands.add_parser("verify")
    verify.add_argument("--bank", required=True)
    review_parser = commands.add_parser("review")
    review_parser.add_argument("--bank", required=True)
    review_parser.add_argument("--review-a", required=True)
    review_parser.add_argument("--review-b", required=True)
    review_parser.add_argument("--adjudications")
    review_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        sizes = {role: getattr(args, role + "_roots") for role in DEFAULT_SIZES}
        manifest = prepare(args.output, sizes=sizes, seed=args.seed)
        print(canonical({"sizes": manifest["sizes"], "review": manifest["review"]["status"]}))
    elif args.command == "verify":
        _, manifest = load_and_check(args.bank)
        print(canonical({"sizes": manifest["sizes"], "isolation": manifest["isolation"]}))
    else:
        _, manifest = load_and_check(args.bank)
        report = reconcile_review(Path(args.bank), manifest, args.review_a, args.review_b,
                                  args.output, adjudications=args.adjudications)
        print(canonical({key: report[key] for key in ("status", "roots", "unresolved_audit_ids")}))
        return int(report["status"] != "passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
