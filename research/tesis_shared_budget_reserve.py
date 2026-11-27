"""Freeze blind Spanish review packets outside the CPU pilot's data path."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import random

from research.src.artifacts import digest, file_hash, write_json


VERSION = "tesis_shared_budget_reserved_draft_v1"
SCOPES = {"none", "session", "interval", "ongoing", "contextual"}
VALIDITY_FIELDS = ("writable_scope", "valid_from", "valid_through", "context")
GUIDE = [
    "Evalúa tus casos individualmente. Usa únicamente esta carpeta en la primera lectura.",
    "Decide qué respuesta sería admisible hoy y qué preferencia personal queda sustentada para otra sesión.",
    "En valid_from y valid_through usa YYYY-MM-DD. valid_through es el último día incluido. Usa null si no aplica o no se puede determinar.",
    "Usa none si la actualización duradera no se puede justificar. Para una elección local que sí puedes confirmar, usa session.",
    "El presupuesto incluye como máximo una pregunta dirigida al usuario entre ambas visitas. Indica cuándo la gastarías, si lo harías.",
    "Primero completa visitas y justificación con 01_casos.json. Después abre 02_replicas.json para valorar si las réplicas de un mismo caso son plausibles.",
    "Mantén los audit_id y un identificador de revisor distinto al de la otra persona. Devuelve 03_respuestas_en_blanco.json completado.",
]


def _visit(day, user, request):
    return {"date": day, "user": user, "request": request}


def _validity(scope, start=None, end=None, context=None):
    return {"writable_scope": scope, "valid_from": start, "valid_through": end, "context": context}


# These are draft semantic hypotheses for reviewers, not human labels. They are
# completely separate from the synthetic development generator and CPU runner.
CASES = (
    {
        "family": "implicit_correction",
        "memory": "Antes el usuario pedía patacones como desayuno de todos los domingos.",
        "visits": [
            _visit("2041-08-10", "Ya le cogí gusto al bolón. Borra los patacones de mi desayuno de los domingos; deja el bolón en ese lugar.",
                   "¿Qué dejo listo para los domingos?"),
            _visit("2041-08-17", "Mañana ordeno los ingredientes otra vez.", "¿Qué pongo para mi desayuno habitual del domingo?"),
        ],
        "expected": [_validity("contextual", "2041-08-10", context="domingo"), _validity("none")],
    },
    {
        "family": "implicit_correction",
        "memory": "El usuario solía guardar caldo de pollo como cena al llegar del trabajo.",
        "visits": [
            _visit("2041-08-14", "Cuando vuelva del trabajo, cámbiame ese caldo por locro en el menú que dejamos guardado.",
                   "¿Con qué plato queda mi menú de regreso del trabajo?"),
            _visit("2041-08-21", "Hoy salgo tarde del trabajo. Abrí el chat de nuevo para revisar el menú.",
                   "¿Qué cena tenía para cuando regreso?"),
        ],
        "expected": [_validity("contextual", "2041-08-14", context="regreso del trabajo"), _validity("none")],
    },
    {
        "family": "implicit_correction",
        "memory": "La opción de bebida habitual del usuario es agua de coco.",
        "visits": [
            _visit("2041-08-05", "Saca el agua de coco de mi lista de siempre. La nueva del día a día es agua con limón.",
                   "¿Qué pongo en mi lista habitual de bebida?"),
            _visit("2041-08-06", "Para la visita me encargaron horchata, pero yo no cambié mi bebida.",
                   "¿Cuál habíamos dejado para mí?"),
        ],
        "expected": [_validity("ongoing", "2041-08-05"), _validity("none")],
    },
    {
        "family": "temporal_wording",
        "memory": "El usuario elige normalmente tortilla de maíz como desayuno.",
        "visits": [
            _visit("2041-08-08", "La avena sírvemela del ocho al doce de agosto, ambos días cuentan. El trece vuelvo a la tortilla.",
                   "¿Qué desayuno pido hoy?"),
            _visit("2041-08-13", "Ya estamos a trece de agosto.", "¿Qué desayuno debía retomar?"),
        ],
        "expected": [_validity("interval", "2041-08-08", "2041-08-12"), _validity("none")],
    },
    {
        "family": "temporal_wording",
        "memory": "El usuario acostumbra beber guayusa.",
        "visits": [
            _visit("2041-08-17", "Ponme agua de jamaica desde hoy hasta que acabe el domingo 18. El lunes sigo con la guayusa.",
                   "¿Qué bebida me sirvo para hoy?"),
            _visit("2041-08-18", "El almuerzo de hoy todavía cuenta como parte del fin de semana.",
                   "¿Qué bebida aplicaba este domingo?"),
        ],
        "expected": [_validity("interval", "2041-08-17", "2041-08-18"), _validity("none")],
    },
    {
        "family": "temporal_wording",
        "memory": "El usuario suele cenar mote con queso en casa.",
        "visits": [
            _visit("2041-08-27", "Me quedan tamales del cumpleaños; por esta vuelta sirvo eso para la cena, luego seguimos igual.",
                   "¿Qué ceno esta vez?"),
            _visit("2041-08-28", "Hoy el resto de los tamales se lo llevó mi vecino.", "¿Qué era lo que solía cenar en casa?"),
        ],
        "expected": [_validity("session", "2041-08-27"), _validity("none")],
    },
    {
        "family": "other_person",
        "memory": "El usuario dejó guardada una preferencia habitual por café pasado.",
        "visits": [
            _visit("2041-08-23", "Mi tía tomó el teléfono y escribió ‘prefiero té de cedrón’. Era ella la que estaba eligiendo.",
                   "¿Qué anoto en su taza y qué me dejas a mí?"),
            _visit("2041-08-24", "Estoy solo en casa. Lo de la taza de ayer era de mi tía.",
                   "¿Qué bebida era mi opción de siempre?"),
        ],
        "expected": [_validity("none"), _validity("none")],
    },
    {
        "family": "other_person",
        "memory": "El desayuno habitual del usuario es pan integral con aguacate.",
        "visits": [
            _visit("2041-08-25", "Estoy copiando el menú de mi sobrina: ‘prefiero cereales de chocolate’. Para cuando venga, recuérdame que eso es de ella.",
                   "¿Qué puedo comer yo mañana?"),
            _visit("2041-08-26", "Mi sobrina ya se fue y voy a preparar lo mío.", "¿Qué desayuno tengo guardado?"),
        ],
        "expected": [_validity("none"), _validity("none")],
    },
    {
        "family": "competing_questions",
        "memory": "El usuario recuerda como habituales la sopa de fideo para cenar y el jugo de mora para beber. Acostumbra agua en la oficina.",
        "visits": [
            _visit("2041-08-11", "Para la cena me voy por el sancocho. Estoy cerrando el teléfono; mañana quizá coma fuera del sitio habitual.",
                   "¿Qué sirvo para esta cena?"),
            _visit("2041-08-12", "Para la bebida, ya sabes que cambia si estoy en el trabajo o en casa.",
                   "¿Qué bebida me corresponde aquí?"),
        ],
        "replies": {"visit": 2, "question": "¿Ahora estás en el trabajo o en casa?",
                    "answers": ["Vine a la oficina.", "Estoy en casa; cambié de plan."]},
        "expected": [_validity("session", "2041-08-11"), _validity("none")],
    },
    {
        "family": "competing_questions",
        "memory": "El usuario solía tener tostadas con queso como desayuno habitual. Suele beber una infusión de menta.",
        "visits": [
            _visit("2041-08-15", "Estoy armando los desayunos de la semana. Hoy me quedo con humita. Tengo un rato por si hace falta precisar algo.",
                   "¿Qué pongo para mi desayuno de hoy?"),
            _visit("2041-08-16", "Volví a armar la lista de mercado; la bebida queda igual.",
                   "¿Qué desayuno tengo como opción cuando vuelva a hacer compras?"),
        ],
        "replies": {"visit": 1, "question": "¿Quieres repetir la humita como opción habitual o solo hoy?",
                    "answers": ["Solo fue por hoy; para las otras compras piensa en las tostadas.",
                                "Me gustó para dejarla en la lista base de esta semana y las que vienen."]},
        "expected": [_validity("session", "2041-08-15"), _validity("none")],
    },
)


def freeze(output):
    """Keep author hypotheses apart from two differently ordered blind packets."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases, replies, keys = [], [], {}
    for case in CASES:
        audit_id = "caso-" + digest([VERSION, case["memory"], case["visits"]])[:14]
        cases.append({"audit_id": audit_id, "memory": case["memory"], "visits": case["visits"]})
        if "replies" in case:
            replies.append({"audit_id": audit_id, **case["replies"]})
        keys[audit_id] = {"family": case["family"], "draft_validity": case["expected"],
                           "has_reply_pair": "replies" in case}
    artifacts = {}

    def save(name, value):
        path = output / name
        write_json(path, value)
        artifacts[name] = file_hash(path)

    save("coordinacion/clave_borrador.json", keys)
    save("coordinacion/adjudicaciones_en_blanco.json", [])
    for reviewer in ("a", "b"):
        rng = random.Random(digest([VERSION, reviewer]))
        order = rng.sample(cases, len(cases))
        shuffled_replies = [{**row, "answers": rng.sample(row["answers"], len(row["answers"]))}
                            for row in rng.sample(replies, len(replies))]
        prefix = f"revisor_{reviewer}"
        save(f"{prefix}/01_casos.json", {"instructions": GUIDE, "question_budget": 1,
                                        "allowed_scopes": sorted(SCOPES), "cases": order})
        save(f"{prefix}/02_replicas.json", shuffled_replies)
        save(f"{prefix}/03_respuestas_en_blanco.json", [
            {"audit_id": row["audit_id"], "reviewer_id": "", "visits": [
                {"admissible_response": "", **dict.fromkeys(VALIDITY_FIELDS)} for _ in row["visits"]],
             "question_visit": None, "question_text": "", "both_replies_plausible": None, "rationale": ""}
            for row in order
        ])
    manifest = {"version": VERSION, "status": "pending_independent_authorship_and_human_review",
                "evaluation_ready": False, "human_authored": False, "human_reviews_received": 0,
                "root_count": len(cases), "paired_reply_roots": len(replies), "artifacts": artifacts}
    write_json(output / "MANIFEST.json", manifest)
    return manifest


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(folder):
    folder = Path(folder)
    manifest = _read(folder / "MANIFEST.json")
    if manifest["version"] != VERSION or any(file_hash(folder / name) != sha
                                             for name, sha in manifest["artifacts"].items()):
        raise ValueError("Los materiales de revisión difieren del congelado.")
    return manifest


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _read_completed(path, keys):
    rows = _read(path)
    if not isinstance(rows, list):
        raise ValueError("Se requiere una lista de respuestas por archivo.")
    reviewed = {}
    for row in rows:
        identity = row.get("audit_id") if isinstance(row, dict) else None
        if not isinstance(identity, str) or identity not in keys or identity in reviewed:
            raise ValueError(f"Caso de revisión desconocido o duplicado {identity}")
        if not _nonempty(row.get("reviewer_id")) or not _nonempty(row.get("rationale")):
            raise ValueError(f"Falta identificador o justificación en {identity}")
        visits = row.get("visits", [])
        if not isinstance(visits, list) or len(visits) != len(keys[identity]["draft_validity"]):
            raise ValueError(f"Número de visitas inválido en {identity}")
        for visit in visits:
            if (not isinstance(visit, dict) or not set(VALIDITY_FIELDS).issubset(visit)
                    or not _nonempty(visit.get("admissible_response"))
                    or not _nonempty(visit["writable_scope"]) or visit["writable_scope"] not in SCOPES):
                raise ValueError(f"Falta respuesta admisible o alcance en {identity}")
            for field in ("valid_from", "valid_through"):
                if visit[field] is not None:
                    if not isinstance(visit[field], str) or date.fromisoformat(visit[field]).isoformat() != visit[field]:
                        raise ValueError(f"Fecha inválida en {identity}")
            if visit["context"] is not None and not _nonempty(visit["context"]):
                raise ValueError(f"Contexto inválido en {identity}")
        question_visit = row["question_visit"]
        if question_visit is not None and (type(question_visit) is not int or not 1 <= question_visit <= len(visits)
                                           or not _nonempty(row.get("question_text"))):
            raise ValueError(f"Indica en qué visita gastarías la pregunta y cuál harías en {identity}")
        if keys[identity]["has_reply_pair"] and type(row.get("both_replies_plausible")) is not bool:
            raise ValueError(f"Falta juzgar las dos réplicas en {identity}")
        reviewed[identity] = row
    reviewer_ids = {row["reviewer_id"].strip() for row in reviewed.values()}
    if set(reviewed) != set(keys) or len(reviewer_ids) != 1:
        raise ValueError("Cada archivo debe contener todos los casos y pertenecer a una sola persona.")
    return reviewed, reviewer_ids.pop()


def compare_reviews(folder, review_a, review_b):
    """Check completeness and report disagreements for independent adjudication."""
    verify(folder)
    keys = _read(Path(folder) / "coordinacion/clave_borrador.json")
    a, a_id = _read_completed(review_a, keys)
    b, b_id = _read_completed(review_b, keys)
    if a_id == b_id:
        raise ValueError("El protocolo requiere dos revisores distintos.")
    cases = []
    agreement = Counter()
    for identity, key in keys.items():
        notes = []
        for index, (first, second, draft) in enumerate(zip(a[identity]["visits"], b[identity]["visits"],
                                                           key["draft_validity"]), 1):
            for field in VALIDITY_FIELDS:
                agreement[field] += int(first[field] == second[field])
                if first[field] != second[field] or first[field] != draft[field]:
                    notes.append({"visit": index, "field": field, "a": first[field], "b": second[field], "draft": draft[field]})
        if a[identity]["question_visit"] != b[identity]["question_visit"]:
            notes.append({"field": "question_visit", "a": a[identity]["question_visit"], "b": b[identity]["question_visit"]})
        if key["has_reply_pair"] and not all(row[identity]["both_replies_plausible"] for row in (a, b)):
            notes.append({"field": "both_replies_plausible", "a": a[identity]["both_replies_plausible"],
                          "b": b[identity]["both_replies_plausible"]})
        cases.append({"audit_id": identity, "family": key["family"], "structured_differences": notes})
    return {"status": "pending_adjudication_and_independent_authorship", "evaluation_ready": False,
            "review_input_hashes": [file_hash(review_a), file_hash(review_b)],
            "scope_agreement_counts": dict(agreement), "visits_reviewed": sum(len(key["draft_validity"]) for key in keys.values()),
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify"):
        commands.add_parser(name).add_argument("folder", type=Path)
    review = commands.add_parser("compare-reviews")
    review.add_argument("folder", type=Path)
    review.add_argument("--review-a", required=True, type=Path)
    review.add_argument("--review-b", required=True, type=Path)
    review.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.folder)
    elif args.command == "verify":
        result = verify(args.folder)
    else:
        result = compare_reviews(args.folder, args.review_a, args.review_b)
        if args.output.exists():
            raise FileExistsError("Escribe cada comparación en un archivo nuevo.")
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
