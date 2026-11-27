"""Adjudicate blind profiles reviews while preserving the frozen reserve."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from research.src.artifacts import canonical, file_hash, write_json
from research.tesis_extended_review import FIELDS, normalized, read_review
from research.tesis_profiles_heldout import ROOT, load_executable, load_gold, verify


def compare(folder, review_a, review_b):
    manifest = verify(folder)
    cases = load_gold(folder)
    authored = {(case["id"], row["visit"]): row for case in cases for row in case["visits"]}
    if len(cases) != manifest["roots"] or len(authored) != manifest["roots"] * manifest["visits_per_root"]:
        raise ValueError("Frozen visit labels do not match the reserve manifest")
    name_a, left = read_review(review_a, set(authored))
    name_b, right = read_review(review_b, set(authored))
    if name_a.casefold() == name_b.casefold():
        raise ValueError("Two distinct blinded reviewers are required")

    normalized_counts, literal_counts = {}, {}
    differences = []
    for field in FIELDS:
        exact = Counter()
        semantic = Counter()
        for (root_id, visit), gold in authored.items():
            key = (root_id, visit)
            values = {"gold": gold[field], "a": left[key][field], "b": right[key][field]}
            literal = values["gold"], values["a"], values["b"]
            cleaned = tuple(normalized(field, value) for value in literal)
            for counts, (author, a, b) in ((exact, literal), (semantic, cleaned)):
                counts["a_vs_b"] += a == b
                counts["a_vs_gold"] += a == author
                counts["b_vs_gold"] += b == author
                counts["all_three"] += author == a == b
            if not literal[0] == literal[1] == literal[2]:
                presentation = cleaned[0] == cleaned[1] == cleaned[2]
                differences.append({
                    "root_id": root_id, "visit": visit, "field": field,
                    "classification": "presentation_only" if presentation else "substantive",
                    "decision": "retain_frozen_gold" if presentation else "pending_adjudication",
                    "reason": (
                        "Los valores estructurados coinciden al normalizar el texto y el orden de las listas."
                        if presentation else
                        "Los valores estructurados difieren. La etiqueta congelada se conserva mientras se resuelve la diferencia."
                    ),
                    "values": values,
                    "notes": {"a": left[key].get("rationale"), "b": right[key].get("rationale")},
                })
        literal_counts[field] = dict(exact)
        normalized_counts[field] = dict(semantic)

    return {
        "review_method": "two_independent_ai_reviews_of_common_public_packet",
        "reviewers": [name_a, name_b], "human_reviews": 0,
        "roots": len(cases), "visits": len(authored),
        "manifest_sha256": file_hash(folder / "MANIFEST.json"),
        "gold_sha256": file_hash(folder / "PRIVATE_GOLD.json"),
        "files_sha256": [file_hash(review_a), file_hash(review_b)],
        "agreement_after_normalization": normalized_counts,
        "literal_agreement": literal_counts,
        "discrepancies": differences,
        "substantive_discrepancies": sum(item["classification"] == "substantive" for item in differences),
    }


def _show(value):
    if not isinstance(value, list):
        return str(value)
    return " → ".join(
        f"{item['value']} ({item['context'] or 'sin contexto'})" if isinstance(item, dict) else item
        for item in value
    ) or "lista vacía"


def report(result):
    visits = result["visits"]
    lines = [
        "# Adjudicación de perfiles reservados v4", "",
        f"{result['roots']} historias ficticias y {visits} visitas. Dos instancias de IA revisaron por "
        "separado el mismo paquete público. El contraste con las etiquetas del autor se hizo "
        "después de completar las dos fichas. La revisión cubre las visitas base sin respuestas "
        "hipotéticas a aclaraciones. El banco conserva su manifiesto y sus etiquetas congeladas.",
        "",
        "| Campo | IA A / IA B | IA A / referencia | IA B / referencia | Coincidencia de los tres |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for field, name in zip(FIELDS, ("Respuesta", "Escritura nueva", "Recuerdos previos"), strict=True):
        counts = result["agreement_after_normalization"][field]
        cells = " | ".join(f"{counts[key]}/{visits}" for key in ("a_vs_b", "a_vs_gold", "b_vs_gold", "all_three"))
        lines.append(f"| {name} | {cells} |")
    lines.extend((
        "", "La comparación conserva los campos de vigencia, contexto y sesión de cada registro. "
        "Se normalizan mayúsculas y espacios exteriores. Las colecciones se comparan por su contenido.",
        "", "## Discrepancias y decisiones", "",
    ))
    for item in result["discrepancies"]:
        title = f"`{item['root_id']}` visita {item['visit']}, `{item['field']}`"
        if item["classification"] == "presentation_only":
            values = item["values"]
            lines.append(
                f"- **{title}.** La referencia contiene {_show(values['gold'])}. "
                f"IA A anotó {_show(values['a'])} e IA B anotó {_show(values['b'])}. "
                "Se conserva la etiqueta congelada. Cambia la presentación de la lista y se "
                "mantienen tanto los recuerdos como sus ámbitos de aplicación."
            )
        else:
            lines.append(f"- **{title}.** Pendiente de adjudicación. {item['reason']}")
    if not result["discrepancies"]:
        lines.append("Las tres fuentes coinciden en todas las comparaciones literales.")
    lines.extend((
        "", "## Alcance de la puntuación", "",
        "La admisibilidad de una respuesta condicional se registra en la revisión semántica. "
        "`current_supported` mide el acierto de una respuesta con un valor único sustentado. "
        "La evaluación confirmatoria conserva ese criterio. Una sensibilidad secundaria puede "
        "contabilizar como admisibles las respuestas condicionales sustentadas que cubran todas "
        "las alternativas y sus contextos.",
        "", "Las huellas, conteos literales y fichas discrepantes figuran en el archivo de acuerdo. "
        "[Acuerdo](AGREEMENT.json) · [IA A](review_a.json) · [IA B](review_b.json)", "",
    ))
    return "\n".join(lines)


def parser_diagnostic(folder):
    """Inspect authored opportunities and admissibility with hand-set action paths."""
    from research.src.scoring.evidence import score_response
    from research.tesis_interactive_env import public_evidence
    from research.tesis_profiles_env import execute
    from research.tesis_shared_budget_env import QUESTION_ACTIONS

    cases = {case["id"]: case for case in load_gold(folder)}
    families = Counter(case["family"] for case in cases.values())
    visits = sum(len(case["visits"]) for case in cases.values())
    by_opportunity, conditional = {}, []
    for story in load_executable(folder):
        root = story["root_id"]
        case = cases[root]
        authored = {item["visit"]: item for item in case["question_opportunities"]}
        for item in case["visits"]:
            visit = item["visit"]
            prefix = ("PROCEED",) * (visit - 1)
            opening = execute(story, prefix)
            offered = [action for action in opening.legal if action in QUESTION_ACTIONS]
            target = authored.get(visit)
            key = root, visit
            row = by_opportunity.setdefault(key, {
                "root_id": root, "visit": visit,
                "authored_slot": target["slot"] if target else None,
                "authored_style": target["style"] if target else None,
                "worlds": [],
            })
            detail = {
                "world_id": story["world_id"], "offered": offered,
                "parser_slot": public_evidence(opening.observation)["missing_slot"],
            }
            if target and target["style"] == "partial" and target["slot"] in offered:
                following = execute(story, prefix + (target["slot"],))
                detail["follow_up_offered"] = target["slot"] in following.legal
            row["worlds"].append(detail)
            if item["response"]["mode"] != "CONDITIONALIZE":
                continue
            prefixes = [prefix]
            if "CURRENT_CONTEXT" in offered:
                prefixes.append(prefix + ("CURRENT_CONTEXT",))
            pairs = sorted(canonical({"condition": record["context"], "value": record["value"]})
                           for record in item["retrieves"])
            weight = 1 / (len(case["visits"]) * families[case["family"]] * len(families))
            for chosen in prefixes:
                path = chosen + ("PROCEED",) * (len(case["visits"]) - visit + 1)
                trace = execute(story, path)
                entry = trace["visits"][visit - 1]
                before_close = execute(story, chosen)
                output = entry["outcome"]
                response = score_response(output, before_close.observation, entry["expected"])
                branches = sorted(canonical(branch) for branch in output["branches"])
                admissible = int(output["mode"] == "CONDITIONALIZE" and response["utility"] == 0
                                 and branches == pairs)
                delta = admissible - entry["current_supported"]
                conditional.append({
                    "root_id": root, "visit": visit, "world_id": story["world_id"], "path": path,
                    "current_supported": entry["current_supported"], "exact_conditional_admissibility": admissible,
                    "questions": trace["questions"], "root_mean_delta": delta / len(case["visits"]),
                    "visit_pool_delta": delta / visits, "stratified_macro_delta": delta * weight,
                })
    slots = [row for row in by_opportunity.values() if row["authored_slot"]]
    return {
        "method": "probe_with_unused_budget_and_no_previous_optional_replies",
        "roots": len(cases), "visits": visits, "authored_opportunities": len(slots),
        "available_in_all_worlds": sum(
            all(row["authored_slot"] in world["offered"] for world in row["worlds"]) for row in slots
        ),
        "other_visits_with_questions": sum(
            row["authored_slot"] is None and any(world["offered"] for world in row["worlds"])
            for row in by_opportunity.values()
        ),
        "opportunities": [row for row in by_opportunity.values()
                          if row["authored_slot"] or any(world["offered"] for world in row["worlds"])],
        "conditional_paths": conditional,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    result = compare(args.folder, args.a, args.b)
    write_json(args.output / "AGREEMENT.json", result)
    (args.output / "ADJUDICACION.md").write_text(report(result), encoding="utf-8")
    if args.diagnostic:
        write_json(args.output / "DIAGNOSTICO_APERTURA.json", parser_diagnostic(args.folder))


if __name__ == "__main__":
    main()
