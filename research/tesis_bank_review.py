"""Prepare blind materials and reconcile supplied human decisions."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random

from .src.data.audit_splits import audit_human_reviews
from .src.env.user_simulator import QUESTIONS
from .src.artifacts import digest, file_hash, write_json

SCOPES = {"none", "ongoing", "interval", "session", "contextual"}
INSTRUCTIONS = [
    "Completa tus valoraciones individualmente con los casos de tu carpeta.",
    "Valora primero el diálogo inicial sin consultar el archivo de réplicas.",
    "Para information_sufficient responde si el texto permite resolver la consulta actual.",
    "Indica con need_future si hace falta aclarar algo para actualizar con alcance fiable para otra sesión.",
    "Usa decisive_slot para la variable decisiva, sea para hoy o para otra sesión. Usa null si corresponde continuar.",
    "Indica las respuestas actuales admisibles en admissible_answers. Si falta un dato, enumera las opciones plausibles.",
    "writable_scope es el alcance máximo sustentado del cambio antes de preguntar. Usa none si no se justifica una escritura confirmada.",
    "Luego abre las réplicas. Indica con both_plausible si las dos caben en el mismo diálogo inicial.",
    "Justifica cada decisión en rationale. Mantén los audit_id y usa un reviewer_id propio y constante.",
]


def prepare_review(episodes, output, *, seed):
    """Serialize only visible stimuli and blank response forms in shareable folders."""
    output.mkdir()
    groups = defaultdict(list)
    for episode in episodes:
        if episode["split"] not in {"train", "validation"}:
            raise ValueError("El banco reservado queda fuera de la revisión de desarrollo.")
        groups[episode["root_id"]].append(episode)
    keys = {}
    stimuli, alternatives = [], []
    for root, rows in sorted(groups.items()):
        first = rows[0]
        audit_id = "caso-" + digest([seed, "review", root])[:14]
        truth = first["truth"]
        slot = truth["decisive_slot"]
        expected_now = slot != "CURRENT_CONTEXT"
        expected_future = slot == "TEMPORAL_SCOPE"
        writable_scope = first["authoring"]["writable_scope_before_ask"]
        values = sorted({row["truth"]["current_value"] for row in rows})
        keys[audit_id] = {
            "root_id": root, "family": first["scenario_family"], "split": first["split"],
            "literal": {"information_sufficient": expected_now, "need_future": expected_future,
                        "decisive_slot": slot, "writable_scope": writable_scope, "admissible_answers": values},
            "has_pair": len(rows) == 2,
        }
        stimuli.append({"audit_id": audit_id, "memory": first["initial_memory"],
                        "conversation": first["user_messages"], "query": first["current_query"],
                        "reuse_regime": "Habrá otra consulta del mismo atributo sin posibilidad de preguntar."})
        if len(rows) == 2:
            answers = sorted(row["replies"][slot]["text"] for row in rows)
            random.Random(digest([seed, audit_id])).shuffle(answers)
            alternatives.append({"audit_id": audit_id, "pregunta": QUESTIONS[slot], "réplicas": answers})
    artifacts = {}
    key_path = output / "private_key.json"
    write_json(key_path, keys)
    artifacts["private_key"] = {"path": key_path.name, "sha256": file_hash(key_path)}
    for reviewer in ("a", "b"):
        folder = output / f"reviewer_{reviewer}"
        folder.mkdir()
        rng = random.Random(digest([seed, "review_order", reviewer]))
        rows, replies = list(stimuli), list(alternatives)
        rng.shuffle(rows)
        rng.shuffle(replies)
        packet = {"instructions": INSTRUCTIONS, "allowed_slots": list(QUESTIONS), "allowed_scopes": sorted(SCOPES),
                  "cases": rows}
        forms = [{"audit_id": row["audit_id"], "reviewer_id": "", "information_sufficient": None,
                  "need_future": None, "decisive_slot": None, "admissible_answers": [], "writable_scope": None,
                  "both_plausible": None, "rationale": ""} for row in rows]
        for name, content in (("01_casos.json", packet), ("02_replicas.json", replies),
                              ("03_respuestas_en_blanco.json", forms)):
            path = folder / name
            write_json(path, content)
            artifacts[f"{reviewer}/{name}"] = {"path": str(path.relative_to(output)), "sha256": file_hash(path)}
    write_json(output / "adjudicaciones_en_blanco.json", [])
    return {"status": "pending_human_audit", "roots": len(groups), "paired_roots": len(alternatives),
            "artifacts": artifacts, "minimum_roots_for_protocol": 100,
            "human_decisions_supplied": 0}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _required_text(value):
    return isinstance(value, str) and bool(value.strip())


def _read_decisions(path, keys):
    rows = _read(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Se espera una lista JSON con las decisiones originales de una persona.")
    decisions, reviewers = {}, set()
    for row in rows:
        aid = row.get("audit_id") if isinstance(row, dict) else None
        if not isinstance(aid, str) or aid not in keys or aid in decisions:
            raise ValueError(f"Identificador de revisión repetido o desconocido: {aid}")
        if not _required_text(row.get("reviewer_id")) or not _required_text(row.get("rationale")):
            raise ValueError(f"Falta el identificador de la persona o la justificación en {aid}")
        if any(type(row.get(field)) is not bool for field in ("information_sufficient", "need_future")):
            raise ValueError(f"Faltan juicios de suficiencia en {aid}")
        if "decisive_slot" not in row or row["decisive_slot"] not in [None, *QUESTIONS]:
            raise ValueError(f"Variable decisiva inválida en {aid}")
        if row.get("writable_scope") not in sorted(SCOPES):
            raise ValueError(f"Alcance de escritura inválido en {aid}")
        answers = row.get("admissible_answers")
        if not isinstance(answers, list) or not answers or not all(map(_required_text, answers)):
            raise ValueError(f"Faltan las respuestas admisibles en {aid}")
        if keys[aid]["has_pair"] and type(row.get("both_plausible")) is not bool:
            raise ValueError(f"Falta el juicio de plausibilidad del par en {aid}")
        reviewer_id = row["reviewer_id"].strip()
        reviewers.add(reviewer_id)
        decisions[aid] = {**row, "reviewer_id": reviewer_id,
                          "admissible_answers": sorted({item.strip().casefold() for item in answers})}
    if len(reviewers) != 1:
        raise ValueError("Cada archivo debe corresponder a una única persona.")
    return decisions, next(iter(reviewers))


def _adjudications(path, keys):
    if path is None:
        return {}
    rows = _read(path)
    if not isinstance(rows, list):
        raise ValueError("Se espera una lista JSON de adjudicaciones.")
    decisions = {}
    for row in rows:
        aid = row.get("audit_id") if isinstance(row, dict) else None
        if not isinstance(aid, str) or aid not in keys or aid in decisions:
            raise ValueError(f"Identificador de adjudicación repetido o desconocido: {aid}")
        if row.get("decision") not in ("keep_as_written", "revise") or not _required_text(row.get("rationale")):
            raise ValueError(f"Se requiere keep_as_written o revise y su justificación en {aid}")
        decisions[aid] = row
    return decisions


def reconcile_review(bank, manifest, review_a, review_b, output, *, adjudications=None):
    """Report observed agreement and block release while contradictions remain."""
    bank, output = Path(bank), Path(output)
    if output.exists():
        raise FileExistsError("Guarda cada adjudicación en un archivo de informe nuevo.")
    info = manifest["review"]["artifacts"]["private_key"]
    key_path = bank / "review" / info["path"]
    if file_hash(key_path) != info["sha256"]:
        raise ValueError("Cambió la clave privada de la revisión.")
    keys = _read(key_path)
    left, reviewer_a = _read_decisions(review_a, keys)
    right, reviewer_b = _read_decisions(review_b, keys)
    if reviewer_a == reviewer_b:
        raise ValueError("Se requieren dos personas distintas.")
    resolution = _adjudications(adjudications, keys)
    reviews, agreements, problems = [], [], {}
    for aid, key in keys.items():
        if aid not in left or aid not in right:
            problems[aid] = ["missing_review"]
            continue
        pair = (left[aid], right[aid])
        fields = key["literal"]
        agreements.append(all(left[aid][field] == right[aid][field]
                              for field in ("information_sufficient", "decisive_slot", "need_future")))
        disagreement = [field for field in fields if left[aid][field] != right[aid][field]]
        conflicts = [field for field in fields if any(row[field] != fields[field] for row in pair)]
        issue = (["reviewer_disagreement:" + field for field in disagreement] +
                 ["literal_mismatch:" + field for field in conflicts])
        if key["has_pair"] and not all(row["both_plausible"] for row in pair):
            issue.append("paired_replies_not_plausible")
        if issue:
            problems[aid] = issue
        reviews.extend({**row, "root_id": key["root_id"], "family": key["family"],
                        "f4_both_plausible": row.get("both_plausible")} for row in pair)

    unresolved = sorted(aid for aid, issue in problems.items()
                        if "missing_review" in issue or resolution.get(aid, {}).get("decision") != "keep_as_written")
    revisable = sorted(aid for aid, row in resolution.items() if row["decision"] == "revise")
    accepted = [{"root_id": keys[aid]["root_id"], "resolved": True, "rationale": row["rationale"]}
                for aid, row in resolution.items() if row["decision"] == "keep_as_written"]
    summary = audit_human_reviews(reviews, accepted)
    agreement = sum(agreements) / len(agreements) if agreements else None
    passed = (summary["status"] == "passed" and not unresolved and not revisable and
              agreement is not None and agreement >= .9)
    report = {"status": "passed" if passed else "pending_human_audit", "roots": len(agreements),
              "expected_roots": len(keys), "reviewer_ids": [reviewer_a, reviewer_b], "protocol_summary": summary,
              "joint_agreement_including_future": agreement, "unresolved_audit_ids": unresolved,
              "revision_required": revisable, "findings": problems,
              "source_sha256": {"bank_manifest": file_hash(bank / "manifest.json"),
                                "review_a": file_hash(review_a), "review_b": file_hash(review_b),
                                "adjudications": file_hash(adjudications) if adjudications else None},
              "supplied_review_files": [str(Path(path).resolve()) for path in (review_a, review_b)],
              "provenance": "Reconciles supplied attestations. Independence and human authorship are external responsibilities."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report
