"""Pre-result AI-authored challenge distributions and explicit external proxy.

This generator was authored separately from build_bank. Its held-out mechanisms
combine temporal scopes, context and version chains in the structured S1 state.
Neither distribution is a substitute for independent human-authored stories.
No policy result, model or reward is read by this module.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
import random

from .build_bank import FAMILIES, FOODS, SLOTS, scope


def _token(seed, *items):
    return hashlib.sha256(json.dumps([seed, *items], sort_keys=True).encode()).hexdigest()[:14]


def _record(mid, value, at, text, *, kind="ongoing", start=None, end=None, context=None, session=None,
            supersedes=(), attribute="dinner_preference"):
    return {"memory_id": mid, "attribute": attribute, "value": value,
            "recorded_at": at, "scope": scope(kind, start=start or at, end=end, context=context, session=session),
            "status": "confirmed", "source_turn_ids": ["s1_" + mid],
            "evidence_quotes": [{"turn_id": "s1_" + mid, "text": text}], "supersedes": list(supersedes)}


def build_independent_bank(split="test_ood", roots=150, seed=87119):
    """Build balanced roots with paired latent worlds and novel compositions.

    The external split is always an AI-authored proxy, even when the caller
    supplies a different source label. Authorship is never upgraded to human.
    """
    if split not in {"test_ood", "test_external"} or roots < 5 or roots % 5:
        raise ValueError("Challenge bank requires OOD/external split and balanced roots")
    proxy = split == "test_external"
    rows = []
    for index in range(roots):
        family = FAMILIES[index % 5]
        rid = "r" + _token(seed, split, index)
        mids = ["m" + _token(seed, rid, j) for j in range(7)]
        old_value, base, changed, office = [FOODS[k % len(FOODS)] for k in (index + 1, index + 4, index + 6, index)]
        now = date(2026, 5, 4) + timedelta(days=index % 29)
        historical = now - timedelta(days=40)
        revised = now - timedelta(days=15)
        later = now + timedelta(days=9)
        memories = [
            _record(mids[0], old_value, historical.isoformat(), f"Antes de esta revisión mi cena habitual era {old_value}."),
            _record(mids[1], base, revised.isoformat(), f"Sustituyo aquella pauta: mi cena habitual pasa a ser {base}.", supersedes=[mids[0]]),
        ]
        # The order of user narration differs from chronological record order.
        # Relative boundaries and context combinations were absent from train.
        if family in {"F2", "F3", "F4"}:
            memories.append(_record(mids[2], office, (now - timedelta(days=7)).isoformat(),
                f"Cuando ceno en la oficina, elijo {office}; este acuerdo vence el {(now + timedelta(days=3)).isoformat()} inclusive.",
                kind="contextual", start=(now - timedelta(days=7)).isoformat(),
                end=(now + timedelta(days=4)).isoformat(), context="oficina"))
        else:
            memories.append(_record(mids[2], office, (now - timedelta(days=6)).isoformat(),
                f"Durante la estancia anterior en la oficina elegía {office}.", kind="contextual",
                start=(now - timedelta(days=6)).isoformat(), end=(now - timedelta(days=2)).isoformat(), context="oficina"))
        if family == "F3":
            # An interval-scoped contextual preference expires before reuse;
            # after expiry the general version chain is again decisive.
            memories.append(_record(mids[3], changed, (now - timedelta(days=5)).isoformat(),
                f"En casa elijo {changed} hasta el {(now + timedelta(days=1)).isoformat()} inclusive.",
                kind="contextual", start=(now - timedelta(days=5)).isoformat(),
                end=(now + timedelta(days=2)).isoformat(), context="casa"))
        else:
            memories.append(_record(mids[3], old_value, (now - timedelta(days=1)).isoformat(),
                f"La excepción de aquella sesión era {old_value}; terminó al cerrar esa sesión.",
                kind="session", session="S1"))
        memories.extend([
            _record(mids[4], "agua", (now - timedelta(days=9)).isoformat(), "Conservo agua como bebida.", attribute="drink_preference"),
            _record(mids[5], "fruta", (now - timedelta(days=8)).isoformat(), "Mi desayuno continúa siendo fruta.", attribute="breakfast_preference"),
        ])
        current_query = {"attribute": "dinner_preference", "date": now.isoformat(), "session_id": "S2", "context": "casa"}
        future_query = {"attribute": "dinner_preference", "date": later.isoformat(), "session_id": "S3", "context": "casa"}
        if family == "F0":
            utterance = "Retomemos el criterio que sigue vigente, aunque la última anotación fuera de otra sesión. ¿Qué plato corresponde hoy?"
            worlds = [("stable", base, base, None, None)]
        elif family == "F1":
            # The explicit operative clause stays within the public parser;
            # independent discourse and S1 composition are held out.
            utterance = f"Revisé el acuerdo anterior. Desde hoy prefiero {changed} de forma habitual. ¿Con qué plato cerramos el día?"
            worlds = [("ongoing", changed, changed, scope(start=now.isoformat()), None)]
        elif family == "F2":
            inclusive = now + timedelta(days=2)
            utterance = f"Hago una pausa en la pauta general. Solo hasta el {inclusive.isoformat()} inclusive prefiero {changed}. ¿Cuál es la cena de esta fecha?"
            worlds = [("interval", changed, base, scope("interval", start=now.isoformat(), end=(inclusive + timedelta(days=1)).isoformat()), None)]
        elif family == "F3":
            current_query["context"] = None
            future_query["context"] = "oficina"
            utterance = "Tengo acuerdos distintos según dónde esté; aún no te he dicho dónde cenaré. ¿Qué plato me corresponde ahora?"
            worlds = [("casa", changed, base, None, "CURRENT_CONTEXT"),
                      ("oficina", office, base, None, "CURRENT_CONTEXT")]
        else:
            # The contextual agreement expires between acquisition and reuse.
            # The ambiguous general change must survive only if made ongoing.
            utterance = f"Reconsiderando el plato anterior: ahora prefiero {changed}. Resuelve primero qué cenar hoy; conservarlo para después requiere conocer su alcance."
            worlds = [("session", changed, base, scope("session", start=now.isoformat(), session="S2"), "TEMPORAL_SCOPE"),
                      ("ongoing", changed, changed, scope(start=now.isoformat()), "TEMPORAL_SCOPE")]
        for name, current_value, future_value, real_scope, decisive in worlds:
            replies = {s: {"turn_id": "u3", "text": "Sobre ese punto no tengo información adicional."} for s in SLOTS}
            if family == "F3":
                replies["CURRENT_CONTEXT"] = {"turn_id": "u3", "text": f"Hoy estoy en {name}."}
            if family == "F4":
                replies["TEMPORAL_SCOPE"] = {"turn_id": "u3", "text": "Solo en esta conversación." if name == "session" else "Como preferencia habitual desde ahora."}
            rows.append({"root_id": rid, "latent_world_id": "w" + _token(seed, rid, name),
                         "realization_id": "independent_0", "split": split, "scenario_family": family,
                         "initial_memory": deepcopy(memories), "user_messages": [{"turn_id": "u2", "text": utterance}],
                         "current_query": deepcopy(current_query), "future_query": deepcopy(future_query), "replies": replies,
                         "truth": {"current_value": current_value, "future_value": future_value,
                                   "initial_value": base, "new_value": changed, "scope": real_scope,
                                   "scope_kind": name, "current_date": now.isoformat(), "decisive_slot": decisive,
                                   "current_context": name if family == "F3" else "casa",
                                   "future_exclusive_benefit": family == "F4"},
                         "opportunity": {"eligible": family == "F4", "slot": decisive if family == "F4" else None,
                                         "negative_control": family in {"F0", "F1", "F2"}},
                         "evaluation_distribution": "ai_authored_external_proxy" if proxy else "independent_ai_ood",
                         "authoring": {"source": "AI_AUTHORED_PROXY" if proxy else "INDEPENDENT_AI_OOD",
                                       "author_id": "codex_data_statistics_independent_author_v1", "human_authored": False,
                                       "independent": True, "human_review_status": "not_executed",
                                       "held_out_mechanisms": ["supersession_chain_with_session_exception",
                                                               "contextual_preference_with_interval_expiry"],
                                       "frozen_without_policy_scores": True,
                                       "limitation": "AI-authored challenge; not independent human evidence. Grammar coverage is evaluator-limited."}})
    random.Random(seed).shuffle(rows)
    return rows


def build_ai_external_proxy(roots=50, seed=87121):
    return build_independent_bank("test_external", roots, seed)


def bank_manifest(episodes):
    """Counts and provenance only; no claim that human review was performed."""
    roots = {e["root_id"]: e["scenario_family"] for e in episodes}
    return {"status": "executed", "roots": len(roots), "episodes": len(episodes),
            "families": {f: sum(v == f for v in roots.values()) for f in FAMILIES},
            "sha256": hashlib.sha256(json.dumps(episodes, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "human_authored_external_status": "not_executed", "human_review_status": "not_executed",
            "authorship": sorted({e["authoring"]["source"] for e in episodes}),
            "independent_authoring_scope": "Separate AI-generated challenge code and composed S1 mechanisms, not independent humans"}
