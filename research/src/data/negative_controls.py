"""Predefined clarification controls for an unaffected attribute.

Controls retain their parent root/partition: they are repeated measurements, not
new independent stories. Construction never consumes a policy's predictions.
"""
from __future__ import annotations

from copy import deepcopy


def unaffected_attribute_controls(episodes, roots=20):
    from ..readers.symbolic import read
    if not isinstance(roots, int) or isinstance(roots, bool) or roots < 1:
        raise ValueError("POSITIVE_CONTROL_ROOT_COUNT_REQUIRED")
    episodes = list(episodes)
    identities = sorted({episode["root_id"] for episode in episodes})
    if len(identities) < roots:
        raise ValueError("INSUFFICIENT_ROOTS_FOR_UNAFFECTED_CONTROLS")
    selected = set(identities[:roots])
    controls = []
    for episode in episodes:
        if episode["root_id"] not in selected:
            continue
        control = deepcopy(episode)
        query = {**control["current_query"], "attribute": "drink_preference", "context": None}
        future = {**control["future_query"], "attribute": "drink_preference", "context": None}
        current_answer = read(control["initial_memory"], query)
        future_answer = read(control["initial_memory"], future)
        if (current_answer["mode"] != "ANSWER" or future_answer["mode"] != "ANSWER"
                or current_answer["value"] != future_answer["value"]):
            raise ValueError("UNAFFECTED_CONTROL_REQUIRES_STABLE_DRINK_MEMORY")
        value = episode.get("truth", {}).get("new_value")
        if not isinstance(value, str) or not value:
            dinner = [r for r in episode["initial_memory"] if r["attribute"] == "dinner_preference"]
            if not dinner:
                raise ValueError("CONTROL_REQUIRES_DINNER_VALUE")
            value = dinner[0]["value"]
        control["realization_id"] += ":unaffected_drink"
        control["current_query"], control["future_query"] = query, future
        control["user_messages"] = [{"turn_id": "u2", "text": f"Para la cena, ahora prefiero {value}. ¿Qué me aconsejas beber?"}]
        control["truth"] = {"current_value": current_answer["value"], "future_value": future_answer["value"],
                            "initial_value": current_answer["value"], "new_value": value,
                            "scope": None, "scope_kind": "unaffected_attribute_control", "decisive_slot": None,
                            "current_context": None, "future_exclusive_benefit": False,
                            "current_date": query["date"]}
        control["opportunity"] = {"eligible": False, "negative_control": True,
                                  "scope_slot": "TEMPORAL_SCOPE", "slot": "TEMPORAL_SCOPE",
                                  "affected_attribute": "dinner_preference", "tested_attribute": "drink_preference"}
        replies = control.setdefault("replies", {})
        original = replies.get("TEMPORAL_SCOPE", {}).get("text", "")
        if not any(marker in original.lower() for marker in ("habitual", "solo para hoy", "solo en esta conversación", "hasta el")):
            replies["TEMPORAL_SCOPE"] = {"turn_id": "u3", "text": "Como preferencia habitual desde ahora."}
        control["authoring"] = {"source": "derived_unaffected_attribute_control_v1", "independent": False,
                                "parent_authoring": deepcopy(episode.get("authoring", {}))}
        controls.append(control)
    return controls
