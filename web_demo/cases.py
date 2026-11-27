"""Recorded episode pairs and browser-facing, public projections."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from research.src.artifacts import file_hash


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "research/runs/reward_review_20261104_002"
TARGETED_RUN = RUN.parent / "reward_review_targeted_20261105_002"
BASE = ROOT / "models/Qwen2.5-0.5B-Instruct"
PROMPTS = ROOT / "research/prompts/conversation_v3"


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    method: str
    path: Path
    run: Path = RUN
    training_epoch: int | None = None

    @property
    def trajectories(self):
        return self.run / "evaluation" / self.method / "trajectories.jsonl"

    def public(self):
        return {"id": self.id, "label": self.label, "checkpoint": self.method}


MODELS = {
    model.id: model for model in (
        Model("sft", "SFT · LoRA", "SFT-write", RUN / "warmup/optimizer/last"),
        Model("grpo", "SFT + GRPO", "RL-revised-terminal", RUN / "continuations/RL-revised/optimizer/last"),
        Model("sft-extra", "SFT adicional", "B3",
              RUN.parent / "weak_sft_natural_20261104_001/continuations/B3/optimizer/step_000634"),
    )
}
TARGETED_MODELS = {
    "sft": Model(
        "sft", "SFT · LoRA", "SFT-frozen",
        RUN.parent / "imperfections_campaign_20261101_001/campaigns/arm_00/adaptation/101/training/step_000840",
        TARGETED_RUN,
    ),
    "grpo": Model("grpo", "SFT + GRPO", "RL-credit", TARGETED_RUN / "rl/optimizer/step_000008", TARGETED_RUN),
}

# Chosen for distinct evidence scopes. The last two share the same prior
# conversation, but have different simulated replies to the scope question.
CASES = (
    ("estable", "Preferencia estable",
     "El usuario cambia otra preferencia. La recomendación consultada debe conservarse.",
     "F0", "nw2760f70b57fa7333b259"),
    ("correccion", "Corrección permanente",
     "El usuario cambia su preferencia habitual. La consulta futura usa la fecha posterior.",
     "F1", "nw70cbc80f1d8af9447d21"),
    ("temporal", "Excepción temporal",
     "La preferencia nueva tiene una fecha de vencimiento. La consulta futura cae después.",
     "F2", "nw4ebc0378207ac4947a7c"),
    ("contextual", "Cambio por contexto",
     "La preferencia aplica en casa y dentro de un intervalo. El lector vuelve a consultar en casa después.",
     "F3", "nw3c22d27460fb6560faf9"),
    ("ambigua", "Duración ambigua",
     "El usuario anuncia un cambio y deja abierta su duración. Una aclaración puede resolver el alcance.",
     "F4", "nwa55928c422c46fe6697d"),
    ("ambigua-permanente", "Duración ambigua · otro desenlace",
     "Repite la conversación inicial del caso ambiguo. La respuesta simulada a la aclaración cambia.",
     "F4", "nwf85ff99a01a6b09d5d5f"),
)
TARGETED_CASES = (
    ("grpo-ambigua", "GRPO · Cambio solo para hoy",
     "Ensayo con crédito por fase. El alcance del cambio de cena se revela al preguntar.",
     "F4", "tw27c685c493e265f6cf0a"),
    ("grpo-ambigua-permanente", "GRPO · Cambio desde ahora",
     "Mismo inicio que el caso anterior. La aclaración revela una preferencia habitual.",
     "F4", "tw943ead7f917efe99ecea"),
)


def load_cases():
    return {
        **_load_suite(CASES, MODELS, live_available=True),
        **_load_suite(TARGETED_CASES, TARGETED_MODELS, live_available=False),
    }


def _load_suite(suite, models, *, live_available):
    worlds = {row[4] for row in suite}
    rows = {}
    for model_id, model in models.items():
        completion = json.loads(model.trajectories.with_name("completion.json").read_text(encoding="utf-8"))
        if completion["status"] != "completed" or (
            file_hash(model.trajectories) != completion["artifacts_sha256"]["trajectories.jsonl"]
        ):
            raise ValueError(f"La grabación de {model.method} difiere del artefacto original.")
        with model.trajectories.open(encoding="utf-8") as source:
            rows[model_id] = {}
            for line in source:
                row = json.loads(line)
                world = row["latent_world_id"]
                if world not in worlds or row["condition"] != "O2" or row["route"] != "natural":
                    continue
                if world in rows[model_id]:
                    raise ValueError(f"Más de una trayectoria para {model.method} / {world}.")
                rows[model_id][world] = row
    cases = {}
    for case_id, title, description, family, world in suite:
        pair = {model_id: data[world] for model_id, data in rows.items()}
        episode = pair["sft"]["trace"]["episode"]
        if episode["scenario_family"] != family or any(
            row["trace"]["episode"] != episode or row["method"] != models[model_id].method
            or row["technical_status"] != "completed"
            for model_id, row in pair.items()
        ):
            raise ValueError(f"Trayectorias incompatibles para el caso {case_id}.")
        question = question_turn(episode)["text"]
        cases[case_id] = {
            "public": {
                "id": case_id, "title": title, "description": description, "family": family,
                "question": question, "attribute": episode["current_query"]["attribute"],
                "current_date": episode["current_query"]["date"],
                "future_date": episode["future_query"]["date"],
                "live_available": live_available, "model_ids": list(models),
            },
            "episode": episode,
            "rows": pair,
            "models": models,
        }
    return cases


def question_turn(episode):
    return next(message for message in reversed(episode["user_messages"]) if message.get("role", "user") == "user")


def expected(episode):
    truth = episode["truth"]
    return {
        "current": truth["current_value"], "future": truth["future_value"],
        "scope_kind": truth["scope_kind"], "clarify_slot": truth["decisive_slot"],
    }


def _message(role, text):
    return {"role": role, "text": text}


def _answer_message(action):
    descriptions = {
        "DEFER": "Respuesta aplazada.",
    }
    mode = action.get("mode")
    if mode == "ANSWER":
        return [_message("assistant", action["value"])]
    if mode == "CONDITIONALIZE":
        return [_message("assistant", "\n".join(
            f"{_condition_text(branch['condition'])}: {branch['value']}" for branch in action["branches"]))]
    if mode in descriptions:
        return [_message("assistant", descriptions[mode])]
    return []


def _condition_text(condition):
    if not isinstance(condition, dict):
        return condition
    return " · ".join(f"si aplica la preferencia {value}" if key == "compatible_value"
                      else f"{key.replace('_', ' ')} = {value}" for key, value in condition.items())


def _step(phase, observation, messages, action, memory, *, record=None, error=None, operations=(), reward=None):
    return {
        "phase": phase, "observation": observation, "messages": messages, "action": action,
        "memory": memory, "raw": record["raw_output"] if record else None,
        "error": error["error"] if error else None, "operations": list(operations), "reward": reward,
    }


def show_steps(row, *, score=True):
    trace = row["trace"]
    observations = trace["observations"]
    response = _answer_message(trace["response"])
    steps = []
    for phase in ("select", "respond", "write"):
        observation = observations[phase]
        visible_messages = observation["user_messages"]
        if phase == "select" and trace.get("asked_slot"):
            # The exchange occurs after the SELECT observation and before the
            # respondent sees the new turn. Keep the pre-action input intact.
            visible_messages = observations["respond"]["user_messages"]
        messages = [{"role": "user", **message} for message in visible_messages]
        if phase in ("respond", "write"):
            messages.extend(response)
        result = row["phases"][phase]
        memory = trace["final_memory"] if phase == "write" else observation["memory"]
        steps.append(_step(
            phase, observation, messages, result["action"], memory,
            record=next(iter(result["records"]), None), error=result["error"],
            operations=trace["executed_operations"] if phase == "write" else (),
        ))
    reuse = trace["reuse_observation"]
    query = reuse["query"]
    attribute = {
        "dinner_preference": "la cena", "breakfast_preference": "el desayuno", "drink_preference": "beber",
    }[query["attribute"]]
    context = f" en {query['context']}" if query["context"] else ""
    next_question = f"¿Qué prefiero para {attribute} el {query['date']}{context}?"
    steps.append(_step(
        "reuse", reuse, [_message("user", next_question), *_answer_message(trace["future_response"])],
        trace["future_response"], reuse["memory"], reward=row["metrics"]["R_long"] if score else None,
    ))
    return steps
