from collections import defaultdict

from research.tesis_interactive_env import public_evidence
from research.tesis_shared_budget_bank import build_development
from research.tesis_shared_budget_planner import simulate
from research.tesis_shared_budget_qwen import prompt_messages
from research.tesis_shared_budget_qwen_context import with_public_context


def test_context_uses_visible_text_and_preserves_the_simulator_parser():
    by_family = defaultdict(list)
    for story in build_development(roots_per_regime=1)["validation"]:
        by_family[story["scenario_family"]].append(story)
    for family, worlds in by_family.items():
        decision = simulate(worlds[0], ())
        raw = decision.observation
        enhanced = with_public_context(raw)
        assert enhanced == with_public_context(enhanced)
        assert enhanced["user_messages"][:-1] == raw["user_messages"]
        assert public_evidence(enhanced) == public_evidence(raw)
        note = enhanced["user_messages"][-1]["text"]
        availability = "limitada" if family.startswith("busy") else "disponible"
        later = "incierto" if family.endswith("contingent") else "previsto como rutina"
        assert f"Disponibilidad para responder ahora: {availability}." in note
        assert f"Plan de una visita posterior: {later}." in note

        hidden = {"strategy_cue": "PRIVATE CUE", "world_id": "PRIVATE WORLD", "future_queries": ["PRIVATE QUERY"]}
        tampered = with_public_context({**raw, **hidden})
        assert prompt_messages(tampered, decision.legal) == prompt_messages(enhanced, decision.legal)


def test_context_distinguishes_second_visit_with_and_without_temporal_ambiguity():
    worlds = build_development(roots_per_regime=1)["validation"]
    for missing in (None, "TEMPORAL_SCOPE"):
        story = next(world for world in worlds if world["visits"][1]["slot"] == missing)
        decision = simulate(story, ("PROCEED",))
        note = with_public_context(decision.observation)["user_messages"][-1]["text"]
        assert "Visita 2, preguntas disponibles 1" in note
        expected = "alcance de la preferencia" if missing else "ninguno detectado"
        assert f"Dato pendiente detectado: {expected}" in note
