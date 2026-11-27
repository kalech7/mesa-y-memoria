from copy import deepcopy
import unittest
from unittest.mock import patch

from web_demo.__main__ import Lab, RequestError
from web_demo.cases import RUN, question_turn
from web_demo.interactive import simulate


class WebDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RUN.is_dir():
            raise unittest.SkipTest(
                "El visor histórico requiere las corridas originales en research/runs/"
            )
        cls.lab = Lab("cpu")

    def test_paired_replay_uses_persisted_memory_and_recorded_actor_outputs(self):
        result = self.lab.compare({"scenario_id": "correccion"})
        self.assertEqual(result["mode"], "replay")
        self.assertEqual([model["id"] for model in result["models"]], ["sft", "grpo"])
        self.assertEqual([model["metrics"]["J_primary"] for model in result["models"]], [0, 1])
        for model in result["models"]:
            select, respond, write, reuse = model["steps"]
            self.assertEqual([step["phase"] for step in model["steps"]], ["select", "respond", "write", "reuse"])
            self.assertIsNotNone(respond["raw"])
            self.assertEqual(write["operations"][0]["op"], "SUPERSEDE")
            self.assertEqual(write["memory"], reuse["memory"])
            self.assertEqual(set(reuse["observation"]), {"memory", "query"})
            self.assertIsNone(reuse["raw"])
            self.assertNotIn("truth", select["observation"])
            self.assertEqual(model["provenance"]["reader"], "fixed_symbolic")

    def test_same_ambiguous_dialogue_can_hide_two_different_replies(self):
        results = [self.lab.compare({"scenario_id": case, "model_ids": ["sft-extra"]})
                   for case in ("ambigua", "ambigua-permanente")]
        short, long = (result["models"][0]["steps"][0] for result in results)
        self.assertEqual(short["observation"], long["observation"])
        self.assertEqual(short["action"], {"action": "ASK", "slot": "TEMPORAL_SCOPE"})
        self.assertEqual(long["action"], short["action"])
        self.assertNotEqual(short["messages"][-1]["text"], long["messages"][-1]["text"])
        self.assertEqual([result["expected"]["scope_kind"] for result in results], ["session", "ongoing"])

    def test_targeted_replay_uses_its_matched_checkpoints_and_reproduces_clarification(self):
        for case, value in (("grpo-ambigua", "pollo al horno"),
                            ("grpo-ambigua-permanente", "salmón a la plancha")):
            with self.subTest(case=case):
                result = self.lab.compare({"scenario_id": case})
                self.assertFalse(result["scenario"]["live_available"])
                self.assertEqual(result["scenario"]["model_ids"], ["sft", "grpo"])
                sft, grpo = result["models"]
                self.assertEqual([model["checkpoint"] for model in (sft, grpo)], ["SFT-frozen", "RL-credit"])
                self.assertEqual(sft["steps"][0]["action"]["action"], "PROCEED")
                self.assertEqual(grpo["steps"][0]["action"], {"action": "ASK", "slot": "TEMPORAL_SCOPE"})
                self.assertEqual(grpo["steps"][-1]["action"]["value"], value)
                self.assertEqual([model["metrics"]["J_primary"] for model in (sft, grpo)], [0, 1])
                with self.assertRaises(RequestError):
                    self.lab.compare({"scenario_id": case, "mode": "live"})

    def test_custom_question_is_sent_to_live_actor_and_never_shown_as_scored(self):
        case = self.lab.cases["correccion"]
        original = deepcopy(case["episode"])
        custom = "¿Qué podría cenar ahora?"

        def compare(episode, models):
            self.assertEqual([model.id for model in models], ["sft"])
            basis = deepcopy(episode)
            self.assertEqual(question_turn(basis)["text"], custom)
            question_turn(basis)["text"] = question_turn(original)["text"]
            self.assertEqual(basis, original)
            return {"sft": case["rows"]["sft"]}

        with patch.object(self.lab.live, "status", return_value=(True, None)), patch.object(
            self.lab.live, "compare", side_effect=compare
        ):
            result = self.lab.compare({
                "scenario_id": "correccion", "mode": "live", "question": custom, "model_ids": ["sft"],
            })
        self.assertTrue(result["custom"])
        self.assertEqual(result["scenario"]["question"], custom)
        self.assertIsNone(result["expected"])
        self.assertIsNone(result["models"][0]["metrics"])
        self.assertTrue(all(step["reward"] is None for step in result["models"][0]["steps"]))
        self.assertEqual(case["episode"], original)

    def test_rejects_custom_replay_and_invalid_model_lists(self):
        with self.assertRaises(RequestError):
            self.lab.compare({"scenario_id": "temporal", "question": "¿Qué bebo?"})
        for model_ids in ([], ["sft", "sft"], ["other"], [{}], "sft"):
            with self.subTest(model_ids=model_ids), self.assertRaises(RequestError):
                self.lab.compare({"scenario_id": "temporal", "model_ids": model_ids})

    def test_guard_replay_preserves_shared_interaction_and_exposes_changed_future(self):
        result = self.lab.guard_cases["temporal"]
        self.assertEqual([model["id"] for model in result["models"]], ["original", "veto", "repair"])
        original, veto, repaired = result["models"]
        self.assertEqual(original["steps"][:2], veto["steps"][:2])
        self.assertEqual(original["steps"][:2], repaired["steps"][:2])
        self.assertEqual(original["metrics"]["FJS"], 0)
        self.assertEqual(repaired["metrics"]["FJS"], 1)
        self.assertEqual(repaired["steps"][2]["memory"], repaired["steps"][3]["memory"])
        self.assertEqual(repaired["verification"]["status"], "repaired")
        self.assertEqual(result["study"]["roots"], 25)

    def test_write_verification_is_explicit_and_applies_to_live_inference(self):
        with self.assertRaises(RequestError):
            self.lab.compare({"scenario_id": "correccion", "verify_writes": True})
        rows = self.lab.cases["correccion"]["rows"]
        with patch.object(self.lab.live, "status", return_value=(True, None)), patch.object(
            self.lab.live, "compare", return_value=rows,
        ) as compare:
            result = self.lab.compare({"scenario_id": "correccion", "mode": "live", "verify_writes": True})
        self.assertTrue(compare.call_args.kwargs["verify_writes"])
        self.assertTrue(result["verified_writes"])

    def test_reserved_results_and_paired_clarification_cases(self):
        catalog = self.lab.catalog()
        choices = catalog["chat_choices"]
        self.assertEqual([choice["label"] for choice in choices], ["SFT", "RL (GRPO)"])
        self.assertEqual(len({choice["id"] for choice in choices}), 2)
        self.assertTrue({choice["id"] for choice in choices} <= {item["id"] for item in catalog["chat_models"]})
        study = catalog["study_results"]
        self.assertEqual(study["scope"]["independent_roots"], 50)
        self.assertEqual(study["scope"]["complete_paired_cells"], 201)
        self.assertEqual(list(study["grpo_selected_steps"].values()), [0, 0, 0])
        short, long, unresolved = [self.lab.guard_cases[f"reservado-{name}"]
                                   for name in ("sesion", "duradero", "sin-aclarar")]
        self.assertEqual(short["models"][0]["steps"][0]["observation"], long["models"][0]["steps"][0]["observation"])
        self.assertNotEqual(short["expected"]["future"], long["expected"]["future"])
        self.assertEqual([row["metrics"]["FJS"] for row in unresolved["models"]], [0, 0, 0])

    def test_multi_visit_demo_keeps_private_world_out_of_the_observation(self):
        catalog = self.lab.catalog()["interactive_study"]
        self.assertEqual(catalog["roots"], 50)
        self.assertAlmostEqual(catalog["terminal_contrasts"]["future_supported"]["estimate"], .1888888889)
        self.assertAlmostEqual(catalog["means"]["GRPO-longitudinal-terminal"]["future_supported"], .9105555556)
        self.assertAlmostEqual(catalog["means"]["GRPO-inmediato-terminal"]["future_supported"], .7216666667)
        self.assertEqual(catalog["guard"]["confirmed_operations_retained"], 261)
        story = self.lab.interactive_cases["aclaracion-1"]["story"]
        other = self.lab.interactive_cases["aclaracion-2"]["story"]
        partial = ["TEMPORAL_SCOPE"]
        self.assertEqual(simulate(story, partial), simulate(other, partial))
        complete = partial + ["TEMPORAL_SCOPE", "PROCEED", "PROCEED"]
        with_questions = simulate(story, complete)
        without_questions = simulate(story, ["PROCEED", "PROCEED"])
        self.assertTrue(with_questions["done"])
        first, second = with_questions["visits"]
        self.assertEqual(first["memory_after"], second["memory_before"])
        self.assertEqual([q["supported"] for q in with_questions["future"]], [1, 1])
        self.assertEqual([q["supported"] for q in without_questions["future"]], [0, 1])
        with self.assertRaises(ValueError):
            simulate(story, complete + ["PROCEED"])
