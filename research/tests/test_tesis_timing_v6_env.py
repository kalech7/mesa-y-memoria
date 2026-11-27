from copy import deepcopy
from itertools import groupby

import pytest

from research.tesis_timing_v6_bank import build_worlds
from research.tesis_timing_v6_env import (
    ACTIONS, cpu_audit, enumerate_world, fit_public_planner, prompt_text, replay, traverse,
)


def pair_worlds(split):
    worlds = build_worlds(split)
    pair = worlds[0]["pair_id"]
    return [world for world in worlds if world["pair_id"] == pair]


def test_disclosure_only_after_paid_ask_and_replay_cites_sqlite():
    story = next(w for w in pair_worlds("train")
                 if all(v["scope"] == "ongoing" for v in w["visits"][:2]))
    other_plan = deepcopy(story)
    other_plan["world_id"] = "SECRET_WORLD"
    other_plan["variant"] = "SECRET_VARIANT"
    other_plan["future_queries"] = [{"date": "2099-01-01", "attribute": "dinner_preference", "expected_value": "SECRET"}]
    other_plan["visits"][0]["reply"] = "SECRET_IF_NOT_ASKED"
    assert replay(story, (), 1) == replay(other_plan, (), 1)
    assert replay(story, ("WAIT",), 1) == replay(other_plan, ("WAIT",), 1)
    assert "SECRET" not in prompt_text(replay(other_plan, ("WAIT",), 1).observation)
    assert replay(story, ("ASK",), 1).legal == ("WAIT",)
    assert replay(story, ("WAIT",), 1).legal == ACTIONS
    with pytest.raises(ValueError, match="Illegal action"):
        replay(story, ("ASK", "ASK"), budget=1)
    finished = replay(story, ("ASK", "ASK"), budget=2)
    assert len(finished["memory"]) == 2
    assert finished["metrics"]["future_supported"] == 1
    assert finished["metrics"]["questions"] == 2
    texts = {row["turn_id"]: row["text"] for row in finished["messages"]}
    for record in finished["memory"]:
        assert all(q["text"] == texts[q["turn_id"]] for q in record["evidence_quotes"])
        assert len(record["evidence_quotes"]) == 2
    assert all(row["response"]["evidence_ids"] for row in finished["future"])


def test_session_answer_expires_and_unrecognized_replies_cannot_create_habits():
    story = next(w for w in pair_worlds("train") if all(v["scope"] == "session" for v in w["visits"][:2]))
    finished = replay(story, ("ASK", "ASK"), budget=2)
    assert all(m["scope"]["kind"] == "session" for m in finished["memory"])
    assert finished["metrics"]["future_correct_abstentions"] == 6
    assert finished["metrics"]["future_positive_supported"] == 0
    assert all(x["response"]["mode"] == "DEFER" for x in finished["future"])
    malformed = deepcopy(story)
    malformed["visits"][0].update(reply="Lo decidiré después.", scope="ongoing")
    assert replay(malformed, ("ASK", "WAIT"))["memory"] == []
    unsupported = deepcopy(story)
    unsupported["visits"][0]["value"] = "un plato ausente del turno"
    end = replay(unsupported, ("ASK", "WAIT"))
    assert not end["memory"] and end["metrics"]["current_first_supported"] == 0


def test_cpu_measures_aggregate_choice_before_observing_latent_world():
    dev = [enumerate_world(w) for w in pair_worlds("train")]
    heldout_dev = [enumerate_world(w) for w in pair_worlds("validation")]
    planner = fit_public_planner(dev)
    with pytest.raises(ValueError, match="TRAIN"):
        fit_public_planner(heldout_dev)
    metrics = cpu_audit(dev, heldout_dev)
    assert metrics["paired_timing"]["robust_flip_rate"] == 1
    assert metrics["paired_timing"]["mean_gain_over_best_constant"] == pytest.approx(.1)
    assert metrics["policies"]["preguntar-primero"]["questions"] == 1
    assert metrics["policies"]["preguntar-segundo"]["questions"] == 1
    assert metrics["policies"]["nunca"]["questions"] == 0
    assert metrics["policies"]["planificador-publico"]["future_supported"] == pytest.approx(.85)
    for tree in heldout_dev:
        node = tree.decisions[()]
        assert (planner.margin(node.observation, node.legal) > 0) == (tree.story["variant"] == "ask_now")
        path, end = traverse(tree, planner)
        assert sum(x == "ASK" for x in path) == end["questions"]


def test_equal_prefixes_and_terminal_metrics_in_every_world_of_each_root():
    forest = sorted(pair_worlds("validation"), key=lambda story: story["root_id"])
    for _, worlds in groupby(forest, key=lambda story: story["root_id"]):
        trees = [enumerate_world(w, budget=2) for w in worlds]
        for path in ((), ("WAIT",)):
            assert all(t.decisions[path] == trees[0].decisions[path] for t in trees)
        for tree in trees:
            assert len(tree.outcomes) == 4
            for path, leaf in tree.outcomes.items():
                assert leaf["current_supported"] == 1
                assert leaf["questions"] == path.count("ASK")
                assert leaf["future_positive_supported"] + leaf["future_correct_abstentions"] == pytest.approx(
                    6 * leaf["future_supported"])
