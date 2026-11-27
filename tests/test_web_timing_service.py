"""Exercise the final adapter selection and the live actor boundary without a GPU."""

import pytest

from web_demo import timing_service as timing


class RecordingScorer:
    def __init__(self):
        self.cache = {}
        self.encodings = {}
        self.tokenizer = self
        self.adapters = []
        self.observations = []

    def load(self, adapter):
        self.adapters.append(adapter)

    def apply_chat_template(self, messages, **_):
        return " ".join(row["content"] for row in messages).split()

    def margins(self, observations):
        self.observations.extend(observations)
        return [1.0 if item["visit"] == 2 else -0.75 for item in observations]


@pytest.fixture
def runtime(monkeypatch):
    # The actual frozen selection and adapter signatures are available locally.
    # Only verification of the 1.5B base/parent needs the deployment host.
    monkeypatch.setattr(timing, "check_sources", lambda run: timing.read(run / "PLAN.json"))
    scorer = RecordingScorer()
    return timing.TimingLab(scorer_factory=lambda: scorer), scorer


def test_deploys_one_validation_selected_checkpoint_per_arm(runtime):
    lab, scorer = runtime
    catalog = lab.catalog()
    assert [opt["id"] for opt in catalog["arms"]] == ["SFT-ramas", "GRPO-continuacion"]
    assert set(catalog["results"]["summary"]) == {"SFT-ramas", "GRPO-continuacion"}
    assert "seeds" not in catalog and "seed" not in catalog["defaults"]
    for arm, seed in (("SFT-ramas", 101), ("GRPO-continuacion", 303)):
        descriptor = lab.describe_model(arm)
        result = lab.run_episode({"arm": arm})
        assert (descriptor["seed"], descriptor["round"]) == (seed, 2)
        assert result["model"] == {**descriptor, "used": True}
        assert scorer.adapters[-1] == lab.run / lab.lock["selected"][str(seed)][arm]["adapter"]
        assert [v["action"] for v in result["visits"]] == ["WAIT", "ASK", "READ"]
        assert [v["margin"] for v in result["visits"]] == [-0.75, 1.0, None]
    with pytest.raises(timing.ProfilesError):
        lab.run_episode({"arm": "GRPO-inmediato"})
    with pytest.raises(timing.ProfilesError):
        lab.run_episode({"arm": "SFT-ramas", "seed": 303})


def test_live_actor_only_sees_public_observations_until_after_its_question(runtime):
    lab, scorer = runtime
    ongoing = lab.run_episode({"scope": "ongoing"})
    visits = scorer.observations.copy()
    assert [obs["visit"] for obs in visits] == [1, 2]
    session = lab.run_episode({"scope": "session"})
    assert scorer.observations[2:] == visits
    assert [row["valid"] for row in ongoing["memory"]] == [True]
    assert [row["valid"] for row in session["memory"]] == [False]
    response = lab.score_public("SFT-ramas", visits[1])
    assert response["source"] == "qwen" and response["action"] == "ASK"
    assert response["margin"] == 1.0 and response["model"] == lab.describe_model("SFT-ramas")
    count = len(scorer.observations)
    forced = lab.score_public("SFT-ramas", {**visits[1], "questions_remaining": 0})
    assert forced["source"] == "forced" and forced["margin"] is None
    assert len(scorer.observations) == count


def test_live_context_preserves_the_last_message_and_question(runtime):
    lab, scorer = runtime
    lab.run_episode({})
    visit = scorer.observations[-1]
    long = {"role": "user", "text": "cenar " * 1500}
    newest = {"role": "user", "text": "Ahora prefiero sopa."}
    oversized = {**visit, "messages": [long, long, newest]}
    result = lab.score_public("SFT-ramas", oversized)
    assert result["context"]["messages_omitted"] == 1
    assert result["context"]["tokens"] <= result["context"]["max_tokens"] == 2048
    assert scorer.observations[-1]["messages"] == [long, newest]
    assert scorer.observations[-1]["question"] == visit["question"]
    with pytest.raises(timing.ProfilesError) as error:
        lab.score_public("SFT-ramas", {**visit, "messages": [{**long, "text": long["text"] * 2}]})
    assert error.value.status == 413
