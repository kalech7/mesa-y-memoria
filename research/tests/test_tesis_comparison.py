from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from research.competence_evaluation import _run_trajectory
from research.src.policy.rule_controller import RuleController
from research.tesis_comparison import (
    FrozenOutputCache, METHODS, SEEDS, TrainingCase, ordered_train, parent_records,
    selected_rows, summarize, train_sft_group,
)
from research.tesis_comparison_audit import verify_trajectory
from research.tests.test_competence_evaluation import _episode


def episode(root, family, world="one", split="train"):
    return {"root_id": root, "scenario_family": family, "latent_world_id": world,
            "realization_id": "first", "split": split}


def test_shared_train_order_preserves_roots_and_is_input_order_independent():
    rows = [episode("c", "F0"), episode("d", "F0"), episode("a", "F4", "temporary"),
            episode("a", "F4", "ongoing"), episode("b", "F1")]
    order = ordered_train(rows, 101)
    assert order == ordered_train(list(reversed(rows)), 101)
    assert Counter(row["root_id"] for row in order) == {"a": 2, "b": 1, "c": 1, "d": 1}
    assert [row["root_id"] for row in order[:2]] == ["a", "a"]
    with pytest.raises(ValueError, match="TRAIN-only"):
        ordered_train([episode("x", "F0", split="evaluation")], 101)


def test_clean_parents_are_seed_specific_and_require_all_selection_evidence():
    parents = parent_records(SEEDS)
    assert set(parents) == set(SEEDS)
    assert len({record["adapter_sha256"] for record in parents.values()}) == 3
    training = {seed: {method: {"selection": {**parents[seed], "method": method}}
                       for method in METHODS[1:]} for seed in SEEDS}
    assert len(selected_rows(SEEDS, parents, training)) == 9
    training[101][METHODS[1]]["selection"]["adapter_sha256"] = "wrong"
    with pytest.raises(ValueError, match="artifacts changed"):
        selected_rows(SEEDS, parents, training)


def test_frozen_cache_keys_exact_prompt_and_cap_and_preserves_original_output():
    class Policy:
        config = SimpleNamespace(max_new_tokens=100)
        calls = []

        def generate(self, prompt, *, sample):
            self.calls.append((prompt, self.config.max_new_tokens, sample))
            return {"prompt": prompt, "metadata": {}}

    policy = Policy()
    cache = FrozenOutputCache(policy)
    cache.generate("a")["metadata"]["mutable"] = True
    assert cache.generate("a")["metadata"] == {}
    assert cache.hits == 1
    cache.generate("b")
    policy.config.max_new_tokens = 101
    cache.generate("a")
    assert policy.calls == [("a", 100, False), ("b", 100, False), ("a", 101, False)]
    with pytest.raises(ValueError, match="greedy"):
        cache.generate("a", sample=True)


def test_additional_sft_raises_observable_action_probability():
    torch = pytest.importorskip("torch")
    weights = torch.nn.Parameter(torch.zeros(5))

    class Actor:
        def __init__(self):
            self.torch = torch

        def likelihood_mode(self):
            return nullcontext()

        def _action_distribution(self, prefix):
            assert prefix == (7, 9)
            return weights, weights.log_softmax(dim=-1)

    optimizer = torch.optim.AdamW([weights], lr=.1, weight_decay=0)
    case = TrainingCase(episode("r", "F4"), "exact-prefix", (7, 9), 1)
    result = train_sft_group(Actor(), optimizer, [weights], case, 1)
    assert result["target_action"] == {"action": "ASK", "slot": "TEMPORAL_SCOPE"}
    assert len(result["optimization"]) == 2
    assert weights[1] > weights[0]


def test_comparison_bootstrap_uses_paired_roots_not_seeds_as_histories():
    def panel(values):
        metrics = {key: values for key in ("strict_joint", "FJS", "uninformative_questions", "unsupported_write")}
        return {"metrics": metrics, "by_root": {"a": metrics, "b": metrics},
                "families": {"a": "F0", "b": "F4"}}

    panels = {seed: {METHODS[0]: panel(.0), METHODS[1]: panel(.25), METHODS[2]: panel(.5)}
              for seed in SEEDS}
    report = summarize(panels, resamples=20)
    contrast = report["paired_root_contrasts"][f"{METHODS[2]} vs {METHODS[1]}"]["strict_joint"]
    assert contrast["estimate"] == .25
    assert contrast["independent_roots"] == 2
    assert contrast["training_seeds"] == list(SEEDS)


@pytest.mark.parametrize("metric", ("questions", "unsupported_write_episode", "any_invalid_output", "R_long"))
def test_comparison_audit_rejects_corrupt_reward_inputs(tmp_path, metric):
    row = _run_trajectory(RuleController(), _episode(ambiguous=True), tmp_path / "trace.sqlite")
    verify_trajectory(row)
    row["metrics"][metric] += 1
    with pytest.raises(ValueError, match=f"Archived metric {metric}"):
        verify_trajectory(row)
