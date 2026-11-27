"""Frozen multicheckpoint caching and prospective conversation-v3 execution."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_local_001.model import Generation
from research.src.artifacts import canonical, digest, file_hash, write_json
from research.src.closure_environment import _episode, _record
from research.src.policy.llm_controller import LLMController, PHASE_LIMITS
from research.src.policy.rule_controller import RuleController
from research.varied_evaluation import (
    ACTIONS, GenerationBudget, MultiPolicyCache, enumerate_cases, execute,
    load_bank, prepare, sampled_episodes, summarize, verify_plan,
)


def encode(prompt):
    return [10, 11, *hashlib.sha256(prompt.encode()).digest()[:8]]


class FakeBackend:
    def __init__(self):
        self.config = SimpleNamespace(max_new_tokens=64, precision="float32", temperature=1.)
        self.model = SimpleNamespace(parameters=lambda: [SimpleNamespace(requires_grad=False)])
        self.calls = []
        self.teacher = RuleController()
        self.current_seed = 0

    def encode_prompt(self, prompt):
        return encode(prompt)

    def seed(self, seed):
        self.current_seed = seed

    def weights_fingerprint(self):
        return "immutable fake test weights"

    def generate(self, prompt, sample=False):
        self.calls.append((prompt, sample, self.current_seed))
        phase = prompt.splitlines()[0].split()[-1]
        observation = json.loads(prompt.split("\n\nOBSERVACION\n")[1])
        action = getattr(self.teacher, phase)(observation)
        if sample and phase == "select":
            selected = ACTIONS[self.current_seed % len(ACTIONS)]
            action = {"action": "PROCEED", "slot": None} if selected == "PROCEED" else {"action": "ASK", "slot": selected.split(":")[1]}
        return Generation(encode(prompt), [20, 21, 2], canonical(action), metadata={"fixture": True, "sampled": sample})


def registry(prompt="PHASE select\n\nOBSERVACION\n{}", limit=64):
    return {digest({"prompt": prompt, "max_new_tokens": limit}): {
        "prompt": prompt, "prompt_ids": encode(prompt), "max_new_tokens": limit, "phase": "select"}}


def test_equivalent_weights_reuse_greedy_outputs_across_labels(tmp_path):
    prompt = "PHASE select\n\nOBSERVACION\n{}"
    first, second = FakeBackend(), FakeBackend()
    first.generate = lambda prompt, sample=False: Generation(encode(prompt), [1, 2], '{"action":"PROCEED","slot":null}')
    second.generate = lambda *a, **k: pytest.fail("Equivalent greedy output must be cached")
    budget = GenerationBudget(5)
    one = MultiPolicyCache(first, "verified-effective-fingerprint", {"torch": "locked"}, tmp_path, registry(), budget)
    two = MultiPolicyCache(second, "verified-effective-fingerprint", {"torch": "locked"}, tmp_path, registry(), budget)
    a, b = one.generate(prompt), two.generate(prompt)
    assert a.text == b.text
    assert b.metadata["cache_hit"]
    assert budget.calls == 1 and budget.hits == 1


def test_distinct_effective_weights_or_precision_do_not_reuse_cache(tmp_path):
    prompt = "PHASE select\n\nOBSERVACION\n{}"
    backend = FakeBackend()
    backend.generate = lambda prompt, sample=False: Generation(encode(prompt), [1, 2], "{}")
    budget = GenerationBudget(5)
    MultiPolicyCache(backend, "a", {}, tmp_path, registry(), budget).generate(prompt)
    MultiPolicyCache(backend, "b", {}, tmp_path, registry(), budget).generate(prompt)
    backend.config.precision = "bfloat16"
    MultiPolicyCache(backend, "a", {}, tmp_path, registry(), budget).generate(prompt)
    assert budget.calls == 3 and budget.hits == 0


def test_stochastic_seeds_are_never_served_as_cached_on_policy_samples(tmp_path):
    prompt = "PHASE select\n\nOBSERVACION\n{}"
    backend = FakeBackend()
    backend.generate = lambda prompt, sample=False: Generation(encode(prompt), [1, 2], "{}")
    budget = GenerationBudget(5)
    cached = MultiPolicyCache(backend, "a", {}, tmp_path, registry(), budget)
    with pytest.raises(RuntimeError, match="seed required"):
        cached.generate(prompt, sample=True)
    cached.seed(101)
    cached.generate(prompt, sample=True)
    with pytest.raises(FileExistsError, match="seed/prefix reuse"):
        cached.generate(prompt, sample=True)
    cached.seed(102)
    cached.generate(prompt, sample=True)
    assert budget.calls == 2 and budget.hits == 0


def test_budget_counts_failed_attempts_and_never_retries(tmp_path):
    prompt = "PHASE select\n\nOBSERVACION\n{}"
    backend = FakeBackend()
    def failure(*args, **kwargs):
        raise RuntimeError("transport failure")
    backend.generate = failure
    budget = GenerationBudget(1)
    cached = MultiPolicyCache(backend, "a", {}, tmp_path, registry(), budget)
    with pytest.raises(RuntimeError, match="transport failure"):
        cached.generate(prompt)
    with pytest.raises(RuntimeError, match="BUDGET_EXHAUSTED"):
        cached.generate(prompt)
    assert budget.calls == 1
    assert not list(tmp_path.glob("*.json"))


def test_cache_integrity_and_unplanned_prefix_fail_as_technical_errors(tmp_path):
    prompt = "PHASE select\n\nOBSERVACION\n{}"
    backend = FakeBackend()
    backend.generate = lambda prompt, sample=False: Generation(encode(prompt), [1, 2], "{}")
    budget = GenerationBudget(5)
    cached = MultiPolicyCache(backend, "a", {}, tmp_path, registry(), budget)
    cached.generate(prompt)
    path = next(tmp_path.glob("*.json"))
    item = json.loads(path.read_text())
    item["generation"]["text"] = "tampered"
    write_json(path, item)
    with pytest.raises(RuntimeError, match="Corrupt"):
        cached.generate(prompt)
    with pytest.raises(RuntimeError, match="UNPLANNED"):
        cached.generate(prompt + " hidden truth")
    assert budget.calls == 1


def fixture_bank(tmp_path):
    episodes = []
    for index in range(2):
        e = _episode()
        e.update(root_id=f"fixture_{index}", latent_world_id="only_world", realization_id="v0", split="validation")
        e["initial_memory"] += [_record("drink", "agua", attribute="drink_preference"), _record("breakfast", "avena", attribute="breakfast_preference")]
        e["user_messages"] = [{"turn_id": "before", "role": "user", "text": "Para beber prefiero agua."},
            {"turn_id": "assistant_hypothesis", "role": "assistant", "text": "¿Quizá sopa?"},
            {"turn_id": "u2", "role": "user", "text": "Ahora prefiero pasta. ¿Qué ceno hoy?"}]
        e["authoring"] = {"stratum": f"stratum_{index}"}
        episodes.append(e)
    bank = tmp_path / "validation.jsonl"
    bank.write_text("\n".join(canonical(e) for e in episodes) + "\n")
    write_json(bank.with_name("manifest.json"), {"data_role": "prospective_development_only", "bank_sha256": file_hash(bank)})
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for phase in ("select", "respond", "write", "reader"):
        (prompts / (phase + ".txt")).write_text("PHASE " + phase)
    return episodes, bank, prompts


def test_varied_bank_refuses_test_and_unfrozen_labels(tmp_path):
    episodes, bank, _ = fixture_bank(tmp_path)
    assert len(load_bank(bank, expected_roots=2, historical_splits={})[0]) == 2
    episodes[0]["split"] = "test_id"
    bank.write_text("\n".join(canonical(e) for e in episodes))
    write_json(bank.with_name("manifest.json"), {"data_role": "prospective_development_only", "bank_sha256": file_hash(bank)})
    with pytest.raises(ValueError, match="non-development"):
        load_bank(bank, expected_roots=2, historical_splits={})


def test_preflight_preserves_roles_and_plans_I1_question_context(tmp_path):
    episodes, _, prompts = fixture_bank(tmp_path)
    cases, prefixes = enumerate_cases(episodes, prompts, encode)
    assert len(cases) == 2
    assert cases[0]["select_observation"]["user_messages"][1]["role"] == "assistant"
    assert all(len(entry["prompt_ids"]) <= 2048 for entry in prefixes.values())
    assert all("PRIVATE_GOLD" not in entry["prompt"] for entry in prefixes.values())
    from research.src.env.episode import EpisodeEnvironment
    env = EpisodeEnvironment(episodes[0], tmp_path / "separate.sqlite")
    try:
        env.ask("TEMPORAL_SCOPE")
        i1, i2 = env.observe("write", "O2", "I1"), env.observe("write", "O2", "I2")
        assert i1 != i2
        prompt = LLMController(object(), prompt_root=prompts).prompt("write", i1)
        assert digest({"prompt": prompt, "max_new_tokens": PHASE_LIMITS["write"]}) in prefixes
    finally:
        env.dispose()


def test_means_respect_world_and_realization_hierarchy():
    episodes = [{"root_id": "r", "authoring": {"stratum": "s"}}]
    def row(world, realization, success):
        return {"root_id": "r", "latent_world_id": world, "realization_id": realization,
            "metrics": {"C2": success, "FJS": success, "unsupported_operations": 0,
                "unsupported_write_episode": 0, "invalid_outputs": 0, "questions": 0},
            "rewards_by_component": {"R_long": success * 2, "R_short": success}}
    rows = [row("a", "one", 1), row("a", "two", 1), row("a", "three", 1), row("b", "one", 0)]
    assert summarize(rows, episodes)["root_weighted"]["J_primary"] == .5


def test_prepare_and_execute_injected_fixture_preserves_physical_alias_states(tmp_path, monkeypatch):
    episodes, bank, prompts = fixture_bank(tmp_path)
    monkeypatch.setattr("research.varied_evaluation.historical_banks", lambda: ({}, {}))
    labels = [{"method": m, "training_seed": 101, "checkpoint_path": "fake_" + m, "selected_step": 0}
              for m in ("B1", "B3", "B2-I", "B2-L")]
    classes = [{"fingerprint": "effective_same", "representative": labels[0], "labels": labels}]
    next_label = {"method": "B1", "training_seed": 202, "checkpoint_path": "fake_202", "selected_step": 0}
    classes.append({"fingerprint": "effective_distinct", "representative": next_label, "labels": [next_label]})
    monkeypatch.setattr("research.varied_evaluation.verify_catalog", lambda path: (classes, {}))
    output = tmp_path / "run"
    plan = prepare(bank, tmp_path / "review.json", prompts, output, encoder=encode, expected_roots=2, device="cpu")
    assert plan["training"] is False and plan["checkpoint_selection"] is False
    assert plan["maximum_input_tokens"] == 10
    assert not (output / "execution_started.json").exists()
    with pytest.raises(ValueError, match="tokenizer fixture"):
        execute(output, device="cpu")
    weak_models = []
    def factory(path, device):
        assert all(ref() is None for ref in weak_models), "Previous GPU model survived into the next model load"
        backend = FakeBackend()
        weak_models.append(weakref.ref(backend))
        return backend
    result = execute(output, device="cpu", backend_factory=factory)
    assert result["status"] == "executed"
    assert result["training"] is False and result["optimizer_steps"] == 0
    assert len(result["primary"]) == 5
    assert result["full_sampled_groups"] == 4
    assert result["selector_samples"] == plan["observable_states"] * 4 * 2
    assert result["execution_scope"] == "injected_test_backend"
    assert result["PI2_pairs_verified"] == 2
    assert result["contrasts_101"]["B1_101_O1_no_question"]["episodes"] == 2
    assert result["contrasts_101"]["B1_101_O2_no_question"]["episodes"] == 2
    assert result["calls"]["cache_hits"] > 0
    assert len(list((output / "states").glob("*.sqlite"))) == result["trajectories"]
    rows = [json.loads(line) for line in (output / "trajectories.jsonl").open()]
    assert len({row["sqlite_path"] for row in rows}) == len(rows)
    assert all(file_hash(row["sqlite_path"]) == row["sqlite_sha256"] for row in rows)
    primary = [r for r in rows if r["review_arm"] in result["primary"]]
    assert len(primary) == 10
    assert all(r["loaded_representative"]["method"] == "B1" for r in primary)
    assert len({r["intended_model"]["method"] for r in primary}) == 4
    assert len(weak_models) == 2 and all(ref() is None for ref in weak_models)
    assert json.loads((output / "progress.json").read_text())["status"] == "executed"
    with pytest.raises(FileExistsError, match="Execution already started"):
        execute(output, device="cpu", backend_factory=lambda path, device: FakeBackend())


@pytest.mark.parametrize("count", [1, 3, 4, 6])
def test_varied_memory_counts_are_permitted_without_global_fixture_distribution(tmp_path, count):
    episodes, bank, _ = fixture_bank(tmp_path)
    for episode in episodes:
        original = episode["initial_memory"]
        episode["initial_memory"] = (original * 2)[:count]
    bank.write_text("\n".join(canonical(e) for e in episodes))
    write_json(bank.with_name("manifest.json"), {"data_role": "prospective_development_only", "bank_sha256": file_hash(bank)})
    assert len(load_bank(bank, expected_roots=2, historical_splits={})[0][0]["initial_memory"]) == count


@pytest.mark.parametrize("count", [0, 7])
def test_varied_memory_counts_outside_frozen_support_are_rejected(tmp_path, count):
    episodes, bank, _ = fixture_bank(tmp_path)
    for episode in episodes:
        episode["initial_memory"] = (episode["initial_memory"] * 3)[:count]
    bank.write_text("\n".join(canonical(e) for e in episodes))
    write_json(bank.with_name("manifest.json"), {"data_role": "prospective_development_only", "bank_sha256": file_hash(bank)})
    with pytest.raises(ValueError, match="1..6 memory"):
        load_bank(bank, expected_roots=2, historical_splits={})


def test_role_renaming_cannot_bypass_historical_v3_identity_guard(tmp_path):
    from research.src.data.semantic_partition_v2 import SemanticSplitLeakage
    episodes, bank, _ = fixture_bank(tmp_path)
    historical = deepcopy(episodes)
    for episode in historical:
        episode["root_id"] = "old_" + episode["root_id"]
        for message in episode["user_messages"]:
            if message.get("role") == "user":
                message.pop("role")
    with pytest.raises(SemanticSplitLeakage):
        load_bank(bank, expected_roots=2, historical_splits={"legacy_train": historical})
