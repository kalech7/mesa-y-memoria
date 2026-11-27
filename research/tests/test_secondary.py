from copy import deepcopy
import json
from types import SimpleNamespace

from research.src.data.build_bank import build_bank
from research.src.evaluation.runner import run_episode
from research.src.evaluation import secondary
from research.src.policy.rule_controller import RuleController
from research.src.readers.symbolic import read


class ExtractionBackend:
    def __init__(self, output):
        self.output = output
        self.config = SimpleNamespace(max_new_tokens=32)
        self.prompts = []

    def encode_prompt(self, prompt):
        return [1] * 30

    def generate(self, prompt, sample=False):
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.output, truncated=False)


def test_extraction_actual_invalid_output_cached_empty_without_gold_repair(tmp_path):
    episode = build_bank()[0]
    backend = ExtractionBackend("invalid JSON")
    cache = secondary.build_extraction_cache([episode], backend, tmp_path)
    entry = cache[episode["root_id"]]
    assert entry["technical_status"] == "failed"
    assert entry["records"] == []
    assert entry["gold_repair"] is False
    assert "future_query" not in backend.prompts[0]
    assert "current_value" not in backend.prompts[0]
    assert "u2" not in backend.prompts[0]
    assert backend.config.max_new_tokens == 32
    secondary.build_extraction_cache([episode], backend, tmp_path)
    assert len(backend.prompts) == 1


def test_false_extraction_cannot_validate_itself_against_its_own_memory():
    episode = build_bank()[0]
    false_record = deepcopy(episode["initial_memory"][0])
    false_record["value"] = "inventado"
    false_record["memory_id"] = "extracted_false"
    audit = secondary._extraction_provenance_audit(episode["initial_memory"], [false_record], episode["current_query"])
    assert audit["unsupported_extracted_records"] == 1


def test_extract_evaluation_rejects_factually_lucky_false_provenance(monkeypatch):
    episode = next(e for e in build_bank() if e["scenario_family"] == "F0")
    row = run_episode(episode, RuleController())
    forged = deepcopy(episode["initial_memory"][1])
    forged.update(memory_id="false_extracted", value=episode["truth"]["future_value"])
    forged["scope"] = deepcopy(episode["initial_memory"][0]["scope"])
    row["trace"]["episode"]["initial_memory"] = [forged]
    row["trace"]["final_memory"] = [forged]
    for observation in row["trace"]["observations"].values():
        observation["memory"] = [forged]
    answer = {"mode": "ANSWER", "value": forged["value"], "branches": [], "evidence_ids": ["false_extracted"]}
    row["trace"]["response"] = answer
    row["trace"]["future_response"] = answer
    monkeypatch.setattr(secondary, "evaluate_batched", lambda *args, **kwargs: iter([deepcopy(row)]))
    cache = {episode["root_id"]: {"records": [forged], "technical_status": "executed"}}
    result = list(secondary.evaluate_extracted([episode], object(), cache))[0]
    assert result["metrics"]["C2"] == 0
    assert result["metrics"]["FJS"] == 0
    assert result["extraction_audit"]["unsupported_extracted_records"] == 1


def test_b4_secondary_controls_accept_no_llm_backend():
    episode = next(e for e in build_bank() if e["scenario_family"] == "F0")
    cache = {episode["root_id"]: {"records": deepcopy(episode["initial_memory"]), "technical_status": "executed"}}
    extracted = list(secondary.evaluate_extracted([episode], None, cache, method="B4", batch_size=8, seed=202))
    assert extracted[0]["method"] == "B4"
    assert extracted[0]["seed"] == 202
    assert extracted[0]["metrics"]["FJS"] == 1
    noresponse = list(secondary.evaluate_nonresponse([episode], None, method="B4", seed=202))
    assert [r["nonresponse_rate"] for r in noresponse] == [.2, .4]


def test_frozen_reader_receives_actual_memory_and_no_dialogue(monkeypatch):
    episode = next(e for e in build_bank() if e["opportunity"]["eligible"])
    row = run_episode(episode, RuleController())
    row["trace"]["final_memory"] = []
    received = []

    def fake_generate(backend, phase, observations, **options):
        received.extend(deepcopy(observations))
        assert phase == "reader"
        return [{"parsed_action": read(o["memory"], o["query"]), "technical_status": "completed"} for o in observations]

    monkeypatch.setattr(secondary, "generate_batch", fake_generate)
    result = list(secondary.reread_frozen([row], object()))[0]
    assert set(received[0]) == {"memory", "query"}
    assert received[0]["memory"] == []
    assert result["metrics"]["FJS"] == 0
    assert result["reader_id"] == "frozen_base_llm"


def test_reuse_interaction_has_only_memory_query_and_own_new_reply(monkeypatch):
    episode = next(e for e in build_bank() if e["opportunity"]["eligible"] and e["truth"]["scope_kind"] == "ongoing")
    row = run_episode(episode, RuleController())
    seen = []

    def fake_generate(backend, phase, observations, **options):
        seen.append((phase, deepcopy(observations)))
        actions = [{"action": "ASK", "slot": "TEMPORAL_SCOPE"} for _ in observations] if phase == "select" else [RuleController().respond(o) for o in observations]
        return [{"parsed_action": action, "technical_status": "completed"} for action in actions]

    monkeypatch.setattr(secondary, "generate_batch", fake_generate)
    result = list(secondary.reuse_with_interaction([row], object()))[0]
    assert seen[0][1][0]["user_messages"] == []
    assert seen[1][1][0]["user_messages"][0]["turn_id"] == "s3_reply"
    assert all(m["turn_id"] != "u2" for m in seen[1][1][0]["user_messages"])
    assert result["metrics"]["s3_questions"] == 1
    assert result["metrics"]["repeated_scope_questions"] == 1
    assert result["metrics"]["FJS"] == 1
