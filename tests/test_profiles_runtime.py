from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from research.src.artifacts import file_hash
from web_demo.__main__ import Handler
from web_demo.profiles_chat import ProfilesChats
from web_demo.profiles_runtime import ProfilesError, ProfilesRuntime, actor_observation

from tests.test_profiles_chat import Runtime


def publish(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    base = tmp_path / "models/Qwen2.5-1.5B-Instruct"
    base.mkdir(parents=True)
    model_files = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    for name in model_files:
        (base / name).write_text("{}")
    (run / "PLAN.json").write_text(json.dumps({
        "model_hashes": {name: file_hash(base / name) for name in model_files},
    }))
    methods = {}
    for method, name, eligible, epoch in (
        ("SFT-ramas", "sft", True, 1), ("GRPO-temperatura", "grpo", False, 0),
        ("GRPO-ramas", "original", False, 0), ("Qwen-inicial", "parent", True, 0),
    ):
        path = run / name
        path.mkdir()
        (path / "adapter_model.safetensors").write_bytes(name.encode())
        (path / "adapter_config.json").write_text("{}")
        methods[method] = {"adapter": name, "epoch": epoch, "eligible": eligible,
                           "weights": file_hash(path / "adapter_model.safetensors"),
                           "config": file_hash(path / "adapter_config.json")}
    panels = {"validation_selected": {"101": methods},
              "last_epoch": {"101": {"GRPO-temperatura": {**methods["GRPO-temperatura"], "epoch": 2}}}}
    (run / "LOCK.json").write_text(json.dumps({"panels": panels}))
    return run


def test_selection_preserves_exact_locked_grpo_and_defaults_to_eligible_sft(tmp_path, monkeypatch):
    monkeypatch.setattr("web_demo.profiles_runtime.find_spec", lambda name: object())
    monkeypatch.setattr("web_demo.profiles_runtime.ROOT", tmp_path)
    runtime = ProfilesRuntime(run=publish(tmp_path))
    assert runtime.resolve()["id"] == "sft-101"
    requested = runtime.resolve("grpo-101")
    assert requested["id"] == "grpo-101" and requested["adapter"] == "grpo"
    assert requested["epoch"] == 0 and not requested["eligible"]
    assert "punto de partida" in requested["fallback"]
    assert runtime.resolve("grpo-101-terminal")["diagnostic"]


def test_result_catalog_removes_private_dialogues_and_checks_selection_link(tmp_path):
    run = publish(tmp_path)
    results = {
        "selection_sha256": file_hash(run / "LOCK.json"),
        "population": {"roots": 18}, "parser": {"summary": {"v5": {"opened": 12}}, "details": ["private"]},
        "semantic": {"parsers": {"v5": {"matches": {"response_exact": 40}, "cases": ["private"]}}},
        "panels": {"validation_selected": {"summary": {}, "comparisons": {}, "rows": ["private"]}},
    }
    (run / "RESERVED_RESULTS.json").write_text(json.dumps(results))
    runtime = ProfilesRuntime(run=run)
    public = runtime.results()
    assert public["reserved"]["selection_verified"]
    assert "private" not in json.dumps(public)
    (run / "LOCK.json").write_text("{}")
    assert not runtime.results()["reserved"]["selection_verified"]


def test_default_prefers_a_trained_eligible_arm_over_ineligible_sft_or_parent(tmp_path, monkeypatch):
    monkeypatch.setattr("web_demo.profiles_runtime.find_spec", lambda name: object())
    monkeypatch.setattr("web_demo.profiles_runtime.ROOT", tmp_path)
    run = publish(tmp_path)
    lock = json.loads((run / "LOCK.json").read_text())
    selected = lock["panels"]["validation_selected"]["101"]
    selected["SFT-ramas"].update(epoch=0, eligible=False)
    selected["Qwen-inicial"].update(eligible=False)
    selected["GRPO-temperatura"].update(epoch=2, eligible=True)
    (run / "LOCK.json").write_text(json.dumps(lock))
    assert ProfilesRuntime(run=run).resolve()["id"] == "grpo-101"


@pytest.mark.parametrize("missing", ["model.safetensors", "tokenizer.json"])
def test_adapters_are_unavailable_when_required_base_or_tokenizer_file_is_missing(tmp_path, monkeypatch, missing):
    monkeypatch.setattr("web_demo.profiles_runtime.find_spec", lambda name: object())
    monkeypatch.setattr("web_demo.profiles_runtime.ROOT", tmp_path)
    run = publish(tmp_path)
    (tmp_path / "models/Qwen2.5-1.5B-Instruct" / missing).unlink()
    catalog = ProfilesRuntime(run=run).catalog()
    assert catalog["models"] and not any(model["available"] for model in catalog["models"])
    assert catalog["default_model"] is None
    assert missing in catalog["reason"]
    assert all("modelo base o del tokenizador" in model["reason"] and missing in model["reason"]
               for model in catalog["models"])


def test_catalog_checks_common_requirements_once_without_hashing_weights(tmp_path, monkeypatch):
    monkeypatch.setattr("web_demo.profiles_runtime.ROOT", tmp_path)
    run = publish(tmp_path)
    imports, reads = [], []

    def dependency(name):
        imports.append(name)
        return object()

    def document(path):
        reads.append(path.name)
        return json.loads(path.read_text())

    def unexpected_hash(path):
        raise AssertionError(f"The catalog must only check file presence: {path}")

    monkeypatch.setattr("web_demo.profiles_runtime.find_spec", dependency)
    monkeypatch.setattr("web_demo.profiles_runtime.read_json", document)
    monkeypatch.setattr("web_demo.profiles_runtime.file_hash", unexpected_hash)
    catalog = ProfilesRuntime(run=run).catalog()
    assert all(model["available"] for model in catalog["models"])
    assert imports == ["torch", "transformers", "peft"]
    assert reads.count("PLAN.json") == 1
    assert reads.count("LOCK.json") == 1


def test_missing_dependency_disables_catalog_with_an_explanation(tmp_path, monkeypatch):
    monkeypatch.setattr("web_demo.profiles_runtime.ROOT", tmp_path)
    monkeypatch.setattr("web_demo.profiles_runtime.find_spec", lambda name: None if name == "peft" else object())
    catalog = ProfilesRuntime(run=publish(tmp_path)).catalog()
    assert not any(model["available"] for model in catalog["models"])
    assert "dependencias de inferencia" in catalog["reason"] and "peft" in catalog["reason"]


def test_prompt_annotation_is_exact_training_helper(tmp_path):
    observation = {
        "user_messages": [{"turn_id": "u1", "role": "user", "text": "Para la cena, prefiero sopa."}],
        "memory": [], "query": {"attribute": "dinner_preference", "date": "2026-12-03",
                               "context": None, "session_id": "S1"},
        "visit": 1, "questions_remaining": 2, "asked_slots": [], "action_history": (),
        "reuse_regime": "Habrá visitas posteriores.",
    }
    from research.tesis_profiles_v5 import actor_observation as training_view

    actual = actor_observation(observation, ("PROCEED", "TEMPORAL_SCOPE"))
    assert actual == training_view(observation, ("PROCEED", "TEMPORAL_SCOPE"))
    assert len(observation["user_messages"]) == 1


def test_http_profiles_routes_persist_by_owner_token(tmp_path):
    class Lab:
        profiles = ProfilesChats(storage=tmp_path, runtime=Runtime())

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.lab = Lab()
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        address = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(address + "/api/profiles/catalog") as response:
                assert json.load(response)["default_model"] == "sft-101"
            def post(body):
                request = Request(address + "/api/profiles/chat", data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
                with urlopen(request) as response:
                    return json.load(response)
            chat = post({"action": "start", "date": "2026-12-03"})
            loaded = post({"action": "load", "id": chat["id"], "owner_token": chat["owner_token"]})
            assert loaded["id"] == chat["id"] and loaded["memory"] == []
            with pytest.raises(HTTPError) as error:
                post({"action": "load", "id": chat["id"], "owner_token": "wrong"})
            assert error.value.code == 404
        finally:
            server.shutdown()
            worker.join(timeout=5)
