from copy import deepcopy
import json

import pytest

from research.src.env.episode import EpisodeEnvironment
from research.src.memory.executor import Executor
from research.src.memory.store import Store
from research.src.readers.symbolic import read


def record(value="ensalada", *, mid="m1", kind="ongoing", start="2026-01-01", end=None, session=None, context=None, supersedes=None):
    return {"memory_id": mid, "attribute": "dinner_preference", "value": value,
            "scope": {"kind": kind, "session_id": session, "context": context,
                      "valid_from": start, "valid_until_exclusive": end},
            "status": "unresolved" if kind == "unknown" else "confirmed",
            "source_turn_ids": ["u1"], "evidence_quotes": [{"turn_id": "u1", "text": "Prefiero " + value}],
            "supersedes": supersedes or [], "recorded_at": start}


def episode():
    return {"root_id": "r1", "latent_world_id": "secret", "scenario_family": "F4", "split": "dev_smoke",
            "initial_memory": [record()], "user_messages": [{"turn_id": "u2", "text": "Ahora prefiero pasta"}],
            "current_query": {"attribute": "dinner_preference", "date": "2026-01-12", "session_id": "S2", "context": None},
            "future_query": {"attribute": "dinner_preference", "date": "2026-01-14", "session_id": "S3", "context": None},
            "replies": {"TEMPORAL_SCOPE": {"turn_id": "u3", "text": "Solo para hoy"}},
            "truth": {"future_value": "ensalada"}}


def operation(value="pasta", **options):
    value = record(value, **options)
    value.pop("memory_id")
    value.pop("recorded_at")
    value.pop("supersedes")
    return {"op": "ADD", "record": value}


def test_T02_T04_T05_prefix_no_hidden_future(tmp_path):
    a = episode()
    b = deepcopy(a)
    b["latent_world_id"] = "other"
    b["truth"]["future_value"] = "pasta"
    b["future_query"]["date"] = "2030-10-10"
    b["replies"]["TEMPORAL_SCOPE"]["text"] = "Como preferencia habitual desde ahora"
    ea = EpisodeEnvironment(a, tmp_path / "a.db")
    eb = EpisodeEnvironment(b, tmp_path / "b.db")
    assert ea.observe("select") == eb.observe("select")
    assert set(ea.observe("select")) == {"memory", "query", "user_messages", "reuse_regime"}
    assert ea.ask("TEMPORAL_SCOPE") != eb.ask("TEMPORAL_SCOPE")
    ea.dispose()
    eb.dispose()


def test_T08_T09_reopens_database_only(tmp_path):
    env = EpisodeEnvironment(episode(), tmp_path / "memory.db")
    env.ask("TEMPORAL_SCOPE")
    env.close_session()
    assert env._reply is None and env._messages == [] and env._initial_episode is None
    observation = env.reopen_for_reuse()
    assert set(observation) == {"memory", "query"}
    assert "Solo para hoy" not in json.dumps(observation)
    assert read(**observation)["value"] == "ensalada"
    env.store.conn.execute("DELETE FROM cm_records")
    env.store.conn.commit()
    assert read(**env.reopen_for_reuse())["mode"] == "DEFER"
    env.dispose()


def test_T10_exception_expires(tmp_path):
    store = Store(tmp_path / "a.db")
    store.initialize([record()])
    op = operation(kind="interval", start="2026-01-12", end="2026-01-13")
    assert Executor(store).execute([op])["valid"]
    query = episode()["current_query"]
    assert read(store.list_records(), query)["value"] == "pasta"
    query["date"] = "2026-01-14"
    assert read(store.list_records(), query)["value"] == "ensalada"
    assert len(store.list_records()) == 2
    store.close()


def test_T11_historical_substitution(tmp_path):
    store = Store(tmp_path / "a.db")
    store.initialize([record()])
    op = operation(start="2026-01-12")
    op.update(op="SUPERSEDE", target_ids=["m1"])
    assert Executor(store).execute([op])["valid"]
    query = episode()["current_query"]
    assert read(store.list_records(), query)["value"] == "pasta"
    query["date"] = "2026-01-05"
    assert read(store.list_records(), query)["value"] == "ensalada"
    assert store.list_records()[0]["value"] == "ensalada"
    store.close()


def test_T12_context_is_query_local(tmp_path):
    records = [record("ensalada", kind="contextual", context="casa"), record("pasta", mid="m2", kind="contextual", context="oficina")]
    query = episode()["current_query"]
    query["context"] = "casa"
    assert read(records, query)["value"] == "ensalada"
    query["context"] = "oficina"
    assert read(records, query)["value"] == "pasta"
    query["context"] = None
    assert read(records, query)["mode"] == "CONDITIONALIZE"


def test_T13_T14_T15_intervention_bytes(tmp_path):
    env = EpisodeEnvironment(episode(), tmp_path / "a.db")
    env.ask("TEMPORAL_SCOPE")
    serialize = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    assert serialize(env.observe("respond", intervention="I1")) == serialize(env.observe("respond", intervention="I2"))
    assert serialize(env.observe("write", intervention="I0")) == serialize(env.observe("write", intervention="I1"))
    assert "Solo para hoy" not in serialize(env.observe("write", intervention="I1"))
    assert "Solo para hoy" in serialize(env.observe("write", intervention="I2"))
    assert "assistant" not in serialize(env.observe("write"))
    env.dispose()


def test_T16_irrelevant_question(tmp_path):
    env = EpisodeEnvironment(episode(), tmp_path / "a.db")
    assert env.ask("GENERIC")["text"] == "No puedo precisar ese dato ahora."
    with pytest.raises(ValueError, match="QUESTION_BUDGET"):
        env.ask("TEMPORAL_SCOPE")
    env.dispose()


def test_T17_executor_does_not_repair_semantics_and_is_atomic(tmp_path):
    store = Store(tmp_path / "a.db")
    store.initialize([record()])
    wrong = operation(value="arroz", kind="ongoing")
    result = Executor(store).execute([wrong])
    assert result["valid"]
    assert store.list_records()[-1]["value"] == "arroz"
    before = store.hash()
    invalid = operation()
    invalid.update(op="SUPERSEDE", target_ids=["nonexistent"])
    assert not Executor(store).execute([wrong, invalid])["valid"]
    assert store.hash() == before
    assert not Executor(store).execute([{"op": "NOOP"}, wrong])["valid"]
    store.close()


def test_T23_rollout_isolation(tmp_path):
    a = EpisodeEnvironment(episode(), tmp_path / "a.db")
    b = EpisodeEnvironment(episode(), tmp_path / "b.db")
    a.apply_memory_operations([operation()])
    assert len(a.store.list_records()) == 2
    assert len(b.store.list_records()) == 1
    a.dispose()
    b.dispose()


def test_unknown_scope_blocks_unjustified_commitment():
    records = [record(), record("pasta", mid="m2", kind="unknown")]
    assert read(records, episode()["future_query"])["mode"] == "CONDITIONALIZE"


def test_T08_reuse_survives_fresh_python_process(tmp_path):
    import subprocess
    import sys
    env = EpisodeEnvironment(episode(), tmp_path / "memory.db")
    op = operation(start="2026-01-12")
    op.update(op="SUPERSEDE", target_ids=["m1"])
    assert env.apply_memory_operations([op])["valid"]
    env.close_session()
    code = """
import json, sys
from research.src.memory.store import Store
from research.src.readers.symbolic import read
store = Store(sys.argv[1])
print(json.dumps(read(store.list_records(), json.loads(sys.argv[2]))))
store.close()
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "memory.db"),
                             json.dumps(episode()["future_query"])],
                            capture_output=True, text=True, check=True, timeout=10)
    assert json.loads(result.stdout)["value"] == "pasta"
