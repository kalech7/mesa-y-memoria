from copy import deepcopy
from contextlib import closing
from datetime import date, timedelta
import json

import pytest

from research.competence_evaluation import _run_trajectory
from research.src.memory.executor import Executor
from research.src.memory.store import Store
from research.src.policy.llm_controller import InvalidOutput
from research.src.policy.rule_controller import RuleController
from research.src.readers.symbolic import read
from research.tesis_bank import build_splits
from research.tesis_memory_guard import GuardedWriter, plan_write
from research.tesis_memory_guard_cli import main
from research.tests.test_scoring import case, observation


def test_duplicate_or_invalid_members_do_not_discard_a_faithful_operation():
    visible = observation(case("F1"))
    good = RuleController().write(visible)["operations"][0]
    for rejected, reason in ((deepcopy(good), "duplicate_operation"), ({"op": "DELETE"}, "invalid_structure:INVALID_OPERATION")):
        proposed = {"operations": [good, rejected]}
        before = deepcopy(proposed)
        result = plan_write(visible, proposed, mode="veto")
        assert result["operations"] == [good]
        assert result["accepted_indices"] == [0]
        assert result["status"] == "partially_kept"
        assert result["rejected"] == [{"index": 1, "reason": reason}]
        assert proposed == before


def test_only_a_faithful_session_scope_survives_a_temporary_preference():
    visible = observation(case("F4", world="session"), reply=True)
    good = RuleController().write(visible)["operations"][0]
    bad = deepcopy(good)
    bad["record"]["scope"].update(kind="ongoing", session_id=None)
    assert plan_write(visible, {"operations": [bad]}, mode="veto")["operations"] == [{"op": "NOOP"}]
    repaired = plan_write(visible, {"operations": [bad]}, mode="repair")
    assert repaired["operations"] == repaired["recovered_operations"] == [good]
    assert repaired["rejected"] == [{"index": 0, "reason": "unsupported_evidence"}]
    assert repaired["operations"][0]["record"]["scope"]["kind"] == "session"


def test_ongoing_repair_preserves_a_distinct_contextual_exception():
    visible = observation(case("F1"))
    exception = deepcopy(visible["memory"][0])
    exception.update(memory_id="office", value="sopa especial")
    exception["scope"].update(kind="contextual", context="oficina")
    visible["memory"].append(exception)
    good = RuleController().write(visible)["operations"][0]
    bad = deepcopy(good)
    bad["target_ids"].append("office")
    result = plan_write(visible, {"operations": [bad]})
    assert result["operations"] == [good]
    assert "office" not in result["operations"][0]["target_ids"]


def test_repair_omission_when_explicit_but_never_invent_unresolved_scope():
    unresolved = observation(case("F4", world="ongoing"))
    noop = {"operations": [{"op": "NOOP"}]}
    assert plan_write(unresolved, noop)["status"] == "noop"
    marker = RuleController().write(unresolved)
    assert plan_write(unresolved, marker)["operations"] == marker["operations"]
    visible = observation(case("F4", world="ongoing"), reply=True)
    repaired = plan_write(visible, noop)
    assert repaired["proposal_kind"] == "noop"
    assert repaired["status"] == "repaired"
    assert repaired["operations"][0]["record"]["scope"]["kind"] == "ongoing"
    with pytest.raises(ValueError, match="public WRITE"):
        plan_write({**visible, "future_query": visible["query"]}, noop)


def test_invalid_model_write_is_kept_in_diagnostics_and_repair_persists(tmp_path):
    class BrokenWriter(RuleController):
        def write(self, observation):
            raise InvalidOutput("Model output was truncated")

    episode = next(row for row in build_splits({"train": 5, "validation": 5, "evaluation": 5})["train"]
                   if row["scenario_family"] == "F4" and row["truth"]["scope_kind"] == "ongoing")
    baseline = _run_trajectory(BrokenWriter(), episode, tmp_path / "baseline.sqlite")
    guard = GuardedWriter(BrokenWriter())
    repaired = _run_trajectory(guard, episode, tmp_path / "repaired.sqlite")
    assert baseline["phases"]["select"] == repaired["phases"]["select"]
    assert baseline["phases"]["respond"] == repaired["phases"]["respond"]
    assert baseline["metrics"]["C2"] == repaired["metrics"]["C2"] == 1
    assert baseline["metrics"]["FJS"] == 0 and repaired["metrics"]["FJS"] == 1
    assert repaired["metrics"]["unsupported_write_count"] == 0
    assert guard.last_write["model_error"] == "Model output was truncated"
    assert guard.last_write["proposal_kind"] == "malformed"


def test_cli_preview_is_json_and_refuses_overwrite(tmp_path, capsys):
    visible = observation(case("F4", world="ongoing"), reply=True)
    payload = tmp_path / "write.json"
    payload.write_text(json.dumps({"observation": visible, "proposed": {"operations": [{"op": "NOOP"}]}}))
    main(["--input", str(payload), "--mode", "veto"])
    assert json.loads(capsys.readouterr().out)["status"] == "noop"
    output = tmp_path / "plan.json"
    main(["--input", str(payload), "--output", str(output)])
    assert json.loads(output.read_text())["status"] == "repaired"
    with pytest.raises(SystemExit):
        main(["--input", str(payload), "--output", str(output)])
    assert json.loads(output.read_text())["status"] == "repaired"


def test_repair_preserves_the_other_attribute_and_restores_default_at_expiry(tmp_path):
    visible = observation(case("F1"))
    query = visible["query"]
    attribute = query["attribute"]
    prior = next(record for record in visible["memory"] if record["attribute"] == attribute)
    other = deepcopy(prior)
    other.update(memory_id="unaffected", attribute="drink_preference", value="agua")
    assert attribute != other["attribute"]
    visible["memory"].append(other)
    start = date.fromisoformat(query["date"])
    last_day = (start + timedelta(days=2)).isoformat()
    expiry = (start + timedelta(days=3)).isoformat()
    visible["user_messages"] = [{"turn_id": "manual", "text": f"Prefiero estofado. Hasta el {last_day} inclusive."}]
    result = plan_write(visible, {"operations": [{"op": "NOOP"}]})
    assert result["status"] == "repaired"
    database = tmp_path / "after.sqlite"
    with closing(Store(database)) as store:
        store.initialize(visible["memory"])
        assert Executor(store).execute(result["operations"], recorded_at=query["date"])["valid"]
    with closing(Store(database)) as store:
        persisted = store.list_records()
    future = {**query, "session_id": "S3", "context": "casa"}
    assert read(persisted, {**future, "date": last_day})["value"] == "estofado"
    assert read(persisted, {**future, "date": expiry})["value"] == prior["value"]
    assert read(persisted, {**future, "date": expiry, "attribute": "drink_preference"})["value"] == "agua"
