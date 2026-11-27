"""Automatic checks are exploratory evidence, never fabricated human review."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from research.src.artifacts import ROOT
from research.src.data.automatic_audit import AUDIT_SOURCE_FILES, audit_development
from research.src.data.build_bank import build_bank


@pytest.fixture(scope="module")
def audit_fixture(tmp_path_factory):
    from unittest.mock import patch
    root = tmp_path_factory.mktemp("automatic_audit")
    for relative in AUDIT_SOURCE_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    report = root / "reports/tests.xml"
    report.parent.mkdir()
    report.write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0" /></testsuites>')
    with patch("research.src.protocol.require_gate"):
        episodes = build_bank("train", 100)
    return root, episodes


def test_automatic_audit_executes_all_checks_without_human_claim(audit_fixture):
    root, episodes = audit_fixture
    result = audit_development(episodes, root=root, tests_report="reports/tests.xml")
    assert result["status"] == "passed", result["unresolved_inconsistencies"][:5]
    assert result["roots_checked"] == 100 and result["f4_pairs"] == 20
    assert all(result["families"][f"F{i}"] == 20 for i in range(5))
    assert result["T07"] == {"human_status": "not_executed_human", "disposition": "substituted_for_exploration"}
    assert result["human_audit_status"] == "pending_human_audit"
    assert result["not_human"] and not result["independent_human_review"]
    assert result["teacher_phases"]["select"] == len(episodes)
    assert result["teacher_phases"]["respond"] == 3 * len(episodes)


def test_automatic_audit_rejects_test_gold_before_inspection(audit_fixture):
    root, episodes = audit_fixture
    test_episode = {"split": "test_id"}  # no private labels need even exist
    with pytest.raises(ValueError, match="DEVELOPMENT_ONLY"):
        audit_development([test_episode], root=root)


def test_automatic_audit_detects_false_generator_truth(audit_fixture):
    root, episodes = audit_fixture
    corrupted = deepcopy(episodes)
    corrupted[0]["truth"]["future_value"] = "un alimento no expresado"
    report = audit_development(corrupted, root=root, tests_report="reports/tests.xml")
    assert report["status"] == "failed"
    assert any(i["code"] == "visible_evidence_future_differs_from_truth" for i in report["unresolved_inconsistencies"])


def test_automatic_audit_cannot_pass_smoke_or_missing_tests(audit_fixture):
    root, episodes = audit_fixture
    report = audit_development(episodes[:3], root=root)
    assert report["status"] == "failed"
    codes = {item["code"] for item in report["unresolved_inconsistencies"]}
    assert "minimum_100_roots_20_per_family_not_met" in codes
    assert "automated_test_evidence_missing_or_failed" in codes


def test_automatic_audit_detects_hidden_query_field(audit_fixture):
    root, episodes = audit_fixture
    corrupted = deepcopy(episodes[:1])
    corrupted[0]["current_query"]["gold_action"] = "ASK"
    report = audit_development(corrupted, root=root, tests_report="reports/tests.xml")
    assert any(i["code"] == "private_or_extra_query_fields" for i in report["unresolved_inconsistencies"])
