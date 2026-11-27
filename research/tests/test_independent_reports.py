from copy import deepcopy
import json

from research.src.data.independent_bank import build_independent_bank, build_ai_external_proxy, bank_manifest
from research.src.data.build_bank import build_bank
from research.src.data.audit_splits import actor_prefix, canonical_signature
from research.src.memory.executor import validate_record
from research.src.readers.symbolic import read
from research.src.analysis.executed_reports import generate_executed_reports


def test_independent_annotation_integrity_without_model_scores():
    for factory, count in ((build_independent_bank, 150), (build_ai_external_proxy, 50)):
        episodes = factory()
        manifest = bank_manifest(episodes)
        assert manifest["roots"] == count
        assert set(manifest["families"].values()) == {count // 5}
        assert manifest["human_authored_external_status"] == "not_executed"
        for episode in episodes:
            assert 4 <= len(episode["initial_memory"]) <= 8
            for record in episode["initial_memory"]:
                validate_record(record, initial=True)
            assert episode["authoring"]["human_authored"] is False
            oracle_memory = deepcopy(episode["initial_memory"])
            truth = episode["truth"]
            if truth["scope"] is not None:
                oracle_memory.append({"memory_id": "annotation_only", "attribute": "dinner_preference",
                    "value": truth["new_value"], "scope": truth["scope"], "status": "confirmed",
                    "supersedes": [r["memory_id"] for r in oracle_memory if r["scope"]["kind"] == "ongoing"]
                                  if truth["scope"]["kind"] == "ongoing" else []})
            now = {**episode["current_query"], "context": truth["current_context"]}
            assert read(oracle_memory, now)["value"] == truth["current_value"]
            assert read(oracle_memory, episode["future_query"])["value"] == truth["future_value"]
        pairs = {}
        for episode in episodes:
            if episode["scenario_family"] == "F4":
                pairs.setdefault(episode["root_id"], []).append(episode)
        assert all(len({actor_prefix(e) for e in pair}) == 1 for pair in pairs.values())


def test_ood_contains_compositions_not_just_food_renaming():
    development = {canonical_signature(e) for e in build_bank()}
    challenge = {canonical_signature(e) for e in build_independent_bank()}
    assert not development & challenge


def test_reports_show_all_missing_contrasts_without_fabricated_results(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("")
    output = tmp_path / "reports"
    report = generate_executed_reports(source, output, resamples=20)
    assert report["status"] == "not_executed"
    assert len(report["primary_contrasts"]) == 5
    assert all(c["status"] == "not_executed" for c in report["primary_contrasts"])
    assert "not_executed" in (output / "INFORME_SFT_RL.md").read_text()


def test_reports_macro_cluster_and_negative_result_from_real_rows(tmp_path):
    source = tmp_path / "input.jsonl"
    rows = []
    for method in ("B1", "B2-L", "B3", "B2-I"):
        for seed in (101, 202, 303):
            for root in range(10):
                rows.append({"experiment": "E2", "split": "test_id", "method": method, "seed": seed,
                             "root_id": str(root), "latent_world_id": "w", "realization_id": "v",
                             "scenario_family": "F" + str(root % 5), "metrics": {"FJS": int(method != "B2-L"),
                             "C2": 1, "unsupported_current": 0, "unsupported_write_episode": 0}})
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = generate_executed_reports(source, tmp_path / "reports", resamples=20)
    contrasts = [c for c in report["primary_contrasts"] if c["status"] == "executed"]
    assert len(contrasts) == 3
    assert all(c["statistics"]["estimate"] == -1 for c in contrasts)
    assert all(c["statistics"]["independent_roots"] == 10 for c in contrasts)
    assert all(c["conclusion"] == "negative_difference" for c in contrasts)


def test_constant_paired_outcomes_do_not_claim_zero_uncertainty():
    from research.src.analysis.executed_reports import _constant_difference_bound
    result=_constant_difference_bound({'estimate':0.,'interval':[0.,0.]},[0.]*300)
    assert result['interval'][0]<0<result['interval'][1]
    assert result['boundary_correction']['roots']==300


def test_current_error_guard_includes_invalid_and_deferred_responses():
    from research.src.analysis.executed_reports import _flatten
    row={'experiment':'E2','technical_status':'executed','metrics':{'C2':0,'FJS':1,'unsupported_current':0,'invalid_outputs':1}}
    flat=_flatten(row)
    assert flat['current_error']==1
    assert flat['unsupported_current']==0
