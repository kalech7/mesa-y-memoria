from copy import deepcopy

from research.tesis_memory_faults_v3 import fixture, replay


def _probes(trace):
    return {row["kind"]: row for row in trace["query_audit"]}


def test_wrong_write_keeps_quote_and_context_but_extends_scope():
    story = fixture(101, "intervalo_contextual", 1)
    baseline = replay(story, writer_fault=False, reader_fault=False, verifier="veto")
    blocked = replay(story, writer_fault=True, reader_fault=False, verifier="veto")
    persisted = replay(story, writer_fault=True, reader_fault=False, verifier="off")
    assert blocked["actions"] == persisted["actions"] == baseline["actions"]
    valid = baseline["write_audit"][0]
    vetoed = blocked["write_audit"][0]
    applied = persisted["write_audit"][0]
    assert valid["evidence_ok"] and valid["applied"]
    assert vetoed["injected"] and not vetoed["evidence_ok"] and vetoed["vetoed"] and not vetoed["applied"]
    assert applied["proposal"] == vetoed["proposal"] and applied["applied"] and not applied["vetoed"]
    revalidating_end = deepcopy(applied["proposal"])
    revalidating_end["record"]["scope"]["valid_until_exclusive"] = (
        valid["proposal"]["record"]["scope"]["valid_until_exclusive"])
    assert revalidating_end == valid["proposal"]
    assert _probes(blocked)["ultimo_dia_vigente"]["candidate"]["correct"] == 0
    assert _probes(blocked)["dia_de_vencimiento"]["candidate"]["supported"] == 1
    after = _probes(persisted)["dia_de_vencimiento"]
    assert after["candidate"]["unsupported_write_citation_ids"] == [applied["persisted_id"]]
    assert not after["candidate"]["expired_citation_ids"]
    assert not after["reader_veto"]
    assert after["guarded"]["unsupported_commitment"] == 1
    assert _probes(persisted)["fuera_de_contexto"]["candidate"]["supported"] == 1


def test_read_guard_abstains_even_if_an_expired_citation_returns_right_value():
    story = fixture(202, "solo_recuperacion", 0)
    trace = replay(story, writer_fault=False, reader_fault=True, verifier="veto")
    obsolete = _probes(trace)["vencido_preexistente"]
    assert not trace["write_audit"]
    assert obsolete["candidate"]["correct"] == 1 and obsolete["candidate"]["supported"] == 0
    assert obsolete["candidate"]["expired_citation_ids"]
    assert obsolete["reader_veto"]
    assert obsolete["guarded"]["response"] == {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": []}
    assert obsolete["guarded"]["unsupported_commitment"] == 0
    assert _probes(trace)["preferencia_estable"]["guarded"]["supported"] == 1


def test_fault_replay_retains_exact_initial_clarification():
    story = fixture(303, "intervalo_aclarado", 1)
    normal = replay(story, writer_fault=False, reader_fault=False, verifier="veto")
    both = replay(story, writer_fault=True, reader_fault=True, verifier="off")
    assert normal["actions"] == both["actions"] == ["VALID_UNTIL", "PROCEED", "PROCEED", "PROCEED"]
    assert normal["visits"][0]["events"] == both["visits"][0]["events"]
    assert normal["visits"][0]["events"][0]["delivery"] == "resolved"
    assert normal["visits"][0]["expected"] == both["visits"][0]["expected"]
    assert _probes(both)["vencido_preexistente"]["candidate"]["expired_citation_ids"]
    assert _probes(both)["dia_de_vencimiento"]["candidate"]["unsupported_write_citation_ids"]
