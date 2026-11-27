"""Executable V4 audit of the existing environment; no substitute simulator.

Expected values below are literal, hand-authored fixtures, independent of the
generator and reader. Passing them certifies these contracts, not model skill.
All hidden information is confined to audit objects and trace artifacts.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from .artifacts import canonical, file_hash, write_json
from .env.episode import EpisodeEnvironment
from .env.user_simulator import NON_RESPONSE, QUESTIONS, UserSimulator, nonresponse_uniform
from .evaluation.runner import run_episode, select_valid
from .memory.executor import Executor
from .memory.store import Store
from .policy.llm_controller import InvalidOutput, LLMController, validate_action
from .policy.rule_controller import RuleController, authorized_operations
from .readers.symbolic import read
from .scoring.evidence import Evaluator, audit_operation, score_response


def _scope(kind="ongoing", *, start="2026-01-01", end=None, context=None, session=None):
    return {"kind": kind, "valid_from": start, "valid_until_exclusive": end,
            "context": context, "session_id": session}


def _record(mid="m1", value="sopa", *, scope=None, attribute="dinner_preference", supersedes=()):
    return {"memory_id": mid, "attribute": attribute, "value": value, "scope": scope or _scope(),
            "status": "unresolved" if scope and scope["kind"] == "unknown" else "confirmed",
            "source_turn_ids": ["s1_" + mid], "evidence_quotes": [{"turn_id": "s1_" + mid, "text": "Preferencia documentada: " + value}],
            "supersedes": list(supersedes), "recorded_at": "2026-01-01"}


def _query(*, date="2026-01-12", session="S2", context=None, attribute="dinner_preference"):
    return {"attribute": attribute, "date": date, "session_id": session, "context": context}


def _episode():
    # Literal ambiguous case; changing the world must not change its input.
    return {"root_id": "fixture_root", "latent_world_id": "world_ongoing", "realization_id": "v0",
            "scenario_family": "F4", "split": "dev_smoke", "initial_memory": [_record()],
            "user_messages": [{"turn_id": "u2", "text": "Ahora prefiero pasta. ¿Qué ceno hoy?"}],
            "current_query": _query(), "future_query": _query(date="2026-01-19", session="S3"),
            "replies": {"TEMPORAL_SCOPE": {"turn_id": "u3", "text": "Como preferencia habitual desde ahora."},
                        "GENERIC": {"turn_id": "u3", "text": "No tengo más detalles sobre eso."}},
            "truth": {"current_value": "pasta", "future_value": "pasta", "scope_kind": "ongoing",
                      "decisive_slot": "TEMPORAL_SCOPE", "scope": _scope(start="2026-01-12")},
            "opportunity": {"eligible": True, "slot": "TEMPORAL_SCOPE"}}


def infrastructure_fixtures():
    """Thirty independently expected semantics, each with an incorrect mutation."""
    base = _record()
    interval = _record("m2", "pasta", scope=_scope("interval", start="2026-01-10", end="2026-01-13"))
    session = _record("m3", "arroz", scope=_scope("session", start="2026-01-12", session="S2"))
    home = _record("m4", "pasta", scope=_scope("contextual", context="casa"))
    office = _record("m5", "arroz", scope=_scope("contextual", context="oficina"))
    unknown = _record("m6", "pasta", scope=_scope("unknown", start="2026-01-12"))
    replacement = _record("m7", "pasta", scope=_scope(start="2026-01-12"), supersedes=("m1",))
    future = _record("m8", "arroz", scope=_scope(start="2026-02-01"), supersedes=("m1",))
    drink = _record("m9", "agua", attribute="drink_preference")
    # name, stored records, query, expected mode, expected value/alternatives.
    cases = [
        ("stable", [base], _query(), "ANSWER", "sopa"),
        ("stable_later", [base], _query(date="2027-01-01", session="S3"), "ANSWER", "sopa"),
        ("before_valid_from", [base], _query(date="2025-12-31"), "DEFER", None),
        ("empty", [], _query(), "DEFER", None),
        ("irrelevant_attribute", [drink], _query(), "DEFER", None),
        ("unaffected_attribute", [base, drink], _query(attribute="drink_preference"), "ANSWER", "agua"),
        ("interval_inside", [base, interval], _query(), "ANSWER", "pasta"),
        ("interval_inclusive_start", [base, interval], _query(date="2026-01-10"), "ANSWER", "pasta"),
        ("interval_before", [base, interval], _query(date="2026-01-09"), "ANSWER", "sopa"),
        ("interval_exclusive_end", [base, interval], _query(date="2026-01-13"), "ANSWER", "sopa"),
        ("interval_archived", [base, interval], _query(date="2026-02-01"), "ANSWER", "sopa"),
        ("session_current", [base, session], _query(), "ANSWER", "arroz"),
        ("session_other_same_day", [base, session], _query(session="S3"), "ANSWER", "sopa"),
        ("session_other_later", [base, session], _query(date="2026-01-19", session="S3"), "ANSWER", "sopa"),
        ("context_home", [base, home], _query(context="casa"), "ANSWER", "pasta"),
        ("context_elsewhere", [base, home], _query(context="oficina"), "ANSWER", "sopa"),
        ("context_pair_home", [home, office], _query(context="casa"), "ANSWER", "pasta"),
        ("context_pair_office", [home, office], _query(context="oficina"), "ANSWER", "arroz"),
        ("context_missing", [home, office], _query(), "CONDITIONALIZE", ["arroz", "pasta"]),
        ("duration_unknown", [base, unknown], _query(), "CONDITIONALIZE", ["pasta", "sopa"]),
        ("duration_unknown_later", [base, unknown], _query(date="2026-01-19", session="S3"), "CONDITIONALIZE", ["pasta", "sopa"]),
        ("replacement", [base, replacement], _query(), "ANSWER", "pasta"),
        ("replacement_historical", [base, replacement], _query(date="2026-01-11"), "ANSWER", "sopa"),
        ("future_update_pending", [base, future], _query(), "ANSWER", "sopa"),
        ("future_update_effective", [base, future], _query(date="2026-02-01"), "ANSWER", "arroz"),
        ("session_over_interval", [base, interval, session], _query(), "ANSWER", "arroz"),
        ("interval_over_context", [base, home, interval], _query(context="casa"), "ANSWER", "pasta"),
        ("unrelated_write_preserves", [base, replacement, drink], _query(attribute="drink_preference"), "ANSWER", "agua"),
        ("base_after_temporary_supersede", [base, {**interval, "supersedes": ["m1"]}], _query(date="2026-01-13"), "ANSWER", "sopa"),
        ("context_partial_dictionary", [base, _record("m10", "pasta", scope=_scope("contextual", context={"place": "casa"}))],
         _query(context={"place": "casa", "company": "familia"}), "ANSWER", "pasta"),
    ]
    return [{"id": f"FIX-{index:02d}", "name": name, "memory": deepcopy(memory), "query": query,
             "expected_mode": mode, "expected_value": value}
            for index, (name, memory, query, mode, value) in enumerate(cases, 1)]


def _run_fixture(fixture):
    memory, query = fixture["memory"], fixture["query"]
    response = read(memory, query)
    assert response["mode"] == fixture["expected_mode"], (fixture["name"], response)
    if response["mode"] == "ANSWER":
        assert response["value"] == fixture["expected_value"], response
    elif response["mode"] == "CONDITIONALIZE":
        assert sorted(b["value"] for b in response["branches"]) == fixture["expected_value"], response
    observation = {"memory": memory, "query": query, "user_messages": []}
    # The expected factual label is independent of the score/reader.
    truth = fixture["expected_value"] if fixture["expected_mode"] == "ANSWER" else "not_identifiable"
    score = score_response(response, observation, truth)
    assert score["success"] == int(fixture["expected_mode"] == "ANSWER"), score
    assert score["utility"] == (1 if fixture["expected_mode"] == "ANSWER" else 0), score
    mutation = {"mode": "ANSWER", "value": "valor_inventado", "branches": [],
                "evidence_ids": [r["memory_id"] for r in memory]}
    bad = score_response(mutation, observation, truth)
    assert bad["success"] == 0 and bad["utility"] == -1, bad
    return {"fixture": fixture, "response": response, "score": score, "mutation": mutation, "mutation_score": bad}


def environment_contracts():
    return {
        "source": "existing research.src modules; V4 control correction is versioned",
        "families": {"F0": "stable preference with archived expired exception", "F1": "explicit ongoing change",
                     "F2": "explicit bounded interval", "F3": "context-dependent preference",
                     "F4": "ambiguous duration and paired explicit controls; not every F4 realization is eligible"},
        "observations": {"O1": "latest recorded memory record of queried attribute plus received current messages",
                         "O2": "all persisted memory plus same received current messages"},
        "interventions": {"I0": "no question; same prior input to WRITE as I1",
                          "I1": "fixed question; reply visible to RESPOND, withheld from WRITE",
                          "I2": "same question/reply as I1; reply visible to RESPOND and WRITE"},
        "session3": "fresh SQLite connection; persisted memory and explicit future query only; fixed reader",
        "scope_priority": ["session", "contextual_with_expiry", "interval", "contextual", "ongoing"],
        "unknown_scope": "kind=unknown means unresolved; ongoing with null end means no explicit expiry",
        "retrieval": "not_applicable: full memory serialized, no learned selective retriever",
        "slots": QUESTIONS, "question_budget": 1,
        "nonresponse": "sha256('7331:<root_id>:<slot>') first 64 bits / 2**64 < p; shared across methods",
        "controls": {"C-NEVER": "zero executed questions", "C-ALWAYS": "one executed question; observable rule slot with GENERIC fallback",
                     "ASK-REFERENCE": "observable missing-slot rule; diagnostic only", "KEEP-INITIAL": "NOOP writer",
                     "REFERENCE-WRITER": "authorized_operations(observed_evidence), never latent truth"},
        "scoring": "factually correct AND justified by visible evidence/provenance; correct hidden guess fails justified success",
        "grammar_limit": "finite public Spanish grammar; unrecognized natural paraphrases remain unverified",
    }


def run_environment_checks():
    """Run deterministically on CPU, without weights, network, or historical writes."""
    checks = []

    def check(identifier, action):
        try:
            evidence = action()
            checks.append({"id": identifier, "status": "passed", "evidence": evidence, "details": "deterministic executed fixture"})
        except Exception as exc:
            checks.append({"id": identifier, "status": "failed", "evidence": {"exception": repr(exc)}, "details": "fixture failed"})

    for fixture in infrastructure_fixtures():
        check(fixture["id"], lambda fixture=fixture: _run_fixture(fixture))

    with tempfile.TemporaryDirectory(prefix="closure_v4_contracts_") as temporary:
        work = Path(temporary)
        episode = _episode()
        env = EpisodeEnvironment(episode, work / "original.sqlite")
        observation = env.observe("select")
        prompt = LLMController(object()).prompt("select", observation)
        alternate = deepcopy(episode)
        alternate["truth"]["future_value"] = "sopa"
        alternate["truth"]["scope_kind"] = "session"
        alternate["latent_world_id"] = "world_temporary"
        alternate["replies"]["TEMPORAL_SCOPE"]["text"] = "Solo en esta conversación."
        other = EpisodeEnvironment(alternate, work / "alternate.sqlite")
        other_obs = other.observe("select")

        def inf1():
            candidate = LLMController(object()).prompt("select", other_obs)
            assert prompt.encode() == candidate.encode()
            return {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "comparison": "exact UTF-8 bytes within same realization"}

        def inf2():
            assert canonical(observation) == canonical(other_obs)
            assert env.store.hash() == other.store.hash()
            assert {s for s in QUESTIONS if select_valid({"action": "ASK", "slot": s})} == set(QUESTIONS)
            return {"same_visible_state": True, "same_memory": True, "legal_slots": sorted(QUESTIONS)}

        def inf3():
            changed = deepcopy(episode)
            changed.update(root_id="arbitrary", latent_world_id="private_world", scenario_family="F0")
            isolated = EpisodeEnvironment(changed, work / "metadata.sqlite")
            try:
                assert RuleController().select(isolated.observe("select")) == RuleController().select(observation)
                assert LLMController(object()).prompt("select", isolated.observe("select")) == prompt
            finally:
                isolated.dispose()
            return {"metadata_invariant": True, "policy": "observable_rule_control"}

        def inf4():
            assert all(select_valid({"action": "ASK", "slot": slot}) for slot in QUESTIONS)
            assert select_valid({"action": "PROCEED", "slot": None})
            return {"legal_action_count": 5, "gold_mask_used": False}

        def inf5():
            env.private_trace["secret"] = "HIDDEN_AUDIT_SENTINEL"
            assert "HIDDEN_AUDIT_SENTINEL" not in LLMController(object()).prompt("select", env.observe("select"))
            try:
                LLMController(object()).prompt("select", {**observation, "truth": episode["truth"]})
            except ValueError:
                return {"private_field_rejected": True}
            raise AssertionError("private fields accepted")

        for identifier, action in (("INF-01", inf1), ("INF-02", inf2), ("INF-03", inf3), ("INF-04", inf4), ("INF-05", inf5)):
            check(identifier, action)
        env.dispose()
        other.dispose()

        def simulator_checks():
            replies = {"TEMPORAL_SCOPE": {"turn_id": "u3", "text": "Solo para hoy.", "gold": "forbidden"},
                       "GENERIC": {"turn_id": "u3", "text": "No tengo más detalles sobre eso."}}
            sim = UserSimulator(replies, root_id="fixture")
            assert sim.reply("TEMPORAL_SCOPE") == sim.reply("TEMPORAL_SCOPE") == {"turn_id": "u3", "text": "Solo para hoy."}
            assert sim.reply("GENERIC")["text"] == replies["GENERIC"]["text"]
            assert sim.last_response_status == "unavailable"
            try:
                sim.reply("ILLEGAL_SLOT")
            except ValueError:
                pass
            else:
                raise AssertionError("invalid slot delivered")
            silent = UserSimulator(replies, root_id="fixture", nonresponse_rate=1)
            assert silent.reply("TEMPORAL_SCOPE") == NON_RESPONSE
            assert silent.last_response_status == "unavailable"
            draws = []
            for index in range(100):
                root_id = f"assignment_{index}"
                a = UserSimulator(replies, root_id=root_id, nonresponse_rate=.2)
                b = UserSimulator(replies, root_id=root_id, nonresponse_rate=.4)
                at20 = a.reply("TEMPORAL_SCOPE") == NON_RESPONSE
                at40 = b.reply("TEMPORAL_SCOPE") == NON_RESPONSE
                draw = nonresponse_uniform(root_id, "TEMPORAL_SCOPE")
                assert at20 == (draw < .2) and at40 == (draw < .4)
                assert not at20 or at40
                draws.append({"root_id": root_id, "uniform": draw, "p20": at20, "p40": at40})
            return {"deterministic": True, "invalid_rejected": True, "shared_nested_assignment": draws}

        check("SIM-01", simulator_checks)

        def meaning_checks():
            neutral = deepcopy(observation)
            neutral["user_messages"].append(deepcopy(NON_RESPONSE))
            unresolved = authorized_operations(neutral)[0]
            assert unresolved["op"] == "MARK_UNRESOLVED"
            assert unresolved["record"]["scope"]["kind"] == "unknown"
            assert unresolved["record"]["source_turn_ids"] == ["u2"]
            short = deepcopy(observation)
            short["user_messages"].append({"turn_id": "u3", "text": "Solo en esta conversación."})
            lasting = deepcopy(observation)
            lasting["user_messages"].append({"turn_id": "u3", "text": "Como preferencia habitual desde ahora."})
            a, b = authorized_operations(short)[0], authorized_operations(lasting)[0]
            assert a["record"]["scope"]["kind"] == "session"
            assert b["record"]["scope"]["kind"] == "ongoing"
            redundant = deepcopy(lasting)
            redundant["user_messages"].append({"turn_id": "u4", "text": "Como preferencia habitual desde ahora."})
            repeated = authorized_operations(redundant)[0]["record"]
            assert all(repeated[key] == b["record"][key] for key in ("value", "scope", "attribute"))
            # Wrong scope/unsupported provenance are semantic mutations of a
            # literal valid operation, independent of authorized_operations.
            text = "Solo en esta conversación prefiero pasta. ¿Qué ceno?"
            visible = {"memory": [_record()], "query": _query(), "user_messages": [{"turn_id": "u2", "text": text}]}
            payload = {"attribute": "dinner_preference", "value": "pasta", "status": "confirmed",
                       "scope": _scope("session", start="2026-01-12", session="S2"),
                       "source_turn_ids": ["u2"], "evidence_quotes": [{"turn_id": "u2", "text": text}]}
            correct = {"op": "ADD", "record": payload}
            assert audit_operation(correct, visible)
            broad = deepcopy(correct)
            broad["record"]["scope"] = _scope(start="2026-01-12")
            assert not audit_operation(broad, visible)
            invented = deepcopy(correct)
            invented["record"]["scope"]["valid_until_exclusive"] = "2026-01-13"
            assert not audit_operation(invented, visible)
            return {"no_response_adds_no_hidden_data": True, "unknown_preserved": True,
                    "only_today_is_session": True, "habitual_is_ongoing": True,
                    "redundant_reply_preserves_value_scope": True, "incorrect_scope_rejected": True}

        check("SIM-02-EVA-04", meaning_checks)

        operation = {"op": "ADD", "record": {k: deepcopy(v) for k, v in _record("unused", "pasta", scope=_scope("session", start="2026-01-12", session="S2")).items()
                                               if k not in {"memory_id", "recorded_at", "supersedes"}}}

        def memory_checks():
            stored = Store(work / "engine.sqlite")
            stored.initialize([_record()])
            before = stored.hash()
            assert Executor(stored).execute([{"op": "NOOP"}])["valid"]
            assert stored.hash() == before
            result = Executor(stored).execute([operation])
            assert result["valid"]
            expected = deepcopy(operation["record"])
            expected.update(memory_id="m2", recorded_at="2026-01-12", supersedes=[])
            assert stored.list_records() == [_record(), expected]
            stored.close()
            reopened = Store(work / "engine.sqlite")
            assert reopened.list_records() == [_record(), expected]
            assert read(reopened.list_records(), _query())["value"] == "pasta"
            assert read(reopened.list_records(), _query(session="S3"))["value"] == "sopa"
            assert Executor(reopened).execute([operation])["valid"]
            assert len(reopened.list_records()) == 3  # ADD is legitimately cumulative.
            snapshot = reopened.snapshot()
            broken = deepcopy(operation)
            broken.update(op="SUPERSEDE", target_ids=["missing"])
            assert not Executor(reopened).execute([operation, broken])["valid"]
            assert reopened.snapshot() == snapshot
            reopened.close()
            return {"save_reload": True, "noop": True, "independent_expected_operation": expected,
                    "add_is_cumulative": True, "atomic_invalid_batch": True}

        check("MEM-01-04", memory_checks)

        def temporal_checks():
            assert _run_fixture(infrastructure_fixtures()[9])["response"]["value"] == "sopa"
            assert _run_fixture(infrastructure_fixtures()[25])["response"]["value"] == "arroz"
            return {"expired_exception_restores_base": True, "priority": environment_contracts()["scope_priority"]}

        check("MEM-05-06", temporal_checks)

        def isolation_checks():
            original = deepcopy(episode)
            isolated = EpisodeEnvironment(original, work / "isolation.sqlite")
            isolated.ask("TEMPORAL_SCOPE")
            isolated.observe("write")
            isolated.apply_memory_operations(authorized_operations(isolated.observe("write")))
            isolated.close_session()
            # Mutating the caller's latent/dialogue objects cannot affect S3.
            original["user_messages"][0]["text"] = "MUTATED_TRANSIENT"
            original["truth"]["future_value"] = "wrong_hidden_value"
            observed = isolated.reopen_for_reuse()
            assert set(observed) == {"memory", "query"}
            assert isolated._messages == [] and isolated._reply is None
            assert isolated._simulator._replies == {} and isolated._initial_episode is None
            assert "MUTATED_TRANSIENT" not in canonical(observed)
            assert read(**observed)["value"] == "pasta"
            state = deepcopy(observed)
            assert read(**state) == read(**observed)
            isolated.dispose()
            return {"fresh_sqlite": True, "transient_cleared": True, "only_memory_and_query": True,
                    "reader_has_no_latent_input": True, "kv_cache": "not_applicable for stateless symbolic reader"}

        check("MEM-07-08-10", isolation_checks)

        def branch_order():
            from .evaluation.persistence_interventions import run_interventions
            forward = run_interventions(episode, RuleController, forced_slot="TEMPORAL_SCOPE")
            reverse = {arm: run_episode(episode, RuleController(), intervention=arm, forced_slot="TEMPORAL_SCOPE")
                       for arm in ("I2", "I1", "I0")}
            for arm in reverse:
                assert forward["arms"][arm]["metrics"] == reverse[arm]["metrics"]
                assert forward["arms"][arm]["trace"]["final_memory"] == reverse[arm]["trace"]["final_memory"]
            return {"invariants": forward["invariants"], "effects": forward["effects"], "order_invariant": True}

        check("MEM-09-INTERVENTIONS", branch_order)

        def controls():
            class BrokenSelect(RuleController):
                def select(self, obs):
                    raise ValueError("deliberately malformed model SELECT")
                def select_forced_ask(self, obs):
                    raise ValueError("must not depend on the model's formatting")
            always = run_episode(episode, BrokenSelect(), method="C-ALWAYS")
            never = run_episode(episode, BrokenSelect(), method="C-NEVER")
            assert always["metrics"]["questions"] == 1
            assert never["metrics"]["questions"] == 0
            assert always["metrics"]["invalid_outputs"] == never["metrics"]["invalid_outputs"] == 0
            assert always["metrics"]["ask_delivered"] == 1
            assert always["metrics"]["slot"] == "TEMPORAL_SCOPE"
            return {"always_delivered": 1, "never_delivered": 0, "slot": "TEMPORAL_SCOPE", "version": "closure_v4"}

        check("CONTROLS-01", controls)

        def evaluator_checks():
            obs = {"memory": [_record()], "query": _query(), "user_messages": []}
            answer = {"mode": "ANSWER", "value": "sopa", "branches": [], "evidence_ids": ["m1"]}
            assert score_response(answer, obs, "sopa")["utility"] == 1
            assert score_response({**answer, "evidence_ids": []}, obs, "sopa")["success"] == 0
            assert score_response({**answer, "value": "wrong"}, obs, "sopa")["success"] == 0
            unrelated = {**obs, "memory": [_record(), _record("drink", "agua", attribute="drink_preference")]}
            assert score_response({**answer, "evidence_ids": ["drink"]}, unrelated, "sopa")["success"] == 0
            reordered = json.loads(json.dumps(answer, sort_keys=True))
            assert score_response(reordered, obs, "sopa") == score_response(answer, obs, "sopa")
            for action in ({"action": "ASK", "slot": "ILLEGAL"}, {"action": "PROCEED", "slot": "GENERIC"}):
                try:
                    validate_action("select", action)
                except InvalidOutput:
                    pass
                else:
                    raise AssertionError("illegal action accepted")
            for text, truncated, expected_parse in (("{bad", False, "failed"), (json.dumps(answer), True, "not_attempted")):
                backend = SimpleNamespace(config=SimpleNamespace(max_new_tokens=10), encode_prompt=lambda p: [1],
                                          generate=lambda *a, text=text, truncated=truncated, **kw: SimpleNamespace(text=text, truncated=truncated))
                controller = LLMController(backend)
                try:
                    controller.respond(obs)
                except InvalidOutput:
                    record = controller.phase_records[-1]
                    assert record["parse_status"] == expected_parse
                    assert record["fallback_used"] is False
                else:
                    raise AssertionError("invalid/truncated output accepted")
            # Independent positive/negative scope and context assertions already
            # run through the real reader+evaluator in FIX-07..21.
            return {"correct_supported": True, "deleted_evidence_rejected": True, "irrelevant_citation_rejected": True,
                    "invalid_json_rejected": True, "illegal_slot_rejected": True, "truncation_not_repaired": True,
                    "key_order_invariant": True, "scope_context_evidence": "FIX-07 through FIX-21"}

        check("EVA-01-10-12", evaluator_checks)

        def persisted_conditional():
            row = run_episode(episode, RuleController(), method="C-NEVER")
            trace = row["trace"]
            assert trace["future_response"]["mode"] == "CONDITIONALIZE"
            score = Evaluator().score_future(trace)
            assert score["utility"] == 0 and score["success"] == 0
            forged = deepcopy(trace)
            forged["future_response"]["evidence_ids"].append("fabricated_memory_id")
            assert Evaluator().score_future(forged)["utility"] == -1
            forged = deepcopy(trace)
            forged["final_memory"][-1]["evidence_quotes"][0]["text"] = "invented_source_quote"
            assert Evaluator().score_future(forged)["utility"] == -1
            return {"utility": 0, "FJS": 0, "forged_citations_rejected": True,
                    "forged_quotes_rejected": True, "evaluator_version": score["evaluator_version"]}

        def immutable_initial_citations():
            row = run_episode(episode, RuleController())
            trace = deepcopy(row["trace"])
            original_id = trace["initial_memory"][0]["memory_id"]
            trace["final_memory"][0]["value"] = "pasta"
            trace["future_response"]["evidence_ids"] = [original_id]
            score = Evaluator().score_future(trace)
            assert score["factual"] == 1 and score["success"] == 0
            assert any(e.startswith("mutated_initial_record:") for e in score["provenance_errors"])
            return {"factual_correct_forgery_rejected": True, "initial_ids_cannot_launder_new_claims": True}

        def horizon_invariance():
            row = run_episode(episode, RuleController(), method="C-NEVER")
            trace = deepcopy(row["trace"])
            neutral = Evaluator().score_future(trace)["utility"]
            trace["future_response"]["evidence_ids"] = ["invented"]
            invalid = Evaluator().score_future(trace)["utility"]
            assert neutral == 0 and invalid == -1
            r2 = Evaluator().score_current(trace)["utility"]
            from .scoring.reward import rewards
            assert rewards(r2, neutral)["R_short"] == rewards(r2, invalid)["R_short"]
            return {"future_changes_only_long_horizon": True, "short_return_unchanged": True}

        check("EVA-S3-PERSISTED-CONDITIONAL", persisted_conditional)
        check("EVA-S3-IMMUTABLE-INITIAL", immutable_initial_citations)
        check("REW-S3-SHORT-INVARIANCE", horizon_invariance)

        def paraphrase():
            first = deepcopy(observation)
            second = deepcopy(observation)
            first["user_messages"][0]["text"] = "Desde hoy prefiero pasta de forma habitual. ¿Qué ceno?"
            second["user_messages"][0]["text"] = "Mi preferencia habitual desde hoy es pasta. ¿Qué ceno?"
            a, b = RuleController().respond(first), RuleController().respond(second)
            assert a == b and a["value"] == "pasta"
            return {"admitted_paraphrases": [first["user_messages"][0]["text"], second["user_messages"][0]["text"]],
                    "expected_value": "pasta", "general_natural_language_coverage": "not_identifiable_from_current_artifacts"}

        check("EVA-11", paraphrase)

        def e6_equivalence():
            from .evaluation import secondary
            row = run_episode(episode, RuleController())
            inputs = []
            def frozen_generate(backend, phase, observations, **options):
                assert phase == "reader"
                inputs.extend(deepcopy(observations))
                return [{"parsed_action": read(o["memory"], o["query"])} for o in observations]
            original = secondary.generate_batch
            try:
                secondary.generate_batch = frozen_generate
                e4 = list(secondary.reread_frozen([row], object()))[0]
                e6 = list(secondary.reuse_with_interaction([row], object(), max_questions_s3=0))[0]
            finally:
                secondary.generate_batch = original
            assert inputs[0] == inputs[1]
            assert e4["metrics"]["FJS"] == e6["metrics"]["FJS"]
            assert e4["trace"]["future_response"] == e6["trace"]["future_response"]
            return {"same_input": True, "same_parser_path": True, "same_score": True,
                    "backend": "deterministic fixture; real frozen LLM remains model diagnostic"}

        check("E6-00-E4-EQUIVALENCE", e6_equivalence)

    return {"status": "passed" if all(c["status"] == "passed" for c in checks) else "failed",
            "checks": checks, "fixture_count": 30, "contracts": environment_contracts(),
            "scope": "deterministic CPU software audit; no model competence, human audit or external validity certified"}


def _normalized_record(record, operation="initial"):
    scope = record["scope"]
    return {"memory_id": record["memory_id"], "attribute": record["attribute"], "value": record["value"],
            "context_scope": scope.get("context") if scope.get("context") is not None else "unknown_or_not_restricted",
            "valid_from": scope.get("valid_from"), "valid_until": scope.get("valid_until_exclusive"),
            "duration_mode": scope["kind"], "end_semantics": "unknown" if scope["kind"] == "unknown" else "explicit_exclusive_boundary" if scope.get("valid_until_exclusive") else "no_explicit_expiry",
            "source_evidence_ids": record["source_turn_ids"], "evidence_quotes": record["evidence_quotes"],
            "operation": operation, "supersedes": record.get("supersedes", [])}


def _case_audit(row, *, source=None, line=None):
    trace = row.get("trace", {})
    metrics = row.get("metrics", {})
    observations = trace.get("observations", {})
    actual = trace.get("interaction_observation")
    first_loss = "not_identifiable"
    kind = "observation"
    if trace and observations:
        if metrics.get("invalid_outputs", 0):
            first_loss = "format_or_schema"
        elif metrics.get("necessary_omission_numerator", 0):
            first_loss = "acquisition_not_executed"
        elif metrics.get("questions") and not metrics.get("response_informative", metrics.get("useful_acquisition_numerator", 0)):
            first_loss = "uninformative_or_irrelevant_reply"
        elif metrics.get("unsupported_operations", 0):
            first_loss = "unsupported_write"
        elif metrics.get("memory_coverage_at_reuse") == 0:
            first_loss = "observed_information_not_functionally_persisted"
        elif metrics.get("FJS") == 0:
            first_loss = "future_read_or_insufficient_memory; inspect exact trace"
            kind = "hypothesis"
        else:
            first_loss = "none_observed_for_primary_query"
    return {"source_path": str(source) if source else None, "source_line": line,
            "root_id": row.get("root_id"), "latent_world_id": row.get("latent_world_id"),
            "realization_id": row.get("realization_id"), "method": row.get("method"), "seed": row.get("seed"),
            "evidence_available": observations, "action": row.get("parsed_actions", trace.get("parsed_actions")),
            "reply": trace.get("user_reply"), "write_evidence": observations.get("write"),
            "before": trace.get("initial_memory"), "after": trace.get("final_memory"),
            "future_query": trace.get("episode", {}).get("future_query"), "future_response": trace.get("future_response"),
            "metrics": metrics, "first_observed_loss": first_loss, "explanation_status": kind,
            "reference_memory_from_delivered_evidence": None if actual is None else _reference_memory(actual),
            "normalized_before": [_normalized_record(r) for r in trace.get("initial_memory", [])],
            "normalized_after": [_normalized_record(r, "persisted") for r in trace.get("final_memory", [])]}


def _reference_memory(observation):
    from .policy.rule_controller import expected_memory
    return expected_memory(observation)


def run_diagnostics(parent_run, output, *, max_historical_cases=30):
    """Small CPU diagnosis; archived traces stay read-only and keep their scores.

    D0-D4 below test intervention plumbing with the observable rule controller.
    They do not substitute for B1 frozen-model diagnostics or imply RL benefit.
    """
    parent_run, output = Path(parent_run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    episode = _episode()
    class Never(RuleController):
        def select(self, obs):
            return {"action": "PROCEED", "slot": None}
    class KeepInitial(RuleController):
        def write(self, obs):
            return {"operations": [{"op": "NOOP"}]}
    # Controlled neutral response with exactly the same prior input.
    neutral = deepcopy(episode)
    neutral["replies"]["TEMPORAL_SCOPE"] = {"turn_id": "u3", "text": "No puedo precisar ese dato ahora."}
    branches = {
        "D0": run_episode(episode, RuleController()),
        "D1": run_episode(episode, Never()),
        "D2": run_episode(episode, RuleController(), intervention="I2", forced_slot="TEMPORAL_SCOPE"),
        "D3": run_episode(neutral, RuleController(), intervention="I2", forced_slot="TEMPORAL_SCOPE"),
        "D4": run_episode(episode, RuleController(), intervention="I2", forced_slot="TEMPORAL_SCOPE"),
        "KEEP-INITIAL": run_episode(episode, KeepInitial()),
    }
    from .evaluation.persistence_interventions import run_interventions
    interventions = run_interventions(episode, RuleController, forced_slot="TEMPORAL_SCOPE")
    audited = [{"diagnostic_branch": label, "actor": "observable_rule_fixture_not_B1", **_case_audit(row)}
               for label, row in branches.items()]
    sources = []
    # Fixed seed 101 and fixed filename ordering; never choose favorable cases.
    candidates = []
    for pattern in ("E2_all_test_C-ALWAYS_s101.jsonl", "E3*I2*s101.jsonl", "E3*I1*s101.jsonl",
                    "E4*B1*s101.jsonl", "E5_EXTRACT*B1*s101.jsonl", "E6_B1_s101.jsonl"):
        candidates.extend(sorted((parent_run / "traces").glob(pattern)))
    if max_historical_cases < 0:
        raise ValueError("NONNEGATIVE_CASE_LIMIT_REQUIRED")
    per_file = max(1, max_historical_cases // max(1, len(candidates)))
    historical_count = 0
    for source in candidates:
        if historical_count >= max_historical_cases:
            break
        selected = 0
        with source.open(encoding="utf-8") as stream:
            for line_number, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                row = json.loads(text)
                audited.append({"diagnostic_branch": "historical_read_only", **_case_audit(row, source=source, line=line_number)})
                historical_count += 1
                selected += 1
                if selected >= per_file or historical_count >= max_historical_cases:
                    break
        sources.append({"path": str(source), "sha256": file_hash(source), "selected_cases": selected})
    memory_path = output / "DIAGNOSTICO_MEMORIAS.jsonl"
    with memory_path.open("w", encoding="utf-8") as stream:
        for row in audited:
            stream.write(canonical(row) + "\n")
    # Guard against attributing a model's poor score to a pipeline fault.
    findings = [
        {"id": "H02", "status": "confirmed_cause", "finding": "Historical C-ALWAYS required model-generated valid ASK JSON; current V4 uses observable deterministic slot and guarantees execution.",
         "code_location": "research/src/evaluation/runner.py:run_episode; research/src/evaluation/batched.py:evaluate_batched"},
        {"id": "H03", "status": "verified_fixture_only", "finding": "I1/I2 RESPOND inputs identical; I0/I1 WRITE inputs identical in controlled independent fixture.", "evidence": interventions["invariants"]},
        {"id": "H04", "status": "hypothesis", "finding": "A redundant question can return an ongoing assertion; inspect archived clear-case reply and unsupported-write evidence before attributing aggregate damage."},
        {"id": "H05", "status": "not_identifiable_from_current_artifacts", "finding": "Control current/future query semantics require inspecting complete matched cohort; primary FJS alone cannot validate current unaffected attribute."},
        {"id": "H06", "status": "not_executed", "finding": "Frozen LLM reader intervention not run by CPU diagnostic; archived score is not an infrastructure failure."},
        {"id": "H07", "status": "not_executed", "finding": "Actual extracted memories preserved in selected historical trace audit when present; no new extraction inferred."},
        {"id": "H08", "status": "verified_fixture_only", "finding": "max_questions_s3=0 now uses exactly E4 reader/parser/input; real model positive-reply competence remains unverified."},
    ]
    result = {"status": "partial", "gate_status": "blocked_missing_evidence",
              "new_llm_calls": 0, "llm_call_budget": 600, "cohort": "independent infrastructure fixture plus fixed historical inspection",
              "fixture_matrix": {label: row["metrics"] for label, row in branches.items()},
              "fixture_matrix_actor": "observable_rule_control; B1 diagnostics not executed",
              "historical_cases": historical_count, "sources": sources, "findings": findings,
              "diagnostic_memory_path": str(memory_path), "diagnostic_memory_sha256": file_hash(memory_path),
              "limitations": ["D4 and D2 share the same reference writer in CPU fixture; their equality tests plumbing, not B1 writer quality",
                              "Historical selection is a deterministic error inspection, not a new independent test",
                              "No human review performed", "No frozen B1/LLM-reader calls executed by this offline command"]}
    write_json(output / "DIAGNOSTICO_CAUSAL.json", result)
    text = ["# Diagnóstico causal V4", "", "Diagnóstico parcial: fixtures ejecutados; diagnóstico B1 congelado pendiente. Cero llamadas nuevas al LLM.", "",
            "La matriz D0–D4 usa un controlador de reglas observable y no mide aprendizaje ni capacidad del LLM. D4 comparte escritor con D2 en este fixture.", "",
            "| Rama | C2 | FJS | Preguntas | Escrituras no sustentadas |", "|---|---:|---:|---:|---:|"]
    for label, row in branches.items():
        metrics = row["metrics"]
        text.append(f"| {label} | {metrics['C2']} | {metrics['FJS']} | {metrics['questions']} | {metrics['unsupported_operations']} |")
    text += ["", f"Inspeccionados {historical_count} casos históricos, semilla 101, orden de archivo fijo. Evidencia completa y memorias antes/después en `DIAGNOSTICO_MEMORIAS.jsonl`.", ""]
    for finding in findings:
        text.append(f"- {finding['id']} — {finding['status']}: {finding['finding']}")
    text += ["", "No se adjudica ninguna métrica histórica: los puntajes y archivos originales se conservan. Las explicaciones no identificadas están etiquetadas como hipótesis o pendientes.", ""]
    (output / "DIAGNOSTICO_CAUSAL.md").write_text("\n".join(text), encoding="utf-8")
    return result


def run_model_diagnostics(parent_run, output, *, device="cuda", episodes=None, max_llm_calls=600, backend=None):
    """Execute a small fixed B1 D0-D4 study, after deterministic contracts pass.

    The primary consumer is the existing symbolic reader. No frozen LLM-reader
    capability or human validity is implied. D4 reuses the exact D2 response and
    replaces only writing by the observable reference writer. Greedy D0-D3 each
    get pristine storage; no previous-actor cache is used for training.
    """
    from .data.build_bank import build_bank
    from .policy.rule_controller import evidence_model
    from .policy.llm_controller import SpecLocalPolicy
    from .artifacts import digest
    from statistics import mean
    import gc
    import time

    parent_run, output = Path(parent_run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    episodes = list(episodes if episodes is not None else build_bank(roots=10, seed=4107))
    if not episodes or any(e.get("split") not in {"dev_smoke", "train", "validation"} for e in episodes):
        raise ValueError("MODEL_DIAGNOSTICS_REQUIRE_NONTEST_DEVELOPMENT_COHORT")
    # No generated SELECT in forced arms: 3 + 2 + 2 + 2 + 0 calls per episode.
    planned = 9 * len(episodes)
    if planned > max_llm_calls:
        result = {"status": "not_executed_budget", "critical_checks_passed": False,
                  "planned_llm_calls": planned, "max_llm_calls": max_llm_calls, "new_llm_calls": 0}
        write_json(output / "DIAGNOSTICO_MODELO.json", result)
        return result
    trace_path = output / "DIAGNOSTICO_MODELO_TRACES.jsonl"
    if trace_path.exists():
        raise FileExistsError("PRESERVE_EXISTING_MODEL_DIAGNOSTIC_TRACES: " + str(trace_path))
    checkpoint = None
    if backend is None:
        training_path = parent_run / "sft" / "101" / "B1" / "training.json"
        if not training_path.is_file():
            result = {"status": "blocked_missing_evidence", "critical_checks_passed": False,
                      "missing": str(training_path), "new_llm_calls": 0}
            write_json(output / "DIAGNOSTICO_MODELO.json", result)
            return result
        selected = json.loads(training_path.read_text())["selected_checkpoint"]
        checkpoint = Path(selected)
        if not checkpoint.is_absolute():
            checkpoint = parent_run.resolve().parents[2] / checkpoint
        if not checkpoint.is_dir():
            result = {"status": "blocked_missing_evidence", "critical_checks_passed": False,
                      "missing": str(checkpoint), "new_llm_calls": 0}
            write_json(output / "DIAGNOSTICO_MODELO.json", result)
            return result
        try:
            backend = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=False, device=device)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            result = {"status": "blocked_missing_evidence", "critical_checks_passed": False,
                      "checkpoint": str(checkpoint), "runtime_error": repr(exc), "new_llm_calls": 0}
            write_json(output / "DIAGNOSTICO_MODELO.json", result)
            return result

    class Counter:
        def __init__(self, original):
            self.original, self.calls = original, 0
        def __getattr__(self, name):
            return getattr(self.original, name)
        def generate(self, *args, **kwargs):
            if self.calls >= max_llm_calls:
                raise RuntimeError("DIAGNOSTIC_LLM_CALL_BUDGET_EXCEEDED")
            self.calls += 1
            return self.original.generate(*args, **kwargs)
    counted = Counter(backend)

    class ReferenceWriter(RuleController):
        def __init__(self, response):
            self.response = deepcopy(response)
        def respond(self, observation):
            return deepcopy(self.response)

    rows, errors = [], []
    started = time.monotonic()
    try:
        # Frozen means both eval mode and no trainable tensors, not just a label.
        if hasattr(backend, "model"):
            if any(p.requires_grad for p in backend.model.parameters()):
                raise ValueError("MODEL_DIAGNOSTICS_REQUIRE_FROZEN_B1")
            backend.model.eval()
        with trace_path.open("x", encoding="utf-8") as stream:
            for episode in episodes:
                prior = {"memory": deepcopy(episode["initial_memory"]), "query": deepcopy(episode["current_query"]),
                         "user_messages": deepcopy(episode["user_messages"])}
                # Same observable slot in compatible worlds; fixed GENERIC tie
                # for cases where the public grammar finds nothing missing.
                slot = evidence_model(prior)["missing_slot"] or "GENERIC"
                neutral = deepcopy(episode)
                neutral["replies"][slot] = {"turn_id": "u3", "text": "No puedo precisar ese dato ahora."}
                factories = [
                    ("D0", episode, LLMController(counted), {}),
                    ("D1", episode, LLMController(counted), {"method": "C-NEVER"}),
                    ("D2", episode, LLMController(counted), {"intervention": "I2", "forced_slot": slot}),
                    ("D3", neutral, LLMController(counted), {"intervention": "I2", "forced_slot": slot}),
                ]
                collected = {}
                for label, case, policy, options in factories:
                    row = run_episode(case, policy, rollout_seed=101, **options)
                    row.update(diagnostic_branch=label, diagnostic_actor="frozen_B1_101", reference_slot=slot)
                    collected[label] = row
                    rows.append(row)
                    stream.write(canonical(row) + "\n")
                    stream.flush()
                d2 = collected["D2"]
                d4 = run_episode(episode, ReferenceWriter(d2["trace"]["response"]), rollout_seed=101,
                                 intervention="I2", forced_slot=slot, method="REFERENCE-WRITER")
                d4.update(diagnostic_branch="D4", diagnostic_actor="D2_B1_response_observable_reference_writer",
                          reference_slot=slot, actor_checkpoint=str(checkpoint) if checkpoint else "injected_frozen_backend")
                # Exact intervention boundary: same delivered evidence, same
                # current response, only the writer and resulting memory change.
                assert canonical(d2["trace"]["observations"]["write"]) == canonical(d4["trace"]["observations"]["write"])
                assert d2["trace"]["response"] == d4["trace"]["response"]
                rows.append(d4)
                stream.write(canonical(d4) + "\n")
                stream.flush()
                print(json.dumps({"closure_v4_diagnostic": {"episodes_completed": len(rows) // 5,
                      "scheduled_episodes": len(episodes), "llm_calls": counted.calls,
                      "max_llm_calls": max_llm_calls, "elapsed_seconds": time.monotonic() - started}}), flush=True)
    except (OSError, RuntimeError, ValueError, AssertionError) as exc:
        errors.append(repr(exc))
    finally:
        actual_calls = counted.calls
        torch = getattr(backend, "torch", None)
        del counted, backend
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    grouped = {label: [r for r in rows if r["diagnostic_branch"] == label] for label in ("D0", "D1", "D2", "D3", "D4")}
    summary = {}
    for label, selected_rows in grouped.items():
        summary[label] = {"episodes": len(selected_rows), **{
            metric: mean(r["metrics"].get(metric, 0) for r in selected_rows) if selected_rows else None
            for metric in ("C2", "FJS", "questions", "invalid_outputs", "unsupported_operations", "complete_acquisition_numerator")}}
    critical_pass = not errors and len(rows) == 5 * len(episodes)
    result = {"status": "executed" if critical_pass else "failed", "critical_checks_passed": critical_pass,
              "training_seed": 101, "actor_checkpoint": str(checkpoint) if checkpoint else "injected_frozen_backend",
              "cohort_sha256": digest(episodes), "root_count": len({e["root_id"] for e in episodes}),
              "episode_count": len(episodes), "planned_llm_calls": planned,
              "new_llm_calls": actual_calls,
              "max_llm_calls": max_llm_calls, "summary": summary, "errors": errors,
              "traces": str(trace_path), "traces_sha256": file_hash(trace_path) if trace_path.exists() else None,
              "frozen_reader": "fixed_symbolic", "llm_reader": "not_executed; excluded_secondary_component",
              "limitations": ["Small development cohort; descriptive causal diagnostics, no generalization claim",
                              "Natural-language reader remains outside this primary symbolic-reader scope",
                              "Format errors are measured model failures, not automatically infrastructure failures",
                              "D3 neutral response may be distributionally different", "Human audit not_executed"]}
    write_json(output / "DIAGNOSTICO_MODELO.json", result)
    with (output / "DIAGNOSTICO_CAUSAL.md").open("a", encoding="utf-8") as report:
        report.write("\n## Diagnóstico ejecutado con B1 congelado, semilla 101\n\n")
        report.write(f"Estado: {result['status']}; raíces: {result['root_count']}; episodios: {len(episodes)}; llamadas: {result['new_llm_calls']}/{max_llm_calls}. Lector simbólico común.\n\n")
        report.write("| Rama | C2 | FJS | Preguntas | Inválidas |\n|---|---:|---:|---:|---:|\n")
        for label, metrics in summary.items():
            report.write(f"| {label} | {metrics['C2']} | {metrics['FJS']} | {metrics['questions']} | {metrics['invalid_outputs']} |\n")
        report.write("\nD4 conserva respuesta y evidencia exactas de D2; cambia únicamente al escritor de referencia observable. Lector LLM y revisión humana no ejecutados.\n")
    return result
