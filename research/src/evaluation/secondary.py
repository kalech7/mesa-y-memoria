"""Secondary experiments using actual memories and a common frozen backbone.

No extraction is repaired from labels. Private original evidence is used only
after actor inference to audit false extraction and memory provenance.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import date
import hashlib
import json
from pathlib import Path

from ..artifacts import canonical
from ..memory.executor import validate_record
from ..readers.symbolic import read
from ..scoring.evidence import Evaluator, EVALUATOR_VERSION, score_response, probe_queries
from ..scoring.reward import rewards
from .batched import generate_batch, evaluate_batched


def _ensure_base(backend):
    if getattr(backend, "training_enabled", False) or getattr(getattr(backend, "model", None), "peft_config", None):
        raise ValueError("SECONDARY_READER_REQUIRES_UNADAPTED_FROZEN_BASE")


def _s1_messages(episode):
    messages = {}
    for record in episode["initial_memory"]:
        for quote in record["evidence_quotes"]:
            key = quote["turn_id"]
            text = quote["text"]
            if key in messages and messages[key]["text"] != text:
                raise ValueError("Conflicting original S1 source text")
            messages[key] = {"turn_id": key, "text": text, "at": record.get("recorded_at", "")}
    return sorted(messages.values(), key=lambda m: (m["at"], m["turn_id"]))


def _validate_extracted(records, messages, root_id):
    if not isinstance(records, list):
        raise ValueError("EXTRACT_RECORDS_ARRAY_REQUIRED")
    sources = {m["turn_id"]: m["text"] for m in messages}
    converted = deepcopy(records)
    output_ids = [r.get("memory_id", f"extract_{i}") for i, r in enumerate(converted) if isinstance(r, dict)]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("EXTRACT_DUPLICATE_RECORD_ID")
    id_map = {r.get("memory_id", f"extract_{i}"): "x" + hashlib.sha256(f"{root_id}:{i}".encode()).hexdigest()[:14]
              for i, r in enumerate(converted) if isinstance(r, dict)}
    for index, record in enumerate(converted):
        if not isinstance(record, dict):
            raise ValueError("EXTRACT_RECORD_OBJECT_REQUIRED")
        old = record.get("memory_id", f"extract_{index}")
        record["memory_id"] = id_map[old]
        targets = record.get("supersedes", [])
        if not isinstance(targets, list) or any(t not in id_map for t in targets):
            raise ValueError("EXTRACT_UNDEFINED_SUPERSESSION")
        record["supersedes"] = [id_map[t] for t in targets]
        validate_record(record, initial=True)
        if not set(record["source_turn_ids"]) <= sources.keys():
            raise ValueError("EXTRACT_UNKNOWN_SOURCE")
        if any(q["text"] not in sources.get(q["turn_id"], "") for q in record["evidence_quotes"]):
            raise ValueError("EXTRACT_NONLITERAL_QUOTE")
    return converted


def build_extraction_cache(episodes, base_backend, destination, *, max_new_tokens=1024):
    """One frozen chronological S1 extraction per root; actual invalid output=>[]."""
    _ensure_base(base_backend)
    episodes = list(episodes)
    destination = Path(destination)
    if destination.suffix != ".json":
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination / "extraction_cache.json"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if destination.exists():
        cache = json.loads(destination.read_text())
        for episode in episodes:
            entry = cache.get(episode["root_id"])
            source_hash = hashlib.sha256(canonical(_s1_messages(episode)).encode()).hexdigest()
            if entry is not None and entry["source_sha256"] != source_hash:
                raise ValueError("EXTRACTION_CACHE_SOURCE_MISMATCH")
        if all(episode["root_id"] in cache for episode in episodes):
            return cache
    template = Path(__file__).resolve().parents[2].joinpath("prompts/extractor.txt").read_text()
    skeleton = {"memory_id": "identificador_de_salida", "attribute": "dinner_preference", "value": "valor literal",
                "scope": {"kind": "ongoing|session|interval|contextual|unknown", "session_id": None,
                          "context": None, "valid_from": None, "valid_until_exclusive": None},
                "status": "confirmed|unresolved", "source_turn_ids": ["id_turno"],
                "evidence_quotes": [{"turn_id": "id_turno", "text": "cita literal"}], "supersedes": []}
    for episode in episodes:
        rid = episode["root_id"]
        messages = _s1_messages(episode)
        source_hash = hashlib.sha256(canonical(messages).encode()).hexdigest()
        if rid in cache:
            if cache[rid]["source_sha256"] != source_hash:
                raise ValueError("Counterfactual roots must share S1")
            continue
        prompt = template + "\nCada registro usa exactamente este esquema, con valores concretos: " + json.dumps(skeleton, ensure_ascii=False)
        prompt += "\nDevuelve sólo {\"records\":[...]}; los supersedes referencian memory_id de tu propia salida.\nMENSAJES_S1\n" + canonical(messages)
        entry = {"source_sha256": source_hash, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                 "records": [], "technical_status": "failed", "raw_output": "", "retry_count": 0,
                 "gold_repair": False, "source": "frozen_base_llm_actual_extraction"}
        old_limit = base_backend.config.max_new_tokens
        try:
            tokens = base_backend.encode_prompt(prompt)
            from ..policy.llm_controller import MAX_INPUT_TOKENS
            if len(tokens) > MAX_INPUT_TOKENS:
                raise ValueError("EXTRACT_INPUT_TOKEN_LIMIT")
            base_backend.config.max_new_tokens = max_new_tokens
            generation = base_backend.generate(prompt, sample=False)
            entry["raw_output"] = generation.text
            entry["generation"] = asdict(generation) if hasattr(generation, "__dataclass_fields__") else {"text": generation.text}
            if getattr(generation, "truncated", False):
                raise ValueError("EXTRACT_OUTPUT_TOKEN_LIMIT")
            output = json.loads(generation.text)
            if not isinstance(output, dict) or set(output) != {"records"}:
                raise ValueError("EXTRACT_WRAPPER_REQUIRED")
            entry["records"] = _validate_extracted(output["records"], messages, rid)
            entry["technical_status"] = "executed"
            entry["error"] = None
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            entry["error"] = repr(exc)
            entry["records"] = []
        finally:
            base_backend.config.max_new_tokens = old_limit
        cache[rid] = entry
        # Preserve actual partial cache if later infrastructure fails.
        destination.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n")
    return cache


def reread_frozen(rows, base_backend, *, batch_size=8):
    """Reread exactly each controller's actual final memory, without dialogue."""
    _ensure_base(base_backend)
    rows = list(rows)
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        eligible = [row for row in group if row.get("trace", {}).get("final_memory") is not None]
        observations = [{"memory": deepcopy(row["trace"]["final_memory"]),
                         "query": deepcopy(row["trace"]["episode"]["future_query"])} for row in eligible]
        outputs = generate_batch(base_backend, "reader", observations) if observations else []
        result_map = {id(row): (output, observation) for row, output, observation in zip(eligible, outputs, observations)}
        for original in group:
            row = deepcopy(original)
            row.update(experiment="E4", reader_id="frozen_base_llm")
            if id(original) not in result_map:
                row["technical_status"] = "failed"
                row.setdefault("metrics", {}).update(FJS=0, future_invalid=1)
                row["secondary_error"] = "Missing actual final memory trace"
                yield row
                continue
            output, observation = result_map[id(original)]
            response = output.get("parsed_action") or {}
            row["trace"]["future_response"] = response
            row["trace"]["frozen_reader_observation"] = observation
            row["trace"]["frozen_reader_generation"] = output
            scored = Evaluator().score_future(row["trace"])
            row["evaluator_version"] = EVALUATOR_VERSION
            row["metrics"].update(FJS=scored["success"], future_factual=scored["factual"],
                                  unsupported_future=scored["unsupported_commitment"], future_invalid=int(not scored["valid"]))
            row["secondary_reader_score"] = scored
            current = Evaluator().score_current(row["trace"])
            row["rewards_by_component"] = rewards(current["utility"], scored["utility"], row["metrics"].get("questions", 0),
                                                   row["metrics"].get("unsupported_operations", 0), row["metrics"].get("invalid_outputs", 0))
            yield row


def _extraction_provenance_audit(reference, extracted, query):
    """Audit extracted records against REF, never against their own claims."""
    from ..scoring.evidence import audit_operation
    observation = {"memory": deepcopy(reference), "user_messages": [], "query": deepcopy(query)}
    # Translate graph IDs only within the audit by shared original source turns;
    # actor memory and actual extracted graph are never changed.
    mapping = {}
    for record in extracted:
        matches = [r["memory_id"] for r in reference if set(r["source_turn_ids"]) == set(record["source_turn_ids"])
                   and r["attribute"] == record["attribute"] and r["value"] == record["value"]]
        if len(matches) == 1:
            mapping[record["memory_id"]] = matches[0]
    checks = {}
    for record in extracted:
        candidate = deepcopy(record)
        targets = [mapping.get(t, "unmapped_" + t) for t in candidate.get("supersedes", [])]
        candidate["supersedes"] = targets
        checks[record["memory_id"]] = audit_operation({"op": "SUPERSEDE" if targets else "ADD",
                                                       "record": candidate, "target_ids": targets}, observation)
    probes = probe_queries(observation, *extracted)
    decisive = [(q, read(reference, q)) for q in probes if read(reference, q)["mode"] == "ANSWER"]
    retained = sum(read(extracted, q).get("mode") == "ANSWER" and read(extracted, q).get("value") == expected["value"]
                   for q, expected in decisive)
    return {"record_faithfulness": checks, "unsupported_extracted_records": sum(not v for v in checks.values()),
            "reference_decisive_probes": len(decisive), "semantically_retained_probes": retained,
            "probe_retention_fraction": retained / len(decisive) if decisive else None,
            "definition": "Finite public S1 semantic probe audit; not a claim of complete relation coverage"}


def _evaluate_actor(episodes, backend, **options):
    if options.get("method") != "B4":
        yield from evaluate_batched(episodes, backend, **options)
        return
    from .runner import run_attempts
    from ..policy.rule_controller import RuleController
    options = dict(options)
    options.pop("batch_size", None)
    options.pop("variant", None)
    options["rollout_seed"] = options.pop("seed", 101)
    for episode in episodes:
        yield run_attempts(episode, RuleController, **options)


def evaluate_extracted(episodes, backend, cache, **options):
    """Run actors on the shared actual extraction, rescore fidelity against REF."""
    originals = {e["root_id"]: e for e in episodes}
    changed = []
    for episode in episodes:
        copy = deepcopy(episode)
        copy["initial_memory"] = deepcopy(cache[episode["root_id"]]["records"])
        changed.append(copy)
    for row in _evaluate_actor(changed, backend, **options):
        rid = row["root_id"]
        reference = originals[rid]["initial_memory"]
        row.update(experiment="E5", extraction_track="EXTRACT")
        row["extraction_cache_status"] = cache[rid]["technical_status"]
        trace = row["trace"]
        audit = _extraction_provenance_audit(reference, trace["episode"]["initial_memory"], trace["episode"]["current_query"])
        row["extraction_audit"] = audit
        trace["reference_initial_memory"] = deepcopy(reference)
        trace["interaction_observation"]["memory"] = deepcopy(reference)
        trace["extracted_initial_memory"] = deepcopy(trace["episode"]["initial_memory"])
        from ..policy.rule_controller import expected_memory
        final_audit = _extraction_provenance_audit(expected_memory(trace["interaction_observation"]),
                                                  trace["final_memory"], trace["episode"]["current_query"])
        row["final_memory_reference_audit"] = final_audit
        current = Evaluator().score_current(trace)
        future = Evaluator().score_future(trace)
        row["evaluator_version"] = EVALUATOR_VERSION
        # False extracted records cannot become evidence merely by persisting.
        current_refs = set(trace["response"].get("evidence_ids", []))
        future_refs = set(trace["future_response"].get("evidence_ids", []))
        bad_initial = {mid for mid, faithful in audit["record_faithfulness"].items() if not faithful}
        bad_final = {mid for mid, faithful in final_audit["record_faithfulness"].items() if not faithful}
        for score, used, bad in ((current, current_refs, bad_initial), (future, future_refs, bad_final)):
            if used & bad:
                score.update(success=0, justified=False, unsupported_commitment=1, utility=-1)
        row["metrics"].update(C2=current["success"], FJS=future["success"], unsupported_current=current["unsupported_commitment"],
                              unsupported_future=future["unsupported_commitment"],
                              unsupported_extracted_records=audit["unsupported_extracted_records"],
                              extraction_failure=int(cache[rid]["technical_status"] != "executed"))
        row["rewards_by_component"] = rewards(current["utility"], future["utility"], row["metrics"]["questions"],
                                               row["metrics"].get("unsupported_operations", 0), row["metrics"]["invalid_outputs"])
        yield row


def evaluate_nonresponse(episodes, backend, *, rates=(.2, .4), **options):
    for rate in rates:
        for row in _evaluate_actor(episodes, backend, nonresponse_rate=rate, **options):
            row.update(experiment="E5", nonresponse_rate=rate)
            yield row


def _reuse_reply(episode, slot):
    truth, query = episode["truth"], episode["future_query"]
    if slot == "CURRENT_CONTEXT" and query.get("context") is not None:
        context = query["context"]
        text = f"Hoy estoy en {context}." if isinstance(context, str) else "El contexto es " + canonical(context)
    elif slot == "TEMPORAL_SCOPE" and truth.get("decisive_slot") != "CURRENT_CONTEXT":
        text = f"Para esta ocasión prefiero {truth['future_value']}. Solo para hoy."
    elif slot == "VALID_UNTIL" and (truth.get("scope") or {}).get("valid_until_exclusive"):
        text = "El final exclusivo comunicado era " + truth["scope"]["valid_until_exclusive"] + "."
    else:
        text = "No tengo más detalles sobre esa variable."
    return {"turn_id": "s3_reply", "text": text}


def reuse_with_interaction(rows, base_backend, *, batch_size=8, max_questions_s3=1):
    """E6: common frozen S3 controller sees only actual memory/new query/reply."""
    if max_questions_s3 not in {0, 1}:
        raise ValueError("S3_QUESTION_BUDGET_MUST_BE_ZERO_OR_ONE")
    if max_questions_s3 == 0:
        # This is intentionally the E4 reader/parser/input/scoring path. Merely
        # asking RESPOND with an empty dialogue would change the prompt contract.
        for row in reread_frozen(rows, base_backend, batch_size=batch_size):
            row.update(experiment="E6", max_questions_s3=0)
            row["metrics"].update(s3_questions=0, total_questions=row["metrics"].get("questions", 0))
            row["trace"]["secondary_reuse"] = {"question_count": 0, "e4_equivalent": True,
                                                   "observation": deepcopy(row["trace"].get("frozen_reader_observation"))}
            yield row
        return
    _ensure_base(base_backend)
    rows = list(rows)
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        valid = [r for r in group if "trace" in r]
        observations = [{"memory": deepcopy(r["trace"]["final_memory"]), "user_messages": [],
                         "query": deepcopy(r["trace"]["episode"]["future_query"]),
                         "reuse_regime": "Se permite como máximo una pregunta sobre esta consulta."} for r in valid]
        selected = generate_batch(base_backend, "select", observations) if observations else []
        metadata = []
        for row, observation, output in zip(valid, observations, selected):
            action = output.get("parsed_action") or {}
            asked = action.get("action") == "ASK"
            slot = action.get("slot") if asked else None
            episode = row["trace"]["episode"]
            if asked:
                observation["user_messages"].append(_reuse_reply(episode, slot))
            previous = row.get("parsed_actions", {}).get("select", {})
            unchanged_context = episode["truth"].get("current_context") == episode["future_query"].get("context")
            previous_scope = episode["truth"].get("scope") or {}
            future_date = episode["future_query"]["date"]
            scope_still_applies = previous_scope.get("kind") == "ongoing" or (
                previous_scope.get("kind") == "interval" and previous_scope.get("valid_from", "") <= future_date
                < previous_scope.get("valid_until_exclusive", ""))
            repeated = bool(asked and previous.get("action") == "ASK" and previous.get("slot") == slot
                            and ((slot == "CURRENT_CONTEXT" and unchanged_context)
                                 or (slot == "TEMPORAL_SCOPE" and scope_still_applies)))
            metadata.append((asked, slot, repeated))
        responses = generate_batch(base_backend, "respond", observations) if observations else []
        by_identity = {id(row): (obs, sel, response, meta) for row, obs, sel, response, meta in zip(valid, observations, selected, responses, metadata)}
        for original in group:
            row = deepcopy(original)
            row.update(experiment="E6", reader_id="frozen_base_llm_with_one_question")
            row["primary_regime_rewards"] = row.pop("rewards_by_component", {})
            row["rewards_by_component"] = {}
            row["secondary_reward_status"] = "not_defined_for_reuse_interaction_regime; report task success and question burden separately"
            if id(original) not in by_identity:
                row["technical_status"] = "failed"
                row.setdefault("metrics", {})["FJS"] = 0
                yield row
                continue
            observation, selected_output, response_output, (asked, slot, repeated) = by_identity[id(original)]
            response = response_output.get("parsed_action") or {}
            episode = row["trace"]["episode"]
            score = score_response(response, observation, episode["truth"]["future_value"])
            from ..policy.rule_controller import evidence_model
            if response.get("mode") == "ANSWER" and evidence_model(observation)["new_record"] is None:
                traced = deepcopy(row["trace"])
                traced["future_response"] = response
                score = Evaluator().score_future(traced)
            row["metrics"].update(FJS=score["success"], future_factual=score["factual"],
                                  unsupported_future=score["unsupported_commitment"], s3_questions=int(asked),
                                  total_questions=row["metrics"].get("questions", 0) + int(asked),
                                  repeated_scope_questions=int(repeated), s3_invalid_outputs=int(not selected_output.get("parsed_action")) + int(not score["valid"]))
            row["trace"]["secondary_reuse"] = {"observation": observation, "select_generation": selected_output,
                                                 "respond_generation": response_output, "question_slot": slot,
                                                 "question_count": int(asked), "repeated_scope": repeated,
                                                 "simulator": "fixed_slot_table_for_future_query_no_gold_json"}
            yield row
