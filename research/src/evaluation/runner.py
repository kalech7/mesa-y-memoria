"""One local rollout, with independent storage and three real policy prefixes."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import time
import hashlib

from ..artifacts import digest, canonical
from ..env.episode import EpisodeEnvironment
from ..protocol import assert_split_allowed
from ..readers.symbolic import SymbolicReader
from ..scoring.evidence import Evaluator, EVALUATOR_VERSION, response_valid
from ..scoring.reward import rewards

SLOTS = {"TEMPORAL_SCOPE", "CURRENT_CONTEXT", "VALID_UNTIL", "GENERIC"}


def _model_identity(policy):
    backend = getattr(policy, "backend", None)
    if backend is None or not hasattr(backend.config, "model_path") or not hasattr(backend, "model"):
        return {}
    identity = {"model_id": backend.config.model_path}
    manifest = Path(backend.config.model_path) / "download_manifest.json"
    if manifest.is_file():
        from ..artifacts import load_config
        pinned = load_config(manifest)
        identity.update(model_id=pinned.get("repo", identity["model_id"]), model_revision=pinned.get("revision"),
                        tokenizer_revision=pinned.get("tokenizer_revision", pinned.get("revision")))
    # Hash only adapter tensors, never copy the entire frozen backbone for each
    # episode. This identifies even intermediate, not-yet-saved RL policies.
    hasher, count = hashlib.sha256(), 0
    for name, parameter in backend.model.named_parameters():
        if "lora_" in name:
            tensor = parameter.detach().cpu().contiguous()
            hasher.update(name.encode())
            hasher.update(tensor.view(backend.torch.uint8).numpy().tobytes())
            count += 1
    identity["adapter_hash"] = hasher.hexdigest() if count else None
    if backend.origin_checkpoint:
        from llm_local_001.model import artifact_hashes
        identity["parent_checkpoint_hash"] = digest(artifact_hashes(backend.origin_checkpoint))
    return identity


def select_valid(action):
    return (isinstance(action, dict) and set(action) == {"action", "slot"}
            and ((action["action"] == "PROCEED" and action["slot"] is None)
                 or (action["action"] == "ASK" and action["slot"] in SLOTS)))


def _phase(policy, phase, observation):
    try:
        return getattr(policy, phase)(deepcopy(observation)), None
    except (ValueError, TypeError, KeyError) as exc:
        # Malformed model output is a phase failure, never a retry opportunity.
        return {}, {"phase": phase, "error": repr(exc), "kind": "invalid_output"}


def run_episode(episode, policy, *, condition="O2", intervention=None,
                forced_slot=None, rollout_seed=101, reader=None, nonresponse_rate=0,
                method="B4", root=None, db_path=None, imperfections=None):
    assert_split_allowed(episode["split"], root=root)
    backend = getattr(policy, "backend", None)
    if backend is not None and hasattr(backend, "seed"):
        backend.seed(rollout_seed)
    if hasattr(policy, "phase_records"):
        policy.phase_records.clear()
    started = time.monotonic()
    reader = reader or SymbolicReader()
    with tempfile.TemporaryDirectory(prefix="cm_rollout_") as work:
        env = EpisodeEnvironment(episode, db_path or Path(work) / "memory.sqlite", nonresponse_rate=nonresponse_rate,
                                 imperfections=imperfections)
        errors, outputs = [], {}
        try:
            obs = env.observe("select", condition, intervention)
            if intervention is not None:
                action, error = {}, None
            elif method == "C-NEVER":
                action, error = {"action": "PROCEED", "slot": None}, None
            elif method == "C-ALWAYS":
                from ..policy.rule_controller import RuleController
                action, error = RuleController().select_forced_ask(obs), None
            else:
                action, error = _phase(policy, "select", obs)
            if intervention is not None:
                if forced_slot not in SLOTS:
                    raise ValueError("Interventions require a preregistered specific slot")
                action = {"action": "PROCEED", "slot": None} if intervention == "I0" else {"action": "ASK", "slot": forced_slot}
                error = None
            elif method == "C-NEVER":
                action, error = {"action": "PROCEED", "slot": None}, None
            outputs["select"] = deepcopy(action)
            if error or not select_valid(action):
                errors.append(error or {"phase": "select", "kind": "invalid_output"})
            elif action["action"] == "ASK":
                env.ask(action["slot"])
            obs = env.observe("respond", condition, intervention)
            response, error = _phase(policy, "respond", obs)
            outputs["respond"] = deepcopy(response)
            if error or not response_valid(response):
                errors.append(error or {"phase": "respond", "kind": "invalid_output"})
            obs = env.observe("write", condition, intervention)
            write, error = _phase(policy, "write", obs)
            outputs["write"] = deepcopy(write)
            if error or not isinstance(write, dict) or set(write) != {"operations"}:
                result = {"valid": False, "executed_operations": [], "error": "INVALID_WRITE_WRAPPER"}
            else:
                result = env.apply_memory_operations(write["operations"])
            if not result["valid"]:
                errors.append(error or {"phase": "write", "kind": "invalid_output", "error": result["error"]})
            trace = deepcopy(env.private_trace)
            interaction = {"memory": deepcopy(episode["initial_memory"]),
                           "user_messages": deepcopy(trace["observations"]["respond"]["user_messages"]),
                           "query": deepcopy(episode["current_query"]), "reuse_regime": obs["reuse_regime"]}
            questions = env.question_count
            env.close_session()
            reuse = env.reopen_for_reuse()
            # Capture audit samples then discard policy context before S3.
            phase_records = deepcopy(policy.serialized_records() if hasattr(policy, "serialized_records")
                                     else getattr(policy, "phase_records", []))
            if hasattr(policy, "clear_transient"):
                policy.clear_transient()
            elif hasattr(policy, "phase_records"):
                policy.phase_records.clear()
            future_error = None
            try:
                future_response = reader.read(deepcopy(reuse["memory"]), deepcopy(reuse["query"]))
            except (ValueError, TypeError, KeyError) as exc:
                future_response, future_error = {}, repr(exc)
            trace.update(deepcopy(env.private_trace))
            trace.update(episode=deepcopy(episode), interaction_observation=interaction, response=response,
                         future_response=future_response, errors=errors, parsed_actions=outputs,
                         phase_records=phase_records, future_error=future_error)
            evaluator = Evaluator()
            current, memory, future = evaluator.score_current(trace), evaluator.score_memory(trace), evaluator.score_future(trace)
            reward = rewards(current["utility"], future["utility"], questions,
                             memory["unsupported_operations"], len(errors))
            metrics = {"C2": current["success"], "FJS": future["success"],
                       "current_factual": current["factual"], "future_factual": future["factual"],
                       "unsupported_current": current["unsupported_commitment"],
                       "unsupported_future": future["unsupported_commitment"],
                       "unsupported_write_episode": int(memory["unsupported_operations"] > 0),
                       "questions": questions, "invalid_outputs": len(errors),
                       "future_invalid": int(not response_valid(future_response)), **memory}
            from ..scoring.diagnostics import episode_diagnostics
            metrics.update(episode_diagnostics(trace, metrics))
            cost_sensitivity = {str(cost): rewards(current["utility"], future["utility"], questions,
                memory["unsupported_operations"], len(errors), question_cost=cost) for cost in (0., .05, .20)}
            return {"protocol_version": "1.2", "research_status": "exploratory", "git_commit": None, "model_id": None,
                    "model_revision": None, "tokenizer_revision": None, "adapter_hash": None,
                    "parent_checkpoint_hash": None, "method": method, "seed": rollout_seed,
                    **{key: episode[key] for key in ("root_id", "latent_world_id", "realization_id", "split", "scenario_family")},
                    "observation_condition": condition, "extraction_track": "REF", "intervention": intervention,
                    "reader_id": getattr(reader, "reader_id", type(reader).__name__),
                    "evaluator_version": EVALUATOR_VERSION,
                    "control_contract_version": "closure_v4" if method in {"C-ALWAYS", "C-NEVER"} else None,
                    "decoding_config_hash": digest({"sample": getattr(policy, "sample", False), "top_p": 1.,
                        "temperature": getattr(getattr(getattr(policy, "backend", None), "config", None), "temperature", 1.)}),
                    "evaluator_hash": digest({"evidence": Path(__file__).parents[1].joinpath("scoring/evidence.py").read_text(),
                                              "reward": Path(__file__).parents[1].joinpath("scoring/reward.py").read_text()}),
                    "initial_memory_hash": trace["initial_memory_hash"], "final_memory_hash": trace["final_memory_hash"],
                    "prompt_hashes": ({r["phase"]: digest(r["prompt"]) for r in phase_records} if phase_records
                                      else {phase: digest(value) for phase, value in trace["observations"].items()}),
                    "raw_outputs": phase_records or outputs, "parsed_actions": outputs,
                    "user_reply": trace.get("user_reply"), "executed_operations": trace["executed_operations"],
                    "rewards_by_component": reward, "metrics": metrics, "technical_status": "executed",
                    "retry_count": 0, "resource_counters": {"elapsed_seconds": time.monotonic() - started,
                        "input_characters": sum(len(canonical(x)) for x in trace["observations"].values())},
                    "question_cost_sensitivity": cost_sensitivity, **_model_identity(policy), "trace": trace}
        finally:
            env.dispose()


def run_attempts(episode, policy_factory, **options):
    """Infrastructure may retry once; all attempts remain in the denominator log."""
    attempts = []
    for retry in range(2):
        controller = policy_factory()
        try:
            row = run_episode(episode, controller, **options)
            row["retry_count"] = retry
            row["attempts"] = attempts + [{"attempt": retry, "status": "executed"}]
            return row
        except (OSError, RuntimeError) as exc:
            from ..protocol import GateError
            if isinstance(exc, GateError):
                raise
            partial = controller.serialized_records() if hasattr(controller, "serialized_records") else []
            attempts.append({"attempt": retry, "status": "failed", "error": repr(exc), "raw_outputs": partial})
    return {"method": options.get("method", "unknown"), "seed": options.get("rollout_seed", 101),
            **{key: episode[key] for key in ("root_id", "latent_world_id", "realization_id", "split", "scenario_family")},
            "technical_status": "failed", "retry_count": 1, "attempts": attempts,
            "metrics": {"C2": 0, "FJS": 0, "technical_failure": 1},
            "rewards_by_component": rewards(-1, -1)}
