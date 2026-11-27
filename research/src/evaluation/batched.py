"""Greedy batched local inference with exact phase observations and replay.

Only decoding is batched. Every episode retains its own simulator/SQLite and the
ordinary evaluator executes its actual actions. Replay asserts the observation
bytes equal the prefix used for generation, including intervention boundaries.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

from llm_local_001.model import Generation
from ..artifacts import canonical
from ..env.episode import EpisodeEnvironment
from ..policy.llm_controller import LLMController, InvalidOutput, validate_action, PHASE_LIMITS, MAX_INPUT_TOKENS
from ..protocol import assert_split_allowed
from .runner import run_episode, select_valid


FORCE_ASK = ('En esta condición debes realizar exactamente una pregunta. Devuelve action:"ASK" y elige '
             'un único slot permitido usando la misma evidencia visible. PROCEED no está permitido.')


def generate_batch(backend, phase, observations, *, variant=None, force_ask=False):
    """One immutable unpadded prompt per output; no inference retries for JSON."""
    from transformers import GenerationConfig
    builder = LLMController(backend, prompt_variant=variant)
    prompts = [builder.prompt(phase, obs, instruction_suffix=FORCE_ASK if force_ask else None) for obs in observations]
    encoded = [backend.encode_prompt(prompt) for prompt in prompts]
    active = [i for i, ids in enumerate(encoded) if len(ids) <= MAX_INPUT_TOKENS]
    results = [{"phase": phase, "observation": deepcopy(obs), "prompt": prompt, "raw_output": "",
                "parsed_action": None, "technical_status": "invalid_output", "retry_count": 0,
                "parse_status": "not_attempted", "schema_status": "not_attempted", "truncated": False,
                "fallback_used": False, "fallback_reason": None,
                "error": "INPUT_TOKEN_LIMIT", "generation": asdict(Generation(ids, [], ""))}
               for obs, prompt, ids in zip(observations, prompts, encoded)]
    if not active:
        return results
    torch, tokenizer = backend.torch, backend.tokenizer
    width = max(len(encoded[i]) for i in active)
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([[pad] * (width - len(encoded[i])) + encoded[i] for i in active], device=backend.device)
    attention = torch.tensor([[0] * (width - len(encoded[i])) + [1] * len(encoded[i]) for i in active], device=backend.device)
    config = GenerationConfig(max_new_tokens=PHASE_LIMITS[phase], do_sample=False, use_cache=True,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=pad, bos_token_id=tokenizer.bos_token_id)
    backend.model.eval()
    with torch.no_grad():
        output = backend.model.generate(input_ids=input_ids, attention_mask=attention, generation_config=config)
    completions = output[:, width:].detach().cpu().tolist()
    for index, completion in zip(active, completions):
        eos = tokenizer.eos_token_id
        if eos in completion:
            completion = completion[:completion.index(eos) + 1]
        generated = Generation(encoded[index], completion, tokenizer.decode(completion, skip_special_tokens=True),
                               truncated=not completion or completion[-1] != eos,
                               metadata={"sampled": False, "inference_batch_size": len(active), "padding": "left"})
        record = results[index]
        record.update(generation=asdict(generated), raw_output=generated.text, error=None, truncated=bool(generated.truncated))
        try:
            if generated.truncated:
                raise InvalidOutput("OUTPUT_TOKEN_LIMIT: incomplete generation")
            action = json.loads(generated.text)
            record["parse_status"] = "passed"
            validate_action(phase, action)
            record["schema_status"] = "passed"
            if force_ask and action.get("action") != "ASK":
                raise InvalidOutput("C-ALWAYS requires ASK")
            record.update(parsed_action=action, technical_status="completed")
        except (ValueError, TypeError) as exc:
            record["error"] = str(exc)
            if record["parse_status"] == "passed":
                record["schema_status"] = "failed"
            elif not generated.truncated:
                record["parse_status"] = "failed"
    del output, input_ids, attention
    return results


class ReplayController:
    def __init__(self, backend, records, observation_map):
        self.backend = backend
        self.records = records
        self.observation_map = observation_map
        self.phase_records = []
        self.sample = False

    def _call(self, phase, obs):
        if canonical(obs) != canonical(self.observation_map[phase]):
            raise RuntimeError("BATCH_PREFIX_REPLAY_MISMATCH")
        record = deepcopy(self.records.get(phase))
        if record is None:
            return {"action": "PROCEED", "slot": None}
        self.phase_records.append(record)
        if record["technical_status"] != "completed":
            raise InvalidOutput(record.get("error", "Invalid generated JSON"))
        return deepcopy(record["parsed_action"])

    def select(self, obs): return self._call("select", obs)
    def select_forced_ask(self, obs): return self._call("select", obs)
    def respond(self, obs): return self._call("respond", obs)
    def write(self, obs): return self._call("write", obs)
    def serialized_records(self): return deepcopy(self.phase_records)


def evaluate_batched(episodes, backend, *, method="B1", condition="O2", intervention=None,
                     forced_slot=None, batch_size=8, root=None, reader=None, seed=101,
                     nonresponse_rate=0, variant=None, imperfections=None):
    if batch_size < 1:
        raise ValueError("Positive inference batch size required")
    episodes = list(episodes)
    variant = variant or ("B0-P" if method == "B0-P" else None)
    for start in range(0, len(episodes), batch_size):
        group = episodes[start:start + batch_size]
        for episode in group:
            assert_split_allowed(episode["split"], root=root)
        with tempfile.TemporaryDirectory(prefix="research_batched_") as work:
            envs = [EpisodeEnvironment(e, Path(work) / f"staging_{i}.sqlite", nonresponse_rate=nonresponse_rate,
                                      imperfections=imperfections)
                    for i, e in enumerate(group)]
            records = [{} for _ in group]
            observations = [{} for _ in group]
            try:
                for i, env in enumerate(envs):
                    observations[i]["select"] = env.observe("select", condition, intervention)
                if intervention is None and method not in {"C-NEVER", "C-ALWAYS"}:
                    selected = generate_batch(backend, "select", [o["select"] for o in observations],
                                              variant=variant, force_ask=method == "C-ALWAYS")
                    for i, record in enumerate(selected):
                        records[i]["select"] = record
                for i, env in enumerate(envs):
                    action = records[i].get("select", {}).get("parsed_action")
                    if intervention in {"I1", "I2"}:
                        slot = forced_slot or group[i]["opportunity"]["slot"]
                        env.ask(slot)
                    elif intervention is None and method == "C-ALWAYS":
                        from ..policy.rule_controller import RuleController
                        env.ask(RuleController().select_forced_ask(observations[i]["select"])["slot"])
                    elif intervention is None and method != "C-NEVER" and select_valid(action) and action["action"] == "ASK":
                        env.ask(action["slot"])
                for phase in ("respond", "write"):
                    for i, env in enumerate(envs):
                        observations[i][phase] = env.observe(phase, condition, intervention)
                    outputs = generate_batch(backend, phase, [o[phase] for o in observations], variant=variant)
                    for i, record in enumerate(outputs):
                        records[i][phase] = record
                for i, episode in enumerate(group):
                    controller = ReplayController(backend, records[i], observations[i])
                    row = run_episode(episode, controller, condition=condition, intervention=intervention,
                                      forced_slot=(forced_slot or episode.get("opportunity", {}).get("slot")),
                                      rollout_seed=seed, reader=reader, nonresponse_rate=nonresponse_rate,
                                      method=method, root=root, imperfections=imperfections)
                    row["decoding_batch"] = {"size": batch_size, "padding": "left", "mode": "greedy"}
                    yield row
            finally:
                for env in envs:
                    env.dispose()


def observability_batched(episodes, backend, *, method="B0", batch_size=8, root=None, seed=101):
    """E1 never selects, asks, writes or accesses the subsequent query."""
    from ..scoring.evidence import score_response
    episodes = list(episodes)
    for condition in ("O1", "O2"):
        for start in range(0, len(episodes), batch_size):
            group = episodes[start:start + batch_size]
            observations = []
            for episode in group:
                assert_split_allowed(episode["split"], root=root)
                memory = deepcopy(episode["initial_memory"])
                if condition == "O1":
                    relevant = [r for r in memory if r["attribute"] == episode["current_query"]["attribute"]]
                    memory = [max(enumerate(relevant), key=lambda item: (item[1].get("recorded_at", ""), item[0]))[1]] if relevant else []
                observations.append({"memory": memory, "user_messages": deepcopy(episode["user_messages"]),
                                     "query": deepcopy(episode["current_query"]),
                                     "reuse_regime": "Habrá una consulta posterior sin canal de aclaración."})
            records = generate_batch(backend, "respond", observations, variant="B0-P" if method == "B0-P" else None)
            for episode, observation, record in zip(group, observations, records):
                response = record["parsed_action"] or {}
                score = score_response(response, observation, episode["truth"]["current_value"])
                yield {"protocol_version": "1.2", "experiment": "E1", "method": method, "seed": seed,
                       **{key: episode[key] for key in ("root_id", "latent_world_id", "realization_id", "split", "scenario_family")},
                       "observation_condition": condition, "extraction_track": "REF", "intervention": None,
                       "reader_id": "not_applicable_E1", "technical_status": "executed", "retry_count": 0,
                       "metrics": {"C2": score["success"], "unsupported_current": score["unsupported_commitment"],
                                   "invalid_outputs": int(not score["valid"]), "questions": 0},
                       "raw_outputs": [record], "parsed_actions": {"respond": response},
                       "resource_counters": {"input_tokens": len(record["generation"]["prompt_ids"]),
                                             "generated_tokens": len(record["generation"]["completion_ids"]),
                                             "silent_truncation": False},
                       "trace": {"observations": {"respond": observation}, "response": response}}
