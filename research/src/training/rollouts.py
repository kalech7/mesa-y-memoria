"""Bridge the real three-session runner to the episodic optimizer."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from llm_local_001.model import Generation
from research.src.artifacts import digest
from research.src.policy.llm_controller import LLMController
from research.src.training.episodic_grpo import EpisodeRollout, PhaseSample


def make_rollout_collector(output_dir, *, condition="O2", method="B2-L", root=None,
                           prompt_root=None, imperfections=None):
    """Return collect(policy, episode, rollout_seed), preserving each SQLite file.

    The episode is passed only to the environment/runner. The LLMController sees
    exclusively each environment observation. No gold value enters its prompt.
    The runner applies the sampled operations and computes the S3 consequences.
    """
    from research.src.evaluation.runner import run_episode
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def collect(policy, episode, rollout_seed):
        identifier = digest({"root_id": episode["root_id"],
                             "latent_world_id": episode["latent_world_id"],
                             "realization_id": episode["realization_id"], "seed": rollout_seed})[:24]
        db_path = output_dir / f"{identifier}.sqlite"
        trace_path = output_dir / f"{identifier}.json"
        if db_path.exists() or trace_path.exists():
            raise FileExistsError("Rollout seeds/identities cannot reuse mutable storage")
        controller = LLMController(policy, prompt_root=prompt_root, sample=True)
        row = run_episode(episode, controller, condition=condition, rollout_seed=rollout_seed,
                          method=method, root=root, db_path=db_path, imperfections=imperfections)
        records = row["trace"]["phase_records"]
        phases = []
        for record in records:
            generation = record["generation"]
            if isinstance(generation, dict):
                generation = Generation(**generation)
            phases.append(PhaseSample(record["phase"], generation))
        components = row["rewards_by_component"]["components"]
        metrics = row["metrics"]
        # Keep all raw sampled text and actual prefixes, including malformed JSON.
        serializable = deepcopy(row)
        for source in (serializable.get("raw_outputs"), serializable.get("trace", {}).get("phase_records")):
            if isinstance(source, list):
                for record in source:
                    if isinstance(record, dict) and isinstance(record.get("generation"), Generation):
                        record["generation"] = asdict(record["generation"])
        trace_path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
        return EpisodeRollout(episode["root_id"], episode["latent_world_id"], rollout_seed, phases,
                              components["current"], components["future"],
                              questions=metrics["questions"],
                              unsupported_operations=metrics["unsupported_operations"],
                              invalid_outputs=metrics["invalid_outputs"], memory_path=str(db_path),
                              split=episode["split"])

    return collect
