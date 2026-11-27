"""Completion-only SFT and episodic REINFORCE of the causal LLM itself.

All sampling rewards arrive through the existing environment after execution.
This module never consumes hidden preferences, evaluation labels or future user
answers. Validation callbacks must use VALIDATION only; TEST is absent from this
API. Higher validation scores are better.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Callable, Any

from .model import Generation, LocalPolicy, utc_now


@dataclass
class TrainingConfig:
    updates: int = 8
    learning_rate: float = 1e-4
    batch_size: int = 1
    max_grad_norm: float = 1.0
    validation_every: int = 1
    gamma: float = 1.0
    baseline_decay: float = 0.9
    kl_coefficient: float = 0.02
    seed: int = 0

    def validate(self) -> None:
        if self.updates < 0 or self.batch_size < 1 or self.validation_every < 1:
            raise ValueError("Invalid update/batch/validation budget")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Learning rate and gradient clipping must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.baseline_decay < 1 or self.kl_coefficient < 0:
            raise ValueError("Invalid discount, past-return baseline or KL coefficient")


@dataclass
class EpisodeTrajectory:
    generations: list[Generation]
    rewards: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.generations or len(self.generations) != len(self.rewards):
            raise ValueError("Every generated action needs its post-execution reward")
        if not all(math.isfinite(float(reward)) for reward in self.rewards):
            raise ValueError("Nonfinite environment reward")


@dataclass
class TrainingResult:
    algorithm: str
    created_at: str
    config: dict[str, Any]
    parameters: dict[str, Any]
    history: list[dict[str, Any]]
    checkpoints: dict[str, Any]
    best_step: int
    selected_score: float
    tokens_processed: int
    completion_tokens: int
    elapsed_seconds: float
    changed_parameters: int
    parameter_delta_l2: float
    source_checkpoint: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def discounted_returns(rewards: list[float], gamma: float = 1.0) -> list[float]:
    """S2 receives the consequences of S3 instead of only its immediate reward."""
    result = [0.0] * len(rewards)
    cumulative = 0.0
    for index in reversed(range(len(rewards))):
        cumulative = float(rewards[index]) + gamma * cumulative
        result[index] = cumulative
    return result


def completion_label_mask(prompt_ids: list[int], completion_ids: list[int]) -> list[int]:
    """Document/test the HF label convention used by completion-only SFT."""
    return [-100] * len(prompt_ids) + list(completion_ids)


def update_baseline(previous: float, observed_returns: list[float], decay: float) -> float:
    if not observed_returns:
        return previous
    return decay * previous + (1.0 - decay) * sum(observed_returns) / len(observed_returns)


def _validation(policy: LocalPolicy, callback: Callable[[LocalPolicy], float]) -> float:
    policy.model.eval()
    with policy.torch.no_grad():
        score = float(callback(policy))
    if not math.isfinite(score):
        raise ValueError("Validation score must be finite")
    return score


def _memory(policy: LocalPolicy) -> dict[str, int]:
    if policy.device.type != "cuda":
        return {"cuda_peak_allocated_bytes": 0, "cuda_peak_reserved_bytes": 0}
    return {"cuda_peak_allocated_bytes": policy.torch.cuda.max_memory_allocated(policy.device),
            "cuda_peak_reserved_bytes": policy.torch.cuda.max_memory_reserved(policy.device)}


class _Run:
    def __init__(self, policy: LocalPolicy, destination: str | Path, config: TrainingConfig,
                 validation: Callable[[LocalPolicy], float], algorithm: str):
        config.validate()
        self.policy, self.config, self.validation = policy, config, validation
        self.destination = Path(destination)
        if self.destination.exists() and any(self.destination.iterdir()):
            raise FileExistsError(f"Preserve existing experiment: {self.destination} is not empty")
        self.destination.mkdir(parents=True, exist_ok=True)
        self.algorithm = algorithm
        self.started = time.monotonic()
        policy.seed(config.seed)
        policy.enable_training()
        self.params = [(name, p) for name, p in policy.model.named_parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("No trainable parameters inside the language model")
        self.before = {name: p.detach().cpu().clone() for name, p in self.params}
        self.optimizer = policy.torch.optim.AdamW([p for _, p in self.params],
                                                  lr=config.learning_rate, weight_decay=0.0)
        self.scaler = policy.torch.amp.GradScaler("cuda", enabled=(policy.device.type == "cuda" and
                      next(p for _, p in self.params).dtype == policy.torch.float16))
        if policy.device.type == "cuda":
            policy.torch.cuda.reset_peak_memory_stats(policy.device)
        self.best_score = _validation(policy, validation)
        self.best_step = 0
        self.history = [{"step": 0, "validation_score": self.best_score, "tokens_processed": 0,
                         "created_at": utc_now(), **_memory(policy)}]
        self.checkpoints = {}
        for label in ("initial", "selected"):
            path = self.destination / label
            self.checkpoints[label] = {"path": str(path.resolve()),
                                       "manifest": policy.save_checkpoint(path)}
        self.tokens = 0
        self.completion_tokens = 0

    def backward(self, loss):
        if not self.policy.torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; no optimizer update applied")
        self.scaler.scale(loss).backward()

    def step(self, step: int, metrics: dict[str, Any]) -> None:
        torch = self.policy.torch
        self.scaler.unscale_(self.optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for _, p in self.params):
            raise FloatingPointError("Nonfinite LLM gradients; training stopped")
        grad = float(torch.nn.utils.clip_grad_norm_([p for _, p in self.params], self.config.max_grad_norm))
        if not any(p.grad is not None for _, p in self.params):
            raise RuntimeError("No language-model gradients reached the optimizer")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        row = {"step": step, **metrics, "gradient_norm_before_clip": grad,
               "tokens_processed": self.tokens, "completion_tokens": self.completion_tokens,
               "elapsed_seconds": time.monotonic() - self.started,
               "created_at": utc_now(), **_memory(self.policy)}
        if step % self.config.validation_every == 0 or step == self.config.updates:
            row["validation_score"] = _validation(self.policy, self.validation)
            if row["validation_score"] > self.best_score:
                self.best_score, self.best_step = row["validation_score"], step
                path = self.destination / "selected"
                self.checkpoints["selected"] = {"path": str(path.resolve()),
                                                "manifest": self.policy.save_checkpoint(path)}
        self.history.append(row)
        # Persist evidence each update, including if a later update fails.
        (self.destination / "intermediate_metrics.json").write_text(
            json.dumps({"algorithm": self.algorithm, "config": asdict(self.config),
                        "history": self.history}, indent=2), encoding="utf-8")

    def finish(self) -> TrainingResult:
        path = self.destination / "last"
        self.checkpoints["last"] = {"path": str(path.resolve()), "manifest": self.policy.save_checkpoint(path)}
        delta_squared, changed = 0.0, 0
        for name, parameter in self.params:
            delta = parameter.detach().cpu().float() - self.before[name].float()
            delta_squared += float(delta.square().sum())
            changed += int(self.policy.torch.count_nonzero(delta))
        result = TrainingResult(self.algorithm, utc_now(), asdict(self.config),
                                self.policy.parameter_counts(), self.history, self.checkpoints,
                                self.best_step, self.best_score, self.tokens, self.completion_tokens,
                                time.monotonic() - self.started, changed, math.sqrt(delta_squared),
                                self.policy.origin_checkpoint)
        (self.destination / "training_result.json").write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
        return result


def train_sft(policy: LocalPolicy, dataset: list[dict[str, str]], destination: str | Path,
              config: TrainingConfig, validation_callback: Callable[[LocalPolicy], float], *,
              target_tokens_per_update: list[int] | None = None) -> TrainingResult:
    """B1 or B3; only provided TRAIN prompt/completion pairs are consumed.

    One optimizer update uses batch_size completions. B3 can exactly match the RL
    optimizer-update budget; measured token counts document remaining differences.
    """
    if not dataset:
        raise ValueError("SFT requires nonempty TRAIN examples")
    if target_tokens_per_update is not None and (
        len(target_tokens_per_update) != config.updates or any(t < 1 for t in target_tokens_per_update)
    ):
        raise ValueError("B3 requires one positive RL token target per optimizer update")
    run = _Run(policy, destination, config, validation_callback, "SFT")
    import random
    order = list(range(len(dataset)))
    rng = random.Random(config.seed)
    rng.shuffle(order)
    position = 0
    for step in range(1, config.updates + 1):
        total_loss = 0.0
        policy.model.train()
        batch = []
        batch_tokens = 0
        target_tokens = target_tokens_per_update[step - 1] if target_tokens_per_update is not None else None
        while (batch_tokens < target_tokens if target_tokens is not None else len(batch) < config.batch_size):
            if position == len(order):
                rng.shuffle(order)
                position = 0
            example = dataset[order[position]]
            position += 1
            generated = policy.supervised_generation(example["prompt"], example["completion"])
            batch.append(generated)
            batch_tokens += len(generated.prompt_ids) + len(generated.completion_ids)
        total_completion_tokens = sum(len(item.completion_ids) for item in batch)
        for generated in batch:
            # sequence_logprobs shifts the causal logits and excludes all prompt
            # targets, exactly equivalent to the -100 prompt label mask.
            loss = -policy.sequence_logprobs(generated).sum() / total_completion_tokens
            run.backward(loss)
            total_loss += float(loss.detach())
            run.tokens += len(generated.prompt_ids) + len(generated.completion_ids)
            run.completion_tokens += len(generated.completion_ids)
        run.step(step, {"supervised_loss": total_loss, "examples": len(batch),
                        "training_forward_tokens_this_update": batch_tokens,
                        "rl_target_tokens_this_update": target_tokens,
                        "target_token_overshoot": batch_tokens - target_tokens if target_tokens is not None else None})
    return run.finish()


def train_rl(policy: LocalPolicy, rollout_callback: Callable[[LocalPolicy], EpisodeTrajectory],
             destination: str | Path, config: TrainingConfig,
             validation_callback: Callable[[LocalPolicy], float], *,
             reference_policy: LocalPolicy | None = None) -> TrainingResult:
    """REINFORCE on generated token probabilities and longitudinal returns.

    The frozen reference is reloaded from the exact B1 checkpoint, on CPU by
    default to preserve GPU memory. No top-k/top-p sampling is permitted. The KL
    stabilization is exact full-vocabulary KL(p || frozen B1) at each visited
    generated-token state, averaged over positions. Its direct conditional
    gradient is differentiated through p; reference logits are frozen. This is
    a visited-state conditional regularizer rather than a trajectory KL. Prompt
    forward tokens and additional reference passes are separately accounted for.

    A past-only EMA of episode returns provides a baseline. All episodes of an
    update use its previous value; current returns update it only after gradients
    have been accumulated. There is no privileged-state critic or policy head.
    """
    if not policy.origin_checkpoint:
        raise ValueError("RL must load a fresh branch of the saved B1 checkpoint")
    if reference_policy is None:
        reference_policy = LocalPolicy.from_checkpoint(policy.origin_checkpoint, device="cpu")
    if reference_policy.origin_checkpoint != policy.origin_checkpoint:
        raise ValueError("RL and frozen reference must originate from exactly the same B1")
    if reference_policy is policy:
        raise ValueError("Reference policy must be an independent frozen checkpoint")
    for parameter in reference_policy.model.parameters():
        parameter.requires_grad_(False)
    reference_policy.model.eval()
    run = _Run(policy, destination, config, validation_callback, "REINFORCE_EMA_EXACT_CONDITIONAL_KL")
    baseline = 0.0
    total_reference_tokens = 0
    total_rollout_tokens = 0
    for step in range(1, config.updates + 1):
        used_baseline = baseline
        episode_returns = []
        policy_losses, kl_values = [], []
        # Sampling uses eval mode. The scoped likelihood mode additionally
        # enables only decoder checkpoint dispatch, leaving dropout disabled.
        policy.model.eval()
        for _ in range(config.batch_size):
            episode = rollout_callback(policy)
            episode.validate()
            returns = discounted_returns(episode.rewards, config.gamma)
            episode_returns.append(returns[0])
            for generation, future_return in zip(episode.generations, returns):
                if generation.metadata.get("sampled") is not True:
                    raise ValueError("REINFORCE trajectories must be sampled from the LLM")
                if generation.metadata.get("temperature") != policy.config.temperature:
                    raise ValueError("Rollout and optimization sampling temperature differ")
                with policy.likelihood_mode():
                    logits = policy.sequence_logits(generation, sampling_distribution=True)
                    log_distribution = policy.torch.log_softmax(logits, dim=-1)
                    target = policy.torch.tensor(generation.completion_ids, device=policy.device)
                    logp = log_distribution.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                    with policy.torch.no_grad():
                        reference_logits = reference_policy.sequence_logits(generation, sampling_distribution=True)
                    kl = policy.torch.zeros((), device=policy.device)
                    # Limit the temporary reference transfer to eight positions;
                    # policy logits already exist for the autoregressive loss.
                    for offset in range(0, len(generation.completion_ids), 8):
                        lp = log_distribution[offset:offset + 8]
                        lq = policy.torch.log_softmax(reference_logits[offset:offset + 8].to(policy.device), dim=-1)
                        kl = kl + (lp.exp() * (lp - lq)).sum() / len(generation.completion_ids)
                    reinforce = -(future_return - used_baseline) * logp.sum()
                    loss = (reinforce + config.kl_coefficient * kl) / config.batch_size
                    run.backward(loss)
                    policy_losses.append(float(reinforce.detach()))
                    kl_values.append(float(kl.detach()))
                # Graphs are consumed by backward. Release all vocabulary-sized
                # tensors before the next action rather than retaining the last
                # iteration's locals while allocating its successor.
                del logits, log_distribution, target, logp, reference_logits
                del kl, lp, lq, reinforce, loss
                length = len(generation.prompt_ids) + len(generation.completion_ids)
                run.tokens += length
                run.completion_tokens += len(generation.completion_ids)
                total_reference_tokens += length
                total_rollout_tokens += length
        baseline = update_baseline(used_baseline, episode_returns, config.baseline_decay)
        run.step(step, {"episode_return_mean": sum(episode_returns) / len(episode_returns),
                        "reinforce_loss_sum": sum(policy_losses) / config.batch_size,
                        "exact_conditional_kl_mean": sum(kl_values) / len(kl_values),
                        "baseline_used": used_baseline, "baseline_after_update": baseline,
                        "episodes": config.batch_size, "reference_forward_tokens": total_reference_tokens,
                        "rollout_forward_token_lower_bound": total_rollout_tokens})
    return run.finish()
