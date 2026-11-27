"""Fresh-trajectory RLOO on a finite public action catalogue, without SFT.

One optimizer step follows the complete group. Decisions are averaged within
trajectories, then trajectories within the group. Reference probabilities are
indexed by full decision identity, never by the repeated session phase alone.
"""
from __future__ import annotations

import hashlib
import math
from statistics import fmean, pstdev


def leave_one_out(returns):
    values = [float(x) for x in returns]
    if len(values) < 2 or not all(math.isfinite(x) for x in values):
        raise ValueError("RLOO needs at least two finite episode returns")
    if all(value == values[0] for value in values):
        return [0.] * len(values)
    return [value - math.fsum(other for j, other in enumerate(values) if j != i)
            / (len(values) - 1) for i, value in enumerate(values)]


def categorical_loss(logprobs, reference_logprobs, action_index, advantage, beta):
    """Exact categorical KL; no token-ratio surrogate or length weighting."""
    if logprobs.ndim != 1 or reference_logprobs.shape != logprobs.shape:
        raise ValueError("Categorical distributions must have identical vector shapes")
    if not 0 <= action_index < len(logprobs) or beta < 0:
        raise ValueError("Invalid action index or KL coefficient")
    probabilities = logprobs.exp()
    if logprobs.detach().equal(reference_logprobs):
        kl = logprobs.sum() * 0.  # Exact zero and exact zero gradient at p=q.
    else:
        kl = (probabilities * (logprobs - reference_logprobs)).sum()
    policy_loss = -float(advantage) * logprobs[action_index]
    return policy_loss + beta * kl, policy_loss, kl


def parameter_hashes(policy):
    """Hash rounded tensors actually used, separately for base and adapters."""
    output = {"base": hashlib.sha256(), "adapter": hashlib.sha256()}
    counts = {"base": 0, "adapter": 0}
    for name, parameter in policy.model.named_parameters():
        category = "adapter" if "lora_" in name else "base"
        # PEFT adds a wrapper path; canonicalize it for the same base identity.
        canonical_name = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
        tensor = parameter.detach().cpu().contiguous()
        output[category].update(canonical_name.encode())
        output[category].update(str(tensor.dtype).encode())
        output[category].update(str(tuple(tensor.shape)).encode())
        output[category].update(tensor.view(policy.torch.uint8).numpy().tobytes())
        counts[category] += parameter.numel()
    return {key: {"sha256": value.hexdigest(), "parameters": counts[key]}
            for key, value in output.items()}


def _decision_records(trajectories):
    seen = set()
    records = []
    for trajectory in trajectories:
        decisions = trajectory["decisions"]
        if not decisions:
            raise ValueError("An episode must contain real manager decisions")
        for decision in decisions:
            metadata = decision["metadata"]
            identifier = str(metadata["decision_id"])
            candidates = decision["candidates"]
            index = metadata["action_index"]
            if (metadata["decision_id"] != decision["decision_id"]
                    or metadata["candidates"] != candidates
                    or type(index) is not int or not 0 <= index < len(candidates)
                    or decision["action"] != candidates[index]
                    or not all(math.isfinite(x) for x in metadata["old_logprobs"])):
                raise ValueError("Decision identity, catalogue, probability or chosen action mismatch")
            if identifier in seen:
                raise ValueError("Repeated decision identity across sessions or rollouts")
            seen.add(identifier)
            if len(metadata["old_logprobs"]) != len(decision["candidates"]):
                raise ValueError("Collection catalogue differs from old categorical distribution")
            records.append((identifier, decision))
    return records


def optimize_group(actor, reference, trajectories, returns, optimizer, *, beta=.03):
    """Accumulate one complete on-policy group and take one clipped Adam step.

    Caller moves the frozen reader to CPU before this function. The reference
    visits the actor device only for no-grad forwards and returns to CPU before
    the differentiable actor forwards.
    """
    if len(trajectories) != len(returns):
        raise ValueError("One terminal return per collected trajectory required")
    torch = actor.torch
    if actor.policy_profile() != reference.policy_profile():
        raise ValueError("Actor/reference categorical protocols differ")
    advantages = leave_one_out(returns)
    records = _decision_records(trajectories)
    for _, decision in records:
        metadata = decision["metadata"]
        if (metadata["profile_sha256"] != actor.policy_profile()["profile_sha256"]
                or metadata["candidate_token_ids"] != actor.candidate_token_ids(decision["candidates"])):
            raise ValueError("Collected candidate tokenization or profile changed")
    references = {}
    reference.model.to(actor.device)
    try:
        with torch.no_grad(), reference.likelihood_mode():
            for identifier, decision in records:
                metadata = decision["metadata"]
                if len(decision["candidates"]) == 1:
                    if metadata["old_logprobs"] != [0.] or metadata["action_index"] != 0:
                        raise ValueError("A forced action must have categorical probability one")
                    continue
                distribution = reference.distribution(metadata["prompt_ids"], decision["candidates"])
                if not bool(torch.isfinite(distribution).all()):
                    raise ValueError("Nonfinite reference distribution")
                references[identifier] = distribution.detach().cpu()
    finally:
        reference.model.to("cpu")
        torch.cuda.empty_cache()
    optimizer.zero_grad(set_to_none=True)
    sums = {"loss": 0., "policy_loss": 0., "kl": 0.}
    max_replay_error = 0.
    for trajectory, advantage in zip(trajectories, advantages, strict=True):
        weight = 1. / (len(trajectories) * len(trajectory["decisions"]))
        for decision in trajectory["decisions"]:
            metadata = decision["metadata"]
            if len(decision["candidates"]) == 1:
                continue  # Exactly zero log probability, KL and gradient.
            identifier = str(metadata["decision_id"])
            with actor.likelihood_mode():
                distribution = actor.distribution(metadata["prompt_ids"], decision["candidates"])
                old = torch.tensor(metadata["old_logprobs"], device=actor.device)
                error = float((distribution.detach() - old).abs().max().cpu())
                max_replay_error = max(max_replay_error, error)
                if error > 2e-4:
                    raise ValueError(f"Collection/training distribution mismatch: {error}")
                ref = references[identifier].to(actor.device)
                losses = categorical_loss(distribution, ref, metadata["action_index"], advantage, beta)
                if not all(bool(torch.isfinite(value)) for value in losses):
                    raise ValueError("Nonfinite RLOO loss")
                (losses[0] * weight).backward()
            for key, value in zip(sums, losses, strict=True):
                sums[key] += float(value.detach().cpu()) * weight
    parameters = [p for p in actor.model.parameters() if p.requires_grad]
    if not parameters or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError("Missing trainable parameters or nonfinite gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {**sums, "grad_norm": float(norm.cpu()), "advantages": advantages,
            "returns": list(returns), "return_mean": fmean(returns),
            "return_std": pstdev(returns), "zero_signal": all(x == 0 for x in advantages),
            "decision_count": len(records), "max_old_distribution_error": max_replay_error,
            "reference_cache_keys": len(references)}
