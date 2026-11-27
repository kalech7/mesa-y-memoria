"""Matched SFT and exact-action GRPO updates on a shared, frozen TRAIN bank."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import random
from statistics import mean

from research.src.artifacts import digest
from research.tesis_shared_budget_qwen import ACTIONS, atomic_json, batches, checkpoint
from research.tesis_shared_budget_qwen import model_policy, padded_scoring, status


PARENT = "Qwen-inicial"
SFT = "SFT-ramas"
GRPO = "GRPO-ramas"
ARMS = (SFT, GRPO)
CLIP = .20
KL_WEIGHT = .02


@dataclass(frozen=True)
class Example:
    record: tuple
    rewards: tuple
    legal: tuple
    weight: float


def prepare(records):
    """Align exact branch returns with the seven masked action logits."""
    prepared = []
    for row in records:
        ids, options = row["record"]
        returns = tuple(float(value) for value in row["returns"])
        weight = float(row["weight"])
        if (not ids or len(options) < 2 or len(set(options)) != len(options)
                or not set(options).issubset(ACTIONS) or len(options) != len(returns)):
            raise ValueError("Each TRAIN prompt needs distinct legal actions and one return per action")
        if not all(isfinite(value) for value in returns) or not isfinite(weight) or weight <= 0:
            raise ValueError("TRAIN weights must be positive and branch returns finite")
        if row.get("split", "train") != "train":
            raise ValueError("Both optimizers accept TRAIN decisions only")
        rewards = dict(zip(options, returns, strict=True))
        prepared.append(Example((list(ids), tuple(options)), tuple(rewards.get(a, 0.) for a in ACTIONS),
                                tuple(a in rewards for a in ACTIONS), weight))
    if not prepared:
        raise ValueError("At least one paired TRAIN group is required")
    return prepared


def branch_loss(arm, logits, legal, reward, *, old=None, anchor=None):
    """Return one loss per slate, scoring every legal action with the same policy."""
    import torch

    raw = logits.log_softmax(dim=-1)
    if arm == SFT:
        eligible_reward = reward.masked_fill(~legal, -torch.inf)
        best = legal & torch.isclose(eligible_reward, eligible_reward.max(-1, keepdim=True).values,
                                     rtol=0., atol=1e-6)
        # SFT rewards the probability of taking any optimal action under the
        # same returns and weights supplied to GRPO.
        return -raw.masked_fill(~best, -torch.inf).logsumexp(dim=-1)
    if arm != GRPO or old is None or anchor is None:
        raise ValueError("Branch GRPO requires its behavior probabilities and the common parent")
    logp = raw.masked_fill(~legal, 0.)
    mass = old.exp().masked_fill(~legal, 0.)
    centered = reward - (mass * reward).sum(-1, keepdim=True)
    ratio = (logp - old).clamp(-30., 30.).exp()
    objective = torch.minimum(ratio * centered, ratio.clamp(1. - CLIP, 1. + CLIP) * centered)
    current_mass = raw.exp()
    kl = (current_mass * (logp - anchor)).sum(dim=-1)
    return -(mass * objective).sum(dim=-1) + KL_WEIGHT * kl


def load_adapter(model, adapter):
    """Restore a default LoRA adapter without copying or reloading the base."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    values = load_file(str(adapter / "adapter_model.safetensors"))
    state = set_peft_model_state_dict(model, values)
    if state.unexpected_keys:
        raise ValueError(f"Adapter contains unexpected keys: {state.unexpected_keys}")
    missing_lora = [name for name in state.missing_keys if ".lora_" in name]
    if missing_lora:
        raise ValueError(f"Adapter is missing LoRA weights: {missing_lora}")


def rank_rows(rows, selection_rank):
    rank = selection_rank(rows)
    if rank is None:
        return None
    rank = tuple(rank)
    if not rank:
        raise ValueError("Checkpoint ranking needs at least one criterion")
    return None if rank[0] is False else rank


def validate_model(model, evaluation, tokenizer, labels, device, size, method, validation_rows):
    counter = Counter()
    policy = model_policy(model, evaluation, tokenizer, labels, device, size, counter)
    return {"rows": validation_rows(policy, method), "validation_tokens": dict(counter)}


def reference_logprobs(model, records, labels, pad, device, batch_size):
    import torch

    count = Counter()
    model.eval()
    # A normal detached tensor can safely participate in the gradient-bearing KL
    # computation later. Inference-mode tensors cannot be saved for backward.
    with torch.no_grad():
        reference = torch.cat([
            padded_scoring(model, subset, labels, pad, device, count).log_softmax(dim=-1)
            for subset in batches(records, batch_size)
        ])
    return reference.masked_fill(~reference.isfinite(), 0.), dict(count)


def train_epoch(model, optimizer, arm, examples, schedule, reference, tokenizer, labels, device,
                *, output, epoch, passes, total_weight):
    import torch

    counts, losses = Counter(), []
    pad = tokenizer.pad_token_id
    model.train()
    for indices in schedule:
        batch = [examples[index] for index in indices]
        records = [example.record for example in batch]
        reward = torch.tensor([example.rewards for example in batch], dtype=torch.float32, device=device)
        legal = torch.tensor([example.legal for example in batch], dtype=torch.bool, device=device)
        weights = torch.tensor([example.weight / total_weight for example in batch], device=device)
        anchor = reference[indices]
        old = None
        for _ in range(passes):
            logits = padded_scoring(model, records, labels, pad, device, counts)
            if arm == GRPO and old is None:
                old = logits.detach().log_softmax(dim=-1).masked_fill(~legal, 0.)
            item_losses = branch_loss(arm, logits, legal, reward, old=old, anchor=anchor)
            loss = (weights * item_losses).mean()
            if not bool(loss.isfinite()):
                raise ValueError(f"{arm} produced a nonfinite masked loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            counts["optimizer_updates"] += 1
            losses.append(float(loss.detach()))
            if counts["optimizer_updates"] % 20 == 0:
                status(output, "training", arm=arm, epoch=epoch, **counts)
    return {**counts, "mean_batch_loss": mean(losses)}


def fit_matched(model, output, *, records, evaluation, tokenizer, labels, device, settings,
                validation_rows, selection_rank, resume=False):
    """Train both arms from the same PEFT parameters and select on validation.

    Callers hold the output lock and freeze split/model/code hashes before
    entering. Exact recorded returns are used offline with a fixed continuation.
    The shared reference is scored once. Both arms perform identical
    gradient-bearing passes and process the same tokens in the same order.
    """
    import torch

    output = Path(output)
    examples = prepare(records)
    seed, epochs, size = (settings[key] for key in ("seed", "epochs", "batch_size"))
    passes = settings.get("passes_per_batch", 2)
    if min(epochs, size, passes) < 1:
        raise ValueError("Need a positive number of epochs, batch size, and actions passes")
    schedules = []
    for epoch in range(1, epochs + 1):
        order = list(range(len(examples)))
        random.Random(seed + epoch).shuffle(order)
        schedules.append(list(batches(order, size)))

    source_records = [example.record for example in examples]
    common = output / "common-parent"
    parent = common / "checkpoint-000"
    common.mkdir(exist_ok=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Matched training requires trainable LoRA parameters")
    if parent.exists():
        if not resume:
            raise FileExistsError("Common parent already exists")
        load_adapter(model, parent / "adapter")
        initial = json.loads((parent / "LEDGER.json").read_text(encoding="utf-8"))[0]
    else:
        status(output, "evaluating_common_parent")
        initial = validate_model(model, evaluation, tokenizer, labels, device, size, PARENT, validation_rows)
        initial["epoch"] = 0
        optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"])
        checkpoint(common, 0, model, optimizer, [initial])
    status(output, "scoring_common_reference")
    reference, reference_counts = reference_logprobs(model, source_records, labels,
                                                       tokenizer.pad_token_id, device, size)
    reference_rank = rank_rows(initial["rows"], selection_rank)
    selected = {PARENT: {"epoch": 0, "adapter": str((parent / "adapter").relative_to(output)),
                         "eligible": reference_rank is not None, "validation_rank": reference_rank}}
    rows = list(initial["rows"])
    ledgers = {}
    for arm in ARMS:
        directory = output / arm
        directory.mkdir(exist_ok=True)
        load_adapter(model, parent / "adapter")
        optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"])
        ledger = []
        if saved := sorted(directory.glob("checkpoint-[0-9][0-9][0-9]")):
            if not resume:
                raise FileExistsError(f"Checkpoints already exist for {arm}")
            last = saved[-1]
            load_adapter(model, last / "adapter")
            optimizer.load_state_dict(torch.load(last / "OPTIMIZER.pt", map_location="cpu", weights_only=True))
            ledger = json.loads((last / "LEDGER.json").read_text(encoding="utf-8"))
            if ([entry["epoch"] for entry in ledger] != list(range(1, len(ledger) + 1))
                    or last.name != f"checkpoint-{len(ledger):03d}" or len(ledger) > epochs):
                raise ValueError(f"Invalid checkpoint ledger for {arm}")
        for epoch in range(len(ledger) + 1, epochs + 1):
            torch.manual_seed(seed + epoch)
            completed = train_epoch(
                model, optimizer, arm, examples, schedules[epoch - 1], reference, tokenizer, labels, device,
                output=output, epoch=epoch, passes=passes, total_weight=mean(example.weight for example in examples),
            )
            status(output, "evaluating_checkpoint", arm=arm, epoch=epoch)
            measured = validate_model(model, evaluation, tokenizer, labels, device, size, arm, validation_rows)
            ledger.append({"epoch": epoch, **completed, **measured})
            checkpoint(directory, epoch, model, optimizer, ledger)
            status(output, "checkpoint_saved", arm=arm, epoch=epoch)

        candidate, candidate_rank, chosen = initial, reference_rank, parent
        for entry in ledger:
            rank = rank_rows(entry["rows"], selection_rank)
            if rank is not None and (candidate_rank is None or rank > candidate_rank):
                candidate, candidate_rank = entry, rank
                chosen = directory / f"checkpoint-{entry['epoch']:03d}"
        selected[arm] = {
            "epoch": candidate["epoch"], "adapter": str((chosen / "adapter").relative_to(output)),
            "eligible": candidate_rank is not None, "validation_rank": candidate_rank,
        }
        rows.extend({**row, "method": arm} for row in candidate["rows"])
        ledgers[arm] = ledger

    budget_keys = ("optimizer_updates", "input_tokens", "padded_input_tokens", "decisions", "forward_batches")
    totals = {arm: {key: sum(entry[key] for entry in ledgers[arm]) for key in budget_keys} for arm in ARMS}
    if totals[SFT] != totals[GRPO]:
        raise ValueError("SFT and GRPO received different completed optimization budgets")
    return {
        "rows": rows,
        "selected_checkpoints": selected,
        "accounting": {
            "decision_groups": len(examples),
            "logged_action_returns": sum(len(example.record[1]) for example in examples),
            "schedule_digest": digest(schedules),
            "input_and_return_digest": digest([example.__dict__ for example in examples]),
            "common_reference": reference_counts,
            "common_parent_validation": initial["validation_tokens"],
            "matched_optimization_totals": totals,
            "arms": ledgers,
            "online_environment_interactions": 0,
            "passes_per_batch": passes,
        },
    }
