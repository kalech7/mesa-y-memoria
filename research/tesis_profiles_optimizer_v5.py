"""Compare matched branch SFT, original GRPO and temperature-softened GRPO."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from statistics import mean

from research.src.artifacts import digest
from research.tesis_shared_budget_qwen import ACTIONS, batches, checkpoint, padded_scoring, status
from research.tesis_shared_budget_qwen_matched import (
    GRPO, PARENT, SFT, branch_loss, load_adapter, prepare, rank_rows,
    reference_logprobs, train_epoch, validate_model,
)


GRPO_V5 = "GRPO-temperatura"
ARMS = (SFT, GRPO, GRPO_V5)
TEMPERATURE_V5 = 4.0


def soften(logprobs, legal, temperature=TEMPERATURE_V5):
    """Temper a masked distribution while keeping forbidden actions excluded."""
    raw = (logprobs / temperature).masked_fill(~legal, -float("inf")).log_softmax(-1)
    return raw.masked_fill(~legal, 0.)


def tempered_loss(logits, legal, reward, *, old, anchor):
    """Use the historical exact-action objective with one consistent temperature."""
    return branch_loss(GRPO, logits / TEMPERATURE_V5, legal, reward, old=old, anchor=anchor)


def initial_diagnostics(examples, records, reference):
    """Measure the first policy gradient on shared TRAIN logits without forwards."""
    import torch

    legal = torch.tensor([example.legal for example in examples], device=reference.device)
    rewards = torch.tensor([example.rewards for example in examples], device=reference.device)
    weights = torch.tensor([example.weight for example in examples], device=reference.device)
    feasible = rewards.masked_fill(~legal, -torch.inf)
    best = legal & torch.isclose(feasible, feasible.max(-1, keepdim=True).values, rtol=0., atol=1e-6)
    tied = best.sum(-1) > 1
    all_tied = tied & (best.sum(-1) == legal.sum(-1))
    variants = {}
    for method, temperature in ((GRPO, 1.), (GRPO_V5, TEMPERATURE_V5)):
        logits = reference.detach().clone().requires_grad_(True)
        behavior = soften(logits.detach(), legal, temperature)
        masked = logits.masked_fill(~legal, -torch.inf)
        losses = branch_loss(GRPO, masked / temperature, legal, rewards, old=behavior, anchor=behavior)
        gradient = torch.autograd.grad(losses.sum(), logits)[0]
        mass = behavior.exp().masked_fill(~legal, 0.)
        norms = gradient.norm(dim=-1)
        max_value = (mass * rewards).sum(-1)
        fields = {
            "optimal_probability_mass": (mass * best).sum(-1),
            "entropy": -(mass * behavior).sum(-1),
            "logit_gradient_l2": norms,
            "relative_reward_gain_available": feasible.max(-1).values - max_value,
        }
        metrics = {}
        groups = defaultdict(list)
        raw_optimal_mass = (reference.exp().masked_fill(~legal, 0.) * best).sum(-1)
        for index, row in enumerate(records):
            probability = float(raw_optimal_mass[index])
            probability_bin = "under_0.01" if probability < .01 else "0.01_to_0.10" if probability < .1 else "at_least_0.10"
            groups[f"role:{row.get('role', 'unspecified')}"].append(index)
            groups[f"initial_optimal_mass:{probability_bin}"].append(index)
        groups["all"] = list(range(len(examples)))
        for name, positions in groups.items():
            selection = torch.tensor(positions, device=reference.device)
            selected_weights = weights[selection]
            selected_weights = selected_weights / selected_weights.sum()
            metrics[name] = {
                "groups": len(positions),
                "groups_with_optimal_ties": int(tied[selection].sum()),
                "groups_with_all_returns_tied": int(all_tied[selection].sum()),
                **{key: float((values[selection] * selected_weights).sum()) for key, values in fields.items()},
                "groups_with_logit_gradient_under_1e-6": int((norms[selection] < 1e-6).sum()),
            }
        variants[method] = {"temperature": temperature, "groups": metrics}
    return {
        "data_role": "train",
        "extra_model_forwards": 0,
        "gradient_unit": "unweighted per-group derivative with respect to original action logits",
        "aggregation": "training-weighted means within each reported group",
        "tie_tolerance": 1e-6,
        "methods": variants,
    }


def _train_tempered_epoch(model, optimizer, examples, schedule, reference, tokenizer, labels, device,
                          *, output, epoch, passes, total_weight):
    import torch

    counts, losses = Counter(), []
    model.train()
    for indices in schedule:
        batch = [examples[index] for index in indices]
        rewards = torch.tensor([example.rewards for example in batch], dtype=torch.float32, device=device)
        legal = torch.tensor([example.legal for example in batch], dtype=torch.bool, device=device)
        weights = torch.tensor([example.weight / total_weight for example in batch], device=device)
        anchor = soften(reference[indices], legal)
        old = None
        for _ in range(passes):
            logits = padded_scoring(model, [example.record for example in batch], labels,
                                    tokenizer.pad_token_id, device, counts)
            if old is None:
                old = soften(logits.detach(), legal)
            item_losses = tempered_loss(logits, legal, rewards, old=old, anchor=anchor)
            loss = (weights * item_losses).mean()
            if not bool(loss.isfinite()):
                raise ValueError("Tempered GRPO produced a nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            counts["optimizer_updates"] += 1
            losses.append(float(loss.detach()))
            if counts["optimizer_updates"] % 20 == 0:
                status(output, "training", arm=GRPO_V5, epoch=epoch, **counts)
    return {**counts, "mean_batch_loss": mean(losses)}


def fit_matched_v5(model, output, *, records, evaluation, tokenizer, labels, device, settings,
                   validation_rows, selection_rank, resume=False):
    """Apply three optimizers to the same parent, recorded branches and token budget.

    The caller locks the output and verifies source/model/data hashes. The v5
    intervention divides action logits by a fixed temperature during training,
    including the behavior distribution and reference used for KL. Greedy
    evaluation uses the same raw logits as both historical controls.
    """
    import torch

    output = Path(output)
    examples = prepare(records)
    seed, epochs, size = (settings[key] for key in ("seed", "epochs", "batch_size"))
    passes = settings.get("passes_per_batch", 2)
    if min(epochs, size, passes) < 1 or epochs > 999:
        raise ValueError("Need positive epochs, batch size and optimizer passes")
    schedules = []
    for epoch in range(1, epochs + 1):
        order = list(range(len(examples)))
        random.Random(seed + epoch).shuffle(order)
        schedules.append(list(batches(order, size)))

    common = output / "common-parent"
    parent = common / "checkpoint-000"
    common.mkdir(exist_ok=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Matched training requires trainable LoRA parameters")
    if parent.exists():
        if not resume:
            raise FileExistsError("The common parent already exists")
        load_adapter(model, parent / "adapter")
        initial = json.loads((parent / "LEDGER.json").read_text(encoding="utf-8"))[0]
    else:
        initial = validate_model(model, evaluation, tokenizer, labels, device, size, PARENT, validation_rows)
        initial["epoch"] = 0
        optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"])
        checkpoint(common, 0, model, optimizer, [initial])
    reference, reference_counts = reference_logprobs(
        model, [example.record for example in examples], labels, tokenizer.pad_token_id, device, size)
    diagnostic = initial_diagnostics(examples, records, reference)
    reference_rank = rank_rows(initial["rows"], selection_rank)
    selected = {
        PARENT: {"epoch": 0, "adapter": str((parent / "adapter").relative_to(output)),
                 "eligible": reference_rank is not None, "validation_rank": reference_rank},
    }
    rows, ledgers = list(initial["rows"]), {}
    for arm in ARMS:
        directory = output / arm
        directory.mkdir(exist_ok=True)
        load_adapter(model, parent / "adapter")
        optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"])
        ledger = []
        saved = sorted(directory.glob("checkpoint-[0-9][0-9][0-9]"))
        if saved:
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
            update = _train_tempered_epoch if arm == GRPO_V5 else train_epoch
            arguments = (model, optimizer, examples, schedules[epoch - 1], reference, tokenizer, labels, device)
            if arm != GRPO_V5:
                arguments = (model, optimizer, arm, *arguments[2:])
            completed = update(*arguments, output=output, epoch=epoch, passes=passes,
                               total_weight=mean(example.weight for example in examples))
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
    if any(totals[arm] != totals[SFT] for arm in ARMS):
        raise ValueError("The three optimizers received different token or update budgets")
    return {
        "rows": rows, "selected_checkpoints": selected,
        "accounting": {
            "decision_groups": len(examples), "logged_action_returns": sum(len(example.record[1]) for example in examples),
            "schedule_digest": digest(schedules),
            "input_and_return_digest": digest([example.__dict__ for example in examples]),
            "common_reference": reference_counts, "common_parent_validation": initial["validation_tokens"],
            "matched_optimization_totals": totals, "arms": ledgers,
            "online_environment_interactions": 0, "passes_per_batch": passes,
            "intervention": {"arm": GRPO_V5, "training_temperature": TEMPERATURE_V5,
                             "behavior_temperature": TEMPERATURE_V5, "reference_temperature": TEMPERATURE_V5,
                             "greedy_inference_temperature": 1.},
            "initial_gradient_diagnostics": diagnostic,
        },
    }
