from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from research import tesis_profiles_optimizer_v5 as optimizer
from research.tesis_shared_budget_qwen_matched import GRPO, SFT, branch_loss, prepare


def test_temperature_recovers_gradient_for_saturated_wrong_action():
    legal = torch.tensor([[True, True, False]])
    rewards = torch.tensor([[0., 1., 0.]])
    raw = torch.tensor([[16., 0., -torch.inf]])

    old = optimizer.soften(raw, legal, 1.)
    reference_logits = raw.clone().requires_grad_(True)
    loss = branch_loss(GRPO, reference_logits, legal, rewards, old=old, anchor=old)
    old_gradient = torch.autograd.grad(loss.sum(), reference_logits)[0]

    current = raw.clone().requires_grad_(True)
    softened = optimizer.soften(raw, legal)
    changed = optimizer.tempered_loss(current, legal, rewards, old=softened, anchor=softened)
    new_gradient = torch.autograd.grad(changed.sum(), current)[0]
    assert new_gradient[0, 1] < 0
    assert abs(new_gradient[0, 1]) > 1000 * abs(old_gradient[0, 1])
    assert new_gradient[0, 2] == 0 and old_gradient[0, 2] == 0
    assert bool(torch.isfinite(new_gradient).all())


def test_temperature_handles_ties_and_shifted_logits_without_forbidden_mass():
    legal = torch.tensor([[True, False, True]])
    rewards = torch.tensor([[.4, 20., .4]])
    logits = torch.tensor([[2., -torch.inf, -1.]], requires_grad=True)
    behavior = optimizer.soften(logits.detach(), legal)
    shifted = optimizer.soften(logits.detach() + 85., legal)
    torch.testing.assert_close(behavior, shifted)
    assert behavior[0, 1] == 0
    loss = optimizer.tempered_loss(logits, legal, rewards, old=behavior, anchor=behavior)
    gradient = torch.autograd.grad(loss.sum(), logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0.)


def test_initial_diagnostics_uses_recorded_reference_without_model_forward():
    records = [{
        "record": ([1, 2], ("PROCEED", "TEMPORAL_SCOPE")),
        "returns": (0., 1.), "weight": 1., "split": "train", "role": "ask_or_wait",
    }]
    examples = prepare(records)
    reference = torch.tensor([[0., -16., 0., 0., 0., 0., 0.]]).log_softmax(-1)
    legal = torch.tensor([examples[0].legal])
    reference = reference.masked_fill(~legal, -torch.inf).log_softmax(-1).masked_fill(~legal, 0.)
    diagnostic = optimizer.initial_diagnostics(examples, records, reference)
    old = diagnostic["methods"][GRPO]["groups"]["all"]
    new = diagnostic["methods"][optimizer.GRPO_V5]["groups"]["all"]
    assert diagnostic["extra_model_forwards"] == 0
    assert new["optimal_probability_mass"] > 1000 * old["optimal_probability_mass"]
    assert new["logit_gradient_l2"] > 1000 * old["logit_gradient_l2"]
    assert new["groups_with_all_returns_tied"] == 0
    assert "role:ask_or_wait" in diagnostic["methods"][GRPO]["groups"]


def test_matched_training_budgets_and_resume_preserve_three_arms(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 7, bias=False)
    tokenizer = SimpleNamespace(pad_token_id=0)
    records = [{
        "record": ([1, 2], ("PROCEED", "TEMPORAL_SCOPE")),
        "returns": (0., 1.), "weight": 1., "split": "train", "role": "ask_or_wait",
    }]
    settings = {"seed": 101, "epochs": 1, "batch_size": 1, "passes_per_batch": 2, "learning_rate": .01}

    def validate(model, evaluation, tokenizer, labels, device, size, method, rows):
        return {"rows": [{"method": method, "metric": 1.}], "validation_tokens": {"input_tokens": 2}}

    def reference(model, records, labels, pad, device, batch_size):
        legal = torch.tensor([[True, True, False, False, False, False, False]])
        logits = model(torch.ones(1, 1)).detach()
        values = logits.masked_fill(~legal, -torch.inf).log_softmax(-1).masked_fill(~legal, 0.)
        return values, {"forward_batches": 1}

    def save(output, epoch, model, optimizer, ledger):
        final = output / f"checkpoint-{epoch:03d}"
        final.mkdir()
        (final / "adapter").mkdir()
        torch.save(deepcopy(model.state_dict()), final / "adapter/weights.pt")
        torch.save(optimizer.state_dict(), final / "OPTIMIZER.pt")
        (final / "LEDGER.json").write_text(json.dumps(ledger))
        return final

    def load(model, adapter):
        model.load_state_dict(torch.load(adapter / "weights.pt", weights_only=True))

    def original_epoch(model, optimizer, arm, examples, schedule, reference, tokenizer, labels, device,
                       *, output, epoch, passes, total_weight):
        return {"optimizer_updates": 2, "input_tokens": 4, "padded_input_tokens": 4,
                "decisions": 2, "forward_batches": 2, "mean_batch_loss": 0.}

    def new_epoch(model, optimizer, examples, schedule, reference, tokenizer, labels, device, **kwargs):
        return original_epoch(model, optimizer, None, examples, schedule, reference, tokenizer, labels, device,
                              **kwargs)

    monkeypatch.setattr(optimizer, "validate_model", validate)
    monkeypatch.setattr(optimizer, "reference_logprobs", reference)
    monkeypatch.setattr(optimizer, "checkpoint", save)
    monkeypatch.setattr(optimizer, "load_adapter", load)
    monkeypatch.setattr(optimizer, "train_epoch", original_epoch)
    monkeypatch.setattr(optimizer, "_train_tempered_epoch", new_epoch)
    monkeypatch.setattr(optimizer, "status", lambda *args, **kwargs: None)
    kwargs = dict(records=records, evaluation=None, tokenizer=tokenizer, labels=None, device="cpu",
                  settings=settings, validation_rows=None, selection_rank=lambda rows: (True, 1.))
    result = optimizer.fit_matched_v5(model, tmp_path, **kwargs)
    assert set(result["selected_checkpoints"]) == {optimizer.PARENT, *optimizer.ARMS}
    totals = result["accounting"]["matched_optimization_totals"]
    assert totals[SFT] == totals[GRPO] == totals[optimizer.GRPO_V5]
    assert all(item["epoch"] == 0 for item in result["selected_checkpoints"].values())
    resumed = optimizer.fit_matched_v5(model, tmp_path, resume=True, **kwargs)
    assert resumed == result
