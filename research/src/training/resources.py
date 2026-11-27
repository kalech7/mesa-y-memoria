"""Measured context ceiling for the local GPU, without task/test labels."""
import time
from llm_local_001.model import Generation


def probe_context_capacity(policy, *, input_tokens, output_tokens):
    started = time.monotonic()
    torch = policy.torch
    token = policy.tokenizer.encode(" pasta", add_special_tokens=False)[0]
    generated = Generation([token] * input_tokens, [token] * output_tokens, "synthetic capacity probe")
    policy.model.zero_grad(set_to_none=True)
    with policy.likelihood_mode():
        loss = -policy.sequence_logprobs(generated).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite context capacity loss")
        loss.backward()
    grads = [p.grad for p in policy.model.parameters() if p.requires_grad and p.grad is not None]
    finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
    policy.model.zero_grad(set_to_none=True)
    if not finite:
        raise RuntimeError("Nonfinite or absent capacity probe adapter gradients")
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
    return {"status": "passed", "purpose": "synthetic_memory_capacity_only", "input_tokens": input_tokens,
            "output_tokens": output_tokens, "finite_adapter_gradients": finite,
            "elapsed_seconds": time.monotonic() - started}


def probe_generation_capacity(policy, *, input_tokens, output_tokens):
    """Measure a full unshortened synthetic context and forced output length.

    This profiles cache capacity only. It is not evidence that long synthetic
    repetitions preserve the semantics of real prompts or produce valid JSON.
    """
    from transformers import GenerationConfig
    torch = policy.torch
    token = policy.tokenizer.encode(" pasta", add_special_tokens=False)[0]
    inputs = torch.tensor([[token] * input_tokens], device=policy.device)
    config = GenerationConfig(do_sample=False, min_new_tokens=output_tokens,
        max_new_tokens=output_tokens, use_cache=True,
        eos_token_id=policy.tokenizer.eos_token_id, pad_token_id=policy.tokenizer.pad_token_id,
        bos_token_id=policy.tokenizer.bos_token_id)
    policy.model.eval()
    if str(policy.device).startswith("cuda"):
        torch.cuda.synchronize(policy.device)
    started = time.monotonic()
    with torch.no_grad():
        generated = policy.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=config)
    if str(policy.device).startswith("cuda"):
        torch.cuda.synchronize(policy.device)
    measured_output = generated.shape[-1] - input_tokens
    if measured_output != output_tokens:
        raise RuntimeError("Capacity generation stopped before the declared output length")
    return {"status": "passed", "purpose": "synthetic_generation_capacity_only",
            "input_tokens": input_tokens, "output_tokens": measured_output,
            "elapsed_seconds": time.monotonic() - started}
