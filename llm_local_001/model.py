"""A strictly local, trainable causal language model (no inference services).

Heavy dependencies are imported only when a model is actually loaded, so the
resource audit and blocked-experiment reports work without transformers/PEFT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import random
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_hashes(path: str | Path) -> dict[str, str]:
    """Hash actual model artifacts, avoiding recursive metadata hashes."""
    root = Path(path)
    result = {}
    for item in sorted(root.rglob("*")):
        if item.is_file() and item.name != "local_checkpoint.json":
            digest = hashlib.sha256()
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result[str(item.relative_to(root))] = digest.hexdigest()
    return result


@dataclass
class Generation:
    prompt_ids: list[int]
    completion_ids: list[int]
    text: str
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    model_path: str
    device: str = "cuda"
    precision: str = "auto"
    lora: bool = True
    full_finetune: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    gradient_checkpointing: bool = False
    max_new_tokens: int = 192
    temperature: float = 1.0
    seed: int = 0


class LocalPolicy:
    """HF model whose own autoregressive tokens determine every decision.

    Loading the base never alters its architecture. Training adds PEFT LoRA
    adapters by default; full base-weight training requires an explicit opt-in.
    Neither GGUF conversion nor dependency installation is attempted.
    """

    def __init__(self, model_path: str, **options: Any):
        self.config = ModelConfig(model_path=str(Path(model_path).expanduser().resolve()), **options)
        root = Path(self.config.model_path)
        if not root.is_dir() or not (root / "config.json").is_file():
            raise ValueError("A real local HuggingFace model directory with config.json is required")
        if not any(root.glob("*.safetensors")) and not any(root.glob("pytorch_model*.bin")):
            raise ValueError("No trainable HF safetensors/PyTorch weights found; GGUF is not accepted")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if self.config.temperature <= 0:
            raise ValueError("Sampling temperature must be positive")
        self.seed(self.config.seed)
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but this PyTorch runtime cannot use the GPU")
        dtype = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}.get(self.config.precision)
        if self.config.precision == "auto":
            dtype = (torch.bfloat16 if torch.cuda.is_available() and
                     self.config.device.startswith("cuda") and torch.cuda.is_bf16_supported()
                     else torch.float32)
        if dtype is None:
            raise ValueError("precision must be auto, float32, float16 or bfloat16")
        # Persist the actual base-weight rounding used by B1. A CPU reference
        # must not reinterpret GPU 'auto' as a different precision.
        self.config.precision = str(dtype).removeprefix("torch.")
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS token")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            root, local_files_only=True, trust_remote_code=False, torch_dtype=dtype,
        ).to(self.config.device)
        if getattr(self.model.config, "is_encoder_decoder", False):
            raise ValueError("Only causal decoder language models are supported")
        if getattr(self.model, "is_quantized", False):
            raise ValueError("Quantized weights require a separately audited training setup")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.training_enabled = False
        self.origin_checkpoint: str | None = None
        self._base_artifact_hashes: dict[str, str] | None = None

    def seed(self, seed: int) -> None:
        random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def enable_training(self) -> None:
        if self.training_enabled:
            return
        if self.config.lora:
            from peft import LoraConfig, get_peft_model
            # The PEFT supported-architecture mapping chooses target modules;
            # an unsupported architecture fails rather than training a new head.
            self.model = get_peft_model(self.model, LoraConfig(
                task_type="CAUSAL_LM", r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha, lora_dropout=self.config.lora_dropout,
                bias="none",
            ))
        elif self.config.full_finetune:
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
        else:
            raise ValueError("Training needs LoRA or explicit full_finetune=True")
        self.training_enabled = True
        self._configure_training_memory()

    def _configure_training_memory(self) -> None:
        self.model.config.use_cache = False
        if self.config.gradient_checkpointing:
            # Recompute decoder activations rather than retaining all layers.
            # Non-reentrant checkpointing preserves normal autograd semantics.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()

    def parameter_counts(self) -> dict[str, Any]:
        return {
            "total": sum(p.numel() for p in self.model.parameters()),
            "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "training_scope": "lora_adapters_integrated_in_llm" if self.config.lora and self.training_enabled
            else "base_weights" if self.training_enabled else "none",
        }

    def _check_length(self, length: int) -> None:
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and length > limit:
            raise ValueError(f"Context length {length} exceeds model capacity {limit}; no silent truncation")

    def encode_prompt(self, prompt: str) -> list[int]:
        if getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            )
        else:
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not ids:
            raise ValueError("Empty tokenized prompt")
        return ids

    def supervised_generation(self, prompt: str, completion: str) -> Generation:
        prompt_ids = self.encode_prompt(prompt)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if not completion_ids or completion_ids[-1] != self.tokenizer.eos_token_id:
            completion_ids.append(self.tokenizer.eos_token_id)
        self._check_length(len(prompt_ids) + len(completion_ids))
        return Generation(prompt_ids, completion_ids, completion)

    def generate(self, prompt: str, sample: bool = False) -> Generation:
        from transformers import GenerationConfig

        ids = self.encode_prompt(prompt)
        self._check_length(len(ids) + self.config.max_new_tokens)
        tokens = self.torch.tensor([ids], device=self.device)
        self.model.eval()
        kwargs = {"temperature": self.config.temperature, "top_k": 0, "top_p": 1.0} if sample else {}
        # Discard inherited beam search, repetition penalties and token filters:
        # the sampling distribution must equal the logits used by REINFORCE.
        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens, do_sample=sample,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True, **kwargs,
        )
        with self.torch.no_grad():
            sequence = self.model.generate(
                input_ids=tokens, attention_mask=self.torch.ones_like(tokens),
                generation_config=generation_config,
            )
        completion = sequence[0, len(ids):].detach().cpu().tolist()
        return Generation(ids, completion, self.tokenizer.decode(completion, skip_special_tokens=True),
                          truncated=bool(completion and completion[-1] != self.tokenizer.eos_token_id),
                          metadata={"sampled": sample, "temperature": self.config.temperature,
                                    "created_at": utc_now()})

    def sequence_logits(self, generated: Generation, *, sampling_distribution: bool = False):
        """Differentiable logits at completion positions, including invalid JSON.

        Prompt tokens are masked by slicing the causal shift. RL uses the same
        unfiltered temperature distribution from which its tokens were sampled.
        """
        if not generated.prompt_ids or not generated.completion_ids:
            raise ValueError("Both prompt and completion token sequences must be nonempty")
        full = generated.prompt_ids + generated.completion_ids
        self._check_length(len(full))
        tokens = self.torch.tensor([full], device=self.device)
        start = len(generated.prompt_ids) - 1
        # Some causal backends can project only the requested hidden positions
        # through their vocabulary head. Keep the entire causal context while
        # avoiding a prompt-length x vocabulary logits allocation. PEFT wraps
        # forward with **kwargs, so inspect the underlying causal model.
        causal_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        supports_selection = "logits_to_keep" in inspect.signature(causal_model.forward).parameters
        kwargs = {}
        if supports_selection:
            kwargs["logits_to_keep"] = self.torch.arange(start, len(full) - 1, device=self.device)
        logits = self.model(input_ids=tokens, attention_mask=self.torch.ones_like(tokens),
                            use_cache=False, **kwargs).logits
        selected = (logits[0] if supports_selection else logits[0, start:-1, :]).float()
        if sampling_distribution:
            selected = selected / self.config.temperature
        return selected

    @contextmanager
    def likelihood_mode(self):
        """Differentiate the evaluation distribution with activation recomputation.

        HF Qwen/GPT decoder containers dispatch checkpointing only when their
        own ``training`` flag is true. Enabling that container alone leaves its
        attention, MLP, and adapter children in evaluation mode, including both
        module and functional dropout. Keep this scope through backward so that
        checkpoint recomputation sees precisely the same module flags.
        """
        previous = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            if self.config.gradient_checkpointing and self.training_enabled:
                model_type = getattr(self.model.config, "model_type", None)
                if model_type not in {"qwen2", "gpt2"}:
                    raise ValueError("Deterministic checkpointed likelihoods are audited only for qwen2/gpt2")
                for module in self.model.modules():
                    if getattr(module, "gradient_checkpointing", False):
                        # Assign directly: .train() would recursively enable
                        # functional attention dropout and LoRA dropout.
                        module.training = True
            yield
        finally:
            for module, training in previous:
                module.training = training

    def sequence_logprobs(self, generated: Generation, *, sampling_distribution: bool = False):
        logits = self.sequence_logits(generated, sampling_distribution=sampling_distribution)
        target = self.torch.tensor(generated.completion_ids, device=self.device)
        return self.torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)

    def base_fingerprint(self) -> dict[str, str]:
        if self._base_artifact_hashes is None:
            self._base_artifact_hashes = artifact_hashes(self.config.model_path)
        return dict(self._base_artifact_hashes)

    def save_checkpoint(self, destination: str | Path) -> dict[str, Any]:
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target, safe_serialization=True)
        self.tokenizer.save_pretrained(target)
        self.base_fingerprint()
        manifest = {"created_at": utc_now(), "config": asdict(self.config),
                    "parameters": self.parameter_counts(), "origin_checkpoint": self.origin_checkpoint,
                    "base_artifacts_sha256": self._base_artifact_hashes,
                    "artifacts_sha256": artifact_hashes(target)}
        (target / "local_checkpoint.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, trainable: bool = False,
                        device: str | None = None) -> "LocalPolicy":
        """Fresh B2/B3 branches from exactly the same audited B1 artifacts."""
        checkpoint = Path(checkpoint).resolve()
        manifest = json.loads((checkpoint / "local_checkpoint.json").read_text(encoding="utf-8"))
        if artifact_hashes(checkpoint) != manifest["artifacts_sha256"]:
            raise ValueError("Checkpoint artifact hash mismatch")
        config = dict(manifest["config"])
        if device is not None:
            config["device"] = device
        scope = manifest["parameters"]["training_scope"]
        if scope == "lora_adapters_integrated_in_llm":
            if artifact_hashes(config["model_path"]) != manifest["base_artifacts_sha256"]:
                raise ValueError("Base model changed since the B1 adapter was saved")
        if scope == "base_weights":
            config["model_path"] = str(checkpoint)
        obj = cls(**config)
        if scope == "lora_adapters_integrated_in_llm":
            from peft import PeftModel
            obj.model = PeftModel.from_pretrained(obj.model, checkpoint, is_trainable=trainable,
                                                 local_files_only=True)
            obj.training_enabled = trainable
            if trainable:
                obj._configure_training_memory()
        elif trainable:
            obj.enable_training()
        obj.origin_checkpoint = str(checkpoint)
        obj._base_artifact_hashes = manifest.get("base_artifacts_sha256")
        obj.model.eval()
        return obj
