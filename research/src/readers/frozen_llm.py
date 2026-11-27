"""A common frozen base reader; neither earlier dialogue nor adapters are accepted."""
from research.src.policy.llm_controller import LLMController, SpecLocalPolicy


class FrozenLLMReader:
    reader_id = "frozen_base_llm"

    def __init__(self, backend, prompt_root=None):
        if hasattr(backend.model, "peft_config") or getattr(backend, "training_enabled", False):
            raise ValueError("Secondary reader must be the frozen base, without controller adapters")
        if any(parameter.requires_grad for parameter in backend.model.parameters()):
            raise ValueError("Reader parameters must be frozen")
        self.controller = LLMController(backend, prompt_root=prompt_root, sample=False)

    @classmethod
    def load(cls, model_path, device="cpu", precision="auto", prompt_root=None):
        return cls(SpecLocalPolicy(model_path, device=device, precision=precision, lora=False), prompt_root)

    def read(self, memory, query):
        self.controller.phase_records.clear()
        try:
            return self.controller.call("reader", {"memory": memory, "query": query})
        finally:
            # S3 calls never accumulate another user's memory or earlier prefix.
            self.controller.phase_records.clear()
