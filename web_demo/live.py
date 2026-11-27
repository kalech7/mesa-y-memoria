"""Execute the same simulator and controller used to record the comparison."""
from __future__ import annotations

from importlib.util import find_spec
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
import json

from .cases import BASE, MODELS, PROMPTS, RUN


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class LiveInference:
    def __init__(self, device):
        self.device = device
        self._backend = None
        self.compute_precision = None
        self._loaded = set()
        self._lock = Lock()

    def status(self):
        missing = [name for name in ("torch", "transformers", "peft") if find_spec(name) is None]
        if missing:
            return False, "Faltan dependencias de inferencia (" + ", ".join(missing) + "). Usa el entorno de investigación."
        if not (BASE / "model.safetensors").is_file() or any(
            not (model.path / "adapter_model.safetensors").is_file() for model in MODELS.values()
        ):
            return False, "Faltan los pesos locales de Qwen o algún adaptador LoRA."
        return True, None

    def _verify_assets(self):
        from llm_local_001.model import artifact_hashes
        from research.src.artifacts import file_hash

        base_hashes = artifact_hashes(BASE)
        # Copied weights can omit HF's download locks. Checkpoint manifests
        # historically hashed these transient files alongside real assets.
        def without_download_locks(hashes):
            return {name: digest for name, digest in hashes.items()
                    if not (name.startswith(".cache/huggingface/download/") and name.endswith(".lock"))}

        selection = {row["method"]: row for row in read_json(RUN / "SELECTION.json")["rows"]}
        for model in MODELS.values():
            manifest_path = model.path / "local_checkpoint.json"
            manifest = read_json(manifest_path)
            selected = selection[model.method]
            if file_hash(manifest_path) != selected["checkpoint_manifest_sha256"]:
                raise ValueError(f"Manifiesto del adaptador modificado: {model.method}.")
            if manifest["artifacts_sha256"] != artifact_hashes(model.path):
                raise ValueError(f"Archivos del adaptador modificados: {model.method}.")
            if manifest["artifacts_sha256"]["adapter_model.safetensors"] != selected["adapter_sha256"]:
                raise ValueError(f"El adaptador no coincide con la selección: {model.method}.")
            if without_download_locks(base_hashes) != without_download_locks(manifest["base_artifacts_sha256"]):
                raise ValueError(f"Los archivos de Qwen difieren del manifiesto: {model.method}.")
            if manifest["parameters"]["training_scope"] != "lora_adapters_integrated_in_llm":
                raise ValueError(f"Se esperaba un adaptador LoRA: {model.method}.")

    def _controller(self, model):
        from peft import PeftModel
        from research.src.policy.llm_controller import LLMController, SpecLocalPolicy

        config = read_json(model.path / "local_checkpoint.json")["config"]
        if self._backend is None:
            self._verify_assets()
            config["model_path"] = str(BASE)
            config["device"] = self.device
            backend = SpecLocalPolicy(**config, cpu_threads=4 if self.device == "cpu" else 2)
            # Keep the checkpoint's BF16-rounded backbone values and use FP32
            # kernels for portable CPU inference, including on AVX2.
            if self.device == "cpu":
                backend.model.float()
            self.compute_precision = str(next(backend.model.parameters()).dtype).removeprefix("torch.")
            backend.model = PeftModel.from_pretrained(
                backend.model, model.path, adapter_name=model.id, is_trainable=False, local_files_only=True)
            self._loaded.add(model.id)
            self._backend = backend
        if model.id not in self._loaded:
            if config["precision"] != self._backend.config.precision:
                raise ValueError(f"Precisión incompatible con la base compartida: {model.method}.")
            if model.id.startswith("comidas-"):
                from llm_local_001.model import artifact_hashes

                if artifact_hashes(model.path) != read_json(model.path / "local_checkpoint.json")["artifacts_sha256"]:
                    raise ValueError("Los archivos del adaptador entrenado cambiaron.")
            self._backend.model.load_adapter(
                model.path, adapter_name=model.id, is_trainable=False, local_files_only=True)
            self._loaded.add(model.id)
        self._backend.model.set_adapter(model.id)
        self._backend.model.requires_grad_(False)
        self._backend.model.eval()
        self._backend.config.temperature = config["temperature"]
        return LLMController(self._backend, prompt_root=PROMPTS, sample=False)

    def compare(self, episode, models, *, verify_writes=False):
        from research.competence_evaluation import _run_trajectory

        # Only one request may switch adapters or change torch RNG at a time.
        with self._lock, TemporaryDirectory(prefix="tesis-memoria-") as directory:
            rows = {}
            for model in models:
                controller = self._controller(model)
                if verify_writes:
                    from research.tesis_memory_guard import GuardedWriter

                    controller = GuardedWriter(controller)
                rows[model.id] = _run_trajectory(
                    controller, episode, Path(directory) / f"{model.id}.sqlite",
                    condition="O2", method=model.method,
                )
                if verify_writes:
                    rows[model.id]["guard_decision"] = controller.last_write
            return rows

    @contextmanager
    def chat_controller(self, model):
        from .chat_inference import chat_controller
        from .chat_history import MemoryMealPolicy

        with self._lock:
            specialization = model.id.startswith("comidas-")
            if specialization:
                self._controller(model)
            controller = self._controller(MODELS["sft"] if specialization else model)
            chat = chat_controller(controller)
            chat.meal_policy = MemoryMealPolicy(controller.backend, model.id) if specialization else None
            yield chat
