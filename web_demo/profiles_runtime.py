"""Load locked v5 checkpoints for decisions made on live, public chat evidence."""
from __future__ import annotations

from collections import Counter
from importlib.util import find_spec
import json
from pathlib import Path
from threading import RLock

from research.src.artifacts import file_hash
from research.tesis_shared_budget_env import ACTIONS
from research.tesis_shared_budget_qwen import LETTERS, encode, padded_scoring

from .cases import ROOT


DEFAULT_RUN = ROOT / "reports/tesis_profiles_qwen_a100_v5"
METHODS = {
    "sft": ("SFT-ramas", "Qwen SFT"),
    "grpo": ("GRPO-temperatura", "Qwen GRPO temperatura"),
    "grpo_original": ("GRPO-ramas", "Qwen GRPO original"),
    "parent": ("Qwen-inicial", "Qwen inicial"),
}


class ProfilesError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def confined(folder, relative):
    if not isinstance(relative, str):
        raise ProfilesError("Ruta de checkpoint inválida.", 503)
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ProfilesError("El checkpoint sale de la carpeta de la corrida.", 503)
    return path


def actor_observation(observation, legal):
    from research.tesis_profiles_v5 import actor_observation as annotate
    return annotate(observation, legal)


class ProfilesRuntime:
    def __init__(self, device="cpu", run=DEFAULT_RUN):
        self.device = device
        self.run = Path(run).resolve()
        self.lock = RLock()
        self.backend = None
        self.loaded = set()
        self.checked = set()

    def descriptors(self):
        lock_path = self.run / "LOCK.json"
        if not lock_path.is_file():
            return [], "La corrida v5 aún no está disponible en esta carpeta."
        try:
            lock = read_json(lock_path)
            requirements = []
            missing = [name for name in ("torch", "transformers", "peft") if find_spec(name) is None]
            if missing:
                requirements.append("Faltan dependencias de inferencia: " + ", ".join(missing) + ".")
            try:
                hashes = read_json(self.run / "PLAN.json")["model_hashes"]
                if not isinstance(hashes, dict) or not hashes:
                    raise ValueError("model_hashes debe enumerar los archivos del modelo.")
                base = ROOT / "models/Qwen2.5-1.5B-Instruct"
                absent = [name for name in hashes if not confined(base, name).is_file()]
                if absent:
                    requirements.append("Faltan archivos del modelo base o del tokenizador "
                                        "Qwen2.5-1.5B-Instruct: " + ", ".join(absent) + ".")
            except (KeyError, TypeError, ValueError, OSError) as error:
                requirements.append(f"No se pudo comprobar el modelo base mediante PLAN.json. {error}")
            models = []
            for panel in ("validation_selected", "last_epoch"):
                for seed, methods in lock["panels"][panel].items():
                    for slug, (method, label) in METHODS.items():
                        if method not in methods:
                            continue
                        checkpoint = methods[method]
                        identity = f"{slug}-{seed}" + ("-terminal" if panel == "last_epoch" else "")
                        adapter = confined(self.run, checkpoint["adapter"])
                        exists = all((adapter / name).is_file() for name in
                                     ("adapter_model.safetensors", "adapter_config.json"))
                        available = exists and not requirements
                        problems = requirements + ([] if exists else ["Faltan los pesos del checkpoint seleccionado."])
                        reason = " ".join(problems) or None
                        models.append({
                            "id": identity, "label": label + (" terminal diagnóstico" if panel == "last_epoch" else ""),
                            "method": method, "seed": int(seed), "panel": panel, "epoch": checkpoint["epoch"],
                            "eligible": bool(checkpoint["eligible"]), "available": available, "reason": reason,
                            "adapter": checkpoint["adapter"], "weights": checkpoint["weights"],
                            "config": checkpoint["config"], "diagnostic": panel == "last_epoch",
                        })
            return models, " ".join(requirements) or None
        except (KeyError, TypeError, ValueError, OSError) as error:
            return [], f"No se pudo leer la selección congelada v5. {error}"

    def resolve(self, identity=None):
        models, reason = self.descriptors()
        return self._select(models, reason, identity)

    def _select(self, models, reason, identity=None):
        default = next((model for model in models if model["id"].startswith("sft-")
                        and model["panel"] == "validation_selected" and model["eligible"]
                        and model["epoch"] > 0 and model["available"]), None)
        default = default or next((model for model in models if model["available"] and model["eligible"]
                                   and model["epoch"] > 0 and not model["diagnostic"]), None)
        default = default or next((model for model in models if model["available"] and model["eligible"]
                                   and not model["diagnostic"]), None)
        default = default or next((model for model in models if model["available"] and not model["diagnostic"]), None)
        requested = next((model for model in models if model["id"] == identity), None) if identity else default
        if requested is None:
            raise ProfilesError(reason or "Elige un checkpoint v5 disponible.", 503 if not models else 400)
        fallback = ("Validación conservó el punto de partida. Se sirve ese mismo checkpoint congelado."
                    if requested["epoch"] == 0 and requested["method"] != METHODS["parent"][0] else None)
        return {**requested, "requested_id": requested["id"], "requested_eligible": requested["eligible"],
                "fallback": fallback}

    def results(self):
        result = {}
        for name, filename in (("reserved", "RESERVED_RESULTS.json"), ("validation", "VALIDATION.json"),
                               ("status", "STATUS.json")):
            path = self.run / filename
            if path.is_file():
                document = read_json(path)
                if name == "reserved":
                    result[name] = {"population": document.get("population")}
                    lock = self.run / "LOCK.json"
                    result[name]["selection_verified"] = (
                        lock.is_file() and document.get("selection_sha256") == file_hash(lock))
                    parser = document.get("parser", {})
                    result[name]["parser"] = {key: parser[key] for key in ("version", "roots", "visits", "summary") if key in parser}
                    semantic = document.get("semantic", {})
                    result[name]["semantic"] = {key: semantic[key] for key in ("protocol", "roots", "visits", "delta_matches_v5_minus_v4") if key in semantic}
                    result[name]["semantic"]["parsers"] = {
                        parser: {key: value for key, value in item.items()
                                 if key in ("roots", "visits", "matches", "complete_semantic_visits", "audit_counts")}
                        for parser, item in semantic.get("parsers", {}).items()
                    }
                    result[name]["panels"] = {
                        panel: {key: value for key, value in item.items() if key in ("summary", "comparisons")}
                        for panel, item in document.get("panels", {}).items()
                    }
                else:
                    result[name] = {key: value for key, value in document.items()
                                    if key in ("summary", "current_floor", "unsupported_ceiling", "stage",
                                               "updated_utc", "message")}
        return result

    def catalog(self):
        models, reason = self.descriptors()
        try:
            default = self._select(models, reason)["id"]
        except ProfilesError:
            default = None
        return {"models": models, "default_model": default, "reason": reason,
                "run": self.run.name, "device": self.device, "results": self.results(),
                "inference_scope": "Qwen elige la acción. El escritor y el lector de evidencia son los del entorno v5."}

    def decide(self, descriptor, observation, legal):
        if len(legal) == 1:
            return legal[0], {"source": "single_legal_action", "probabilities": {legal[0]: 1.},
                              "tokens": 0, "model": descriptor}
        if not descriptor["available"]:
            raise ProfilesError(descriptor["reason"] or "El modelo v5 no está disponible.", 503)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with self.lock:
            adapter = confined(self.run, descriptor["adapter"])
            signature = descriptor["weights"], descriptor["config"]
            if signature not in self.checked:
                if (file_hash(adapter / "adapter_model.safetensors") != descriptor["weights"]
                        or file_hash(adapter / "adapter_config.json") != descriptor["config"]):
                    raise ProfilesError("El checkpoint cambió desde la selección congelada.", 503)
                self.checked.add(signature)
            name = "checkpoint_" + descriptor["weights"][:20]
            if self.backend is None:
                if self.device == "cuda" and not torch.cuda.is_available():
                    raise ProfilesError("CUDA no está disponible.", 503)
                base = ROOT / "models/Qwen2.5-1.5B-Instruct"
                plan = read_json(self.run / "PLAN.json")
                if any(file_hash(confined(ROOT, filename)) != checksum
                       for filename, checksum in plan["sources"].items()):
                    raise ProfilesError("El código del entorno difiere de la corrida seleccionada.", 503)
                if any(file_hash(base / filename) != checksum for filename, checksum in plan["model_hashes"].items()):
                    raise ProfilesError("Los pesos base o el tokenizador cambiaron desde el experimento.", 503)
                tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
                tokens = [tokenizer.encode(LETTERS[action], add_special_tokens=False) for action in ACTIONS]
                if any(len(token) != 1 for token in tokens):
                    raise ProfilesError("El tokenizador no admite las acciones del experimento.", 503)
                model = AutoModelForCausalLM.from_pretrained(
                    base, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
                model = model.to(self.device)
                if self.device == "cpu":
                    model.float()
                model = PeftModel.from_pretrained(model, adapter, adapter_name=name,
                                                  is_trainable=False, local_files_only=True)
                self.backend = model.eval(), tokenizer, torch.tensor([token[0] for token in tokens], device=self.device)
                self.loaded.add(name)
            model, tokenizer, labels = self.backend
            if name not in self.loaded:
                model.load_adapter(adapter, adapter_name=name, is_trainable=False, local_files_only=True)
                self.loaded.add(name)
            model.set_adapter(name)
            model.eval()
            try:
                record = encode(tokenizer, actor_observation(observation, legal), legal, 2048)
            except ValueError as error:
                raise ProfilesError("La conversación supera el contexto del modelo. Abre una nueva visita.", 413) from error
            counter = Counter()
            with torch.inference_mode():
                scores = padded_scoring(model, [record], labels, tokenizer.pad_token_id, self.device, counter)[0]
                probabilities = scores.softmax(-1).tolist()
            action = ACTIONS[int(scores.argmax())]
            return action, {"source": "qwen", "model": descriptor, "tokens": counter["input_tokens"],
                            "device": self.device, "precision": str(next(model.parameters()).dtype),
                            "probabilities": {candidate: probabilities[ACTIONS.index(candidate)] for candidate in legal}}
