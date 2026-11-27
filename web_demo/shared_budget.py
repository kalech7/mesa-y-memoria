"""Run shared-budget development cases and the selected Qwen adapters on demand."""
from __future__ import annotations

from collections import Counter
from importlib.util import find_spec
import json
from pathlib import Path
from threading import RLock

from research.src.artifacts import canonical, file_hash
from research.tesis_shared_budget_bank import build_development
from research.tesis_shared_budget_env import ACTIONS, SharedBudgetEpisode, rule_action
from research.tesis_shared_budget_planner import outcome, public_state
from research.tesis_shared_budget_qwen import LETTERS, encode, padded_scoring

from .cases import ROOT


DEFAULT_RUN = ROOT / "reports/tesis_shared_budget_qwen_matched_a100_v2"
TEXT_PROMPT = "text-v1"
PUBLIC_CONTEXT_PROMPT = "public-context-v2"
MAX_ACTIONS = 3
CASE_TITLES = {
    "busy_contingent": "Poco tiempo y otra decisión incierta",
    "ready_regular": "Tiempo para responder y una rutina conocida",
    "busy_regular": "Poco tiempo y una rutina conocida",
    "ready_contingent": "Tiempo para responder y otra decisión incierta",
}
POLICY_NAMES = {
    "rule": "Regla pública", "planner": "Planificador", "frozen": "Qwen congelado",
    "sft": "Qwen con SFT", "grpo": "Qwen con GRPO",
}


class BudgetError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def development_cases():
    """Pick fixed illustrative worlds from development for the four public situations."""
    cases = {}
    rows = build_development()["validation"]
    for cue, title in CASE_TITLES.items():
        style = "refused" if cue.startswith("busy") else "direct"
        competing = cue.endswith("_contingent")
        story = min((row for row in rows if row["scenario_family"] == cue
                     and row["visits"][0]["style"] == style
                     and (row["visits"][1]["slot"] is not None) == competing),
                    key=lambda row: (row["root_id"], row["world_id"]))
        cases[cue] = (title, story)
    return cases


def _asset(parent, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Falta la ruta de un artefacto.")
    path = (parent / name).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError("Un artefacto sale de su carpeta de trabajo.")
    return path


class BudgetLab:
    def __init__(self, device, run=DEFAULT_RUN):
        self.device = device
        self.run = Path(run)
        self.cases = development_cases()
        self._lock = RLock()
        self._backend = None
        self._fingerprint = None
        self._loaded = set()

    def _configuration(self):
        """Discover only published adapters. Report a missing artifact as unavailable."""
        path = self.run / "LAB_MODELS.json"
        if not path.is_file():
            return None, {}, {}, "La corrida aún no publicó adaptadores seleccionados.", TEXT_PROMPT
        try:
            manifest = json.loads(path.read_bytes())
            if manifest["status"] != "completed":
                return None, {}, {}, "La corrida aún no publicó adaptadores seleccionados.", TEXT_PROMPT
            prompt_format = manifest.get("prompt_format", TEXT_PROMPT)
            if prompt_format not in (TEXT_PROMPT, PUBLIC_CONTEXT_PROMPT):
                raise ValueError("Formato de entrada desconocido.")
            base = _asset(ROOT, manifest["base_model"])
            if not isinstance(manifest["models"], dict):
                raise ValueError("Falta el catálogo de modelos.")
            selection = manifest.get("selection", {})
            if not isinstance(selection, dict):
                raise ValueError("La selección del experimento debe ser un objeto.")
            adapters = {}
            for name in ("sft", "grpo"):
                model = manifest["models"].get(name)
                if model is None:
                    continue
                adapter = _asset(self.run, model["adapter"])
                if (adapter / "adapter_model.safetensors").is_file() and (adapter / "adapter_config.json").is_file():
                    adapters[name] = (adapter, model["adapter_sha256"], selection.get(name, {}).get("epoch"))
            planner = {canonical(key): action for key, action in manifest.get("planner", [])}
        except (KeyError, TypeError, OSError, ValueError) as error:
            return None, {}, {}, f"No se pudo leer la publicación del experimento. {error}", TEXT_PROMPT
        if (not (base / "config.json").is_file() or not (base / "tokenizer.json").is_file()
                or not any(base.glob("*.safetensors"))):
            return None, {}, planner, "Faltan los archivos locales del modelo base.", prompt_format
        if any(find_spec(name) is None for name in ("torch", "transformers", "peft")):
            return None, {}, planner, "Para inferencia, abre el laboratorio con el entorno de investigación.", prompt_format
        if PUBLIC_CONTEXT_PROMPT == prompt_format and find_spec("research.tesis_shared_budget_qwen_context") is None:
            return None, {}, planner, "Falta el módulo que prepara el contexto público del experimento.", prompt_format
        return base, adapters, planner, None, prompt_format

    def catalog(self):
        base, adapters, planner, reason, _ = self._configuration()
        availability = {
            "rule": (True, None),
            "planner": (bool(planner), "El planificador se publica con la comparación."),
            "frozen": (base is not None, reason),
            "sft": ("sft" in adapters, reason or "Faltan los archivos seleccionados de SFT."),
            "grpo": ("grpo" in adapters, reason or "Faltan los archivos seleccionados de GRPO."),
        }
        return {
            "cases": [{"id": identity, "title": title} for identity, (title, _) in self.cases.items()],
            "base_model": base.name if base else None,
            "device": self.device,
            "policies": [
                {"id": policy, "label": label, "available": ready, "reason": explanation if not ready else None,
                 "epoch": adapters[policy][2] if policy in adapters else None}
                for policy, label in POLICY_NAMES.items()
                for ready, explanation in [availability[policy]]
            ],
        }

    def _decide(self, policy, observation, legal, configuration):
        if policy == "rule":
            return rule_action(observation, legal)
        base, adapters, planner, reason, prompt_format = configuration
        if policy == "planner":
            if not planner:
                raise BudgetError("Aún no se publicó el planificador.", status=503)
            action = planner.get(canonical(public_state(observation, legal)))
            return action if action in legal else rule_action(observation, legal)
        if base is None or (policy != "frozen" and policy not in adapters):
            raise BudgetError(reason or f"Falta el adaptador seleccionado de {POLICY_NAMES[policy]}.", status=503)
        if prompt_format == PUBLIC_CONTEXT_PROMPT:
            from research.tesis_shared_budget_qwen_context import with_public_context

            observation = with_public_context(observation)
        return self._infer(policy, observation, legal, base, adapters)

    def _infer(self, policy, observation, legal, base, adapters):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with self._lock:
            fingerprint = (base, tuple(adapters.items()))
            if self._fingerprint != fingerprint:
                self._backend = None
                self._loaded.clear()
                self._fingerprint = fingerprint
            if self._backend is None:
                if self.device == "cuda" and not torch.cuda.is_available():
                    raise BudgetError("CUDA no está disponible en el servidor del laboratorio.", status=503)
                precision = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
                tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
                tokens = [tokenizer.encode(LETTERS[action], add_special_tokens=False) for action in ACTIONS]
                if any(len(token) != 1 for token in tokens) or len({token[0] for token in tokens}) != len(ACTIONS):
                    raise BudgetError("El tokenizador no admite el menú de acciones del experimento.", status=503)
                model = AutoModelForCausalLM.from_pretrained(
                    base, local_files_only=True, torch_dtype=precision, attn_implementation="sdpa",
                ).to(self.device).eval()
                self._backend = tokenizer, torch.tensor([token[0] for token in tokens], device=self.device), model
            tokenizer, labels, model = self._backend
            if policy not in self._loaded and policy != "frozen":
                adapter, expected, _ = adapters[policy]
                if file_hash(adapter / "adapter_model.safetensors") != expected:
                    raise BudgetError(f"Las huellas del adaptador de {POLICY_NAMES[policy]} cambiaron.", status=503)
                if isinstance(model, PeftModel):
                    model.load_adapter(adapter, adapter_name=policy, local_files_only=True, is_trainable=False)
                else:
                    model = PeftModel.from_pretrained(
                        model, adapter, adapter_name=policy, local_files_only=True, is_trainable=False,
                    )
                    self._backend = tokenizer, labels, model
                self._loaded.add(policy)
            if policy != "frozen":
                model.set_adapter(policy)
            model.eval()
            record = encode(tokenizer, observation, legal, 2048)
            if policy == "frozen" and self._loaded:
                with model.disable_adapter(), torch.inference_mode():
                    scores = padded_scoring(model, [record], labels, tokenizer.pad_token_id, self.device, Counter())
            else:
                with torch.inference_mode():
                    scores = padded_scoring(model, [record], labels, tokenizer.pad_token_id, self.device, Counter())
            return ACTIONS[int(scores.argmax())]

    def _replay(self, story, prefix, policy, complete, configuration):
        env = SharedBudgetEpisode(story, question_budget=story["question_budget"])
        try:
            actions = []

            def apply(action):
                if env.done or action not in env.legal_actions:
                    raise BudgetError("Esa acción ya no está disponible con el presupuesto restante.")
                env.step(action)
                actions.append(action)

            for action in prefix:
                apply(action)
            if policy:
                if env.done:
                    raise BudgetError("La historia ya terminó.")
                while not env.done:
                    observation = {**env.observe(), "action_history": tuple(actions)}
                    apply(self._decide(policy, observation, tuple(env.legal_actions), configuration))
                    if not complete:
                        break
            if env.done:
                summary = env.summary()
                return {"actions": actions, "done": True, "metrics": outcome(summary),
                        "visits": [_visit(visit) for visit in summary["visits"]], "future": summary["future"],
                        "questions_remaining": summary["questions_remaining"]}
            observation = env.observe()
            return {"actions": actions, "done": False,
                    "visits": [_visit(visit) for visit in env._visits],
                    "observation": {key: observation[key] for key in ("visit", "query", "user_messages", "memory")},
                    "questions_remaining": observation["questions_remaining"], "legal_actions": env.legal_actions}
        finally:
            env.close()

    def play(self, body):
        if not isinstance(body, dict):
            raise BudgetError("Selecciona una historia de desarrollo.")
        identity = body.get("case_id")
        actions, policy, compare = body.get("actions", []), body.get("step_with"), body.get("compare")
        if not isinstance(identity, str) or identity not in self.cases:
            raise BudgetError("Historia de desarrollo desconocida.", status=404)
        if (not isinstance(actions, list) or len(actions) > MAX_ACTIONS
                or any(not isinstance(action, str) or action not in ACTIONS for action in actions)):
            raise BudgetError("Indica hasta tres acciones válidas.")
        if policy is not None and (not isinstance(policy, str) or policy not in POLICY_NAMES):
            raise BudgetError("Política desconocida.")
        _, story = self.cases[identity]
        configuration = self._configuration() if policy or compare is not None else None
        if compare is None:
            return self._replay(story, actions, policy, False, configuration)
        if (policy or not isinstance(compare, list) or not compare or len(compare) > len(POLICY_NAMES)
                or any(not isinstance(name, str) or name not in POLICY_NAMES for name in compare)
                or len(set(compare)) != len(compare) or actions):
            raise BudgetError("Selecciona políticas distintas para comparar desde el inicio.")
        return {"comparisons": [
            {"id": name, "label": POLICY_NAMES[name], "epoch": configuration[1].get(name, (None, None, None))[2],
             **self._replay(story, [], name, True, configuration)}
            for name in compare
        ]}


def _visit(visit):
    return {key: visit[key] for key in
            ("query", "opening_messages", "events", "closing_action", "outcome",
             "memory_after", "questions", "questions_remaining")}
