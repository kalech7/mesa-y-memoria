"""Three local causal-LLM calls, each with its own immutable observed prefix."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from llm_local_001.model import LocalPolicy

PHASE_LIMITS = {"select": 64, "respond": 256, "write": 512, "reader": 256}
MAX_INPUT_TOKENS = 2048
OBSERVATION_KEYS = {"memory", "user_messages", "query", "reuse_regime"}


class InvalidOutput(ValueError):
    """A model format failure; never permission for another inference attempt."""


class SpecLocalPolicy(LocalPolicy):
    """Reuse the audited local backend, with this protocol's explicit LoRA modules."""

    def __init__(self, model_path, **options):
        cpu_threads = options.pop("cpu_threads", 2)
        options.setdefault("lora_rank", 16)
        options.setdefault("lora_alpha", 32)
        options.setdefault("lora_dropout", 0.0)
        options.setdefault("gradient_checkpointing", True)
        super().__init__(model_path, **options)
        self.torch.set_num_threads(cpu_threads)

    def enable_training(self):
        if self.training_enabled:
            return
        if not self.config.lora or self.config.full_finetune:
            raise ValueError("This protocol trains integrated LoRA adapters only")
        from peft import LoraConfig, get_peft_model
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        found = {name.rsplit(".", 1)[-1] for name, _ in self.model.named_modules()}
        if not set(targets) <= found:
            raise ValueError("G0: backbone does not expose all declared LoRA targets")
        self.model = get_peft_model(self.model, LoraConfig(
            task_type="CAUSAL_LM", r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha, lora_dropout=self.config.lora_dropout,
            target_modules=targets, bias="none"))
        self.training_enabled = True
        self._configure_training_memory()


def validate_action(phase, action):
    if not isinstance(action, dict):
        raise InvalidOutput("Expected one JSON object")
    if phase == "select":
        if set(action) != {"action", "slot"}:
            raise InvalidOutput("SELECT requires action and slot")
        if action["action"] == "PROCEED" and action["slot"] is None:
            return action
        if action["action"] == "ASK" and action["slot"] in {
                "TEMPORAL_SCOPE", "CURRENT_CONTEXT", "VALID_UNTIL", "GENERIC"}:
            return action
        raise InvalidOutput("Invalid SELECT action")
    if phase in {"respond", "reader"}:
        if set(action) != {"mode", "value", "branches", "evidence_ids"}:
            raise InvalidOutput("RESPOND requires mode, value, branches and evidence_ids")
        if not isinstance(action["branches"], list) or not isinstance(action["evidence_ids"], list):
            raise InvalidOutput("branches/evidence_ids must be lists")
        if not all(isinstance(x, str) for x in action["evidence_ids"]):
            raise InvalidOutput("Evidence references must be strings")
        mode = action["mode"]
        if mode == "ANSWER" and isinstance(action["value"], str) and bool(action["value"]) and not action["branches"]:
            return action
        if mode == "DEFER" and action["value"] is None and not action["branches"]:
            return action
        if mode == "CONDITIONALIZE" and action["value"] is None and action["branches"]:
            for branch in action["branches"]:
                if not isinstance(branch, dict) or set(branch) != {"condition", "value"}:
                    raise InvalidOutput("Each branch requires condition and value")
                if not isinstance(branch["condition"], (str, dict)) or not isinstance(branch["value"], str):
                    raise InvalidOutput("Invalid conditional branch")
            return action
        raise InvalidOutput("Invalid structured response")
    if phase == "write":
        if set(action) != {"operations"} or not isinstance(action["operations"], list):
            raise InvalidOutput("WRITE requires operations list")
        if not 1 <= len(action["operations"]) <= 2:
            raise InvalidOutput("WRITE permits one or two operations")
        for operation in action["operations"]:
            if not isinstance(operation, dict) or operation.get("op") not in {
                    "NOOP", "ADD", "SUPERSEDE", "MARK_UNRESOLVED"}:
                raise InvalidOutput("Unknown operation")
            if operation["op"] == "NOOP" and (len(action["operations"]) != 1 or set(operation) != {"op"}):
                raise InvalidOutput("NOOP must be alone")
        # Detailed record/reference validation belongs to the atomic executor.
        return action
    raise ValueError("Unknown phase")


class LLMController:
    def __init__(self, backend, prompt_root=None, sample=False, prompt_variant=None):
        self.backend = backend
        self.prompt_root = Path(prompt_root or Path(__file__).resolve().parents[2] / "prompts")
        self.sample = sample
        if prompt_variant not in {None, "B0", "B0-P"}:
            raise ValueError("Unknown registered prompt variant")
        self.prompt_variant = prompt_variant
        self.phase_records = []

    def prompt(self, phase, observation, instruction_suffix=None):
        if not isinstance(observation, dict) or not set(observation) <= OBSERVATION_KEYS:
            raise ValueError("Actor observation contains forbidden or private fields")
        instruction = (self.prompt_root / f"{phase}.txt").read_text(encoding="utf-8")
        if self.prompt_variant == "B0-P":
            instruction = instruction.rstrip() + (
                "\nComprueba internamente: evidencia disponible, variable faltante, alcance temporal o contextual, "
                "decisión y procedencia de escritura. Distingue evidencia del usuario de hipótesis. "
                "Entrega solo el JSON de esta fase; no incluyas razonamiento ni diagnóstico en la salida.")
        if instruction_suffix:
            instruction = instruction.rstrip() + "\n" + instruction_suffix
        return instruction.rstrip() + "\n\nOBSERVACION\n" + json.dumps(
            observation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def call(self, phase, observation, instruction_suffix=None):
        prompt = self.prompt(phase, observation, instruction_suffix=instruction_suffix)
        prompt_ids = self.backend.encode_prompt(prompt)
        if len(prompt_ids) > MAX_INPUT_TOKENS:
            raise ValueError("INPUT_TOKEN_LIMIT: silent truncation is prohibited")
        previous_limit = self.backend.config.max_new_tokens
        self.backend.config.max_new_tokens = PHASE_LIMITS[phase]
        try:
            generated = self.backend.generate(prompt, sample=self.sample)
        finally:
            self.backend.config.max_new_tokens = previous_limit
        record = {"phase": phase, "prompt": prompt, "generation": generated,
                  "raw_output": generated.text, "parsed_action": None,
                  "technical_status": "completed", "retry_count": 0,
                  "parse_status": "not_attempted", "schema_status": "not_attempted",
                  "truncated": bool(generated.truncated), "fallback_used": False, "fallback_reason": None}
        self.phase_records.append(record)
        try:
            if generated.truncated:
                raise InvalidOutput("OUTPUT_TOKEN_LIMIT: incomplete generation")
            action = json.loads(generated.text)
            record["parse_status"] = "passed"
            validate_action(phase, action)
            record["schema_status"] = "passed"
        except (ValueError, TypeError) as exc:
            record["technical_status"] = "invalid_output"
            if record["parse_status"] == "passed":
                record["schema_status"] = "failed"
            elif not generated.truncated:
                record["parse_status"] = "failed"
            record["error"] = str(exc)
            raise InvalidOutput(str(exc)) from exc
        record["parsed_action"] = action
        return action

    def select(self, observation):
        return self.call("select", observation)

    def select_forced_ask(self, observation):
        action = self.call("select", observation, instruction_suffix=(
            'En esta condición debes realizar exactamente una pregunta. Devuelve action:"ASK" y elige '
            "un único slot permitido usando la misma evidencia visible. PROCEED no está permitido."))
        if action["action"] != "ASK":
            self.phase_records[-1]["technical_status"] = "invalid_output"
            self.phase_records[-1]["error"] = "C-ALWAYS requires ASK with one model-selected slot"
            raise InvalidOutput(self.phase_records[-1]["error"])
        return action

    def respond(self, observation):
        return self.call("respond", observation)

    def write(self, observation):
        return self.call("write", observation)

    def serialized_records(self):
        return [{**record, "generation": asdict(record["generation"])} for record in self.phase_records]
