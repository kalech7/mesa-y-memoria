"""Unfiltered per-phase sampling with identical actor/reference likelihoods."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

from research.src.policy.llm_controller import SpecLocalPolicy

VERSION = "phase_temperature_policy_v1"
DEFAULT_PHASE_TEMPERATURES = MappingProxyType({"select": 2.0, "respond": .6, "write": .35})
PROMPT_ROOT = Path(__file__).resolve().parent / "targeted_prompts_v2"
# Unique opening instruction text, never a task label from an observation.
PHASE_MARKERS = MappingProxyType({
    "select": "Selecciona una acción JSON:",
    "respond": "Responde exclusivamente JSON con mode",
    "write": 'Devuelve {"operations":[...]} con máximo dos operaciones',
})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


class PhaseTemperaturePolicy(SpecLocalPolicy):
    """Keep long WRITE samples stable while independently exploring SELECT.

    This changes only unfiltered sampling temperature; greedy generation,
    generated tokens and validation remain those of the audited local backend.
    The checkpoint's legacy global temperature is normalized to base 1.0.
    """

    def __init__(self, model_path, *, phase_temperatures=None, **options):
        self._instructions = {phase: (PROMPT_ROOT / f"{phase}.txt").read_text().rstrip()
                              for phase in DEFAULT_PHASE_TEMPERATURES}
        self._instruction_hashes = {phase: hashlib.sha256((PROMPT_ROOT / f"{phase}.txt").read_bytes()).hexdigest()
                                    for phase in DEFAULT_PHASE_TEMPERATURES}
        self._phase_cache = {}
        self._set_phase_temperatures(phase_temperatures)
        options["temperature"] = 1.0
        super().__init__(model_path, **options)

    def _set_phase_temperatures(self, temperatures):
        values = dict(DEFAULT_PHASE_TEMPERATURES if temperatures is None else temperatures)
        if set(values) != set(DEFAULT_PHASE_TEMPERATURES):
            raise ValueError("Phase temperature profile requires exactly select/respond/write")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("Phase temperatures must be finite positive numbers")
        self.phase_temperatures = MappingProxyType({key: float(values[key]) for key in DEFAULT_PHASE_TEMPERATURES})

    def phase_profile(self):
        profile = {"version": VERSION, "temperatures": dict(self.phase_temperatures),
                   "base_temperature": 1.0, "instruction_sha256": dict(self._instruction_hashes)}
        return {**profile, "profile_sha256": _digest(profile)}

    def phase_from_prompt(self, prompt):
        # Full trusted instruction prefix must match; observations cannot select a phase.
        matches = [phase for phase, instruction in self._instructions.items() if prompt.startswith(instruction)]
        if len(matches) != 1:
            raise ValueError("Prompt does not match exactly one frozen phase instruction")
        return matches[0]

    def phase_from_generation(self, generated):
        key = tuple(generated.prompt_ids[:128])
        if key in self._phase_cache:
            return self._phase_cache[key]
        # Reference has no actor cache. Decode only the instruction preamble,
        # before any observation; metadata and generated text are not trusted.
        prefix = self.tokenizer.decode(list(key), skip_special_tokens=True)
        header = prefix.split("OBSERVACION", 1)[0]
        matches = [phase for phase, marker in PHASE_MARKERS.items() if marker in header]
        if len(matches) != 1:
            raise ValueError("Token preamble does not identify exactly one phase")
        phase = matches[0]
        self._phase_cache[key] = phase
        return phase

    @contextmanager
    def _at_temperature(self, temperature):
        previous = self.config.temperature
        self.config.temperature = temperature
        try:
            yield
        finally:
            self.config.temperature = previous

    def generate(self, prompt, sample=False):
        phase = self.phase_from_prompt(prompt)
        temperature = self.phase_temperatures[phase] if sample else self.config.temperature
        with self._at_temperature(temperature):
            generated = super().generate(prompt, sample=sample)
        key = tuple(generated.prompt_ids[:128])
        if key in self._phase_cache and self._phase_cache[key] != phase:
            raise ValueError("Phase instruction token preamble collision")
        self._phase_cache[key] = phase
        generated.metadata.update(phase_temperature_version=VERSION, phase=phase,
                                  phase_temperature=temperature, phase_temperature_applied=bool(sample),
                                  phase_profile_sha256=self.phase_profile()["profile_sha256"])
        return generated

    def sequence_logprobs(self, generated, *, sampling_distribution=False):
        if not sampling_distribution:
            return super().sequence_logprobs(generated, sampling_distribution=False)
        phase = self.phase_from_generation(generated)
        with self._at_temperature(self.phase_temperatures[phase]):
            return super().sequence_logprobs(generated, sampling_distribution=True)

    def save_checkpoint(self, destination):
        manifest = super().save_checkpoint(destination)
        # local_checkpoint.json is excluded from artifact_hashes by the backend.
        manifest["phase_temperature_profile"] = self.phase_profile()
        (Path(destination) / "local_checkpoint.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    @classmethod
    def from_checkpoint(cls, checkpoint, *, trainable=False, device=None, phase_temperatures=None):
        manifest = json.loads((Path(checkpoint) / "local_checkpoint.json").read_text())
        saved = manifest.get("phase_temperature_profile")
        obj = super().from_checkpoint(checkpoint, trainable=trainable, device=device)
        obj._set_phase_temperatures(phase_temperatures if phase_temperatures is not None
                                    else saved["temperatures"] if saved else None)
        if saved is not None and obj.phase_profile() != saved:
            raise ValueError("Frozen checkpoint phase temperature profile mismatch")
        return obj


def verify_phase_profiles(actor, reference):
    """Complement the historical reference check with the sampling profile."""
    if actor.config.temperature != 1.0 or reference.config.temperature != 1.0:
        raise ValueError("Actor/reference global base temperature must both be 1.0")
    if actor.phase_profile() != reference.phase_profile():
        raise ValueError("Actor/reference phase temperature profiles differ")
    return actor.phase_profile()
