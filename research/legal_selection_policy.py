"""Exact categorical SELECT over five public schema actions; no SFT or repair.

Each action's logit is the mean LM log-probability of its canonical completion,
including EOS, divided by Tcat=1. SELECT likelihoods returned to the historical
optimizer replicate ONE action log-probability over its completion positions.
They are not per-token LM likelihoods: the optimizer's token mean is therefore
exactly the categorical action objective, with unchanged 1/3 phase weighting.
"""
from __future__ import annotations

import json
from types import MappingProxyType

from llm_local_001.model import Generation
from research.phase_temperature_policy import PhaseTemperaturePolicy, _digest
from research.src.policy.llm_controller import SpecLocalPolicy

VERSION = "legal_selection_policy_v1"
CATEGORICAL_TEMPERATURE = 1.0
PHASE_TEMPERATURES = MappingProxyType({"select": 1.0, "respond": .6, "write": .35})
ACTION_CATALOG = tuple(json.dumps(action, separators=(",", ":")) for action in (
    {"action": "PROCEED", "slot": None},
    *({"action": "ASK", "slot": slot} for slot in
      ("TEMPORAL_SCOPE", "CURRENT_CONTEXT", "VALID_UNTIL", "GENERIC")),
))


class LegalSelectionPolicy(PhaseTemperaturePolicy):
    """Choose among all five legal actions at every state, using only the LM."""

    def _set_phase_temperatures(self, temperatures):
        values = dict(PHASE_TEMPERATURES if temperatures is None else temperatures)
        if values != dict(PHASE_TEMPERATURES):
            raise ValueError("Legal selection requires its frozen 1.0/.6/.35 profile")
        super()._set_phase_temperatures(values)

    def phase_profile(self):
        profile = super().phase_profile()
        profile.pop("profile_sha256")
        profile["version"] = VERSION
        profile["selection"] = {
            "likelihood_unit": "action", "temperature": CATEGORICAL_TEMPERATURE,
            "score_normalization": "mean_completion_token_logprob_including_eos",
            "catalog": list(ACTION_CATALOG), "catalog_sha256": _digest(list(ACTION_CATALOG)),
        }
        return {**profile, "profile_sha256": _digest(profile)}

    def _completion_catalog(self):
        if not hasattr(self, "_legal_completion_ids"):
            eos = self.tokenizer.eos_token_id
            if eos is None:
                raise ValueError("Categorical SELECT requires EOS")
            result = []
            for text in ACTION_CATALOG:
                ids = list(self.tokenizer.encode(text, add_special_tokens=False))
                if not ids or ids[-1] != eos:
                    ids.append(eos)
                if self.tokenizer.decode(ids, skip_special_tokens=True) != text:
                    raise ValueError("Canonical SELECT text does not roundtrip through tokenizer")
                result.append(tuple(ids))
            if len(set(result)) != len(ACTION_CATALOG):
                raise ValueError("Canonical actions do not have unique token sequences")
            self._legal_completion_ids = tuple(result)
        return self._legal_completion_ids

    def _action_distribution(self, prompt_ids):
        scores = []
        for text, ids in zip(ACTION_CATALOG, self._completion_catalog()):
            candidate = Generation(list(prompt_ids), list(ids), text)
            # Direct base entry avoids recursive categorical likelihood and does
            # not use the RESPOND/WRITE token-temperature sampling profile.
            token_logprobs = SpecLocalPolicy.sequence_logprobs(self, candidate, sampling_distribution=False)
            scores.append(token_logprobs.mean())
        scores = self.torch.stack(scores)
        return scores, self.torch.log_softmax(scores / CATEGORICAL_TEMPERATURE, dim=-1)

    def generate(self, prompt, sample=False):
        phase = self.phase_from_prompt(prompt)
        if phase != "select":
            generated = super().generate(prompt, sample=sample)
            generated.metadata.update(selection_protocol_version=VERSION, likelihood_unit="token")
            return generated
        prompt_ids = self.encode_prompt(prompt)
        with self.torch.no_grad(), self.likelihood_mode():
            scores, logprobs = self._action_distribution(prompt_ids)
            chosen = int(self.torch.multinomial(logprobs.exp(), 1).item() if sample else logprobs.argmax().item())
        completion = list(self._completion_catalog()[chosen])
        key = tuple(prompt_ids[:128])
        if key in self._phase_cache and self._phase_cache[key] != phase:
            raise ValueError("Phase instruction token preamble collision")
        self._phase_cache[key] = phase
        metadata = {
            "sampled": bool(sample), "phase": phase, "temperature": CATEGORICAL_TEMPERATURE,
            "selection_protocol_version": VERSION, "likelihood_unit": "action",
            "score_normalization": "mean_completion_token_logprob_including_eos",
            "action_catalog": list(ACTION_CATALOG), "action_index": chosen,
            "action_scores": scores.detach().cpu().tolist(),
            "action_logprobs": logprobs.detach().cpu().tolist(),
            "chosen_action_logprob": float(logprobs[chosen].detach().cpu()),
            "phase_profile_sha256": self.phase_profile()["profile_sha256"],
        }
        return Generation(list(prompt_ids), completion, ACTION_CATALOG[chosen], truncated=False, metadata=metadata)

    def sequence_logprobs(self, generated, *, sampling_distribution=False):
        if self.phase_from_generation(generated) != "select":
            return super().sequence_logprobs(generated, sampling_distribution=sampling_distribution)
        try:
            chosen = self._completion_catalog().index(tuple(generated.completion_ids))
        except ValueError as exc:
            raise ValueError("SELECT completion is not one of the five canonical actions including EOS") from exc
        _, logprobs = self._action_distribution(generated.prompt_ids)
        # Categorical for SELECT regardless of the legacy token-distribution
        # flag. Only the direct SpecLocalPolicy call above scores LM tokens.
        return logprobs[chosen].expand(len(generated.completion_ids))
