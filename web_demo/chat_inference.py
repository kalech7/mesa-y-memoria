"""Local structured decoding for interactive use, with explicit phase limits."""
from __future__ import annotations

from dataclasses import dataclass
import json

from llm_local_001.model import Generation
from research.src.policy.llm_controller import LLMController


@dataclass
class ChatBackend:
    local: object

    @property
    def config(self):
        return self.local.config

    def encode_prompt(self, prompt):
        return self.local.encode_prompt(prompt)

    def generate(self, prompt, sample=False):
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

        local = self.local
        tokenizer = local.tokenizer
        ids = self.encode_prompt(prompt)
        limits = {64: 40, 256: 128, 512: 256}
        limit = limits.get(self.config.max_new_tokens, self.config.max_new_tokens)
        local._check_length(len(ids) + limit)
        tokens = local.torch.tensor([ids], device=local.device)
        eos = tokenizer.eos_token_id
        special = {eos, tokenizer.pad_token_id}
        decoder = json.JSONDecoder()

        class EndOfObject(StoppingCriteria):
            complete = False

            def __call__(self, input_ids, scores, **kwargs):
                output = tokenizer.decode(input_ids[0, len(ids):].tolist(), skip_special_tokens=True).strip()
                # A repetitive malformed completion can otherwise exhaust the entire phase.
                tokens = input_ids[0, len(ids):].tolist()
                if len(tokens) >= 24 and tokens[-8:] == tokens[-16:-8] == tokens[-24:-16]:
                    return True
                if output.endswith("}"):
                    try:
                        value, end = decoder.raw_decode(output)
                        self.complete = isinstance(value, dict) and not output[end:].strip()
                    except ValueError:
                        # A closed object with a syntax error already has a diagnosable output.
                        if output.startswith("{") and output.count("{") == output.count("}"):
                            return True
                return self.complete

        stop = EndOfObject()
        with local.torch.inference_mode():
            sequence = local.model.generate(
                input_ids=tokens, attention_mask=local.torch.ones_like(tokens),
                generation_config=GenerationConfig(max_new_tokens=limit, do_sample=False, use_cache=True,
                                                  eos_token_id=eos, pad_token_id=tokenizer.pad_token_id),
                stopping_criteria=StoppingCriteriaList([stop]),
            )
        completion = sequence[0, len(ids):].tolist()
        output = tokenizer.decode(completion, skip_special_tokens=True)
        return Generation(ids, completion, output, truncated=not stop.complete and completion[-1] not in special)


class InteractiveController(LLMController):
    def recommendation(self, observation, choices):
        options = json.dumps(choices, ensure_ascii=False)
        return self.call("respond", observation, instruction_suffix=(
            f"En esta consulta el usuario pide una sugerencia para ahora. Elige un único plato de {options}. "
            "Devuelve ANSWER con value exactamente igual al plato, branches:[] y evidence_ids:[]. "
            "Las opciones son sugerencias, no hechos sobre las preferencias de la persona."))


def chat_controller(controller):
    return InteractiveController(ChatBackend(controller.backend), prompt_root=controller.prompt_root, sample=False)
