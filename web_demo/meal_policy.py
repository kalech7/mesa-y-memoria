"""Qwen's direct categorical recommendation, shared by evaluation and chat."""
from __future__ import annotations

from dataclasses import dataclass
import json
from random import Random

from research.tesis_meal_cases import LETTERS, prompt
from .recommendations import DISHES


def recommendation_menu(turns, attribute, *, shuffle):
    foods = DISHES[attribute]
    if attribute == "dinner_preference":
        foods = (*foods[:-1], ("Pollo con papas", "pollo papa"))
    if len(foods) < len(LETTERS) - 1:
        foods = (*foods, ("Agua mineral", "agua"))
    recipes = [{"name": name, "ingredients": ingredients.split()} for name, ingredients in foods]
    recipes.append({"name": "Pedir una aclaración", "ingredients": []})
    if shuffle:
        # A repeated text gets the same order, while different histories vary it.
        Random("\n".join(turns)).shuffle(recipes)
    return recipes


def option_tokens(tokenizer):
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(tokens) != 1 for tokens in ids) or len({tuple(tokens) for tokens in ids}) != len(ids):
        raise ValueError("Las letras del menú deben corresponder a tokens distintos.")
    return [tokens[0] for tokens in ids]


def final_logits(actor, sequences):
    """Compute logits at the final prompt position, with matching token positions."""
    import inspect

    torch = actor.torch
    width = max(map(len, sequences))
    pad = actor.tokenizer.pad_token_id
    x = torch.tensor([[pad] * (width - len(ids)) + list(ids) for ids in sequences], device=actor.device)
    mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in sequences], device=actor.device)
    position = (mask.cumsum(-1) - 1).clamp_min(0)
    causal_model = actor.model.get_base_model()
    kwargs = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(causal_model.forward).parameters else {}
    return actor.model(input_ids=x, attention_mask=mask, position_ids=position,
                       use_cache=False, **kwargs).logits[:, -1].float()


@dataclass
class MealPolicy:
    actor: object
    adapter: str

    def choose(self, turns, attribute):
        varied = self.adapter.startswith("comidas-variado-")
        recipes = recommendation_menu(turns, attribute, shuffle=varied)
        render_prompt = prompt
        if varied:
            from research.tesis_meal_varied_bank import prompt as varied_prompt

            render_prompt = varied_prompt
        tokens = option_tokens(self.actor.tokenizer)
        return self._predict(self.actor.encode_prompt(render_prompt(turns, recipes)), recipes, tokens)

    def _predict(self, ids, recipes, tokens):
        active = self.actor.model.active_adapter
        try:
            self.actor.model.set_adapter(self.adapter)
            self.actor.model.requires_grad_(False)
            with self.actor.torch.inference_mode():
                array = final_logits(self.actor, [ids])[0]
        finally:
            self.actor.model.set_adapter(active)
            self.actor.model.requires_grad_(False)
        action = int(array[tokens].argmax())
        candidate = recipes[action]
        record = {"phase": "recommendation",
                  "raw": json.dumps({"letra": LETTERS[action], "plato": candidate["name"],
                                     "menú": {letter: option["name"] for letter, option in zip(LETTERS, recipes, strict=True)},
                                     "primera_salida_libre": self.actor.tokenizer.decode([int(array.argmax())])},
                                    ensure_ascii=False)}
        return candidate, record
