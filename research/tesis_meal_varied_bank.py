"""Build fixed dialogue roots before rotating their ingredient-labelled menus."""
from __future__ import annotations

import random
import re
from itertools import combinations
from unicodedata import normalize, combining

from research.src.artifacts import digest
from research.tesis_meal_cases import CAPTURE, CLARIFY, LETTERS, TYPOS, WORD
from research.tesis_meal_dialogues_train import TEMPLATES
from research.tesis_meal_dialogues_eval import VALIDATION, EVALUATION


CATEGORIES = ("rechazo", "errata", "multiple", "variedad", "rectificacion", "ambiguo", "todas", "historial",
              "confirmacion", "terceros", "contrafactual")
CONTENT = {"train": TEMPLATES, "validation": VALIDATION, "evaluation": EVALUATION}
MENUS = {
    "train": (
        ("Pollo al limón", "pollo limon"),
        ("Picadillo de pollo con choclo", "pollo choclo"),
        ("Bolitas de pollo con queso", "pollo queso"),
        ("Arroz con huevo y tomate", "arroz huevo tomate"),
        ("Omurice de champiñones", "arroz huevo champinon"),
        ("Tostadas con queso y albahaca", "pan queso albahaca"),
        ("Pescado con puré de papa", "pescado papa"),
        ("Caldo de pescado con choclo", "pescado choclo"),
        ("Faláfel con pepino", "garbanzo pepino"),
        ("Garbanzos con queso y acelga", "garbanzo queso acelga"),
        ("Aguacate con huevo", "aguacate huevo"),
        ("Ensalada de garbanzos", "garbanzo tomate"),
        ("Quinua con zanahoria", "quinua zanahoria"),
        ("Tortilla de papa y cebolla", "papa huevo cebolla"),
    ),
    "validation": (
        ("Arroz con pollo y arvejas", "arroz pollo arveja"),
        ("Sándwich de pollo", "pan pollo"),
        ("Pollo con pimientos asados", "pollo pimiento"),
        ("Enrollado de queso", "queso harina"),
        ("Tomates al horno con queso", "tomate queso"),
        ("Filete de pescado con yuca", "pescado yuca"),
        ("Pescado con salsa de limón", "pescado limon"),
        ("Papas con huevo pochado", "papa huevo"),
        ("Huevos a la mexicana", "huevo chile tomate"),
        ("Crema de garbanzos", "garbanzo"),
        ("Garbanzos con espinacas", "garbanzo espinaca"),
        ("Ensalada de pepino", "pepino cebolla"),
        ("Sopa de lentejas y zanahoria", "lenteja zanahoria"),
    ),
    "evaluation": (
        ("Guiso de pollo con batata", "pollo batata"),
        ("Tacos de pollo y repollo", "pollo repollo"),
        ("Pollo estofado con vainitas", "pollo vainita"),
        ("Puré de zapallo con queso fresco", "zapallo queso"),
        ("Cazuela de maíz y queso", "maiz queso"),
        ("Brochetas de pescado y calabacín", "pescado calabacin"),
        ("Pescado en salsa de maracuyá", "pescado maracuya"),
        ("Ensalada tibia de papa y huevo", "papa huevo"),
        ("Sopa con huevo y apio", "huevo apio"),
        ("Garbanzos especiados con calabaza", "garbanzo calabaza"),
        ("Arepas rellenas de garbanzos", "garbanzo maiz"),
        ("Verduras asadas con batata", "batata calabacin repollo"),
        ("Tabulé de quinua", "quinua tomate cebolla"),
    ),
}
SIZE = {"train": 530, "validation": 100, "evaluation": 150}
EXTRA_ENTITIES = {"train": ["huevo", "pescado", "pollo", "queso", "garbanzo", "arroz", "tomate", "papa", "choclo"],
                  "validation": ["huevo", "pescado", "pollo", "queso", "garbanzo", "yuca", "espinaca", "arveja"],
                  "evaluation": ["huevo", "pescado", "pollo", "queso", "garbanzo", "maiz", "batata", "calabacin",
                                 "repollo", "apio", "vainita", "zapallo"]}
VOCAB = {**WORD, "maiz": "maíz", "calabacin": "calabacín", "choclo": "choclo",
         "arveja": "arvejas", "vainita": "vainitas", "lenteja": "lentejas"}


def normalized(turns):
    chars = normalize("NFKD", " ".join(turns).casefold())
    return " ".join(re.findall(r"[a-z0-9]+", "".join(c for c in chars if not combining(c))))


def _typo(split, food, rng):
    if food == "pollo":
        return rng.choice(TYPOS[split][:2])
    idx = ("huevo", "queso", "pescado", "garbanzo").index(food)
    return TYPOS[split][2 + idx]


def _spanish(ingredients):
    words = [VOCAB.get(f, f) for f in ingredients]
    return ", ".join(words[:-1]) + " ni " + words[-1] if len(words) > 1 else words[0]


def _option(name, ingredients):
    return {"name": name, "ingredients": sorted(ingredients)}


def _row(split, family, turns, menu, exclusions, *, ambiguous=False, pair=None):
    options = [_option(name, ingredients.split()) for name, ingredients in menu]
    options.append(_option(*CLARIFY))
    accepted = [idx for idx, option in enumerate(options)
                if option["ingredients"] and not (exclusions & set(option["ingredients"])) and not ambiguous]
    if not accepted:
        accepted = [len(LETTERS) - 1]
    public = {"turns": turns, "options": options}
    return {**public, "id": digest(public), "split": split, "family": family, "forbidden": sorted(exclusions),
            "accepted": accepted, "pair": pair, "ambiguous": ambiguous}


def _dialogue(split, family, pool, rng, index):
    templates = CONTENT[split][family]
    template = templates[index % len(templates)]
    if family in {"errata", "rectificacion"}:
        food, other = rng.sample(list(WORD), 2)
    else:
        food, other = rng.sample(EXTRA_ENTITIES[split], 2)
    if family in {"rechazo", "errata", "variedad", "confirmacion"}:
        exclusions = {food}
    elif family == "multiple":
        exclusions = {food, other}
    elif family == "rectificacion":
        exclusions = {other}
    else:
        exclusions = set()
    if family == "todas":
        shortlist = rng.sample(pool, 5)
        exclusions = {rng.choice(ingredients.split()) for _, ingredients in shortlist}
    else:
        mandatory = [recipe for recipe in pool if food in recipe[1].split()]
        shortlist = rng.sample(mandatory, min(2, len(mandatory)))
        if family in {"multiple", "rectificacion"}:
            counterparts = [recipe for recipe in pool if other in recipe[1].split() and recipe not in shortlist]
            shortlist.extend(rng.sample(counterparts, min(2, len(counterparts))))
        shortlist.extend(rng.sample([recipe for recipe in pool if recipe not in shortlist], 5 - len(shortlist)))
    rng.shuffle(shortlist)
    turns = [text.format(food=VOCAB.get(food, food), other=VOCAB.get(other, other),
                         typo=_typo(split, food, rng) if family == "errata" else VOCAB.get(food, food),
                         foods=_spanish(sorted(exclusions)) if exclusions else "")
             for text in template]
    return _row(split, family, turns, shortlist, exclusions, ambiguous=family == "ambiguo")


PAIR_WORDING = {
    "train": ("Evitemos {common}.", "Entre lo que falta, tampoco quiero {food}. ¿Qué preparo?"),
    "validation": ("Se me acabó {common} y hoy prefiero evitar {food}.",
                   "¿Qué opción me serviría?"),
    "evaluation": ("Te actualizo lo de la comida: ya descarté {common}.",
                   "Hay otro ingrediente que tampoco quiero esta vez, {food}. ¿Cuál me llevo?"),
}


def _pairs(split, pool, rng, count):
    choices = []
    for common in EXTRA_ENTITIES[split]:
        blocked = [dish for dish in pool if common in dish[1].split()]
        if len(blocked) < 3:
            continue
        other_dishes = [dish for dish in pool if common not in dish[1].split()]
        for left, right in combinations(other_dishes, 2):
            left_keys, right_keys = set(left[1].split()), set(right[1].split())
            for a in sorted(left_keys - right_keys):
                for b in sorted(right_keys - left_keys):
                    choices.append((common, left, right, a, b, blocked))
    if len(choices) < count:
        raise ValueError(f"Insufficient counterfactual menus in {split}")
    rows = []
    for common, left, right, a, b, blocked in rng.sample(choices, count):
        menu = [left, right, *rng.sample(blocked, 3)]
        rng.shuffle(menu)
        pair = digest([split, common, menu, a, b])
        for food in (a, b):
            turns = [text.format(common=VOCAB.get(common, common), food=VOCAB.get(food, food))
                     for text in PAIR_WORDING[split]]
            row = _row(split, "contrafactual", turns, menu, {common, food}, pair=pair)
            if len(row["accepted"]) != 1:
                raise ValueError("Each counterfactual has a single compatible dish")
            rows.append(row)
    return rows


def build_varied(seed=63211):
    rng = random.Random(seed)
    bank = {}
    for split, size in SIZE.items():
        pair_count = {"train": 35, "validation": 8, "evaluation": 15}[split]
        pool = MENUS[split]
        rows = _pairs(split, pool, rng, pair_count)
        seen = {row["id"] for row in rows}
        if split == "train":
            menu = [pool[0], pool[1], pool[4], pool[-2], pool[-1]]
            row = _row(split, "errata", [CAPTURE], menu, {"pollo"})
            rows.append(row)
            seen.add(row["id"])
        total = len(rows)
        while len(rows) < size:
            family = CATEGORIES[(len(rows) - total) % (len(CATEGORIES) - 1)]
            candidate = _dialogue(split, family, pool, rng, rng.randrange(len(CONTENT[split][family])))
            if candidate["id"] not in seen:
                rows.append(candidate)
                seen.add(candidate["id"])
        rng.shuffle(rows)
        bank[split] = rows
    for first, second in combinations(bank, 2):
        if ({normalized(row["turns"]) for row in bank[first]} & {normalized(row["turns"]) for row in bank[second]}):
            raise ValueError(f"Overlapping text between {first} and {second}")
    return bank


def rotate(row, offset):
    order = list(range(len(row["options"])))
    order = order[offset:] + order[:offset]
    return {**row, "options": [row["options"][idx] for idx in order], "accepted": [order.index(idx) for idx in row["accepted"]],
            "rotation": offset, "order": order}


def prompt(turns, options):
    dialogue = "\n".join(f"Usuario: {text}" for text in turns)
    options_text = "\n".join(f"{letter}. {item['name']} ({', '.join(item['ingredients'])})"
                             if item["ingredients"] else f"{letter}. Pedir una aclaración"
                             for letter, item in zip(LETTERS, options, strict=True))
    return ("Elige un plato del menú para el usuario. Los ingredientes entre paréntesis indican qué contiene cada plato. "
            "Respeta lo que quiere evitar y la última corrección. Lo que comió antes y lo que prefiere otra persona "
            "no son exclusiones por sí solos. Si falta un dato indispensable o ningún plato cumple, pide una aclaración. "
            "Responde únicamente con la letra.\n" + dialogue + "\n" + options_text + "\nLetra:")
