"""Spanish recommendation cases with authored exclusions and independent splits."""
from __future__ import annotations

import random

from research.src.artifacts import digest
from web_demo.recommendations import DISHES


CAPTURE = "que puedo comoer hoy si ayer comi pooolo y no quiero comer pollo"
FAMILIES = ("rechazo", "errata", "multiple", "variedad", "rectificacion", "ambiguo", "todas", "historial")
CLARIFY = ("Pedir una aclaración", frozenset())
LETTERS = "ABCDEF"
# Each split has its own language and dishes. Ingredient concepts intentionally recur.
EXTRA = {
    "validation": (
        ("Hamburguesa de pollo", "pollo tomate pan"),
        ("Crema de zanahoria con queso", "zanahoria queso"),
        ("Arroz con huevo y maíz", "arroz huevo maiz"),
        ("Pescado con patacones", "pescado platano"),
        ("Ensalada de garbanzos con pepino", "garbanzo pepino"),
    ),
    "evaluation": (
        ("Brochetas de pollo con verduras", "pollo pimiento cebolla"),
        ("Puré de papa con queso fresco", "papa queso"),
        ("Arroz salteado con huevo y guisantes", "arroz huevo guisante"),
        ("Pescado al vapor con zanahoria", "pescado zanahoria"),
        ("Ensalada de quinua y garbanzos", "quinua garbanzo pepino"),
    ),
}
TRAIN_RECIPES = (*DISHES["dinner_preference"], ("Pollo con papas", "pollo papa"),
                 ("Tostadas con queso", "pan queso"))
WORD = {"pollo": "pollo", "huevo": "huevo", "queso": "queso", "pescado": "pescado", "garbanzo": "garbanzos"}
TYPOS = {
    "train": ("pooolo", "pollio", "huevvo", "qeso", "pescaddo", "garbansos"),
    "validation": ("polo", "poloo", "uebo", "quezo", "pezcado", "garvanzos"),
    "evaluation": ("polllo", "poyo", "guevo", "keso", "pescao", "garbanzoo"),
}
WORDS = {
    "train": ("¿Qué puedo comer hoy?", "¿Qué me recomiendas?", "¿Qué puedo cenar?"),
    "validation": ("¿Qué preparo para la cena?", "¿Me das una idea para comer?", "¿Qué puedo cocinar esta noche?"),
    "evaluation": ("¿Con qué plato salgo del paso esta noche?", "¿Qué me sugieres para la comida?", "¿Qué cenarías en mi lugar?"),
}


def wording(split, family, forbidden, other, rng):
    query = rng.choice(WORDS[split])
    food = WORD[forbidden]
    second = WORD[other]
    typo = rng.choice(TYPOS[split][:2]) if forbidden == "pollo" else TYPOS[split][
        2 + ("huevo", "queso", "pescado", "garbanzo").index(forbidden)]
    if family == "rechazo":
        phrase = {"train": f"No quiero {food} hoy.", "validation": f"Para esta comida deja fuera el {food}.",
                  "evaluation": f"Esta noche prefiero que la comida no lleve {food}."}[split]
        return [f"{query} {phrase}"], {forbidden}, False
    if family == "errata":
        phrase = {"train": f"hoy no quiero comer {typo}", "validation": f"pero con {typo} no",
                  "evaluation": f"eso sí, nada de {typo}"}[split]
        return [f"{query} {phrase}"], {forbidden}, False
    if family == "multiple":
        phrase = {"train": f"Sin {food} ni {second}.", "validation": f"Quitemos {food} y también {second}.",
                  "evaluation": f"Que no tenga {food}; tampoco tolero {second} en este plato."}[split]
        return [f"{query} {phrase}"], {forbidden, other}, False
    if family == "variedad":
        phrase = {"train": f"Ayer comí {food} y no quiero repetirlo.",
                  "validation": f"Ya almorcé {food}; buscaba algo distinto.",
                  "evaluation": f"Me serví {food} al mediodía. Para ahora quisiera variar."}[split]
        return [f"{phrase} {query}"], {forbidden}, False
    if family == "rectificacion":
        first, correction = {
            "train": (f"Hoy no quiero {food} ni {second}.", f"Al final {food} sí, {second} todavía no."),
            "validation": (f"Evitemos {food} y {second}.", f"Me expresé mal: {food} está bien, deja fuera solo {second}."),
            "evaluation": (f"Por ahora descartemos {food} y {second}.", f"Cambio de plan: ya acepto {food}, el problema es {second}."),
        }[split]
        return [first, f"{correction} {query}"], {other}, False
    if family == "ambiguo":
        phrase = {"train": "Sin eso, por favor.", "validation": "Quítale ese ingrediente que te digo.",
                  "evaluation": "Hay algo de esa lista que no como y olvidé decirte qué."}[split]
        return [f"{phrase} {query}"], set(), True
    if family == "todas":
        phrase = {"train": "Sin papa, huevo, arroz, pescado, quinua, pollo, garbanzos, pan ni queso.",
                  "validation": "Ahora mismo no puedo usar pollo, queso, arroz, pescado ni garbanzos.",
                  "evaluation": "Descarta pollo y queso. Tampoco tengo arroz, pescado ni garbanzos."}[split]
        return [f"{phrase} {query}"], (set(WORD) | {"papa", "arroz", "quinua", "pan"} if split == "train"
                                      else {"pollo", "queso", "arroz", "pescado", "garbanzo"}), False
    return [f"El otro día comí {food}. {query}"], set(), False


def build(seed=40791, sizes=None):
    sizes = sizes or {"train": 96, "validation": 24, "evaluation": 40}
    rng = random.Random(seed)
    bank = {split: [] for split in sizes}
    for split, count in sizes.items():
        pool = TRAIN_RECIPES if split == "train" else EXTRA[split]
        for index in range(count):
            family = FAMILIES[index % len(FAMILIES)]
            food, other = rng.sample(list(WORD), 2)
            recipes = [pool[i] for i in rng.sample(range(len(pool)), 5)]
            turns, excluded, ambiguous = wording(split, family, food, other, rng)
            if split == "train" and index == 0:
                family, turns, excluded = "errata", [CAPTURE], {"pollo"}
            options = [(name, frozenset(ingredients.split())) for name, ingredients in recipes]
            options.append(CLARIFY)
            rng.shuffle(options)
            accepted = [i for i, (_, ingredients) in enumerate(options)
                        if ingredients and not (ingredients & excluded) and not ambiguous]
            if not accepted:
                accepted = [options.index(CLARIFY)]
            # Correction cases require a choice that demonstrates the reversal when possible.
            if family == "rectificacion":
                restored = [i for i in accepted if food in options[i][1]]
                if restored:
                    accepted = restored
            public = {"turns": turns, "options": [{"name": name, "ingredients": sorted(ingredients)}
                                                  for name, ingredients in options]}
            row = {**public, "id": digest(public), "split": split, "family": family, "forbidden": sorted(excluded),
                   "accepted": accepted, "teacher": accepted[0]}
            bank[split].append(row)
    identities = [{digest(row["turns"]) for row in rows} for rows in bank.values()]
    if any(a & b for i, a in enumerate(identities) for b in identities[i+1:]):
        raise ValueError("Conversational prefixes overlap between partitions")
    if any(len({row["id"] for row in rows}) != len(rows) for rows in bank.values()):
        raise ValueError("Duplicate challenge in a partition")
    return bank


def prompt(turns, options):
    dialogue = "\n".join(f"Usuario: {text}" for text in turns)
    menu = "\n".join(f"{letter}. {row['name']}" + (f" ({', '.join(row['ingredients'])})" if row["ingredients"] else "")
                     for letter, row in zip(LETTERS, options, strict=True))
    return ("Elige una opción para lo que pide el usuario ahora. Respeta los ingredientes que descarta y las "
            "correcciones. Si falta un dato o ninguna comida es compatible, pide aclaración. "
            "Contesta solo con la letra de la opción.\n" + dialogue + "\n" + menu + "\nLetra:")
