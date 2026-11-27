"""Resolve present-meal requests separately from claims about stored preferences."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as CalendarDate, timedelta
from difflib import SequenceMatcher
import re
from unicodedata import combining, normalize

from simulator.preferences import FOOD_ALIASES


# Suggestions are simple preparations with a known ingredient list. The list is
# used only when recommending, never as evidence that a person likes a dish.
DISHES = {
    "dinner_preference": (
        ("Tortilla de papa y cebolla", "papa huevo cebolla aceite"),
        ("Arroz con lentejas y zanahoria", "arroz lenteja zanahoria aceite"),
        ("Pescado al horno con papas", "pescado papa limon aceite"),
        ("Ensalada de quinua con pepino y tomate", "quinua pepino tomate aceite"),
        ("Garbanzos salteados con pimiento", "garbanzo pimiento aceite"),
    ),
    "breakfast_preference": (
        ("Avena con banano", "avena banano"),
        ("Huevos revueltos con tomate", "huevo tomate aceite"),
        ("Yogur natural con mango", "yogur leche mango"),
        ("Papaya con banano", "papaya banano"),
        ("Tortilla de papa y cebolla", "papa huevo cebolla aceite"),
    ),
    "drink_preference": (
        ("Agua con limón", "agua limon"),
        ("Infusión de menta", "agua menta"),
        ("Jugo de maracuyá", "agua maracuya"),
        ("Café pasado", "agua cafe cafeina"),
    ),
}
CATEGORIES = {
    "carne": {"pollo", "pescado", "res", "cerdo", "pavo", "marisco", "carne"},
    "lacteo": {"leche", "queso", "yogur", "mantequilla", "lacteo"},
    "gluten": {"gluten", "trigo", "avena", "pan", "pasta", "harina"},
    "verdura": {"verdura", "cebolla", "zanahoria", "pimiento", "pepino", "tomate"},
}
EXCLUSION = re.compile(
    r"\b(?:no\s+(?:quiere[n]?|quiero|queremos|puede[n]?|puedo|podemos|prefiere[n]?|prefiero|preferimos"
    r"|(?:me|le|nos)\s+(?:gusta[n]?|apetece|provoca))"
    r"(?:\s+que\s+(?:(?:(?:el|la|los|las|mi|nuestro|nuestra)\s+)?(?:plato|comida|cena|pedido|receta)\s+)?"
    r"(?:lleve[n]?|tenga[n]?|incluya[n]?|contenga[n]?)"
    r"|(?!\s+que\b)(?:\s+(?:volver a|repetir|comer|cenar|almorzar|desayunar|beber|tomar|consumir|usar))*)"
    r"|(?:no|tampoco)\s+(?:agreg(?:a|ues)|anad(?:e|as)|ech(?:a|es)|met(?:e|as)|pon(?:gas)?|inclu(?:ye|yas)|"
    r"(?:le\s+)?pongas|uses|utilices)"
    r"|tampoco(?!\s+(?:(?:lo|la|los|las)\s+)?voy\s+a\b)(?:\s+(?:quiero|puedo)(?:\s+(?:comer|beber|tomar))?)?"
    r"|que\s+no\s+(?:lleve|tenga|incluya|contenga)"
    r"|sin|nada\s+de|evit[ao]|evitar|descart(?:a|o|amos)|(?:deja\s+)?fuera|quita|saca"
    r"|no\s+tengo|me\s+faltan?|se\s+(?:me\s+)?acabo(?:aron)?"
    r"|en\s+vez\s+de|alergi[ac]\w*\s+(?:a|al)|intoleran\w*\s+(?:a|al))\s*[:.]*\s+"
)
CLAUSE_END = re.compile(r"[,;.!?]|\b(?:pero|porque|aunque|si|para|que|por hoy|hoy|ayer|manana|"
                        r"y\s+(?:si|quiero|tengo|no|tampoco)|"
                        r"y\s+(?=(?:el|la|los|las)\s+[a-z]+\s+sigue[n]?\s+fuera)|"
                        r"no\s+quiero|dame|sugiere(?:me)?|recomienda(?:me)?)\b"
                        r"|\b(?:quiero|quisiera|me gustaria)\s+"
                        r"(?=(?:algo|otra cosa|un desayuno|un almuerzo|un plato|una cena|una comida|una bebida"
                        r"|comer|cenar|almorzar|desayunar)\b)")
QUESTION = re.compile(
    r"\b(?:que\s+(?:(?:(?:bebida|plato|comida|cena|desayuno|almuerzo)\s+)?(?:me\s+)?(?:recomiend\w*|sugier\w*|propon\w*)|"
    r"(?:puedo|podria)\s+(?:comer|cenar|almorzar|desayunar|tomar|beber|preparar|cocinar|pedir)"
    r"|(?:preparo|cocino|como|ceno|desayuno|almuerzo|tomo)\b)"
    r"|(?:dame|sugiere(?:me)?|recomienda(?:me)?)\s+(?:un[ao]s?\s+|algunas?\s+)?"
    r"(?:algo|otra|opciones?|comidas?|platos?|alternativas?|cenas?|almuerzos?|desayunos?|bebidas?"
    r"|(?:cosas|ideas)\s+para\s+(?:comer|cenar|almorzar|desayunar|beber|tomar))"
    r"|cual\s+de\s+(?:es[ao]s|est[ao]s|l[ao]s)\s+(?:platos|opciones)\s+(?:podemos|podriamos)\s+compartir"
    r"|que\s+(?:podemos|podriamos)\s+(?:comer|cenar|preparar|pedir|compartir)"
    r"|que\s+(?:comemos|cenamos|preparamos|pedimos)\b"
    r"|(?:alguna|otra)\s+(?:alternativa|opcion)|algo\s+(?:distinto|diferente))\b"
)
MEAL_WISH = re.compile(
    r"(?<!no )(?<!tampoco )\b(?:quiero|quisiera|me gustaria)\s+(?:"
    r"(?:comer|cenar|almorzar|desayunar|beber|tomar)\s+algo\b"
    r"|(?:un|una)\s+(?:desayuno|almuerzo|cena|comida|bebida|plato)\b"
    r"|(?:algo|otra\s+cosa)\s+(?:(?:[a-z]+\s+){0,3}para\s+(?:comer|cenar|almorzar|desayunar|beber|tomar)\b"
    r"|(?:rico|sabroso|delicioso|sin|que\s+no\s+(?:lleve|tenga))\b))"
)
OCCASION = re.compile(
    r"(?:(?:es|digo|lo digo)\s+)?(?:solo\s+)?"
    r"(?:(?:para|por)\s+)?(?:hoy|esta\s+(?:ocasion|ocacion|vez|comida|cena|noche))[.!?\s]*"
)
CHANGE = re.compile(r"\b(?:si\s+(?:quiere|quiero|puede|puedo)|(?:ahora|al final|mejor)\s+(?:quiero|puedo|si)|"
                    r"ya\s+(?:consegui|encontre|compre)|"
                    r"ya\s+no\s+(?:evito|quiero\s+evitar|hace\s+falta\s+quitar)|"
                    r"no\s+quiero\s+dejar\s+de|no\s+hace\s+falta\s+(?:evitar|quitar))\s+"
                    r"(?:(?:comer|cenar|almorzar|desayunar|beber|tomar|consumir)\s+)*")
NORMAL_WORD = re.compile(r"[a-z]+")
OMIT = {"el", "la", "los", "las", "de", "del", "al", "un", "una", "mas", "con", "a", "tampoco", "por", "favor",
        "comer", "cenar", "almorzar", "desayunar", "beber", "tomar", "consumir", "usar", "todavia"}
UNSPECIFIED = {"eso", "esto", "aquello", "mismo", "lo", "algo", "nada", "repetirlo", "repetirla"}
PAST_DISH = re.compile(r"\b(?:ayer\s+comi|comi\s+ayer|ayer\s+almorce|ayer\s+cene)\s+")
UNKNOWN_FOOD = re.compile(
    r"\b(?:(?:un[oa]?|algun[oa]?|cierto|el)\s+(?:de\s+los\s+)?ingredientes?\b|"
    r"lo\s+que\s+(?:me\s+dijo|me\s+pidio|no\s+como|no\s+quiere)|(?:uno|algo)\s+del\s+menu)\b"
)
UNKNOWN_STATEMENT = re.compile(
    r"\b(?:no\s+(?:recuerdo|se)\s+cual|no\s+(?:me\s+)?(?:dijeron|dijo|dicen|confirman|confirmo)\s+cual|"
    r"no\s+te\s+(?:dije|he\s+dicho)\s+cual|aun\s+no\s+(?:se|me\s+dicen)\s+cual|"
    r"no\s+(?:te\s+)?(?:lo\s+)?(?:he\s+dicho|he\s+contado)|me\s+da\s+cosa\s+decirla)\b"
)
UNKNOWN_REFERENCE = re.compile(r"\b(?:ingrediente|algo\s+del\s+menu|una\s+cosa|hay\s+algo|que\s+evitar)\b")
SEVERE = re.compile(r"\b(?:soy\s+(?:alergic[oa]|celiac[oa]|intolerante)|"
                    r"tengo\s+(?:alergia|intolerancia)|alergia\s+a[l]?)\b")
CLAUSES = re.compile(
    r"[;.!?¿¡]+|\bpero\b|\by\s+(?=no\s+quiero|tampoco)"
    r"|\bsi\s+(?=ayer\b|manana\b|hoy\b)|\by\s+(?=(?:yo|mi\s+herman[oa]|mi\s+amig[oa]|ella|el)\s+(?:sin|no|no\s+quiero|descarta))"
    r"|,\s*(?=(?:yo|hoy|ayer|manana|ahora|para\s+(?:hoy|ayer|manana|mi|la\s+cena|el\s+desayuno)|"
    r"mi\s+(?:herman[oa]|amig[oa]|pareja|madre|padre)|(?:el|la|los|las)\s+[a-z]+\s+sigue[n]?\s+fuera)\b)"
)
MEALS = {
    "dinner_preference": re.compile(r"\b(?:cena(?:r|s|mos|remos)?|ceno|cenemos)\b"),
    "breakfast_preference": re.compile(r"\bdesayun(?:o|os|ar|amos)\b"),
    "drink_preference": re.compile(r"\b(?:bebidas?|beber)\b"),
}
RELATIVE_DAY = re.compile(r"\b(?:pasado\s+manana|manana|hoy|ayer|esta\s+noche)\b|\b\d{4}-\d{2}-\d{2}\b")
DAY_OFFSETS = {"hoy": 0, "ayer": -1, "manana": 1, "pasado manana": 2, "esta noche": 0}
SHARED = re.compile(r"\b(?:compart(?:ir|imos)|cenamos\s+junt[oa]s|para\s+(?:l[oa]s\s+dos|tod[oa]s)|"
                    r"para\s+mi\s+y|(?:que\s+)?(?:comemos|cenamos|cenaremos)\s+(?:junt[oa]s|l[oa]s\s+dos)|"
                    r"(?:cenar|comer|cena|comida)\s+con\s+(?:mi|el|ella))\b")
SOLO = re.compile(r"\b(?:solo\s+para\s+mi|para\s+mi\s+sol[oa]|por\s+mi\s+cuenta|"
                  r"(?:ya\s+)?no\s+viene|(?:comera|come|comiendo)\s+en\s+su\s+casa|pedido\s+aparte)\b")
SELF = re.compile(r"\b(?:yo|para\s+mi\b(?!\s+(?:herman[oa]|amig[oa]|pareja|madre|padre))|mi\s+(?:cena|plato)|a\s+mi)\b")
OTHER = re.compile(r"\b(?:mi\s+(?:herman[oa]|amig[oa]|pareja|madre|padre|mama|papa|ti[oa]|prim[oa])|"
                   r"(?:el|ella)\s+(?:no|si|pidio|quiere|acepta|prefiere|descarta|dice)|"
                   r"(?:el|su)\s+(?:pedido|plato)\s+(?:de\s+el|de\s+ella|es\s+aparte))\b")
RESET = re.compile(r"\b(?:sin\s+restricciones|no\s+tengo\s+restricciones|(?:puedo|podemos)\s+comer\s+"
                   r"(?:cualquiera|de\s+todo)|(?:solo|solamente|unicamente|lo\s+unico)\s+"
                   r"(?:quiero\s+)?(?:sin|quita|excluye|evita|no\s+quiero))\b")
KEEP = re.compile(r"\b(?:y\s+)?(?:sigue(?:n)?\s+(?:sin|fuera)|todavia\s+sin|igual\s+sin)\s+")
REFERENCED_CHANGE = re.compile(
    r"\b(?:(?P<allow>(?:ese|esa|eso|esos|esas)\s+ya\s+(?:lo\s+|la\s+|los\s+|las\s+)?puedo\s+(?:usar|comer))"
    r"|(?:tampoco|no)\s+(?:lo|la|los|las)\s+voy\s+a\s+(?:usar|comer))\b"
)
SUFFIX_CHANGE = re.compile(r"\b(?P<food>[a-z]+)\s*\*?\s+"
                           r"(?:(?P<allow>si(?:\s+(?:esta\s+bien|me\s+(?:sirve|vale)|lo\s+puedo\s+usar|"
                           r"puedo\s+(?:usar|comer)|se\s+puede(?:\s+usar)?|puede\s+ir|va)|(?=\s*(?:\*|,|$)))|"
                           r"(?:queda\s+)?permitid[oa]s?|(?:se\s+)?puede\s+(?:ir|entrar|usar|comer))|sigue[n]?\s+fuera|tampoco)\b")


def _normalize(text):
    text = normalize("NFKD", text.casefold())
    return "".join(c for c in text if not combining(c))


def _stem(word):
    if word.endswith("ces") and len(word) > 4:
        return word[:-3] + "z"
    if word.endswith("es") and len(word) > 5 and word[-3] not in "aeiou":
        return word[:-2]
    return word.removesuffix("s") if len(word) > 3 else word


ALIASES = {
    _stem(_normalize(value)): _stem(_normalize(food))
    for food, aliases in FOOD_ALIASES.items() for value in (food, *aliases)
    if " " not in value
}
ALIASES.update({food: food for recipes in DISHES.values() for _, ingredients in recipes
                for food in ingredients.split()})
ALIASES.update({"lacteo": "lacteo", "res": "res", "cerdo": "cerdo", "pavo": "pavo"})
ALIASES.update({"poyo": "pollo", "pollito": "pollo", "pescao": "pescado", "peskao": "pescado", "huebo": "huevo",
                "patata": "papa", "quinoa": "quinua", "chicken": "pollo", "asiete": "aceite"})
INTENT_TYPOS = {"comoer": "comer", "comerr": "comer", "cemar": "cenar", "recomendas": "recomiendas",
                "agl": "algo", "qiero": "quiero"}


def _intent_text(text):
    normalized = NORMAL_WORD.sub(lambda m: INTENT_TYPOS.get(m[0], m[0]), _normalize(text))
    return " ".join(normalized.split())


def _requests_meal(normalized):
    return bool(QUESTION.search(normalized) or MEAL_WISH.search(normalized))


def asks_for_meal(text):
    return _requests_meal(_intent_text(text))


def _canonical(word, choices, fuzzy=False):
    key = _stem(word)
    if key in choices or key in ALIASES:
        return ALIASES.get(key, key)
    squeezed = re.sub(r"(.)\1{2,}", r"\1\1", key)
    variants = (key, squeezed, re.sub(r"([aeiou])\1+", r"\1", squeezed))
    for variant in variants:
        if variant in choices or variant in ALIASES:
            return ALIASES.get(variant, variant)
    if fuzzy and len(key) >= 4:
        matches = {candidate for candidate in choices
                   if any(SequenceMatcher(None, variant, candidate, autojunk=False).ratio() >= .86
                          for variant in variants)}
        if len(matches) == 1:
            return matches.pop()
        if not matches and len(key) >= 5:
            transposed = {key[:i] + key[i + 1] + key[i] + key[i + 2:] for i in range(len(key) - 1)}
            candidates = {ALIASES.get(variant, variant) for variant in transposed if variant in choices or variant in ALIASES}
            if len(candidates) == 1:
                return candidates.pop()
    return key


def _menu(attribute, dishes):
    return tuple((name, frozenset(ingredients.split() if isinstance(ingredients, str) else ingredients))
                 for name, ingredients in (DISHES[attribute] if dishes is None else dishes))


def preference_ingredients(value, attribute, *, dishes=None):
    """Get catalog ingredients supported by a confirmed positive preference."""
    menu = _menu(attribute, dishes)
    normalized = _normalize(value).strip(" .")
    for dish, ingredients in menu:
        if _normalize(dish) == normalized:
            return ingredients
    candidates = set().union(*(ingredients for _, ingredients in menu))
    return frozenset(_canonical(word, candidates) for word in NORMAL_WORD.findall(normalized)) & candidates


def _day(marker, anchor):
    if marker is None:
        return anchor
    if marker in DAY_OFFSETS:
        if anchor is None:
            return DAY_OFFSETS[marker]
        return (CalendarDate.fromisoformat(anchor) + timedelta(days=DAY_OFFSETS[marker])).isoformat()
    try:
        return CalendarDate.fromisoformat(marker).isoformat()
    except ValueError:
        return anchor


def _target_day(parts, date):
    questions = [part for part in parts if _requests_meal(part) or part.startswith("que prefiero")]
    if questions and (matches := list(RELATIVE_DAY.finditer(questions[-1]))):
        marker = next((match[0] for match in reversed(matches) if match[0] != "ayer"), matches[-1][0])
        return _day(marker, date)
    for part in reversed(parts):
        if (not PAST_DISH.search(part) and (matches := list(RELATIVE_DAY.finditer(part)))
                and matches[-1][0] != "ayer"):
            return _day(matches[-1][0], date)
    return date


def meal_date(text, date):
    """Return the ISO date targeted in the current question, anchored to its turn."""
    day = _target_day(_parts(_intent_text(text)), date)
    return day if isinstance(day, str) else None


def _parts(text):
    return [part.strip() for part in CLAUSES.split(text) if part.strip()]


def _named_meal(text):
    return next((name for name, marker in MEALS.items() if marker.search(text)), None)


def _foods(part, known, *, fuzzy=False):
    """Read the ingredient list at the start of a restriction."""
    foods = set()
    first = True
    for chunk in re.split(r"[,/+]", part):
        head = CLAUSE_END.split(chunk, maxsplit=1)[0]
        if UNKNOWN_FOOD.match(head.lstrip()):
            foods.add("eso")
            break
        words = [word for word in NORMAL_WORD.findall(head) if word not in OMIT | {"ni", "y", "o"}]
        if not words:
            break
        canonical = [_canonical(word, known, fuzzy) for word in words]
        if not first and canonical[0] not in known:
            break
        if first and len(canonical) == 1 and canonical[0] not in {"restriccion", "cosa", "ingrediente"}:
            foods.add(canonical[0])
        else:
            foods.update(word for word in canonical if word in known or word in UNSPECIFIED)
        if head != chunk:
            break
        first = False
    return foods


def _scopes(parts, attribute, anchor, default_day):
    """Assign separate scopes when a turn mentions two meals, recipients or days."""
    day, meal, owner = default_day, attribute, "self"
    for part in parts:
        markers = list(RELATIVE_DAY.finditer(part))
        if markers and not PAST_DISH.search(part):
            day = _day(markers[-1][0], anchor)
        meal = _named_meal(part) or meal
        first_other = OTHER.search(part)
        first_self = SELF.search(part)
        if first_other and not (first_self and first_self.start() > first_other.start()):
            owner = "other"
        elif first_self or SOLO.search(part) or _requests_meal(part):
            owner = "self"
        yield part, meal, day, owner


def _changes(text, known):
    releases = list(CHANGE.finditer(text))
    events = [(match.start(), "remove", _foods(text[match.end():], known, fuzzy=True)) for match in releases]
    if reset := RESET.search(text):
        events.append((reset.start(), "reset", set()))
    for match in EXCLUSION.finditer(text):
        if not any(change.start() <= match.start() < change.end() for change in releases):
            events.append((match.start(), "add", _foods(text[match.end():], known, fuzzy=True)))
    for match in KEEP.finditer(text):
        events.append((match.start(), "add", _foods(text[match.end():], known, fuzzy=True)))
    for match in SUFFIX_CHANGE.finditer(text):
        food = _canonical(match["food"], known, fuzzy=True)
        if food in known:
            events.append((match.start(), "remove" if match["allow"] else "add", {food}))
    for match in REFERENCED_CHANGE.finditer(text):
        prior = (_canonical(word, known) for word in reversed(NORMAL_WORD.findall(text[:match.start()])))
        food = next((word for word in prior if word in known), None)
        if food:
            events.append((match.start(), "remove" if match["allow"] else "add", {food}))
    return sorted(events, key=lambda event: event[0])


def _expanded(excluded):
    return excluded | set().union(*(CATEGORIES.get(food, set()) for food in excluded))


@dataclass(frozen=True)
class MealRequest:
    excluded: frozenset[str]
    unspecified: bool
    severe: bool
    suggestions: tuple[tuple[str, frozenset[str]], ...]
    current_excluded: frozenset[str] = frozenset()
    memory_excluded: frozenset[str] = frozenset()
    date: str | None = None

    def accepts(self, value):
        return any(_normalize(value).strip(" .") == _normalize(dish) for dish, _ in self.suggestions)

    def allows(self, ingredients):
        return _expanded(self.excluded).isdisjoint(ingredients) and not (self.unspecified or self.severe)

    @property
    def choices(self):
        return [dish for dish, _ in self.suggestions]

    def replacement(self):
        if self.unspecified:
            return "¿Qué alimento quieres evitar en esta comida?"
        if self.severe:
            return ("Para tomar en cuenta esa restricción necesito saber qué ingredientes tienes "
                    "y cómo se preparan. ¿Qué opciones estás considerando?")
        if not self.suggestions:
            return "¿Qué ingredientes tienes a mano? Puedo buscar una opción que respete lo que quieres evitar."
        return "Podrías preparar " + self.suggestions[0][0].lower() + "."


def meal_request(history, attribute, *, follow_up=False, date=None, memory_excluded=(), dishes=None,
                 classified_request=False):
    """Recognize recommendation turns and carry exclusions within this conversation."""
    users = [m for m in history if m["role"] == "user"]
    if not users:
        return None
    answering = users[-1].get("answers_food_exclusion", False)
    current = _intent_text(users[-1]["text"])
    menu = _menu(attribute, dishes)
    known = set(ALIASES.values()).union(*(ingredients for _, ingredients in menu))
    if not (classified_request or _requests_meal(current) or answering or
            (follow_up and (_changes(current, known) or OCCASION.fullmatch(current)))):
        return None
    date = date or users[-1].get("date")
    parts = _parts(current)
    wanted_day = _target_day(parts, date)
    # Undated unit tests may still express relative dates. Preserve that scope internally.
    if wanted_day is None:
        wanted_day = 0
    request_parts = [part for part in parts if _requests_meal(part)]
    wanted_meal = next((meal for part in reversed(request_parts or parts) if (meal := _named_meal(part))), attribute)
    restrictions = {}
    severe = set()
    shared_meals = {}
    selected_meal = None
    previous_anchor = None
    previous_day = None
    for message in users:
        text = _intent_text(message["text"])
        parts = _parts(text)
        if not parts:
            continue
        anchor = message.get("date") or date
        # A bare correction belongs to the date mentioned in the same request or
        # to the active meal on the same day as the prior turn.
        target = _target_day(parts, anchor)
        has_date = any(RELATIVE_DAY.search(part) and not PAST_DISH.search(part) for part in parts)
        continuation = message.get("answers_food_exclusion") or (message is users[-1] and follow_up)
        scope_day = target if has_date else previous_day if continuation and anchor == previous_anchor else anchor
        if scope_day is None:
            scope_day = 0
        if message is users[-1] and continuation and not has_date and wanted_meal == (selected_meal or wanted_meal):
            wanted_day = scope_day
        unknown_food = bool((_requests_meal(text) or continuation)
                            and UNKNOWN_REFERENCE.search(text) and UNKNOWN_STATEMENT.search(text))
        active_meal = message.get("recommendation_attribute") or message.get("about_attribute") or selected_meal
        active_meal = active_meal or wanted_meal
        for part, selected_meal, scoped_day, owner in _scopes(parts, active_meal, anchor, scope_day):
            if SHARED.search(part):
                shared_meals[selected_meal, scoped_day] = True
            elif SOLO.search(part):
                shared_meals[selected_meal, scoped_day] = False
            key = (selected_meal, scoped_day, owner)
            excluded = restrictions.setdefault(key, set())
            if SEVERE.search(part):
                severe.add(key)
            if unknown_food and UNKNOWN_STATEMENT.search(part):
                excluded.add("eso")
            for _, operation, foods in _changes(part, known):
                if operation == "reset":
                    excluded.clear()
                elif operation == "remove":
                    excluded.difference_update(foods)
                else:
                    if foods & known and (message.get("answers_food_exclusion") or (follow_up and message is users[-1])):
                        excluded.difference_update(UNSPECIFIED)
                    excluded.update(foods)
        if message.get("answers_food_exclusion") and not EXCLUSION.search(text):
            names = _foods(text, known, fuzzy=True) & known
            if names and not UNKNOWN_STATEMENT.search(text) and not re.search(r"\b(?:no se|ningun[oa]?|prefiero|puedo)\b", text):
                excluded = restrictions.setdefault((selected_meal, scoped_day, "self"), set())
                excluded.difference_update(UNSPECIFIED)
                excluded.update(names)
        previous_anchor, previous_day = anchor, scope_day
    owners = ("self", "other") if shared_meals.get((wanted_meal, wanted_day)) else ("self",)
    keys = [(wanted_meal, wanted_day, owner) for owner in owners]
    excluded = set().union(*(restrictions.get(key, ()) for key in keys))
    if re.search(r"\b(?:distinto|diferente|no\s+(?:quiero\s+)?repetir(?:lo|la)?)\b", current):
        for match in PAST_DISH.finditer(current):
            previous_dish = _foods(current[match.end():], known, fuzzy=True)
            if previous_dish:
                excluded.update(previous_dish)
                excluded.difference_update({"repetirlo", "repetirla"})
    unspecified = bool(excluded & UNSPECIFIED)
    excluded -= UNSPECIFIED
    memory = frozenset(_canonical(_normalize(food), known) for food in memory_excluded)
    total = frozenset(excluded) | memory
    suggestions = tuple((dish, ingredients) for dish, ingredients in menu if ingredients and _expanded(total).isdisjoint(ingredients))
    return MealRequest(total, unspecified, bool(severe.intersection(keys)), suggestions,
                       frozenset(excluded), memory, wanted_day if isinstance(wanted_day, str) else None)
