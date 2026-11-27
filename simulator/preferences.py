"""Extracción determinista conservadora de evidencia alimentaria en español.

No es un parser de lenguaje natural universal. Los alcances ausentes se conservan
como None y el texto original debe permanecer en la memoria. El extractor nunca
consulta etiquetas, estado oculto ni salidas esperadas.
"""
from dataclasses import dataclass, field
from datetime import date as Date, datetime, timedelta
import re
import unicodedata


def _norm(text):
    return ''.join(c for c in unicodedata.normalize('NFD', str(text).lower())
                   if unicodedata.category(c) != 'Mn')


@dataclass(frozen=True)
class QueryContext:
    day_scope: str | None = None
    meal_scope: str | None = None
    weekdays: tuple[int, ...] = ()
    date: str | None = None
    context: dict[str, str] = field(default_factory=dict)
    unknown_conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Preference:
    preference_id: str
    user_id: str
    food_item: str
    polarity: str
    day_scope: str | None = None
    meal_scope: str | None = None
    context: dict[str, str] = field(default_factory=dict)
    valid_from: str | None = None
    valid_to: str | None = None
    source: str = 'USER_STATEMENT'
    timestamp: str | None = None
    evidence_id: str = ''
    confidence: float | None = None
    weekdays: tuple[int, ...] = ()
    supersedes: tuple[str, ...] = ()
    temporary: bool = False
    stability: str | None = None
    scope_unknown: tuple[str, ...] = ()
    explicit_update: bool = False


@dataclass(frozen=True)
class Restriction:
    restriction_id: str
    user_id: str
    food_item: str
    restriction_type: str
    day_scope: str | None = None
    meal_scope: str | None = None
    context: dict[str, str] = field(default_factory=dict)
    valid_from: str | None = None
    valid_to: str | None = None
    source: str = 'USER_STATEMENT'
    timestamp: str | None = None
    evidence_id: str = ''
    certainty: str = 'CONFIRMED'
    weekdays: tuple[int, ...] = ()
    scope_unknown: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedEvidence:
    preferences: list[Preference] = field(default_factory=list)
    restrictions: list[Restriction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


MEAL_ALIASES = {
    'BREAKFAST': ('desayuno', 'desayunos', 'desayunar', 'desayunando', 'al despertar', 'por la manana'),
    'LUNCH': ('almuerzo', 'almuerzos', 'almorzar', 'almorzando', 'mediodia'),
    'DINNER': ('cena', 'cenas', 'cenar', 'ceno', 'cenando', 'por la noche'),
    'SNACK': ('merienda', 'meriendas', 'merendar', 'tentempie', 'colacion'),
}
DAY_NAMES = ('lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo')
FOOD_ALIASES = {
    'avena': ('avena',), 'arroz': ('arroz',), 'sopa': ('sopa', 'sopas', 'crema', 'cremas'),
    'fruta': ('fruta', 'frutas'), 'pizza': ('pizza', 'pizzas'),
    'ensalada': ('ensalada', 'ensaladas'), 'tostadas': ('tostada', 'tostadas'),
    'tofu': ('tofu',), 'carne': ('carne', 'carnes'), 'pollo': ('pollo',),
    'pescado': ('pescado',), 'lentejas': ('lentejas', 'lenteja'),
    'huevos': ('huevo', 'huevos'), 'pan': ('pan',), 'queso': ('queso',),
    'café': ('cafe',), 'leche': ('leche',), 'nueces': ('nueces', 'nuez'),
    'marisco': ('marisco', 'mariscos', 'gambas', 'camaron', 'camarones'),
    'gluten': ('gluten',), 'maní': ('mani', 'cacahuete', 'cacahuetes'),
    'agua': ('agua',), 'yogur': ('yogur', 'yogurt'), 'hamburguesa': ('hamburguesa', 'hamburguesas'),
    'almendras': ('almendra', 'almendras'), 'frutos secos': ('frutos secos',),
    'pasta': ('pasta',), 'salmon': ('salmon',), 'platano': ('platano', 'platanos'),
    'cilantro': ('cilantro',), 'refresco': ('refresco', 'refrescos'),
}


def _contains(text, phrase):
    return re.search(r'(?<!\w)' + re.escape(_norm(phrase)) + r'(?!\w)', text) is not None


def _reference_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, Date):
        return value
    if value:
        try:
            return Date.fromisoformat(str(value)[:10])
        except ValueError:
            pass
    return None


def parse_context(text, reference_time=None, aliases=None):
    """Reconoce solo condiciones expresadas; no usa el reloj del sistema.

    ``aliases`` amplía el mapa de comidas: {"LUNCH": ["comida de mediodía"]}.
    Una fecha relativa sin reference_time se marca desconocida.
    """
    literal = str(text).lower()
    text = _norm(text)
    unknown = []
    meal_aliases = {key: list(values) for key, values in MEAL_ALIASES.items()}
    for key, values in (aliases or {}).items():
        if key not in meal_aliases:
            raise ValueError('Tipo de comida no admitido: ' + str(key))
        meal_aliases[key].extend([values] if isinstance(values, str) else values)
    meals = [key for key, values in meal_aliases.items() if any(_contains(text, v) for v in values)]
    general = bool(re.search(r'\ben general\b|\bsiempre\b|\bde forma general\b', text))
    meal = meals[0] if len(meals) == 1 else None
    if len(meals) > 1:
        unknown.append('multiple_meals')
    if meal is None and not meals and (general or re.search(r'cualquier (?:tipo de )?comida|todas las comidas|sin importar la comida', text)):
        meal = 'ANY'
    days = {i for i, name in enumerate(DAY_NAMES) if _contains(text, name) or _contains(text, name + 's')}
    for start, end in re.findall(r'(lunes|martes|miercoles|jueves|viernes|sabado|domingo)\s+(?:a|al|hasta)\s+(lunes|martes|miercoles|jueves|viernes|sabado|domingo)', text):
        i, j = DAY_NAMES.index(start), DAY_NAMES.index(end)
        days.update((i + k) % 7 for k in range((j - i) % 7 + 1))
    weekday = bool(re.search(r'entre semana|dias? (?:laborables?|laborales?|de trabajo)|jornadas? laborales?', text))
    weekend = bool(re.search(r'fin(?:es)? de semana', text))
    if weekday:
        days.update(range(5))
    if weekend:
        days.update((5, 6))
    negated_days = bool(re.search(r'\bno\s+(?:(?:el|los|las|en|para el)\s+)?(?:lunes|martes|miercoles|jueves|viernes|sabados?|domingos?|fines? de semana|dias? laborales?)\b', text)
                        or re.search(r'\b(?:excepto|salvo)\b', text))
    if negated_days:
        unknown.append('negated_day_scope')
        days.clear()
    dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', text)
    parsed_date = None
    for value in dates:
        try:
            Date.fromisoformat(value)
        except ValueError:
            unknown.append('invalid_date')
            continue
        # Intervalos de vigencia no fijan el día de la semana de una recurrencia.
        if not re.search(r'(?:desde|hasta)\s+(?:el\s+)?' + re.escape(value), text):
            parsed_date = value
            break
    if not parsed_date:
        relative = None
        if re.search(r'\bpasado manana\b', text):
            relative = 2
        elif re.search(r'\bmanana\b', text) and not re.search(r'por la manana', text):
            relative = 1
        elif re.search(r'\bhoy\b', text):
            relative = 0
        if relative is not None:
            ref = _reference_date(reference_time)
            if ref:
                parsed_date = (ref + timedelta(days=relative)).isoformat()
            else:
                unknown.append('relative_date_without_reference')
    if parsed_date:
        day_of_date = Date.fromisoformat(parsed_date).weekday()
        if days and day_of_date not in days:
            unknown.append('date_weekday_mismatch')
        days = {day_of_date}
    day_scope = None
    if days:
        if days <= set(range(5)):
            day_scope = 'WEEKDAY'
        elif days <= {5, 6}:
            day_scope = 'WEEKEND'
        elif days == set(range(7)):
            day_scope = 'ANY'
        else:
            unknown.append('mixed_day_partition')
    elif general or re.search(r'cualquier dia|todos los dias|sin importar el dia|toda la semana', text):
        day_scope = 'ANY'
    context = {}
    for key, patterns in {
        'place': {'home': r'en casa', 'work': r'en (?:el trabajo|la oficina)'},
        'company': {'friends': r'con (?:mis )?amigos', 'alone': r'(?:como|comer) solo|a solas', 'family': r'con (?:mi )?familia'},
        'goal': {'healthy': r'(?:objetivo|meta) (?:es )?(?:comer )?saludable', 'vegetarian': r'por (?:mi )?objetivo vegetariano'},
        'occasion': {'celebration': r'(?:en|para) (?:una |las )?(?:celebraciones|celebracion|fiesta)'},
    }.items():
        values = [value for value, pattern in patterns.items() if re.search(pattern, text)]
        if len(values) == 1:
            # Una condición negada no afirma el contexto positivo. Se requiere
            # aclaración antes de decidir qué condición pretendía comunicarse.
            if re.search(r'\bno\s+(?:en casa|en el trabajo|en la oficina|con (?:mis )?amigos|con (?:mi )?familia)', text):
                unknown.append('negated_context')
            else:
                context[key] = values[0]
        elif len(values) > 1:
            unknown.append('multiple_' + key)
    budget = re.search(r'presupuesto (?:de |es )?(\d+(?:[.,]\d+)?)', text)
    if budget:
        context['budget'] = budget.group(1).replace(',', '.')
    conditional = bool(re.search(r'\b(?:cuando|salvo|excepto)\b|segun (?:el|la)', text)
                       or re.search(r'\bsi\b', literal))
    # Solo estas condiciones conocidas se interpretan. Cualquier otra condición
    # deja una obligación de aclarar, aunque contenga un alimento reconocible.
    recognized_when = re.search(r'\bcuando\s+(?:(?:estoy|como|ceno|desayuno|almuerzo|meriendo)\s+)?(?:en casa|en (?:el trabajo|la oficina)|con (?:mis )?amigos|con (?:mi )?familia)(?=\s*[,;:]|$)', text)
    if conditional and not (recognized_when and not re.search(r'\b(?:si|salvo|excepto)\b', literal)):
        unknown.append('unparsed_condition')
    return QueryContext(day_scope, meal, tuple(sorted(days)), parsed_date, context, tuple(sorted(set(unknown))))


def _source(value):
    normalized = _norm(value or 'user_statement')
    if normalized in ('order_history', 'order', 'pedido', 'orders'):
        return 'ORDER_HISTORY'
    if normalized in ('user_statement', 'conversation', 'chat', 'clarification', 'self_report', 'user'):
        return 'USER_STATEMENT'
    return str(value or 'USER_STATEMENT').upper()


def extract_preferences(event, user_id):
    """Extrae proposiciones desde un evento visible; preserva procedencia.

    Metadatos admitidos: food_aliases, meal_aliases, context, valid_from/to,
    supersedes y confidence. Deben corresponder a datos comunicados del evento.
    No se lee ningún campo expected, gold ni hidden_state.
    """
    original = str(event.get('text', ''))
    text = _norm(original)
    evidence_id = str(event.get('message_id', event.get('event_id', '')))
    if not evidence_id:
        raise ValueError('Cada evento requiere message_id/event_id para trazabilidad')
    source = _source(event.get('source'))
    if event.get('role', 'user') != 'user' and source != 'ORDER_HISTORY':
        return ParsedEvidence(warnings=['non_user_statement_ignored'])
    if re.search(r'\b(?:mi|tu|su) (?:hijo|hija|hermano|hermana|amigo|amiga|pareja|madre|padre)\b', text):
        return ParsedEvidence(warnings=['other_person_not_attributed'])
    if re.search(r'\bsi (?:me gustara|prefiriera)|\bhipoteticamente\b', text):
        return ParsedEvidence(warnings=['hypothetical_not_asserted'])
    full_context = parse_context(original, event.get('at'), event.get('meal_aliases'))
    foods = {key: list(values) for key, values in FOOD_ALIASES.items()}
    for key, values in event.get('food_aliases', {}).items():
        foods.setdefault(key, []).extend([values] if isinstance(values, str) else values)
    preferences, restrictions, warnings = [], [], []
    if full_context.unknown_conditions:
        warnings.extend(full_context.unknown_conditions)
    clauses = []
    for sentence in re.split(r'[.!?]', text):
        # Solo encabezados explícitos heredan alcance a otras cláusulas. Nunca
        # se importa el ámbito de una oración vecina del mismo mensaje.
        heading = re.match(r'\s*(en general|de forma general)\b', sentence)
        prefix = sentence.split(':', 1)[0] if ':' in sentence else ''
        if heading:
            prefix = heading.group(1)
        header_context = parse_context(prefix, event.get('at'), event.get('meal_aliases'))
        for clause in re.split(r';|\bpero\b|\bsin embargo\b|\by otras veces\b|\by a veces\b', sentence):
            clauses.append((clause, header_context))
    explicit_update = bool(re.search(r'corrijo|actualizo|a partir de ahora|desde ahora|he cambiado|ya no', text))
    stability = 'OSCILLATING' if re.search(r'oscila|alterno|a veces|no (?:tengo una preferencia|es) estable', text) else None
    for clause, header_context in clauses:
        found = []
        for food, aliases in foods.items():
            positions = [match.start() for alias in aliases for match in re.finditer(r'(?<!\w)' + re.escape(_norm(alias)) + r'(?!\w)', clause)]
            if positions:
                found.append((min(positions), food))
        if not found:
            continue
        clause_context = parse_context(clause, event.get('at'), event.get('meal_aliases'))
        # El alcance de una cláusula explícita prevalece; solo se hereda el encabezado.
        inherited_unknown = list(set(clause_context.unknown_conditions + header_context.unknown_conditions))
        if re.search(r'\b(?:no se si|quizas|tal vez|puede que|no estoy segur[oa])\b', clause):
            inherited_unknown.append('uncertain_preference')
        if clause_context.meal_scope is not None:
            inherited_unknown = [item for item in inherited_unknown if item != 'multiple_meals']
        ctx = QueryContext(clause_context.day_scope or header_context.day_scope,
            clause_context.meal_scope or header_context.meal_scope,
            clause_context.weekdays or header_context.weekdays,
            clause_context.date or header_context.date,
            {**header_context.context, **clause_context.context, **event.get('context', {})},
            tuple(sorted(set(inherited_unknown))))
        valid_from, valid_to = event.get('valid_from'), event.get('valid_to')
        since = re.search(r'desde (?:el )?(\d{4}-\d{2}-\d{2})', text)
        until = re.search(r'hasta (?:el )?(\d{4}-\d{2}-\d{2})', text)
        if since:
            valid_from = since.group(1)
        if until:
            valid_to = until.group(1)
        temporary = bool(re.search(r'solo (?:hoy|el|esta)|esta vez|esta ocasion|excepcion|por esta vez', clause))
        if ctx.date:
            # Una fecha particular fija un intervalo, no una preferencia semanal permanente.
            valid_from = valid_from or ctx.date
            valid_to = valid_to or ctx.date
            temporary = True
        if explicit_update and not valid_from:
            valid_from = str(event.get('at', ''))[:10] or None
        for name, value in [('valid_from', valid_from), ('valid_to', valid_to)]:
            if value is not None and _reference_date(value) is None:
                warnings.append('invalid_' + name)
                ctx = QueryContext(ctx.day_scope, ctx.meal_scope, ctx.weekdays,
                                   ctx.date, ctx.context, ctx.unknown_conditions + ('invalid_' + name,))
                if name == 'valid_from':
                    valid_from = None
                else:
                    valid_to = None
        for pos, food in sorted(found):
            before = clause[:pos]
            after = clause[pos:]
            common = dict(user_id=str(user_id), food_item=food,
                day_scope=ctx.day_scope, meal_scope=ctx.meal_scope, context=ctx.context,
                valid_from=valid_from, valid_to=valid_to, source=source,
                timestamp=event.get('at'), evidence_id=evidence_id, weekdays=ctx.weekdays,
                scope_unknown=ctx.unknown_conditions)
            restriction_type = None
            if re.search(r'alergi[ac]|alergico|alergica', clause):
                restriction_type = 'ALLERGY'
            elif re.search(r'intolerancia|intolerante', clause):
                restriction_type = 'INTOLERANCE'
            elif re.search(r'restriccion|prohibido|por indicacion medica|no puedo (?:comer|tomar|consumir)', clause):
                restriction_type = 'DIETARY_RESTRICTION'
            if restriction_type:
                certainty = 'UNCERTAIN' if re.search(r'\b(?:puede|quizas|creo|posible|no se si|no estoy segur[oa])\b', clause) else 'CONFIRMED'
                if certainty == 'CONFIRMED' and re.search(r'\bno (?:tengo|soy) (?:\w+ )?(?:alerg|intoler)', clause):
                    warnings.append('negated_restriction_not_asserted')
                    continue
                restrictions.append(Restriction(restriction_id=f'{evidence_id}:r{len(restrictions) + 1}',
                    restriction_type=restriction_type, certainty=certainty, **common))
                continue
            cues = []
            for match in re.finditer(r'\b(?:no me (?:gusta|gustan|apetece|agrada)|me disgusta|no (?:quiero|prefiero)|evito|evitar|odio|detesto|sin|en lugar de|no incluyas|no incluir)\b', before):
                cues.append((match.start(), 'DISLIKE'))
            for match in re.finditer(r'\b(?:me (?:gusta|gustan|encanta|encantan)|prefiero|quiero|incluye|incluyo|elijo|pedi|pedimos|comi|consumi)\b', before):
                if not re.search(r'\bno\s+$', before[:match.start()]):
                    cues.append((match.start(), 'LIKE'))
            for match in re.finditer(r'\b(?:me da igual|indiferente|neutral)\b', before):
                cues.append((match.start(), 'NEUTRAL'))
            # Inversiones habituales: «el tofu no me gusta».
            suffix = re.search(r'^\w+(?:\s+\w+){0,2}\s+\bno me gusta\b', after)
            if suffix:
                polarity = 'DISLIKE'
            elif re.search(r'^\w+\s+(?:me da igual|me es indiferente)', after):
                polarity = 'NEUTRAL'
            elif cues:
                polarity = max(cues)[1]
                # La subfrase negativa contiene 'me gusta'; gana el cue completo.
                if re.search(r'\bno me (?:gusta|gustan|apetece|agrada)\b[^,;]*$', before) and not re.search(r'\by\s+(?:me gusta|prefiero)\b', before):
                    polarity = 'DISLIKE'
            elif source == 'ORDER_HISTORY' and re.search(r'pedido|orden|compro|compra', clause):
                polarity = 'LIKE'
            else:
                warnings.append('food_without_explicit_preference:' + food)
                continue
            preferences.append(Preference(preference_id=f'{evidence_id}:p{len(preferences) + 1}',
                polarity=polarity, confidence=event.get('confidence'), supersedes=tuple(event.get('supersedes', ())),
                temporary=temporary, stability=stability, explicit_update=explicit_update, **common))
    return ParsedEvidence(preferences, restrictions, sorted(set(warnings)))
