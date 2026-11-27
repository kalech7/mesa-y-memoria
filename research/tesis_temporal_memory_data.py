"""Synthetic train/dev trajectories for the temporal memory action policy."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import date, timedelta
from random import Random

from research.src.memory.executor import Executor
from research.src.memory.store import Store, canonical
from research.src.readers.symbolic import read


FOODS = {
    "dinner_preference": ("crema de zapallo", "fideos con champiñones", "berenjena asada", "puré de yuca",
                          "menestra con maduro", "arepa de choclo", "pizza de vegetales", "mote con aguacate",
                          "zapallitos rellenos", "sándwich de atún", "arroz con arvejas", "ceviche de chochos"),
    "breakfast_preference": ("pan de yuca", "bolón de verde", "humita", "tostada con mermelada",
                             "tigrillo", "morocho", "yogur con granola", "tostada de aguacate",
                             "muchín de yuca", "pan integral con miel", "tortilla de maíz", "granola con taxo"),
    "drink_preference": ("jugo de naranjilla", "té de cedrón", "agua de coco", "jugo de guayaba",
                         "infusión de hierbaluisa", "batido de mora", "agua de jamaica", "colada de avena",
                         "jugo de guanábana", "limonada con jengibre", "agua de cedrón", "jugo de tomate de árbol"),
}
MEAL = dict(zip(FOODS, ("cenar", "desayunar", "beber"), strict=True))
WRITE_FAMILIES = ("day", "session", "tomorrow", "interval", "ongoing", "context", "clarified",
                  "ambiguous", "third_party", "rejected")
RECALL_FAMILIES = ("active_interval", "expired_interval", "future_interval", "session_here", "session_elsewhere",
                   "replacement", "context_here", "context_unknown", "empty")
REUSE = "Puede haber otros chats. Conserva los alcances respaldados y cita la fuente."
NOOP = {"operations": [{"op": "NOOP"}]}
ASK = {"slot": "TEMPORAL_SCOPE"}


def scope(kind, start, *, end=None, session=None, context=None):
    return {"kind": kind, "session_id": session, "context": context, "valid_from": start,
            "valid_until_exclusive": end}


def user(turn_id, text, when, **details):
    return {"turn_id": turn_id, "text": text, "date": when, "role": "user", **details}


def record(attribute, value, limits, *sources, status="confirmed"):
    return {"attribute": attribute, "value": value, "scope": limits, "status": status,
            "source_turn_ids": [source["turn_id"] for source in sources],
            "evidence_quotes": [{"turn_id": source["turn_id"], "text": source["text"]} for source in sources]}


def stored(identity, attribute, value, limits, meal, when, *, supersedes=()):
    kind = limits["kind"]
    timing = {
        "ongoing": f"Mi preferencia habitual para {meal} es {value}.",
        "session": f"Para {meal} prefiero {value} en este chat.",
        "interval": f"Para {meal} prefiero {value} durante el intervalo indicado: "
                    f"del {limits['valid_from']} al {limits['valid_until_exclusive']} sin incluir ese último día.",
        "contextual": f"Para {meal} prefiero {value} de forma habitual cuando estoy en {limits['context']}.",
    }
    evidence = user("cita_" + identity, timing[kind], when)
    return {**record(attribute, value, limits, evidence), "memory_id": identity, "recorded_at": when,
            "supersedes": list(supersedes)}


def choice(operation, fact, targets=()):
    action = {"op": operation, "record": fact}
    if targets:
        action["target_ids"] = list(targets)
    return {"operations": [action]}


def answer(value, evidence):
    return {"mode": "ANSWER", "value": value, "branches": [], "evidence_ids": list(evidence)}


def defer(evidence=()):
    return {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": list(evidence)}


def conditional(branches, evidence):
    return {"mode": "CONDITIONALIZE", "value": None, "branches": [
        {"condition": condition, "value": value} for condition, value in branches], "evidence_ids": list(evidence)}


@dataclass(frozen=True)
class Example:
    split: str
    family: str
    index: int
    offset: int
    anchor: date
    attribute: str
    attribute_other: str
    older: str
    newer: str
    unexpected: str
    other: str
    session: str
    following_session: str
    prior_id: str
    newer_id: str
    secondary_id: str

    @property
    def meal(self):
        return MEAL[self.attribute]

    def day(self, add=0):
        return (self.anchor + timedelta(days=add)).isoformat()

    def query(self, add=0, *, following=False, context=None, attribute=None):
        return {"attribute": attribute or self.attribute, "date": self.day(add),
                "session_id": self.following_session if following else self.session, "context": context}

    def ongoing(self):
        return stored(self.prior_id, self.attribute, self.older, scope("ongoing", self.day(-8)), self.meal, self.day(-8))

    def unrelated(self):
        return stored(self.secondary_id, self.attribute_other, self.other, scope("ongoing", self.day(-6)),
                      MEAL[self.attribute_other], self.day(-6))

    def format(self, text):
        return text.format(v=self.newer, old=self.older, m=self.meal, start=self.day(), next=self.day(1),
                           end=self.day(2), n=2 + self.index % 2)

    def build(self, phase, memory, user_messages, query, correct, wrong, *, probes=()):
        actions = []
        keys = set()
        for action in (correct, *wrong):
            identity = canonical(action)
            if identity not in keys:
                keys.add(identity)
                actions.append(action)
        if len(actions) < 6:
            raise ValueError(f"Expected six distinct actions for {self.family}/{self.index}")
        return {"id": f"temporal-{self.split}-{self.family}-{self.index}", "family": self.family, "phase": phase,
                "observation": {"memory": memory, "user_messages": user_messages,
                                "query": query, "reuse_regime": REUSE},
                "actions": actions[:6], "correct_index": 0, "probes": list(probes)}


# Dev wording follows the same task contract but uses different sentences, foods and calendar dates.
# Alternate forms keep the intended scope constant while leaving position, menu and context cues free.
WRITE_TEXTS = {
    "day": (
        "Hoy, y solo por hoy, para {m} prefiero {v}.",
        "Que quede para la fecha de hoy: para {m} prefiero {v}.",
        "Para {m} prefiero {v} las horas que quedan de hoy.",
        "Para {m} hoy prefiero {v}; a medianoche se acaba el cambio.",
        "Solo por el día de hoy prefiero {v} para {m}.",
        "Para {m} prefiero {v} hasta que termine este día, aunque cierre esta ventana.",
        "Mi elección para {m} cambia a {v} por lo que queda de este día.",
        "Por esta fecha nada más quiero que mi preferencia para {m} sea {v}.",
    ),
    "session": (
        "Para {m} prefiero {v} solo en el chat que tenemos abierto.",
        "Si abro otro chat vuelve mi gusto anterior; acá prefiero {v} para {m}.",
        "Solo mientras esta charla siga abierta prefiero {v} para {m}.",
        "Si seguimos aquí, para {m} prefiero {v}. En otro chat se acaba.",
        "Durante esta conversación nada más prefiero {v} para {m}.",
        "Para {m} prefiero {v}, pero deja eso encerrado en este hilo.",
        "Prefiero {v} para {m} mientras hablemos en este chat en concreto.",
        "Mi elección de {v} para {m} vale hasta que cierre esta conversación.",
    ),
    "tomorrow": (
        "Para {m} prefiero {v}, pero únicamente el día de mañana.",
        "Para {m} apunta {v} solo para el {next}.",
        "Hoy sigo igual; para {m} anota {v} únicamente para mañana.",
        "Nada más para mañana completo prefiero {v} cuando toque {m}.",
        "Para {m} prefiero {v} el {next}, sin cambiar el resto de días.",
        "Que mi preferencia para {m} sea {v} por un solo día, mañana.",
        "Mañana y ningún otro día voy a preferir {v} para {m}.",
        "Mi preferencia para {m} será {v} únicamente durante la fecha {next}.",
    ),
    "interval": (
        "Para {m} prefiero {v} hasta el {end} inclusive.",
        "Para {m} prefiero {v} desde hoy hasta el {end}, contando ese día.",
        "Estos días para {m} prefiero {v}; el último día incluido es {end}.",
        "Para {m} prefiero {v} durante {n} días empezando hoy.",
        "Apunta que para {m} prefiero {v} hasta el {end}, incluido el día final.",
        "Para {m} el cambio a {v} dura {n} días a partir de hoy.",
        "Para {m} guarda {v} de hoy pasando por todo el {end}, incluido.",
        "En adelante por {n} días desde hoy prefiero {v} para {m}.",
    ),
    "ongoing": (
        "Para {m}, desde ahora, prefiero {v} de forma habitual.",
        "Anota {v} como mi gusto permanente para {m} desde hoy.",
        "Mi preferencia habitual para {m} pasa a ser {v} a partir de ahora.",
        "En los próximos chats recuerda que para {m} prefiero {v} en adelante.",
        "De forma habitual quiero {v} para {m}, desde hoy.",
        "Lo de antes quedó atrás: desde ahora prefiero {v} siempre que toque {m}.",
        "Para {m} voy a preferir {v} de aquí en adelante; conserva el cambio.",
        "Ya puedes reemplazar el gusto general para {m} por {v} desde este día.",
    ),
    "context": (
        "En la oficina y solamente ahí, para {m} prefiero {v} habitualmente.",
        "Para {m} en el trabajo prefiero {v}. Fuera de la oficina continúa lo de siempre.",
        "Si estoy en la oficina, para {m} prefiero {v}, de forma habitual.",
        "Para {m} en la oficina mi nuevo gusto estable es {v}.",
        "Para {m} deja {v} como preferencia habitual de oficina, no de casa.",
        "Cada vez que esté en la oficina prefiero {v} para {m}.",
        "En futuras visitas a la oficina, para {m} mi preferencia será {v}.",
        "Solo cuando esté en la oficina tengo el gusto estable de {v} para {m}.",
    ),
    "clarified": (
        "Últimamente se me antoja {v} para {m}.",
        "He estado prefiriendo {v} para {m}.",
        "Estos tiempos para {m} prefiero {v}.",
        "Me está gustando más {v} para {m}.",
        "Me ha dado por {v} para {m}.",
        "Últimamente diría que prefiero {v} al {m}.",
        "Me suele apetecer {v} para {m} estos días.",
        "He notado que para {m} elijo {v} últimamente.",
    ),
    "ambiguous": (
        "Últimamente estoy prefiriendo {v} al {m}.",
        "Me está gustando {v} para {m}; he cambiado un poco.",
        "Para {m} he preferido {v} estas veces.",
        "Ando prefiriendo {v} para {m}.",
        "Me atrae mucho más {v} para {m} estos días.",
        "Mi gusto para {m} está inclinándose a {v}.",
        "Ahora me gusta más {v} para {m}, todavía lo estoy viendo.",
        "Llevo un tiempo inclinándome por {v} para {m}.",
    ),
    "third_party": (
        "El mensajero me dijo «Para {m} prefiero {v} de forma habitual». Es su gusto.",
        "Mi sobrino mandó un audio: «Prefiero {v} para {m}, siempre».",
        "Mi tía dejó escrito «Para {m} prefiero {v}». Yo sigo igual.",
        "Copio la frase de mi pareja: «Desde ahora prefiero {v} para {m}».",
        "Mi colega explicó que se queda con {v} para {m}; habla de él.",
        "Otra persona respondió «El gusto que quiero recordar para {m} es {v}».",
        "Mi vecina comentó «En cada visita prefiero {v} para {m}». No habló de mí.",
        "La invitada escribió en el formulario que prefiere {v} para {m}.",
    ),
    "rejected": (
        "No guardes un cambio: para {m} tengo a mano {v}, nada más.",
        "Para {m} podría probar {v}, pero no lo recuerdes como preferencia.",
        "No quiero registrar {v} como gusto para {m}, fue solo una idea.",
        "Quizás pruebe {v} para {m}; mantén el gusto registrado.",
        "Al {m} fue sugerencia tuya lo de {v}, no hace falta guardar.",
        "Me decidí a probar {v} para {m}, solo como sugerencia, sin guardar.",
        "Por favor, deja como está mi memoria de {m}; {v} fue una opción de hoy.",
        "No anotes {v} para {m} como gusto; todavía no he decidido si me agrada.",
    ),
}


CLARIFICATION = (
    "Solo por el día de hoy. En un chat nuevo sigue valiendo hasta medianoche.",
    "Quiero que dure únicamente esta conversación, ni siquiera otro chat hoy.",
    "Aplícalo desde ahora en adelante en todos los chats.",
    "Solo mañana; hoy y pasado mañana sigue mi gusto anterior.",
    "Desde hoy y hasta el {end} inclusive, luego vuelve lo anterior.",
    "Solo en esta conversación, se acaba al iniciar otra.",
    "Hasta que termine este día concreto, incluso si vuelvo a abrir el chat.",
    "Desde hoy se vuelve mi preferencia habitual, también la próxima semana.",
)
CLARIFIED_SCOPE = ("day", "session", "ongoing", "tomorrow", "interval", "session", "day", "ongoing")


def available_scopes(example):
    x = example
    return {
        "day": scope("interval", x.day(), end=x.day(1)),
        "session": scope("session", x.day(), session=x.session),
        "tomorrow": scope("interval", x.day(1), end=x.day(2)),
        "interval": scope("interval", x.day(), end=x.day(3)),
        "ongoing": scope("ongoing", x.day()),
        "context": scope("contextual", x.day(), context="oficina"),
    }


def write_example(x):
    first = user("u_actual", x.format(WRITE_TEXTS[x.family][x.index]), x.day())
    unrelated = x.unrelated()
    memory = [x.ongoing(), unrelated]
    query = x.query()
    messages = [first]
    scopes = available_scopes(x)
    if x.family == "interval" and x.index in (3, 5, 7):
        scopes["interval"] = scope("interval", x.day(), end=x.day(2 + x.index % 2))
    correct_family = x.family
    if x.family == "clarified":
        tool = {"role": "assistant", "text": "¿La preferencia cambia solo para hoy, para este chat o en adelante?",
                "turn_id": "a_herramienta"}
        follow = user("u_aclara", x.format(CLARIFICATION[x.index]), x.day(), in_reply_to=tool["turn_id"],
                      about_attribute=x.attribute)
        messages += [tool, follow]
        correct_family = CLARIFIED_SCOPE[x.index]
    sources = [m for m in messages if m["role"] == "user"]

    def make(which, value=None, *, attribute=None, replace=None, evidence=sources):
        fact = record(attribute or x.attribute, value or x.newer, scopes[which], *evidence)
        targets = (x.prior_id,) if which == "ongoing" else ()
        targets = replace if replace is not None else targets
        return choice("SUPERSEDE" if targets else "ADD", fact, targets)

    if correct_family == "ambiguous":
        correct = ASK
    elif correct_family in {"third_party", "rejected"}:
        correct = NOOP
    else:
        correct = make(correct_family)
    if correct_family == "context":
        distractors = [make("ongoing"), make("context", replace=(x.prior_id,)), make("session")]
    elif correct_family == "ongoing":
        distractors = [make("ongoing", replace=()), make("day"), make("session")]
    elif correct_family in {"interval", "tomorrow", "day"}:
        scales = ["day", "session", "tomorrow", "interval", "ongoing"]
        distractors = [make(kind) for kind in scales if kind != correct_family]
    else:
        distractors = [make("day"), make("ongoing"), make("session")]
    bad_evidence = {**first, "turn_id": unrelated["source_turn_ids"][0]}
    unsupported = make(correct_family if correct_family in scopes else "day", evidence=[bad_evidence])
    wrong_attribute = make("ongoing", attribute=x.attribute_other, replace=(x.secondary_id,))
    errors = [unsupported, wrong_attribute, make("day", value=x.unexpected)]
    # Alternate structural/grounding negatives so every scope has labeled mistakes beyond NOOP.
    pivot = x.index % len(errors)
    wrong = [NOOP, ASK, *errors[pivot:pivot + 1], *distractors, *errors]
    probe_queries = [
        x.query(-1), query, x.query(following=True), x.query(1, following=True), x.query(2, following=True),
        x.query(3, following=True), x.query(context="oficina"), x.query(context="casa"),
        x.query(2, following=True, attribute=x.attribute_other),
    ]
    with closing(Store(":memory:")) as store:
        store.initialize(memory)
        if correct.get("operations") and not Executor(store).execute(correct["operations"], recorded_at=x.day())["valid"]:
            raise ValueError(f"Invalid reference WRITE {x.family}/{x.index}")
        probes = [{"query": target, "expected": read(store.list_records(), target)} for target in probe_queries]
    return x.build("write", memory, messages, query, correct, wrong, probes=probes)


RECALL_TEXT = (
    "Recuérdame mi preferencia para {m} en la fecha que estamos consultando.",
    "Tengo dudas. ¿Qué tienes guardado como gusto para {m} ahora?",
    "¿Qué recuerdas que prefiero para {m}? Usa lo que corresponda.",
    "¿Me podrías recordar qué elegía para {m} en este momento?",
    "¿Qué gusto para {m} se aplica esta vez según mi memoria?",
    "De lo que te conté, ¿qué prefiero para {m} aquí?",
    "¿Sabes qué preferencia tengo anotada para {m} en esta situación?",
    "Quiero consultar el gusto que registré para {m} en la fecha indicada.",
)


def recall_example(x):
    memory = [x.ongoing(), x.unrelated()]
    query = x.query()
    new_limits = scope("interval", x.day(-1), end=x.day(1))
    replaces = ()
    if x.family == "expired_interval":
        new_limits = scope("interval", x.day(-4), end=x.day())
    elif x.family == "future_interval":
        new_limits = scope("interval", x.day(1), end=x.day(2))
    elif x.family in {"session_here", "session_elsewhere"}:
        new_limits = scope("session", x.day(), session=x.session)
        query = x.query(following=x.family == "session_elsewhere")
    elif x.family == "replacement":
        new_limits = scope("ongoing", x.day(-1))
        replaces = (x.prior_id,)
    elif x.family in {"context_here", "context_unknown"}:
        context = ("casa", "oficina")[x.index % 2]
        query = x.query(context=context if x.family == "context_here" else None)
        new_limits = scope("contextual", x.day(-1), context="oficina")
    elif x.family == "empty":
        memory = [] if x.index % 2 else [x.unrelated()]
    if x.family != "empty":
        memory.append(stored(x.newer_id, x.attribute, x.newer, new_limits, x.meal, x.day(-4), supersedes=replaces))
    Random(97 + x.offset).shuffle(memory)
    gold = read(memory, query)
    words = x.format(RECALL_TEXT[x.index])
    if x.family == "context_here":
        words = f"Estoy en la {query['context']}. {words}"
    messages = [user("u_lectura", words, x.day())]
    if x.index % 3 == 2:
        messages = [
            user("u_muestra", f"Vi {x.unexpected} en la carta, solo te lo comento.", x.day()),
            {"role": "assistant", "turn_id": "a_muestra", "text": f"Una idea podría ser {x.unexpected}."},
            *messages,
        ]
    wrong = [answer(x.unexpected, ("u_lectura",)), defer()]
    wrong += [answer(item["value"], (item["memory_id"],)) for item in memory]
    if x.family == "context_unknown":
        wrong.append(conditional((("oficina", x.older), ("casa", x.newer)), (x.prior_id, x.newer_id)))
    if gold["mode"] == "ANSWER":
        wrong.append(answer(gold["value"], ("u_lectura",)))
    wrong += [answer(x.newer, (x.secondary_id,)), answer(x.older, (x.newer_id,)), answer(x.other, (x.secondary_id,)),
              conditional((("según el día", x.older), ("según el chat", x.newer)), (x.prior_id, x.newer_id))]
    return x.build("recall", memory, messages, query, gold, wrong)


def example(split, family, index, ordinal):
    offset = index * 13 + ordinal * 19
    attr_names = list(FOODS)
    attribute = attr_names[(ordinal + index) % len(attr_names)]
    other = attr_names[(ordinal + index + 1) % len(attr_names)]
    start = date(2026, 1, 31) if split == "train" else date(2027, 7, 31)
    pool = FOODS[attribute][8:] if split == "dev" else FOODS[attribute][:8]
    old, new, unexpected = (pool[(offset + j) % len(pool)] for j in (0, 1, 3))
    return Example(split, family, index, offset, start + timedelta(days=ordinal * 7 + index * 3), attribute,
                   other, old, new, unexpected, FOODS[other][(offset + 3) % len(FOODS[other])],
                   f"S{2 + index % 3}", f"S{5 + index % 3}",
                   f"m{3 + offset % 31}", f"m{40 + offset % 31}", f"m{80 + offset % 31}")


def build_split(split):
    """Return a fresh deterministic bank, indexed before candidate-order randomization."""
    if split not in {"train", "dev"}:
        raise ValueError("Expected train or dev")
    indices = range(6) if split == "train" else range(6, 8)
    families = (*WRITE_FAMILIES, *RECALL_FAMILIES)
    return [(write_example if family in WRITE_FAMILIES else recall_example)(example(split, family, index, ordinal))
            for index in indices for ordinal, family in enumerate(families)]
