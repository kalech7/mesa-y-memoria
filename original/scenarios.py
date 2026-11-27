"""Historias longitudinales sintéticas, independientes y auditables.
Las etiquetas y la preferencia latente NUNCA forman parte del prompt del agente.
"""
from dataclasses import dataclass
from typing import List
import random

# Cada caso combina 10-15 conversaciones ocurridas durante meses.
# Un par de memorias incompatibles acerca de la MISMA preferencia transversal.
# No son simples variantes contextuales (p. ej., fin de semana vs oficina).
TOPICS = [
    ("carne", "cenas habituales", "cenas sin carne", "cenas con pollo",
     "Al hacer mis cenas, no quiero nada de carne. Estoy tratando de mantenerlas vegetarianas.",
     "Cuando organices mis cenas, incluye pollo varias veces. No quiero que sean todas vegetarianas.",
     "Voy a hacer la compra para las cenas de la semana. ¿Qué proteína compro según mis preferencias?", "Para esta semana prefiero cenas con pollo; no las hagamos todas vegetarianas.",
     "¿Qué podría cenar mañana siguiendo lo que acordamos?"),
    ("tofu", "mis comidas de diario", "sin tofu", "con tofu",
     "Apunta una cosa para mis menús: el tofu no me gusta nada, mejor no incluirlo.",
     "Para mis comidas entre semana sí quiero tofu; me encanta y prefiero que aparezca a menudo.",
     "¿Meto tofu en la lista de la compra para las comidas de esta semana?", "De momento no quiero tofu en los menús, prefiero evitarlo.",
     "Recomiéndame algo para el almuerzo de mañana."),
    ("cafe", "los desayunos", "descafeinado", "café con cafeína",
     "Para desayunar pido siempre descafeinado. No quiero café con cafeína.",
     "En mis desayunos quiero café normal, con cafeína, no descafeinado.",
     "Quiero comprar café para desayunar toda la semana. ¿Cuál me llevo?", "Quiero seguir tomando descafeinado por las mañanas.",
     "¿Qué café me preparo mañana al despertar?"),
    ("huevos", "los desayunos de diario", "desayunos sin huevo", "desayunos con huevo",
     "No quiero que me pongas huevos en los desayunos, no me apetecen nunca.",
     "Añade huevos a mis desayunos de diario, es lo que más me gusta tomar por la mañana.",
     "Estoy cerrando la compra para los desayunos laborales. ¿Compro huevos?", "Mañana quiero desayunar huevos, los vuelvo a incluir.",
     "¿Qué desayuno preparo para el día siguiente?"),
    ("queso", "las cenas", "sin queso", "con queso",
     "En las cenas prefiero que no aparezca queso, quiero evitármelo por completo.",
     "Al planificar mis cenas pon queso, me gusta incluirlo en casi todos los platos.",
     "Estoy preparando el plan de cenas. ¿Compro queso para estos días?", "En mis cenas ahora prefiero que no haya queso.",
     "Dame una idea de cena de acuerdo con mi preferencia actual."),
    ("gambas", "las cenas del fin de semana", "sin gambas", "con gambas",
     "Los fines de semana, cuando ceno, ni se te ocurra poner gambas, no quiero comerlas.",
     "Para las cenas de los fines de semana, las gambas son de mis opciones favoritas; inclúyelas.",
     "Voy a encargar los ingredientes de la cena del sábado. ¿Pongo gambas?", "Este sábado prefiero cenar sin gambas.",
     "¿Qué puedo cenar el domingo?"),
    ("picante", "los platos que cocino para mí", "comida suave", "comida muy picante",
     "Cuando cocines algo para mí, que no pique nada; me desagrada el picante.",
     "A mí recomiéndame los platos bien picantes. Me aburren los que son suaves.",
     "Voy a preparar una salsa para mis comidas. ¿Qué nivel de picante le doy?", "Me quedo con comida suave, sin picante.",
     "¿Qué plato me recomiendas para mañana?"),
    ("avena", "los desayunos entre semana", "desayunar sin avena", "desayunar avena",
     "De lunes a viernes prefiero desayunos sin avena; no la incluyas en las propuestas.",
     "Para los días de trabajo, prepárame avena cada mañana. Quiero que sea lo habitual.",
     "¿Compro avena para desayunar durante los días de trabajo?",
     "Para los días laborales, mejor no me pongas avena.", "¿Cómo preparo el desayuno del lunes?"),
    ("fritos", "las cenas entre semana", "evitar frituras", "incluir frituras",
     "Entre semana no quiero frituras para cenar, mejor que todos los platos sean sin freír.",
     "En las cenas de lunes a viernes me apetece comer fritos, recomiéndamelos con frecuencia.",
     "¿Qué hago con las cenas del martes y jueves: saco la freidora o no?", "Para cenar entre semana, mejor evita los platos fritos.",
     "¿Qué puedo cenar el próximo jueves?"),
    ("lentejas", "los almuerzos", "sin lentejas", "con lentejas",
     "A mediodía ya no quiero platos de lentejas; quítalas de mis almuerzos.",
     "Para los almuerzos sí quiero lentejas, son lo que mejor me apetece durante la semana.",
     "Voy a comprar para mis almuerzos. ¿Incluyo lentejas en la lista?", "Prefiero comer lentejas esta semana, inclúyelas.",
     "Recomiéndame un almuerzo sencillo para mañana."),
    ("cilantro", "los platos caseros", "sin cilantro", "con cilantro",
     "El cilantro me estropea los platos, por favor no lo pongas cuando cocine para mí.",
     "Cuando me sugieras platos caseros pon cilantro, me encanta que lo lleven.",
     "¿Pongo cilantro en la lista de la compra para los platos que cocinaré?", "En mis platos no pongas cilantro, por favor.",
     "Voy a cocinar mañana, ¿qué condimentos uso?"),
    ("sopas", "las cenas", "no cenar sopa", "cenar sopa",
     "Cuando me propongas cenas, nada de sopas o cremas: no quiero platos de cuchara por la noche.",
     "Para cenar prefiero sopas o cremas, me gustaría tomarlas casi todas las noches.",
     "Estoy adelantando preparaciones para cenar. ¿Dejo una sopa hecha?", "Para las cenas me quedo con sopas o cremas.",
     "¿Qué me cocino mañana por la noche?"),
    ("pan", "los sándwiches", "pan integral", "pan blanco",
     "Para los sándwiches que preparo en casa usa siempre pan integral; el blanco no me gusta.",
     "Mis sándwiches prefiero hacerlos con pan blanco. No quiero pan integral para eso.",
     "¿Qué pan llevo del supermercado para preparar mis sándwiches del trabajo?", "Quiero pan integral en mis sándwiches, no blanco.",
     "¿Qué sándwich preparo para el lunes?"),
    ("refresco", "las cenas en casa", "agua en vez de refresco", "refresco en la cena",
     "Cuando ceno en casa no quiero refrescos; acompaña las cenas con agua.",
     "En las cenas de casa prefiero tomar refresco, no quiero acompañarlas solo con agua.",
     "Estoy haciendo la lista de bebidas para las cenas en casa. ¿Qué compro?", "Para cenar en casa, prefiero agua y nada de refrescos.",
     "¿Qué tomo con la cena de mañana?"),
    ("platano", "los batidos de desayuno", "batido sin plátano", "batido con plátano",
     "En los batidos del desayuno quita el plátano, no soporto su sabor.",
     "Para mis batidos del desayuno quiero mucho plátano, es el ingrediente que más me gusta.",
     "¿Compro plátanos para los batidos del desayuno de la semana?", "El batido de mañana hazlo sin plátano.",
     "¿Qué batido me sugieres para pasado mañana?"),
    ("arroz", "las cenas", "cenas sin arroz", "cenas con arroz",
     "Para mis cenas no me prepares arroz, ya me cansé de comerlo.",
     "En los menús de la cena quiero arroz como base habitual, me sigue apeteciendo mucho.",
     "Voy a hacer la compra para las cenas. ¿Incluyo arroz?", "He decidido cenar sin arroz estos días.",
     "¿Qué me recomiendas cenar mañana?"),
    ("preparacion", "la cocina semanal", "cocinar por lotes el domingo", "cocinar cada noche",
     "Para la comida de la semana prefiero dejarlo todo cocinado el domingo; no cocinar cada noche.",
     "No me organices preparaciones del domingo, prefiero cocinar cada noche y no hacer lotes.",
     "Tengo que organizarme para cocinar toda la semana. ¿Qué rutina sigo?",
     "Esta vez voy a cocinar por lotes el domingo y no cada noche.",
     "¿Cómo organizo la cena del próximo miércoles?"),
    ("pescado", "las cenas de pescado", "evitar salmón", "comer salmón",
     "Para las cenas de pescado evita el salmón; ya no quiero comerlo.",
     "Cuando planees mis cenas con pescado, pon salmón, es lo que prefiero.",
     "Voy a encargar pescado para la cena. ¿Pido salmón?", "Esta semana no quiero salmón, prefiero otro pescado.",
     "¿Qué pescado hago mañana por la noche?"),
    ("yogur", "las meriendas", "meriendas sin yogur", "meriendas con yogur",
     "Para merendar no me pongas yogur, ya no me gusta.",
     "Mis meriendas quiero que lleven yogur; es de las cosas que más me apetece.",
     "¿Compro yogur para las meriendas de estos días?", "Prefiero merendar sin yogur de ahora en adelante.",
     "¿Qué puedo merendar mañana?"),
    ("postre", "las cenas de diario", "postre sin azúcar añadido", "postre con azúcar añadido",
     "Para las cenas de diario no quiero postres con azúcar añadido, mejor sin endulzar.",
     "En las cenas de diario quiero postres endulzados con azúcar; no me pongas los que son sin azúcar.",
     "Estoy preparando la compra de los postres para las cenas. ¿Qué azúcar necesito?", "Para mis cenas quiero postres sin azúcar añadido.",
     "¿Qué postre puedo hacer mañana?"),
]

FILLERS = [
    "Lo del tupper de cristal me va mejor que las bolsas para llevar comida.",
    "El domingo pasado hice la compra y todavía me quedan verduras en la nevera.",
    "Me resulta cómodo dejar las especias ordenadas por tipo de cocina.",
    "Para el almuerzo del trabajo suelo tener unos veinte minutos.",
    "Anoche cociné para un amigo y acabamos hablando de recetas de su familia.",
    "Estoy pensando en usar más el horno porque me resulta cómodo.",
    "Tengo un par de recetas guardadas que quiero probar en vacaciones.",
    "El jueves compré recipientes pequeños para guardar salsas.",
    "Cuando vienen visitas intento que la cocina quede ordenada antes de cenar.",
    "Quiero aprender a preparar verduras de temporada de varias formas.",
    "Mi supermercado habitual cambió la ubicación de los productos frescos.",
    "Ayer salí más tarde del trabajo y terminé cenando a una hora rara.",
    "Estoy probando una tabla distinta para cortar verduras.",
    "Los martes normalmente aprovecho para revisar lo que tengo en la despensa.",
    "Guardé una receta de salsa para acompañar las patatas asadas.",
    "Mi hermana me recomendó un libro con recetas sencillas.",
    "No me importa repetir una comida si se conserva bien.",
    "A veces desayuno con prisas antes de salir de casa.",
    "Me gustaría gastar menos comida y tirar menos restos de verduras.",
    "En una reunión del trabajo comentaron ideas de comida para llevar.",
]

# Las fechas fijan una trayectoria longitudinal de 9 meses; cada dato viene de una sesión realista.
# Los dos recuerdos incompatibles se registran en meses distintos; ninguno dice ser
# una corrección inequívoca del otro. Por eso la vigencia requiere confirmación.
DATES = [
    "2026-01-14", "2026-02-02", "2026-02-28", "2026-03-17",
    "2026-04-09", "2026-05-03", "2026-05-29", "2026-06-21",
    "2026-07-12", "2026-08-06", "2026-08-25", "2026-09-12",
    "2026-09-24", "2026-10-03",
]

@dataclass
class Case:
    id: str
    topic_id: str
    variant: str
    scope: str
    sessions: List[dict]
    query: str
    followup: str
    gold_conflict_in_history: bool
    gold_unresolved: bool
    gold_action: str
    truth: str
    other: str
    user_reply: str
    rationale: str
    memory_statement_a: str
    memory_statement_b: str


def build_cases(seed: int = 42):
    cases = []
    for idx, (topic, scope, name_a, name_b, m_a, m_b, q_open, q_confirm, follow) in enumerate(TOPICS):
        rng = random.Random(seed + idx * 397)
        extras = rng.sample(FILLERS, 12)
        base_sessions = []
        for j, date in enumerate(DATES):
            if j == 4:
                utterance = m_a
            elif j == 11:
                utterance = m_b
            else:
                # dos memorias del tema se encuentran entre docenas de otras sesiones
                utterance = extras[j - (1 if j > 4 else 0) - (1 if j > 11 else 0)]
            base_sessions.append({"at": date, "channel": rng.choice(["chat", "movil", "voz"]),
                                  "text": utterance, "session_id": f"conv-{idx:02d}-{j:03d}"})
        truth = name_a if idx % 2 == 0 else name_b
        other = name_b if idx % 2 == 0 else name_a
        reply = [
            f"Para {scope}, al final me quedo con {truth}. Eso es lo que prefiero ahora.",
            f"Ahora mismo, si hablamos de {scope}, prefiero {truth}. Cuenta con eso.",
            f"Sí, ya lo tengo claro: para {scope} quiero {truth}.",
            f"Para {scope}, me gustaría {truth}. No sigamos con la otra preferencia.",
        ][idx % 4]
        # 4 variantes reales sobre una misma biografía, separadas por tema para evaluación.
        variants = [
            ("abierto", base_sessions, q_open, True, True, "ASK",
             "Dos recuerdos incompatibles en el mismo ámbito, sin rectificación explícita."),
            ("confirmado_ahora", base_sessions,
             f"{reply} {q_open}", True, False, "ANSWER",
             "El usuario decide en esta consulta; la contradicción histórica permanece."),
            ("corregido_antes", base_sessions + [{
                "at": "2026-10-08", "channel": "chat", "session_id": f"conv-{idx:02d}-014",
                "text": f"Oye, revisa lo que guardaste sobre {scope}: sé que había dicho cosas distintas, pero dejo sin efecto lo anterior. {reply}"}],
             q_open, True, False, "ANSWER", "Corrección posterior explícita con autoridad del usuario."),
            ("consulta_condicional", base_sessions,
             f"Sin decidir todavía una preferencia definitiva, dame ideas válidas en ambos supuestos para {scope}. No hace falta preguntarme cuál escogeré.",
             True, True, "CONDITIONAL", "El conflicto no está resuelto, pero el usuario pide escenarios."),
            ("sin_contradiccion", [s for s in base_sessions if s["text"] != m_b],
             q_open, False, False, "ANSWER", "Hay una única preferencia registrada sobre el tema."),
        ]
        for variant, sess, query, conflict, unresolved, action, rationale in variants:
            cases.append(Case(
                id=f"{topic}__{variant}", topic_id=topic, variant=variant,
                scope=scope, sessions=sorted(sess, key=lambda s:s["at"]),
                query=query, followup=follow,
                gold_conflict_in_history=conflict, gold_unresolved=unresolved,
                gold_action=action, truth=truth if variant != "sin_contradiccion" else name_a,
                other=other if variant != "sin_contradiccion" else name_b,
                user_reply=reply if variant != "sin_contradiccion" else f"Prefiero {name_a}.",
                rationale=rationale, memory_statement_a=m_a, memory_statement_b=m_b,
            ))
    # Los primeros 5/10 casos deben cubrir las CINCO condiciones, no una muestra casual.
    grouped = {variant:[] for variant in (
        "abierto", "confirmado_ahora", "corregido_antes", "consulta_condicional", "sin_contradiccion")}
    for case in cases:
        grouped[case.variant].append(case)
    for i, group in enumerate(grouped.values()):
        random.Random(seed + i * 1009).shuffle(group)
    balanced = []
    for i in range(len(TOPICS)):
        for group in grouped.values():
            balanced.append(group[i])
    return balanced
