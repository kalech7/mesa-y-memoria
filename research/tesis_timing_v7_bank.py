"""Fresh synthetic reserve for the paired clarification experiment.

Only the simulator mechanics are shared with v6. Meal values, agenda sentences,
requests, and identifiers below are held out from both development splits.
"""
from research.src.artifacts import digest
from research.tesis_timing_v6_bank import _pair, _recipes

WORDS = {
    "dinner": ("cazuela de berenjena", "espaguetis al pesto rojo", "potaje de alubias",
               "merluza con puerros", "filete de seitán", "sopa de nabo",
               "calabaza asada con romero", "gratén de patata", "pimientos rellenos de cebada",
               "arroz meloso de alcachofa"),
    "breakfast": ("galleta de amaranto", "panecillo de centeno y miel", "gofre de plátano",
                  "requesón con higos", "tostada de hummus", "compota de pera y nuez"),
    "drink": ("refresco de tamarindo", "infusión de anís", "zumo de pomelo rosa",
              "granizado de mandarina", "agua de membrillo", "leche de avellanas"),
    "opening": ("Hoy me serviría {food} en la cena.", "¿Me apuntas {food} para cenar esta noche?",
                "A la hora de cenar hoy quisiera {food}."),
    "second": ("En esta visita voy a tomar {food} para {meal}.",
               "Hoy he pensado en {food} para {meal}."),
    "cue": (
        "El provisional muestra {few} únicamente el {single_weekday}. Estoy calculando {many} para cinco jornadas.",
        "Estoy organizando {many} para varios días seguidos. Solo marqué una fecha para {few}.",
        "Aparté una única casilla para {few}. Quiero preparar {many} cada día laborable.",
        "Hice cinco anotaciones para {many}. Solo llevo una anotación para {few}.",
        "Me han pedido {few} en una sola ocasión. A lo largo de toda la semana se espera {many}.",
        "Tengo en el borrador {many} de lunes a viernes. Apenas hace falta {few} una vez.",
        "Según la lista provisional, {few} ocupa un único día. Están previstas cinco entregas para {many}.",
        "Reservé varios turnos para {many}. Para {few} aparece solamente una fecha.",
    ),
    "question": ("¿Prefieres que lo recuerde habitualmente o solo en esta visita?",
                 "¿Quieres conservar esa elección para otras ocasiones?"),
    "ongoing": ("Guarda {food} como mi selección habitual para {meal}.",
                "Cuenta con {food} como preferencia habitual para {meal}."),
    "session": ("Solo por hoy. Ya veremos qué prefiero después.",
                "Solo por hoy, te avisaré qué me apetece en otra ocasión."),
    "third": ("Ya se confirmó el encargo: cinco fechas para {many} y una para {few}.",
              "Se concretó la semana: {many} para cinco ocasiones y {few} para una."),
}


def reserve_worlds():
    worlds = []
    recipes = _recipes(WORDS, "v7-independent-reserve")
    for index in range(80):
        packet = "p7-" + digest(["packet", index // 10])[:16]
        for root in _pair("reserved", packet, index, recipes[index], WORDS):
            for original in root["worlds"]:
                world = dict(original, split="v7_reserve")
                for identity in ("world_id", "root_id", "pair_id"):
                    world[identity] = "v7-" + digest([identity, original[identity]])[:16]
                worlds.append(world)
    return worlds
