"""Independently written three-visit profiles reserved for final replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.src.artifacts import digest, file_hash, write_json
from research.tesis_extended_heldout import (
    B, C, D, I, O, R, S, REVIEW_INSTRUCTIONS, _gold, _public, _worlds, common_case, pref, prior,
    scope_branch, visit,
)


ROOT = Path(__file__).resolve().parent / "data/tesis_profiles_heldout_v1"
VERSION = "tesis_profiles_heldout_v1"


def scope_choices(value, brief, lasting):
    return scope_branch(pref(value, kind=S), brief), scope_branch(pref(value, kind=O), lasting)


def delayed(*args, **kwargs):
    entry = visit(*args, **kwargs)
    entry["style"] = "deferred"
    return entry


CASES = (
    common_case("perfil-01", "timing", (
        prior(D, "pescado al horno", "Mi ficha de cena habitual dice pescado al horno."),
        prior(B, "avena con canela", "De desayuno suelo dejar lista avena con canela."),
        prior(R, "agua mineral", "Normalmente elijo agua mineral para beber."),
    ), (
        delayed(D, 11, "Ahorita estoy pagando unas compras. Para la cena, prefiero menestra de fréjol. "
                "Cuando vuelva a abrir el chat podré revisar cuánto dura el cambio.",
                "¿Qué sirvo de cena ahora?", "menestra de fréjol", stored="pescado al horno",
                alternatives=scope_choices("menestra de fréjol",
                    "La menestra fue solo para aquella sesión. Mi cena habitual sigue siendo pescado al horno.",
                    "Anota menestra de fréjol como mi cena habitual para las próximas veces."),
                slot="TEMPORAL_SCOPE"),
        visit(R, 13, "Ya guardé la compra. La bebida de mi lista sigue igual.",
              "¿Cuál era mi bebida habitual?", "agua mineral", stored="agua mineral"),
        visit(B, 15, "Resulta que el desayuno también cambió. Para el desayuno, prefiero pan de yuca.",
              "¿Qué dejo sobre la mesa ahora?", "pan de yuca", stored="avena con canela",
              alternatives=scope_choices("pan de yuca", "El pan de yuca es solo para esta sesión.",
                    "Sí, deja pan de yuca como mi desayuno habitual."), slot="TEMPORAL_SCOPE"),
    ), ((D, 19, None), (D, 23, None), (B, 19, None)),
        "Una consulta puede quedar pendiente hasta la reapertura del chat. Queda otra oportunidad al "
        "aparecer un cambio de desayuno. La réplica de la cena y la consulta de bebida son independientes."),
    common_case("perfil-02", "timing", (
        prior(R, "agua mineral", "Mi bebida habitual es agua mineral."),
        prior(B, "papaya picada", "Normalmente pido papaya picada de desayuno."),
        prior(D, "estofado de garbanzos", "Guarda estofado de garbanzos como mi cena habitual."),
    ), (
        visit(R, 11, "Ya terminé lo urgente y puedo contestar. Para la bebida, prefiero jugo de guayaba.",
              "¿Qué pongo de bebida ahora?", "jugo de guayaba", stored="agua mineral",
              alternatives=scope_choices("jugo de guayaba", "Esta bebida es solo para esta sesión.",
                    "Guarda jugo de guayaba como bebida habitual de ahora en adelante."), slot="TEMPORAL_SCOPE"),
        visit(B, 13, "Me cambiaron el horario y no tengo todavía el resto de desayunos resuelto. "
              "Para el desayuno, prefiero tortilla de verde.",
              "¿Qué me sirve para desayunar en esta visita?", "tortilla de verde", stored="papaya picada",
              alternatives=scope_choices("tortilla de verde", "Solo para esta sesión, después sigo con papaya picada.",
                    "La tortilla de verde queda como desayuno habitual."),
              slot="TEMPORAL_SCOPE", style="partial",
              partial="Lo de hoy sí lo sé. Dame un segundo y te confirmo si es también para más adelante."),
        visit(D, 15, "Al final en la cena no hubo cambios. Retoma lo que dejé anotado.",
              "¿Cuál era mi cena habitual?", "estofado de garbanzos", stored="estofado de garbanzos"),
    ), ((R, 19, None), (R, 24, None), (B, 20, None)),
        "La réplica temprana llega directamente. La siguiente aclara una preferencia que hoy puede usarse "
        "y podría necesitar dos preguntas para justificar su persistencia."),
    common_case("perfil-03", "timing", (
        prior(B, "maduros asados", "Habitualmente dejo maduros asados para desayunar."),
        prior(R, "café pasado", "Mi opción de bebida de siempre es café pasado."),
        prior(D, "mote con queso", "Suelo cenar mote con queso."),
    ), (
        visit(B, 11, "Puedo seguir en el chat mientras se calienta el desayuno. Para el desayuno, prefiero cachapa con queso.",
              "¿Qué desayuno preparo hoy?", "cachapa con queso", stored="maduros asados",
              alternatives=scope_choices("cachapa con queso", "La cachapa es para esta sesión solamente.",
                    "Quiero cachapa con queso como mi desayuno habitual a partir de ahora."),
              slot="TEMPORAL_SCOPE", style="partial",
              partial="Puedo asegurar lo de hoy. Dame un momento para mirar qué pasa más adelante."),
        visit(R, 13, "Me llaman de recepción. Para la bebida, prefiero helarla; la bebida de la ficha queda igual.",
              "¿Cuál es la bebida que puedo enfriar?", "café pasado", stored="café pasado"),
        visit(D, 15, "Pensaba cenar fuera y se canceló la salida. Para la cena, prefiero ensalada de quinua.",
              "¿Qué ceno esta vez?", "ensalada de quinua", stored="mote con queso",
              alternatives=scope_choices("ensalada de quinua", "La ensalada corresponde solo a esta sesión.",
                    "Pon ensalada de quinua como cena habitual."), slot="TEMPORAL_SCOPE"),
    ), ((B, 19, None), (D, 19, None), (D, 22, None)),
        "El primer turno admite respuesta parcial y su seguimiento compite con una pregunta posterior. "
        "Enfriar una bebida no sustituye su valor almacenado."),
    common_case("perfil-04", "timing", (
        prior(D, "locro de papa", "Para la cena suelo escoger locro de papa."),
        prior(R, "té de jazmín", "Lo habitual para beber es té de jazmín."),
        prior(B, "avena con canela", "Tengo guardada avena con canela como desayuno habitual."),
    ), (
        visit(D, 11, "Se me va el bus y no voy a poder seguir leyendo. Para la cena, prefiero arroz con lentejas.",
              "¿Qué voy a cenar hoy?", "arroz con lentejas", stored="locro de papa",
              alternatives=scope_choices("arroz con lentejas", "El arroz con lentejas era solo para esta sesión.",
                    "El arroz con lentejas pasa a ser mi cena habitual."),
              slot="TEMPORAL_SCOPE", style="refused"),
        visit(R, 13, "Creía que llevaría bebida de casa y al final la preparo aquí. Para la bebida, prefiero café pasado.",
              "¿Qué bebida preparo ahora?", "café pasado", stored="té de jazmín",
              alternatives=scope_choices("café pasado", "El café es solo para esta sesión.",
                    "Café pasado queda anotado como mi bebida habitual."), slot="TEMPORAL_SCOPE"),
        visit(B, 15, "Se suspendió la comida del hotel. Para el desayuno, prefiero tamal de maíz.",
              "¿Qué pongo de desayuno ahora?", "tamal de maíz", stored="avena con canela",
              alternatives=scope_choices("tamal de maíz", "El tamal es solo para esta sesión.",
                    "Desde ahora quiero tamal de maíz como desayuno habitual."), slot="TEMPORAL_SCOPE"),
    ), ((D, 20, None), (R, 20, None), (B, 20, None)),
        "Una primera interrupción probablemente se pierde. Los planes cambian en las dos visitas siguientes y "
        "cada cambio puede justificar su propia aclaración."),
    common_case("perfil-05", "timing", (
        prior(D, "tortilla de yuca", "Normalmente preparo tortilla de yuca para cenar."),
        prior(R, "infusión de menta", "Mi bebida habitual es infusión de menta."),
        prior(B, "tostadas con tomate", "Para desayunar suelo elegir tostadas con tomate."),
    ), (
        delayed(D, 11, "Estoy de paso en la tienda. Para la cena, prefiero tacos de hongos. "
                "Tengo la agenda en casa. Te digo el alcance en el próximo chat.",
                "¿Qué sirvo de cena ahora?", "tacos de hongos", stored="tortilla de yuca",
                alternatives=scope_choices("tacos de hongos",
                    "Entonces era por esa sola sesión. Conserva mi habitual.",
                    "A partir de entonces tacos de hongos es mi cena habitual."),
                slot="TEMPORAL_SCOPE"),
        visit(R, 13, "Llegué a casa. Para esta consulta quiero repasar mi bebida de siempre.",
              "¿Qué bebida figuraba en mi lista?", "infusión de menta", stored="infusión de menta"),
        visit(B, 15, "Tengo visitas esta mañana. Para el desayuno, prefiero humita de choclo.",
              "¿Qué desayuno sirvo ahora?", "humita de choclo", stored="tostadas con tomate",
              alternatives=scope_choices("humita de choclo", "La humita solo va para esta sesión.",
                    "Conserva humita de choclo como desayuno habitual."), slot="TEMPORAL_SCOPE"),
    ), ((D, 19, None), (B, 20, None), (D, 24, None)),
        "La pregunta inicial puede habilitar una réplica en la visita posterior. Recuperar la bebida habitual "
        "sigue siendo una acción independiente de la nueva evidencia sobre la cena."),
    common_case("perfil-06", "temporal", (
        prior(R, "agua de cedrón", "Agua de cedrón es mi bebida habitual."),
        prior(D, "sopa de zapallo", "De cena suelo escoger sopa de zapallo."),
    ), (
        visit(R, 11, "Mientras dure la mudanza, desde hoy hasta el 13 de junio inclusive, "
              "quiero agua mineral como bebida. Después recupera mi hábito.",
              "¿Qué bebida sirvo hoy?", "agua mineral", stored="agua de cedrón",
              write=pref("agua mineral", kind=I, start=11, through=13)),
        visit(R, 13, "Queda un solo viaje de mudanza y sigo dentro de la excepción.",
              "¿Qué bebida quedó anotada para este último día?", "agua mineral", stored="agua mineral"),
        visit(D, 15, "La mudanza terminó y del menú de cena no he dicho ningún cambio.",
              "¿Qué cena uso normalmente?", "sopa de zapallo", stored="sopa de zapallo"),
    ), ((R, 16, None), (R, 19, None), (D, 19, None)),
        "El día final es parte del intervalo. Una vez vencido vuelve la bebida habitual; la cena permanece aparte."),
    common_case("perfil-07", "temporal", (
        prior(B, "yogur con mango", "Me gusta yogur con mango como desayuno habitual."),
        prior(R, "limonada con hierbabuena", "Para beber habitualmente guardé limonada con hierbabuena."),
    ), (
        visit(B, 11, "Mi nuevo horario empieza el 14 de junio. A partir de ese día mi desayuno habitual será "
              "pan de centeno. Hasta entonces sigue el yogur.",
              "¿Qué desayuno me corresponde hoy?", "yogur con mango", stored="yogur con mango",
              write=pref("pan de centeno", start=14)),
        visit(B, 13, "Mañana es el primer día de ese horario. Hoy corresponde todavía mi pauta anterior.",
              "¿Qué desayuno sigo usando esta mañana?", "yogur con mango", stored="yogur con mango"),
        visit(B, 15, "Ya empecé el horario nuevo. Abro la ficha para recuperar lo que dejamos desde el 14.",
              "¿Cuál es el desayuno que corresponde hoy?", "pan de centeno", stored="pan de centeno"),
    ), ((B, 19, None), (R, 19, None), (B, 22, None)),
        "Se autoriza escribir antes de que empiece el alcance. La respuesta actual sigue usando el habitual aplicable."),
    common_case("perfil-08", "temporal", (
        prior(D, "verduras al wok", "Mi cena habitual es verduras al wok."),
        prior(R, "agua de coco", "Mi bebida habitual es agua de coco."),
    ), (
        visit(D, 11, "La muestra del taller me dejó choclo con queso. Para la cena, prefiero choclo con queso "
              "solo para esta sesión. Mantén mi cena habitual en la ficha.",
              "¿Qué sirvo esta vez para cenar?", "choclo con queso", stored="verduras al wok",
              write=pref("choclo con queso", kind=S, start=11)),
        visit(D, 13, "Ya no tengo las sobras de aquella muestra. Hoy abro una sesión nueva.",
              "¿Qué cena habitual tenía que retomar?", "verduras al wok", stored="verduras al wok"),
        visit(R, 15, "La bebida no forma parte del cambio anterior. Déjala como estaba.",
              "¿Qué bebida me queda en la ficha?", "agua de coco", stored="agua de coco"),
    ), ((D, 19, None), (R, 19, None), (D, 22, None)),
        "El registro explícito de una sesión no sustituye el habitual y no sobrevive como aplicable a nuevas visitas."),
    common_case("perfil-09", "temporal", (
        prior(R, "horchata lojana", "Suelo beber horchata lojana."),
        prior(R, "jugo de tomate de árbol", "Del día 4 al 8 inclusive tengo una excepción con jugo de tomate de árbol.",
              start=4, kind=I, through=8),
    ), (
        visit(R, 11, "El jugo fue para la feria y se terminó aquel día 8. Hoy quiero la pauta de siempre.",
              "¿Cuál es mi bebida habitual?", "horchata lojana", stored="horchata lojana"),
        visit(R, 13, "Volví a comprar la bebida normal. Lo de la feria no se prolonga.",
              "¿Qué bebida tengo que conservar como hábito?", "horchata lojana", stored="horchata lojana"),
        visit(R, 15, "Estoy repasando la ficha para el pedido de la próxima semana.",
              "¿Qué bebida quedó como habitual?", "horchata lojana", stored="horchata lojana"),
    ), ((R, 18, None), (R, 21, None), (R, 24, None)),
        "Nombrar una excepción vencida no aporta autorización de escritura ni la vuelve un recuerdo aplicable."),
    common_case("perfil-10", "context", (
        prior(R, "agua de coco", "En casa mi bebida habitual es agua de coco.", kind=C, context="casa"),
        prior(R, "jugo de maracuyá", "Mi bebida de siempre en la oficina es jugo de maracuyá.", kind=C, context="oficina"),
    ), (
        visit(R, 11, "Cambia solo la bebida que tengo anotada para la oficina. Allí deja té de jazmín de ahora en adelante.",
              "¿Qué dejo de bebida en la oficina?", "té de jazmín", stored="jugo de maracuyá", context="oficina",
              write=pref("té de jazmín", kind=C, start=11, context="oficina")),
        visit(R, 13, "Mandaron a todo el equipo a casa y hoy trabajo allí.",
              "¿Qué bebida me corresponde en casa?", "agua de coco", stored="agua de coco", context="casa"),
        visit(R, 15, "Al final hoy sí toca volver al escritorio de la oficina.",
              "¿Qué bebida había quedado para la oficina?", "té de jazmín", stored="té de jazmín", context="oficina"),
    ), ((R, 19, "casa"), (R, 19, "oficina"), (R, 23, "oficina")),
        "La sustitución se aplica únicamente a oficina. El cambio de plan altera la consulta y el valor recuperado."),
    common_case("perfil-11", "context", (
        prior(D, "tortilla de yuca", "Si estoy en casa suelo cenar tortilla de yuca.", kind=C, context="casa"),
        prior(D, "tacos de hongos", "En la oficina suelo cenar tacos de hongos.", kind=C, context="oficina"),
    ), (
        visit(D, 11, "Aún me queda la duda de dónde terminaré. Tengo una cena distinta según la sede.",
              "¿Qué cena me correspondería?", ("CONDITIONALIZE", ["tacos de hongos", "tortilla de yuca"])),
        visit(D, 13, "Me avisaron que no abra la oficina. Hoy me quedo en casa.",
              "¿Qué cena habíamos anotado para casa?", "tortilla de yuca", stored="tortilla de yuca", context="casa"),
        visit(D, 15, "Esta noche sí me asignaron el escritorio de la oficina.",
              "¿Qué cena va para un día normal de oficina?", "tacos de hongos", stored="tacos de hongos", context="oficina"),
    ), ((D, 19, "casa"), (D, 19, "oficina"), (D, 20, "casa")),
        "La ubicación sin resolver requiere preservar las dos alternativas. Ninguna visita autoriza una nueva escritura."),
    common_case("perfil-12", "context", (
        prior(B, "yogur con mango", "Normalmente desayuno yogur con mango en casa.", kind=C, context="casa"),
        prior(B, "pan de yuca", "Mi desayuno habitual en la oficina es pan de yuca.", kind=C, context="oficina"),
    ), (
        visit(B, 11, "Para la semana del taller, únicamente en la oficina y hasta el día 13 inclusive, prefiero "
              "tamal de maíz para el desayuno.",
              "¿Qué desayuno dejo en la oficina hoy?", "tamal de maíz", stored="pan de yuca", context="oficina",
              write=pref("tamal de maíz", kind=I, start=11, through=13, context="oficina")),
        visit(B, 13, "Movieron la última sesión a remoto. La haré desde casa.",
              "¿Qué desayuno se recupera para casa?", "yogur con mango", stored="yogur con mango", context="casa"),
        visit(B, 15, "Después del taller volví a trabajar presencialmente en la oficina.",
              "¿Qué desayuno toca de nuevo en la oficina?", "pan de yuca", stored="pan de yuca", context="oficina"),
    ), ((B, 18, "oficina"), (B, 18, "casa"), (B, 20, "oficina")),
        "La excepción requiere simultáneamente fecha y ubicación. El último día está en casa y el regreso a "
        "la oficina sucede cuando la excepción ya venció."),
    common_case("perfil-13", "ownership", (
        prior(B, "humita de choclo", "La humita de choclo figura como mi desayuno habitual."),
        prior(D, "pescado al horno", "Normalmente dejo pescado al horno para mi cena."),
    ), (
        visit(B, 11, "A Rocío le llevo maduros asados para el desayuno. Ella lo pidió para sí misma. "
              "Mi desayuno sigue como ayer.",
              "¿Qué dejo anotado para mi desayuno?", "humita de choclo", stored="humita de choclo"),
        visit(D, 13, "Rocío llegará por la noche y quiere sopa de zapallo para ella. Mi plato lo tengo en la ficha.",
              "¿Qué cena habitual corresponde a mi ficha?", "pescado al horno", stored="pescado al horno"),
        visit(B, 15, "Terminó la visita. Para mi desayuno mantengo humita de choclo como antes.",
              "¿Qué desayuno tengo guardado?", "humita de choclo", stored="humita de choclo"),
    ), ((B, 19, None), (D, 19, None), (B, 22, None)),
        "Las dos elecciones de Rocío se citan con titular explícita. No cambian las preferencias de quien conversa."),
    common_case("perfil-14", "ownership", (
        prior(R, "jugo de maracuyá", "De bebida siempre escojo jugo de maracuyá."),
        prior(B, "papaya picada", "Mi desayuno habitual es papaya picada."),
    ), (
        visit(R, 11, "En el grupo de cocina me escribieron «mi bebida de siempre es café pasado». "
              "Era un mensaje de Martín. Para mí usa lo que ya anotaste.",
              "¿Cuál es mi bebida de costumbre?", "jugo de maracuyá", stored="jugo de maracuyá"),
        visit(B, 13, "Martín me pidió que le compre pan de centeno mañana; él es quien lo desayuna.",
              "¿Cuál era el desayuno de mi ficha?", "papaya picada", stored="papaya picada"),
        visit(R, 15, "Ya hice el pedido colectivo. Quiero volver a usar mi propia lista.",
              "¿Qué bebida guardaste para mí?", "jugo de maracuyá", stored="jugo de maracuyá"),
    ), ((R, 19, None), (B, 19, None), (R, 21, None)),
        "La primera persona dentro de la cita pertenece al tercero. La lista del usuario conserva su otra bebida."),
    common_case("perfil-15", "ownership", (
        prior(D, "arroz con lentejas", "En mi ficha antigua, el plato habitual para la cena es arroz con lentejas."),
        prior(R, "limonada con hierbabuena", "Suelo pedir limonada con hierbabuena para beber."),
    ), (
        visit(D, 11, "Corrige mi ficha. Ya cambié mi elección habitual para cenar a estofado de garbanzos. "
              "El arroz con lentejas déjalo para el plato de Emilia.",
              "¿Qué cena figura para mí desde hoy?", "estofado de garbanzos", stored="arroz con lentejas",
              write=pref("estofado de garbanzos", start=11)),
        visit(R, 13, "Emilia pidió agua mineral para ella. De mi bebida no cambia nada.",
              "¿Qué prefiero beber yo normalmente?", "limonada con hierbabuena", stored="limonada con hierbabuena"),
        visit(D, 15, "Emilia volverá a pedir lo suyo. Para mí conserva la rectificación que hice.",
              "¿Qué había dejado como mi cena habitual?", "estofado de garbanzos", stored="estofado de garbanzos"),
    ), ((D, 19, None), (R, 19, None), (D, 22, None)),
        "La escritura explícita actualiza una preferencia personal. Las menciones a Emilia no actualizan otras fichas."),
    common_case("perfil-16", "temporal", (
        prior(D, "sopa de zapallo", "Siempre preparo sopa de zapallo para mi cena de costumbre."),
        prior(R, "agua de cedrón", "Mi bebida habitual es agua de cedrón."),
    ), (
        visit(D, 11, "Mientras revisan mi cocina estaré usando la otra olla. Por unos días prefiero locro de papa para la cena.",
              "¿Qué ceno hoy mientras dura el arreglo?", "locro de papa", stored="sopa de zapallo",
              alternatives=(
                  scope_branch(pref("locro de papa", kind=I, start=11, through=12),
                               "La excepción dura hasta el día 12 inclusive."),
                  scope_branch(pref("locro de papa", kind=I, start=11, through=16),
                               "Lo tendré hasta el día 16 inclusive."),
              ), slot="VALID_UNTIL", style="direct"),
        visit(R, 13, "Finalmente solo cambié la cena. La bebida se mantiene.",
              "¿Qué bebida tenía anotada de siempre?", "agua de cedrón", stored="agua de cedrón"),
        visit(R, 15, "Para beber aún dejo lo de la ficha. La cena es otro tema.",
              "¿Qué bebida se recupera ahora?", "agua de cedrón", stored="agua de cedrón"),
    ), ((D, 16, None), (D, 17, None), (R, 19, None)),
        "Se conoce que la vigencia es limitada. El final exacto depende de una aclaración y cambia la consulta del día 16."),
)


def executable_worlds(case):
    meal_titles = {B: "el desayuno", R: "la bebida", D: "la cena"}
    for row in _worlds(case):
        row["profile_id"] = f"{case['id']}:persona"
        row["recipe_id"] = f"{VERSION}:{case['id']}"
        for index, (original, native) in enumerate(zip(case["visits"], row["visits"], strict=True)):
            if native["style"] != "deferred":
                continue
            attribute = original["attribute"]
            value = native["fact"]["value"]
            if native["fact"]["scope"]["kind"] == S:
                value = next(item[1].value for item in case["prior"] if item[0] == attribute)
            departure = ("Necesito mirar la agenda. Mejor dejarlo ahí; te digo cuando abra el chat otra vez."
                         if case["id"] == "perfil-05" else
                         "Estoy con la compra. Mejor dejarlo ahí; te respondo apenas vuelva.")
            topic = meal_titles[attribute]
            native["deferred_reply"] = departure
            native["recontact"] = {
                "visit": index + 2,
                "message": f"Ya pude revisar el alcance. Para {topic} prefiero {value} de forma habitual.",
            }
        yield row


def freeze(folder=ROOT):
    public = [_public(case) for case in CASES]
    gold = [_gold(case) for case in CASES]
    worlds = [world for case in CASES for world in executable_worlds(case)]
    for item, case in zip(gold, CASES, strict=True):
        paths = [world for world in worlds if world["root_id"] == case["id"]]
        item["recontact_by_world"] = {
            world["world_id"]: [{"origin": index, **visit["recontact"]}
                                for index, visit in enumerate(world["visits"], 1) if "recontact" in visit]
            for world in paths
        }
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError("Create a new reserved version before editing sealed information")
    write_json(folder / "PUBLIC.json", {
        "version": VERSION, "rubric": REVIEW_INSTRUCTIONS, "cases": public,
    })
    write_json(folder / "REVIEW_TEMPLATE.json", {"reviewer_id": "", "cases": [
        {"id": case["id"], "visits": [
            {"visit": visit["visit"], "response": {"mode": None, "value": None},
             "new_write": [], "retrieves": [], "rationale": ""}
            for visit in case["visits"]
        ]} for case in public
    ]})
    write_json(folder / "PRIVATE_GOLD.json", {"version": VERSION, "cases": gold})
    paths = ("PUBLIC.json", "REVIEW_TEMPLATE.json", "PRIVATE_GOLD.json", "PRIVATE_WORLDS.jsonl")
    (folder / paths[-1]).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in worlds),
                                   encoding="utf-8")
    result = {
        "version": VERSION, "split": "reserved", "status": "frozen_for_blind_review",
        "authorship": "handwritten by a separate AI subagent without consulting the development wording",
        "human_reviews": 0, "real_user_data": False, "roots": len(public),
        "worlds": len(worlds), "visits_per_root": 3,
        "recontact_roots": sum(any(visit["style"] == "deferred" for visit in case["visits"])
                               for case in CASES),
        "public_digest": digest(public), "artifacts": {name: file_hash(folder / name) for name in paths},
    }
    write_json(manifest_path, result)
    return result


def verify(folder=ROOT):
    meta = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    if meta["version"] != VERSION or meta["split"] != "reserved":
        raise ValueError("Unexpected reserved version")
    for name, sha256 in meta["artifacts"].items():
        if file_hash(folder / name) != sha256:
            raise ValueError(f"Frozen artifact changed: {name}")
    return meta


def load_executable(folder=ROOT):
    """Load the sealed worlds after policy and checkpoint selection."""
    meta = verify(folder)
    lines = (folder / "PRIVATE_WORLDS.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    if len(rows) != meta["worlds"] or any(row["split"] != "reserved" for row in rows):
        raise ValueError("Invalid reserved split or number of worlds")
    return rows


def load_gold(folder=ROOT):
    verify(folder)
    return json.loads((folder / "PRIVATE_GOLD.json").read_text(encoding="utf-8"))["cases"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify"))
    parser.add_argument("--folder", type=Path, default=ROOT)
    args = parser.parse_args()
    result = freeze(args.folder) if args.action == "freeze" else verify(args.folder)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
