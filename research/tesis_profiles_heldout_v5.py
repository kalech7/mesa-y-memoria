"""Fresh, separately authored users for blind evaluation of three-visit policies."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

from research.src.artifacts import canonical, digest, file_hash, write_json
from research.tesis_extended_heldout import (
    B, C, D, I, O, R, S, REVIEW_INSTRUCTIONS, Preference,
    _expiry, _gold, _native_fact, _prior, _public, _query, _remember, _turn, _update_truth,
    common_case, pref, prior, scope_branch, visit,
)


ROOT = Path(__file__).resolve().parent / "data/tesis_profiles_heldout_v5"
VERSION = "tesis_profiles_heldout_v5"
PROFILES = ("retiro_sin_respuesta", "precisa_tras_repaso", "contesta_ahora",
            "regresa_con_agenda", "respuesta_en_dos_partes", "consulta_de_ficha")


def alternatives(value, brief, durable):
    return scope_branch(pref(value, kind=S), brief), scope_branch(pref(value, kind=O), durable)


def later(*args, return_visit=2, **kwargs):
    item = visit(*args, style="deferred", **kwargs)
    item["return_visit"] = return_visit
    return item


# The text below is authored directly for the reserve. It does not import the
# development generator, its user templates or learned parser features.
CASES = (
    common_case("nuevo-01", "agenda_response", (
        prior(B, "avena con canela", "Mi ficha de desayunos de cada semana usa avena con canela."),
        prior(D, "mote con queso", "La cena que tengo guardada de costumbre es mote con queso."),
        prior(R, "agua mineral", "En mi lista de bebidas, agua mineral es lo de diario."),
    ), (
        visit(B, 12, "Me acaban de entregar un paquete y debo firmar; apenas alcanzo a escribir esto. "
              "Para el desayuno, prefiero humita de choclo. Lo de las próximas comidas sigue por resolver.",
              "¿Qué puedo servir de desayuno en esta visita?", "humita de choclo", stored="avena con canela",
              alternatives=alternatives("humita de choclo", "La humita era por esta sesión y después vuelvo a la avena.",
                    "Quiero que humita de choclo quede como desayuno habitual."),
              slot="TEMPORAL_SCOPE", style="refused"),
        visit(D, 15, "Suspendieron la cena fuera. Para la cena, prefiero tacos de hongos. "
              "Ahora sí puedo completar lo que haga falta.",
              "¿Qué cena elijo para esta noche?", "tacos de hongos", stored="mote con queso",
              alternatives=alternatives("tacos de hongos", "Solo para esta sesión, después toca mi cena anterior.",
                    "Desde ahora tacos de hongos queda como mi cena habitual."),
              slot="TEMPORAL_SCOPE"),
        visit(R, 18, "Cambió el lugar de la reunión. Para la bebida, prefiero agua de coco.",
              "¿Qué bebida me corresponde ahora?", "agua de coco", stored="agua mineral",
              alternatives=alternatives("agua de coco", "El agua de coco es para esta sesión solamente.",
                    "Agua de coco es mi nueva bebida habitual."), slot="TEMPORAL_SCOPE"),
    ), ((B, 22, None), (D, 24, None), (R, 26, None)),
        "La primera oportunidad se pierde si se pregunta. Dos cambios posteriores compiten por las "
        "dos preguntas que pueden preservarse."),
    common_case("nuevo-02", "agenda_response", (
        prior(R, "jugo de guayaba", "El jugo de guayaba es mi bebida habitual."),
        prior(D, "sopa de zapallo", "Para cenar normalmente preparo sopa de zapallo."),
        prior(B, "pan de yuca", "Mi desayuno de costumbre es pan de yuca."),
    ), (
        visit(R, 12, "Puedo sentarme a revisar el pedido contigo. Para la bebida, prefiero café pasado.",
              "¿Qué preparo de bebida hoy?", "café pasado", stored="jugo de guayaba",
              alternatives=alternatives("café pasado", "Es café solo en esta sesión, después jugo como antes.",
                    "Pon café pasado como mi bebida habitual en la ficha."),
              slot="TEMPORAL_SCOPE"),
        visit(D, 15, "Se atrasó la entrega y hoy cocino. Para la cena, prefiero estofado de garbanzos.",
              "¿Qué ceno esta vez?", "estofado de garbanzos", stored="sopa de zapallo",
              alternatives=alternatives("estofado de garbanzos", "El estofado era solo para esta sesión.",
                    "El estofado de garbanzos será mi cena habitual."),
              slot="TEMPORAL_SCOPE", style="partial",
              partial="Te confirmo el plato actual. Déjame mirar cómo queda el alcance para otras fechas."),
        visit(B, 18, "He dejado intacto el desayuno. Puedes leer mi ficha de siempre.",
              "¿Qué desayunaba habitualmente?", "pan de yuca", stored="pan de yuca"),
    ), ((R, 22, None), (R, 26, None), (D, 25, None)),
        "Responder una primera aclaración cuesta un turno. La segunda requiere dos y solo tiene "
        "una consulta futura, por lo que gastar todo en ella puede tener menor utilidad."),
    common_case("nuevo-03", "agenda_response", (
        prior(D, "locro de papa", "En la cena mi elección habitual es locro de papa."),
        prior(B, "papaya picada", "Normalmente desayuno papaya picada."),
        prior(R, "agua de cedrón", "Guarda agua de cedrón como mi bebida habitual."),
    ), (
        visit(D, 12, "Mientras espero a que se enfríe la olla tengo tiempo. Para la cena, prefiero verduras al wok.",
              "¿Qué cena tengo para esta ocasión?", "verduras al wok", stored="locro de papa",
              alternatives=alternatives("verduras al wok", "Las verduras son solo para esta sesión.",
                    "Mi cena habitual cambia a verduras al wok."),
              slot="TEMPORAL_SCOPE", style="partial",
              partial="Lo de hoy está resuelto. Dame un momento para ver si cambia también lo habitual."),
        visit(B, 15, "Ya me llamaron a ventanilla. Para el desayuno, prefiero tamal de maíz.",
              "¿Qué desayuno sirvo ahora?", "tamal de maíz", stored="papaya picada",
              alternatives=alternatives("tamal de maíz", "Solo esta sesión y luego regreso a la papaya.",
                    "Guarda tamal de maíz como mi desayuno habitual."),
              slot="TEMPORAL_SCOPE", style="refused"),
        visit(R, 18, "Hemos decidido hacer aquí la merienda. Para la bebida, prefiero infusión de menta.",
              "¿Qué bebida preparo?", "infusión de menta", stored="agua de cedrón",
              alternatives=alternatives("infusión de menta", "La infusión solo vale para esta sesión.",
                    "Infusión de menta queda como bebida habitual."), slot="TEMPORAL_SCOPE"),
    ), ((D, 23, None), (D, 26, None), (R, 23, None)),
        "El seguimiento de la primera respuesta parcial compite con una oportunidad tardía. "
        "La segunda visita no ofrece respuesta informativa."),
    common_case("nuevo-04", "return_pending", (
        prior(B, "maduros asados", "Mi desayuno usual es maduros asados."),
        prior(R, "jugo de maracuyá", "Lo habitual de bebida es jugo de maracuyá."),
        prior(D, "pescado al horno", "La cena normal de mi lista es pescado al horno."),
    ), (
        later(B, 12, "Me toca entrar al taller y cierro el teléfono. Para el desayuno, prefiero tostadas con tomate. "
              "Si me dejas la duda escrita, te la aclaro al volver.",
              "¿Qué desayuno me sirve hoy?", "tostadas con tomate", stored="maduros asados",
              alternatives=alternatives("tostadas con tomate", "El cambio fue puntual para esa sesión.",
                    "Las tostadas con tomate se quedan de desayuno habitual."), slot="TEMPORAL_SCOPE"),
        visit(R, 15, "Ya salí del taller. La bebida queda como estaba.",
              "¿Qué bebida figura como habitual?", "jugo de maracuyá", stored="jugo de maracuyá"),
        visit(D, 18, "Pensé que cenaría donde mi tía y cambiamos el plan. Para la cena, prefiero arroz con lentejas.",
              "¿Qué puedo ponerme de cena ahora?", "arroz con lentejas", stored="pescado al horno",
              alternatives=alternatives("arroz con lentejas", "El arroz es solo para esta sesión.",
                    "Desde ahora arroz con lentejas es mi cena habitual."), slot="TEMPORAL_SCOPE"),
    ), ((B, 22, None), (B, 25, None), (D, 24, None)),
        "Una pregunta puede habilitar una respuesta cuando la persona vuelve y conservar otra "
        "pregunta para el cambio final de cena."),
    common_case("nuevo-05", "return_pending", (
        prior(D, "ensalada de quinua", "Mi cena habitual es ensalada de quinua."),
        prior(R, "horchata lojana", "Mi bebida de diario es horchata lojana."),
        prior(B, "yogur con mango", "El desayuno habitual que tengo guardado es yogur con mango."),
    ), (
        later(D, 12, "Estoy por coger carretera. Para la cena, prefiero tortilla de yuca. "
              "El papel donde anoté las fechas lo tengo al regreso de este viaje.",
              "¿Qué cena puedo preparar antes de salir?", "tortilla de yuca", stored="ensalada de quinua",
              alternatives=alternatives("tortilla de yuca", "Era la elección de aquella sola sesión.",
                    "La tortilla de yuca queda como cena habitual."),
              slot="TEMPORAL_SCOPE", return_visit=3),
        visit(B, 15, "Me quedan unos minutos en una parada. Para el desayuno, prefiero pan de centeno.",
              "¿Qué desayuno pongo hoy?", "pan de centeno", stored="yogur con mango",
              alternatives=alternatives("pan de centeno", "Para esta sesión únicamente.",
                    "Pan de centeno queda como desayuno habitual."), slot="TEMPORAL_SCOPE"),
        visit(R, 18, "Acabo de volver y abrí de nuevo mi lista. De bebida seguimos igual.",
              "¿Qué bebida habitual tengo anotada?", "horchata lojana", stored="horchata lojana"),
    ), ((D, 23, None), (B, 23, None), (D, 26, None)),
        "El contacto diferido llega en la tercera visita. Antes puede aparecer una segunda pregunta "
        "sin que el alcance pendiente de la cena esté resuelto."),
    common_case("nuevo-06", "return_pending", (
        prior(R, "té de jazmín", "Mi bebida habitual es té de jazmín."),
        prior(D, "menestra de fréjol", "En la cena normalmente repito menestra de fréjol."),
        prior(B, "humita de choclo", "Mi desayuno cotidiano es humita de choclo."),
    ), (
        later(R, 12, "Me están haciendo una prueba de sonido y ahora no escucho bien. Para la bebida, prefiero agua de coco. "
              "En el siguiente descanso puedo confirmar si lo dejo fijo.",
              "¿Qué bebida sirvo ya?", "agua de coco", stored="té de jazmín",
              alternatives=alternatives("agua de coco", "Aquello era solo para una sesión.",
                    "Quiero agua de coco como bebida habitual."), slot="TEMPORAL_SCOPE"),
        visit(D, 15, "Se reprogramó el ensayo y me encargo de la cena. Para la cena, prefiero mote con queso.",
              "¿Qué cena preparo para hoy?", "mote con queso", stored="menestra de fréjol",
              alternatives=alternatives("mote con queso", "El mote es solo para esta sesión.",
                    "Mote con queso queda como mi cena habitual."), slot="TEMPORAL_SCOPE"),
        visit(B, 18, "El desayuno ha seguido tal cual, sin cambios.",
              "¿Qué desayuno tenía como habitual?", "humita de choclo", stored="humita de choclo"),
    ), ((R, 23, None), (D, 23, None), (B, 24, None)),
        "La respuesta postergada se incorpora en una visita que pregunta por otro atributo y no "
        "debe sustituir la cena por la bebida."),
    common_case("nuevo-07", "calendar_scope", (
        prior(D, "sopa de zapallo", "Para cenar habitualmente guardé sopa de zapallo."),
        prior(R, "agua mineral", "La bebida habitual es agua mineral."),
    ), (
        visit(D, 12, "El comedor estará cerrado desde hoy hasta el 16 de junio incluido. "
              "En ese lapso me quedo con choclo con queso para la cena; después retoma mi costumbre.",
              "¿Qué cena uso hoy?", "choclo con queso", stored="sopa de zapallo",
              write=pref("choclo con queso", kind=I, start=12, through=16)),
        visit(D, 16, "Hoy es el último turno con el comedor cerrado.",
              "¿Qué cena sigue vigente en esta fecha?", "choclo con queso", stored="choclo con queso"),
        visit(D, 18, "El comedor reabrió y quiero recuperar mi ficha de diario.",
              "¿Qué cena habitual me toca ahora?", "sopa de zapallo", stored="sopa de zapallo"),
    ), ((D, 22, None), (R, 23, None), (D, 26, None)),
        "El límite final es inclusivo. La segunda visita usa la excepción y la tercera vuelve a la "
        "preferencia normal sin una nueva escritura."),
    common_case("nuevo-08", "calendar_scope", (
        prior(B, "tortilla de verde", "El desayuno que tengo de siempre es tortilla de verde."),
        prior(R, "jugo de tomate de árbol", "Normalmente bebo jugo de tomate de árbol."),
    ), (
        visit(B, 12, "El horario de madrugada arranca el 17 de junio. Desde esa fecha mi desayuno habitual "
              "pasa a huevos con espinaca. Para hoy conserva la ficha anterior.",
              "¿Qué desayuno preparo hoy?", "tortilla de verde", stored="tortilla de verde",
              write=pref("huevos con espinaca", start=17)),
        visit(B, 15, "Todavía faltan dos días para arrancar. Estoy usando mi desayuno previo.",
              "¿Qué desayuno es el de esta fecha?", "tortilla de verde", stored="tortilla de verde"),
        visit(B, 18, "Ya trabajé el primer turno de madrugada.",
              "¿Qué desayuno habíamos fijado a partir del 17?", "huevos con espinaca", stored="huevos con espinaca"),
    ), ((B, 22, None), (R, 22, None), (B, 25, None)),
        "La autorización para registrar el cambio ocurre el día 12 y su aplicabilidad empieza el 17."),
    common_case("nuevo-09", "calendar_scope", (
        prior(R, "café pasado", "Café pasado figura como mi bebida habitual."),
        prior(R, "batido de naranjilla", "Mi excepción fue batido de naranjilla entre el 3 y el 7 inclusive.",
              start=3, kind=I, through=7),
        prior(D, "tacos de hongos", "Mi cena normal es tacos de hongos."),
    ), (
        visit(R, 12, "He encontrado el ticket del batido del día 7. Esa excepción quedó atrás.",
              "¿Qué bebida de diario tengo vigente?", "café pasado", stored="café pasado"),
        visit(D, 15, "Del menú de cena nadie ha cambiado la ficha.",
              "¿Qué cena era la habitual?", "tacos de hongos", stored="tacos de hongos"),
        visit(R, 18, "Voy a repetir el pedido corriente de esta semana.",
              "¿Qué bebida debo recuperar?", "café pasado", stored="café pasado"),
    ), ((R, 22, None), (D, 23, None), (R, 25, None)),
        "El ticket menciona un recuerdo vencido y no modifica su alcance. La recuperación usa el habitual."),
    common_case("nuevo-10", "place_switch", (
        prior(R, "infusión de menta", "Mi bebida habitual en casa es infusión de menta.", kind=C, context="casa"),
        prior(R, "agua de coco", "En la oficina mi bebida normal es agua de coco.", kind=C, context="oficina"),
    ), (
        visit(R, 12, "La ficha que tengo en la oficina se quedó vieja. Allí cambia mi bebida habitual a horchata lojana.",
              "¿Qué pongo de bebida en la oficina?", "horchata lojana", stored="agua de coco", context="oficina",
              write=pref("horchata lojana", kind=C, start=12, context="oficina")),
        visit(R, 15, "Se cortó la luz del edificio y me mandaron a trabajar a casa.",
              "¿Qué bebida recupero para casa?", "infusión de menta", stored="infusión de menta", context="casa"),
        visit(R, 18, "Arreglaron la luz y volví al puesto de la oficina.",
              "¿Qué bebida quedó en mi ficha de oficina?", "horchata lojana", stored="horchata lojana", context="oficina"),
    ), ((R, 22, "casa"), (R, 22, "oficina"), (R, 26, "oficina")),
        "La corrección cambia el valor de oficina. El incidente altera el contexto de la consulta "
        "sin globalizar la escritura."),
    common_case("nuevo-11", "place_switch", (
        prior(D, "pescado al horno", "En casa mi cena habitual es pescado al horno.", kind=C, context="casa"),
        prior(D, "arroz con lentejas", "La cena de costumbre en la oficina es arroz con lentejas.", kind=C, context="oficina"),
    ), (
        visit(D, 12, "Me falta saber dónde acaba el turno. La sede cambia cuál de mis dos cenas uso.",
              "¿Qué cena corresponde si aún no sé dónde estaré?",
              ("CONDITIONALIZE", ["arroz con lentejas", "pescado al horno"])),
        visit(D, 15, "Me confirmaron que estaré en casa durante esta visita.",
              "¿Qué cena recupero en casa?", "pescado al horno", stored="pescado al horno", context="casa"),
        visit(D, 18, "Hoy me quedé en la oficina hasta el cierre.",
              "¿Qué cena me corresponde en la oficina?", "arroz con lentejas", stored="arroz con lentejas", context="oficina"),
    ), ((D, 22, "casa"), (D, 23, "oficina"), (D, 26, "casa")),
        "La consulta inicial admite dos respuestas condicionadas por contexto. La visita siguiente "
        "resuelve la dirección sin cambiar preferencias."),
    common_case("nuevo-12", "place_switch", (
        prior(B, "papaya picada", "En casa habitualmente desayuno papaya picada.", kind=C, context="casa"),
        prior(B, "tostadas con tomate", "En la oficina mi desayuno habitual es tostadas con tomate.", kind=C, context="oficina"),
    ), (
        visit(B, 12, "Para la oficina durante el inventario, desde hoy hasta el 16 inclusive, apúntame "
              "avena con canela como desayuno. Fuera de eso quedan mis fichas.",
              "¿Qué desayuno toca hoy en la oficina?", "avena con canela", stored="tostadas con tomate", context="oficina",
              write=pref("avena con canela", kind=I, start=12, through=16, context="oficina")),
        visit(B, 15, "El inventario sigue, pero hoy me lo asignaron desde casa.",
              "¿Qué desayuno uso para casa?", "papaya picada", stored="papaya picada", context="casa"),
        visit(B, 18, "Se cerró el inventario y estoy otra vez en la oficina.",
              "¿Qué desayuno normal tengo allí?", "tostadas con tomate", stored="tostadas con tomate", context="oficina"),
    ), ((B, 22, "oficina"), (B, 23, "casa"), (B, 26, "oficina")),
        "La excepción requiere contexto y fecha simultáneos. La segunda visita sale del contexto "
        "y la tercera sucede después del final."),
    common_case("nuevo-13", "speaker_and_quote", (
        prior(R, "agua de cedrón", "Agua de cedrón es mi bebida habitual."),
        prior(B, "cachapa con queso", "Mi desayuno habitual es cachapa con queso."),
    ), (
        visit(R, 12, "Me reenviaron una nota de Lidia que dice «mi bebida para diario es jugo de guayaba». "
              "Solo la copio para su encargo; lo mío sigue igual.",
              "¿Qué bebida tengo en mi propia ficha?", "agua de cedrón", stored="agua de cedrón"),
        visit(B, 15, "Lidia me confirmó que ella desayuna humita de choclo. Mi ficha no cambia por ese pedido.",
              "¿Qué desayuno había dejado para mí?", "cachapa con queso", stored="cachapa con queso"),
        visit(R, 18, "Ya entregué el encargo de Lidia. Ahora reviso mi lista de nuevo.",
              "¿Qué bebida tengo como hábito?", "agua de cedrón", stored="agua de cedrón"),
    ), ((R, 22, None), (B, 23, None), (R, 26, None)),
        "La primera persona dentro de la cita es Lidia. Reenviar su nota no crea una preferencia personal."),
    common_case("nuevo-14", "speaker_and_quote", (
        prior(D, "mote con queso", "La cena habitual de mi ficha es mote con queso."),
        prior(R, "limonada con hierbabuena", "Suelo beber limonada con hierbabuena."),
    ), (
        visit(D, 12, "Anota una rectificación para mí: mi cena habitual ahora es menestra de fréjol. "
              "El mote con queso lo preparo para mi sobrino desde hoy.",
              "¿Qué cena pongo en mi ficha personal?", "menestra de fréjol", stored="mote con queso",
              write=pref("menestra de fréjol", start=12)),
        visit(R, 15, "Mi sobrino pidió jugo de maracuyá para él. Mi bebida sigue en la ficha de siempre.",
              "¿Qué bebida tenía yo?", "limonada con hierbabuena", stored="limonada con hierbabuena"),
        visit(D, 18, "El sobrino repite su plato. Yo quiero seguir con la rectificación que te indiqué.",
              "¿Qué cena habitual quedó para mí?", "menestra de fréjol", stored="menestra de fréjol"),
    ), ((D, 22, None), (R, 23, None), (D, 26, None)),
        "Una rectificación propia sí autoriza reemplazo. El sujeto del pedido posterior sigue siendo el sobrino."),
    common_case("nuevo-15", "speaker_and_quote", (
        prior(B, "pan de centeno", "Habitualmente desayuno pan de centeno."),
        prior(D, "verduras al wok", "Para cenar normalmente guardo verduras al wok."),
    ), (
        visit(B, 12, "Mario quiso apuntar en mi cuaderno «prefiero tamal de maíz». Le dije que lo ponga "
              "en su propio cuaderno y que mi desayuno sigue igual.",
              "¿Qué desayuno tengo registrado para mí?", "pan de centeno", stored="pan de centeno"),
        visit(D, 15, "También aparece el plato de Mario, locro de papa, al margen de mi lista. Es para él.",
              "¿Qué cena recupero de mi ficha?", "verduras al wok", stored="verduras al wok"),
        visit(B, 18, "Borré el apunte ajeno y solo quedó mi menú de costumbre.",
              "¿Qué desayuno corresponde a mi ficha?", "pan de centeno", stored="pan de centeno"),
    ), ((B, 22, None), (D, 23, None), (B, 26, None)),
        "Una atribución explícita y la corrección del apunte preservan al titular original. "
        "No hay nuevas escrituras personales en estas visitas."),
    common_case("nuevo-16", "duration_recovery", (
        prior(D, "tortilla de yuca", "Lo habitual de cena para mí es tortilla de yuca."),
        prior(B, "maduros asados", "Mi desayuno normal es maduros asados."),
    ), (
        visit(D, 12, "Voy a usar la cocina pequeña por unos días. Para la cena, prefiero sopa de zapallo.",
              "¿Qué cena puedo hacer hoy?", "sopa de zapallo", stored="tortilla de yuca",
              alternatives=(
                  scope_branch(pref("sopa de zapallo", kind=I, start=12, through=16),
                               "Termina el día 16 inclusive; después vuelve mi cena normal."),
                  scope_branch(pref("sopa de zapallo", kind=I, start=12, through=23),
                               "Es hasta el día 23 inclusive."),
              ), slot="VALID_UNTIL", style="partial",
              partial="Sé que son unos días. Déjame mirar la fecha exacta en el papel."),
        visit(B, 15, "La obra no afectó a mis desayunos. Continúan los que ya tenía.",
              "¿Qué desayuno tenía anotado?", "maduros asados", stored="maduros asados"),
        visit(B, 18, "Solo estoy comprobando el menú del desayuno; no te estoy cambiando la cena otra vez.",
              "¿Qué desayuno sigo usando?", "maduros asados", stored="maduros asados"),
    ), ((D, 22, None), (D, 24, None), (B, 26, None)),
        "La información inicial permite la cena del día y expresa temporalidad. Se requieren dos preguntas "
        "para precisar la fecha que decide la primera consulta futura."),
    common_case("nuevo-17", "duration_recovery", (
        prior(B, "humita de choclo", "Humita de choclo es mi desayuno habitual."),
        prior(R, "horchata lojana", "Mi bebida de siempre es horchata lojana."),
    ), (
        visit(B, 12, "Compré una porción que debo terminar. Para el desayuno, prefiero yogur con mango solo "
              "durante esta sesión. En la ficha deja mi hábito tal cual.",
              "¿Qué desayuno me sirvo en esta ocasión?", "yogur con mango", stored="humita de choclo",
              write=pref("yogur con mango", kind=S, start=12)),
        visit(B, 15, "Terminé aquella porción y esta es otra sesión.",
              "¿Qué desayuno habitual tengo que recuperar?", "humita de choclo", stored="humita de choclo"),
        visit(R, 18, "Al final me encargo también de la bebida. Para la bebida, prefiero jugo de maracuyá.",
              "¿Qué bebida preparo ahora?", "jugo de maracuyá", stored="horchata lojana",
              alternatives=alternatives("jugo de maracuyá", "El jugo solo es para esta sesión.",
                    "Jugo de maracuyá queda como bebida habitual."), slot="TEMPORAL_SCOPE"),
    ), ((B, 22, None), (R, 23, None), (R, 26, None)),
        "No hace falta aclarar un alcance de sesión explícito. Mantener presupuesto permite resolver "
        "la novedad de la tercera visita."),
    common_case("nuevo-18", "duration_recovery", (
        prior(R, "agua de coco", "Lo habitual para beber es agua de coco."),
        prior(D, "arroz con lentejas", "Arroz con lentejas es mi cena habitual."),
    ), (
        visit(R, 12, "El grupo de mi hermana compró té de jazmín para ellos durante el curso. "
              "Mi bebida queda como en la ficha.",
              "¿Qué bebida tengo yo para estos días?", "agua de coco", stored="agua de coco"),
        visit(D, 15, "Hasta que acabe el día 18 de junio inclusive, mi cena será estofado de garbanzos. "
              "Después recupera el arroz con lentejas.",
              "¿Qué cena toca hoy?", "estofado de garbanzos", stored="arroz con lentejas",
              write=pref("estofado de garbanzos", kind=I, start=15, through=18)),
        visit(D, 18, "Hoy se cumple el último día de la excepción. Estoy cerrando el curso.",
              "¿Qué cena sigue válida hoy?", "estofado de garbanzos", stored="estofado de garbanzos"),
    ), ((D, 22, None), (R, 23, None), (D, 26, None)),
        "Se mantienen distintos el pedido de la hermana, la excepción propia y la recuperación después "
        "del último día incluido."),
)


def executable_worlds(case):
    """Enumerate independent hidden scopes without inspecting policy behavior."""
    memory, history = [], {}
    for attribute, preference, text in case["prior"]:
        _remember(memory, history, attribute, preference.value,
                  f"2038-06-{preference.start:02d}", text, kind=preference.kind,
                  end=_expiry(preference), context=preference.context)
    uncertain = [index for index, visit in enumerate(case["visits"]) if visit["alternatives"]]
    assignments = list(product((0, 1), repeat=len(uncertain)))
    meal = {D: "la cena", B: "el desayuno", R: "la bebida"}
    for ordinal, choices in enumerate(assignments, 1):
        selected = dict(zip(uncertain, choices, strict=True))
        facts = [_prior(record) for record in memory]
        visits = []
        for index, item in enumerate(case["visits"]):
            attribute, when = item["attribute"], item["day"]
            session = f"S{index + 2}"
            candidates = [_native_fact(attribute, choice, when, session)
                          for choice, _ in item["alternatives"]]
            preference, reply = item["fact"], None
            if candidates:
                preference, reply = item["alternatives"][selected[index]]
            fact = _native_fact(attribute, preference, when, session) if preference else None
            if fact is not None:
                facts.append(fact)
            mode, expected = item["response"]
            visit = {
                "query": _query(attribute, when, session, item["context"]),
                "messages": [_turn(index + 1, "situacion", "user", item["text"]),
                             _turn(index + 1, "consulta", "user", item["question"])],
                "expected": expected if mode == "ANSWER" else None, "fact": fact,
                "alternatives": candidates or None, "slot": item["slot"],
                "style": item["style"], "answer": reply, "partial_answer": item["partial"],
                "strategy_cue": None,
            }
            if item["style"] == "deferred":
                value = fact["value"]
                if fact["scope"]["kind"] == S:
                    value = next(preference.value for key, preference, _ in case["prior"] if key == attribute)
                visit["deferred_reply"] = (
                    "Deja la pregunta anotada y mejor dejarlo ahí por ahora. Te contesto cuando vuelva."
                )
                visit["recontact"] = {
                    "visit": item["return_visit"],
                    "message": f"Ya revisé lo que me preguntaste. Para {meal[attribute]} prefiero {value} "
                               "de forma habitual. Esa es la ficha para las próximas visitas.",
                }
            visits.append(visit)
        future = []
        for probe_index, (attribute, when, context) in enumerate(case["probes"], 1):
            query = _query(attribute, when, "S5", context)
            future.append({"kind": f"probe_{probe_index}", "query": query,
                           "expected": _update_truth(facts, query)})
        yield {
            "root_id": case["id"], "world_id": f"{case['id']}:w{ordinal}",
            "realization_id": "manual-v5", "profile_id": f"{VERSION}:{case['id']}",
            "split": "reserved", "scenario_family": case["family"],
            "recipe_id": f"{VERSION}:{case['id']}", "question_budget": 2,
            "world_probability": 1. / len(assignments),
            "initial_memory": memory, "prior_sessions": list(history.values()),
            "impatient": False, "visits": visits, "future_queries": future,
            "second_visit_kind": "independent_ambiguity",
        }


def materialize():
    public = [_public(case) for case in CASES]
    gold = [_gold(case) for case in CASES]
    worlds = [world for case in CASES for world in executable_worlds(case)]
    for reference in gold:
        reference["question_responses_by_world"] = {
            row["world_id"]: [
                {"visit": index, "style": visit["style"], "answer": visit["answer"],
                 "partial_answer": visit["partial_answer"], "recontact": visit.get("recontact")}
                for index, visit in enumerate(row["visits"], 1) if visit["slot"] is not None
            ] for row in worlds if row["root_id"] == reference["id"]
        }
    return public, gold, worlds


def _artifacts(folder):
    return {name: file_hash(folder / name)
            for name in ("PUBLIC.json", "REVIEW_TEMPLATE.json", "PRIVATE_GOLD.json", "PRIVATE_WORLDS.jsonl")}


def freeze(folder=ROOT, seal=None):
    public, gold, worlds = materialize()
    if materialize() != (public, gold, worlds):
        raise ValueError("Reserve materialization is not deterministic")
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError("The version is already frozen")
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
    (folder / "PRIVATE_WORLDS.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in worlds), encoding="utf-8")
    frozen = {
        "version": VERSION, "split": "reserved", "status": "frozen_before_blind_review",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "authorship": "fresh manual Spanish narratives from a separate AI author",
        "training_templates_consulted": False, "real_users": False, "human_reviews": 0,
        "roots": len(public), "worlds": len(worlds), "visits_per_root": 3,
        "question_budget": 2, "world_assignment": "independent two-way latent alternatives per ambiguous visit",
        "categories": dict(Counter(case["family"] for case in CASES)),
        "public_digest": digest(public), "artifacts": _artifacts(folder),
    }
    write_json(manifest_path, frozen)
    if seal is not None:
        seal.parent.mkdir(parents=True, exist_ok=True)
        if seal.exists():
            raise FileExistsError("Seal already exists")
        write_json(seal, {
            "version": VERSION, "purpose": "freeze before evaluation and checkpoint selection",
            "sealed_utc": frozen["frozen_utc"], "manifest_sha256": file_hash(manifest_path),
            "artifacts": frozen["artifacts"], "author_source_sha256": file_hash(Path(__file__)),
            "deterministic_materialization": True,
            "public_digest": frozen["public_digest"], "split": "reserved",
        })
    return verify(folder)


def verify(folder=ROOT):
    meta = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    if meta["version"] != VERSION or meta["split"] != "reserved":
        raise ValueError("Unexpected reserve version")
    if _artifacts(folder) != meta["artifacts"]:
        raise ValueError("Frozen artifact was changed")
    return meta


def load_executable(folder=ROOT):
    meta = verify(folder)
    worlds = [json.loads(line) for line in (folder / "PRIVATE_WORLDS.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(worlds) != meta["worlds"] or any(row["split"] != "reserved" for row in worlds):
        raise ValueError("Sealed worlds have invalid size or split")
    return worlds


def load_gold(folder=ROOT):
    verify(folder)
    return json.loads((folder / "PRIVATE_GOLD.json").read_text(encoding="utf-8"))["cases"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify"))
    parser.add_argument("--folder", type=Path, default=ROOT)
    parser.add_argument("--seal", type=Path, default=Path("reports/tesis_profiles_review_v5/SEAL.json"))
    args = parser.parse_args()
    result = freeze(args.folder, args.seal) if args.action == "freeze" else verify(args.folder)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
