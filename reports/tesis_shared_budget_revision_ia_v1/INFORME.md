# Revisión ciega del borrador reservado con IA

Dos instancias de IA revisaron por separado diez historias en español y veinte visitas. Cada una fijó las valoraciones iniciales antes de leer los dos pares de réplicas. Los paquetes de consulta omiten la clave preliminar del autor. Se conservaron las fichas originales y se verificaron contra el manifiesto congelado. [Revisión A](revisor_ia_a.json) · [Revisión B](revisor_ia_b.json) · [Comparación](COMPARACION.json)

| Aspecto | Coincidencias |
| --- | ---: |
| Visita en que harían la pregunta, incluida la abstención | 10/10 |
| Pares de réplicas que ambas consideraron plausibles | 2/2 |
| Alcance de la escritura | 15/20 |
| Fecha de inicio | 15/20 |
| Último día incluido | 19/20 |
| Coincidencia literal del contexto | 9/20 |

Cuatro de los cinco desacuerdos de alcance ocurren en la segunda visita. La revisión A anotó una preferencia anterior que seguía vigente y la B anotó si el turno autorizaba una escritura nueva. Por ejemplo, volver a preguntar por agua con limón recupera lo guardado y deja la escritura del nuevo turno en `none`. Para la próxima versión, la rúbrica debe registrar por separado la operación del turno y el estado aplicable al responder.

El quinto desacuerdo fue el desayuno de los domingos. La lectura contextual conserva el alcance a los domingos aunque la regla se repita indefinidamente. Para el menú de regreso del trabajo, la fecha del cambio en el registro y la primera ocasión en que aplica son dos datos distintos. Conviene representar los contextos como condiciones normalizadas, por ejemplo `día = domingo`, y dejar la comida como atributo. Así variaciones de mayúsculas y expresiones como «cena de esta vez» dejan de mezclarse con diferencias de alcance.

El banco conserva su estado de borrador. Este ejercicio documenta acuerdos de IA y señala cómo mejorar la rúbrica. La validación humana está registrada con cero revisiones y el reservado permanece fuera de la evaluación de las políticas. [Protocolo del borrador](../../research/data/tesis_shared_budget_reserved_draft_v1/README.md)
