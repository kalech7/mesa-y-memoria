# Revisión por IA de 100 historias de preferencias

El 27 de noviembre de 2026 dos subagentes evaluaron, por separado y en distinto orden, las mismas 100 historias de entrenamiento y validación del banco de tesis v1. El revisor A examinó la pragmática del español y el revisor B la vigencia de las preferencias. Cada uno recibió su paquete ciego, registró una primera lectura antes de abrir las réplicas y después justificó qué cambios autorizaba cada respuesta. Se conservaron las cuatro lecturas. Las huellas SHA-256 permiten comprobar su integridad. [Primer juicio A](A_antes.json) · [Primer juicio B](B_antes.json) · [Juicio final A](A_final.json) · [Juicio final B](B_final.json)

| Juicio | Resultado de las dos lecturas |
| --- | --- |
| ¿Basta lo dicho para responder hoy? | Sí en 87 historias. En 13 faltaba saber el lugar de la consulta. |
| ¿Hace falta aclarar para otra visita? | Sí en 20 historias. Se podía servir el plato de hoy, pero el cambio de preferencia tenía una duración incierta. |
| ¿Qué escritura nueva está autorizada? | 20 preferencias habituales, 20 excepciones con fecha y 60 historias que permiten conservar los registros existentes sin añadir otro. |
| ¿Son plausibles las dos continuaciones del diálogo? | Los dos revisores aceptaron las dos respuestas en los 33 pares. En los otros 67 casos el paquete carece de réplicas alternativas. |

Los evaluadores coincidieron en los cinco campos estructurados de las 100 historias. El acuerdo conjunto sobre suficiencia actual, necesidad futura y variable a aclarar fue del 100 %. La kappa de Cohen para cada uno de esos campos fue 1. En plausibilidad hubo 33 coincidencias en 33 pares. Como todos recibieron la misma categoría, su kappa es indefinida. Tras congelar los juicios se compararon los campos estructurados con la clave sintética, que también coincidió en los 100 casos para ambos revisores. Las justificaciones se conservan como texto libre. [Cómputo reproducible](ACUERDO.json)

En el caso `caso-a029da8edf1f8d`, la persona prefiere menestra para la cena actual y tenía un registro previo de estofado. Los dos revisores aceptaron servir menestra ese día y pedir aclaración antes de reemplazar el hábito. En `caso-8b7f91d5e1cd64`, la persona ya había especificado platos distintos para la oficina y la casa. Lo pendiente era saber dónde comería esa vez. En esos dos ejemplos preguntar tiene consecuencias diferentes para SQLite. [Juicios y razones por caso](A_final.json)

Las plantillas repiten construcciones y los dos revisores usan IA dentro del mismo sistema. El acuerdo mide la legibilidad de estas etiquetas para estos dos revisores. El banco de personas mantiene **cero valoraciones humanas**. Para estimar naturalidad en español y el esfuerzo de responder hace falta observación con hablantes. [Revisión humana y distribución de materiales](../../docs/revision_humana_tesis.md)

Desde la raíz, `python -m research.score_ai_reviews reports/tesis_revision_ia_100_v1` recalcula los resultados y verifica que los campos de la primera lectura se conservaron al evaluar las réplicas.
