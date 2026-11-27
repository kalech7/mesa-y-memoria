# Revisión humana del banco de preferencias alimentarias

El banco contiene conversaciones simuladas en español. La revisión busca saber si una persona puede distinguir una preferencia habitual, una excepción temporal, un cambio por contexto y un cambio ambiguo a partir de lo que el usuario dijo. Las 100 historias a revisar provienen de entrenamiento y validación, con 20 historias por familia. Hay 20 pares donde la duración del cambio es ambigua y 13 pares donde falta el contexto de la consulta actual.

## Para quienes revisan

Cada persona recibe su propia carpeta y trabaja de forma independiente. Las dos carpetas contienen los mismos casos en diferente orden.

1. Abre `01_casos.json` y guarda una copia de `03_respuestas_en_blanco.json` para editar. Conserva el identificador `audit_id` de cada caso y usa el mismo `reviewer_id` en tus respuestas.
2. Lee la memoria, los mensajes y la consulta de cada caso. Completa `information_sufficient` para indicar si la respuesta actual se puede determinar. En `need_future`, indica si hace falta aclarar algo para que el recuerdo sea fiable en otra sesión. Es posible conocer la respuesta de hoy y desconocer el alcance del cambio.
3. En `decisive_slot`, elige la variable que una pregunta tendría que resolver. Los valores son `TEMPORAL_SCOPE` para saber si el cambio es temporal o duradero, `CURRENT_CONTEXT` para saber dónde aplica hoy, `VALID_UNTIL` para saber hasta cuándo, `GENERIC` para otra aclaración, y `null` cuando la evidencia basta.
4. Escribe las opciones plausibles en `admissible_answers` usando el nombre del plato o bebida. Selecciona en `writable_scope` el alcance que se podría guardar con la evidencia disponible: `ongoing`, `interval`, `session`, `contextual` o `none` si el diálogo no justifica una escritura confirmada. Añade una razón breve en `rationale`.
5. Cuando termines la primera lectura, abre `02_replicas.json`. Para cada caso con dos réplicas, marca en `both_plausible` si ambas son compatibles con el mismo diálogo inicial. En `rationale`, anota qué guardarías después de cada una de las dos réplicas y durante cuánto tiempo o en qué contexto se debería aplicar. Si una réplica solo aclara el contexto de hoy, indica qué recuerdo debería seguir vigente más adelante. Para los casos sin par conserva `null`. Devuelve únicamente tu archivo de respuestas.

El coordinador debe entregar solo la carpeta `reviewer_a` o `reviewer_b` a cada persona. La clave privada y los conjuntos de datos quedan en el espacio del coordinador.

## Para quien coordina

Verifica que el banco sigue congelado y ejecuta la conciliación con las respuestas originales recibidas.

```bash
python -m research.tesis_bank verify --bank research/data/tesis_bank_v1
python -m research.tesis_bank review \
  --bank research/data/tesis_bank_v1 \
  --review-a valoraciones_a.json --review-b valoraciones_b.json \
  --output reports/revision_humana_tesis_primera.json
```

El informe conserva los hashes de ambos archivos, los denominadores, el acuerdo entre personas y los identificadores con diferencias respecto a la construcción sintética. Al adjudicar, lee también las justificaciones sobre el alcance posterior a cada réplica. El informe automático compara el alcance estructurado anterior a preguntar. Los identificadores de revisor sirven para distinguir archivos. La independencia de las personas y la autoría de las valoraciones las verifica el coordinador.

Documenta cada diferencia mediante una lista JSON de objetos con `audit_id`, `decision` y `rationale`. Usa `keep_as_written` cuando se pueda justificar la versión congelada y `revise` cuando el caso requiera revisión. Vuelve a ejecutar el comando con `--adjudications archivo.json` y un nombre nuevo para el informe. Conserva las valoraciones originales.

El estado `passed` exige dos respuestas por historia, cobertura de 100 historias, acuerdo inicial de al menos 90 %, plausibilidad de los pares temporales y resolución documentada de los desacuerdos. Si una historia requiere cambios, versiona el banco y conserva el anterior como evidencia exploratoria. El estado de esta entrega es `pending_human_audit`.
