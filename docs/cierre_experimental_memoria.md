# Evaluación de aclaraciones y memoria alimentaria

El estudio compara cómo un asistente decide si debe preguntar y qué efecto tiene esa decisión en la respuesta actual y en el recuerdo reutilizado. El simulador conserva la intención privada. El selector solo recibe el diálogo y los recuerdos visibles. El lector posterior recibe la consulta y la memoria reabierta en SQLite.

## Banco y auditoría

```bash
python -m research.tesis_bank verify --bank research/data/tesis_bank_v1

python -m research.tesis_bank prepare --output research/data/mi_banco_tesis
```

El banco completo tiene 60 historias de entrenamiento, 40 de validación y 50 reservadas. Incluye preferencias estables, cambios permanentes, excepciones temporales, preferencias contextuales y cambios de duración ambigua. En los pares ambiguos, la entrada inicial es idéntica en ambos mundos. La respuesta del simulador a la pregunta decisiva y el alcance que corresponde guardar pueden diferir. Las recetas y plantillas se asignan por partición, y el manifiesto conserva los hashes y la auditoría de aislamiento.

La autoría es sintética. Las expresiones principales se mantienen dentro de la gramática que el evaluador puede verificar. En esta etapa se usa auditoría automática del banco. Los materiales de revisión ciega se pueden usar para inspeccionar una muestra cualitativa de desarrollo. [Instrucciones de revisión](revision_humana_tesis.md).

Una [revisión ciega por IA de 15 historias](../reports/tesis_revision_ia_v1/REVISION_CUALITATIVA_IA.md) encontró tres casos en los que hacía falta confirmar la duración para guardar la preferencia y tres en los que el lugar era necesario para responder en el momento. Señaló que «solo por hoy» también puede cubrir otra sesión del mismo día. Los pares temporales auditados consultan en una fecha posterior. La lectura que distingue día y sesión queda abierta para consultas separadas en la misma fecha.

## Comparación de aprendizaje

La comparación tiene tres brazos por semilla (101, 202 y 303). Se evalúa el SFT limpio congelado, SFT adicional del selector y GRPO del selector. Las continuaciones parten del mismo adaptador por semilla y usan el mismo límite de tiempo de entrenamiento. En GRPO, el límite incluye obtener las trayectorias. El informe conserva el tiempo real y cualquier exceso al finalizar un grupo en curso. Los tres brazos mantienen fijos el escritor, el respondedor y el lector para aislar el efecto de entrenar la decisión de preguntar.

Ejecuta con el Python de investigación que tenga PyTorch, Transformers y PEFT. Las instrucciones del [laboratorio](laboratorio_memoria.md#abrir-el-laboratorio) explican cómo recrear el entorno.

```bash
python -m research.tesis_comparison \
  --bank research/data/tesis_bank_v1 --output research/runs/mi_plan_tesis \
  --device cuda --budget-seconds 300 --plan-only

python -u -m research.tesis_comparison \
  --bank research/data/tesis_bank_v1 --output research/runs/mi_comparacion_tesis \
  --device cuda --budget-seconds 300 --validation-only
```

`--plan-only` comprueba los prefijos de las ramas con el tokenizador local. `--validation-only` entrena y selecciona los adaptadores en validación. Una corrida exploratoria sin esa opción evalúa los checkpoints elegidos en el banco reservado después de escribir `SELECTION.json`. Usa un destino nuevo para cada corrida. El paso cero es elegible. El presupuesto de 300 segundos es un parámetro inicial reproducible, sujeto al diagnóstico de validación si se define una comparación confirmatoria.

La corrida con el banco completo terminó para las tres semillas y sus resultados fueron auditados. [Selección, resultados y métricas de las 50 historias reservadas](comparacion_tesis.md#resultado-en-50-historias-reservadas)

El [preflight del banco completo](../research/runs/tesis_comparison_plan_v1/TOKEN_PREFLIGHT.json) comprobó las cinco ramas de los 200 episodios. El máximo fue de 1319 tokens de entrada frente al límite de 2048.

La [prueba en GPU con tres semillas](../research/runs/tesis_comparison_smoke_v1/RESULTS.json) evaluó cinco historias reservadas, siete episodios por semilla y un límite de 20 segundos de optimización por brazo. El acierto futuro sustentado fue 53,3 % en SFT congelado, 43,3 % en SFT adicional y 53,3 % en GRPO. El éxito conjunto estricto fue 0 % en los tres brazos. Los tres grupos GRPO tuvieron retorno de tarea constante dentro del grupo. La variación observada provino del coste de preguntar. El [plan de la campaña en el banco completo](../research/runs/tesis_comparison_full_v1/PLAN.json) fija una asignación de 300 segundos por continuación y semilla. [Auditoría de la prueba](../research/runs/tesis_comparison_smoke_v1/AUDIT.json).

## Métricas por familia

```bash
python -m research.tesis_metrics \
  --comparison-results research/runs/mi_comparacion_tesis/RESULTS.json \
  --reference SFT --output reports/mi_comparacion_tesis/METRICAS.json
```

El comando usa las trayectorias de una corrida que completó la evaluación reservada. Verifica sus hashes y empareja historias, mundos, realizaciones y semillas. También acepta trazas archivadas con `--panel ETIQUETA=RUTA`. Distingue aclaraciones útiles, redundantes y omitidas, cambios permanentes que se podían guardar con la evidencia entregada al escritor, excepciones sin vencimiento y daño a preferencias que seguían vigentes. Publica los numeradores y denominadores de cada tasa.

En una [reevaluación de los pilotos archivados](../reports/tesis_metricas/INFORME.md), más SFT redujo las aclaraciones omitidas de 8/10 a 2/10 en cambios ambiguos y guardó los cinco cambios permanentes explícitos. También convirtió tres excepciones en preferencias sin vencimiento y alteró las respuestas protegidas en esos tres casos.

## Cuánto aporta una pregunta

```bash
python -m research.tesis_question_value \
  --bank research/data/tesis_bank_v1 --output reports/mi_auditoria_preguntas
```

Esta auditoría ejecuta las cinco acciones posibles en cada episodio de validación. Compara continuar directamente con preguntar por duración, contexto, fecha de vencimiento o información adicional. Cada rama genera su propia traza y su estado persistido. Para prefijos idénticos, primero se promedian los mundos simulados y luego se elige una acción común. Los resultados incluyen el coste máximo de preguntar que compensa frente a continuar, la diferencia de acierto actual y futuro y la distancia entre la política evaluada y la mejor acción retrospectiva del simulador.

Cuando la pregunta por duración recupera información pertinente, se repite WRITE con la misma conversación y la misma respuesta actual, ocultando la réplica únicamente al escritor. El contraste comprueba si cambia el acierto futuro. La ejecución por defecto usa reglas públicas. `--checkpoint` permite repetirla con un adaptador LoRA congelado. `--family F4 --roots-per-family 2` acota la ejecución a dos historias ambiguas completas. Los resultados dependen del escritor, del lector y del banco simulado.

`INFORME.md` resume las diferencias por familia, `COSTE_POR_FAMILIA.csv` contiene la sensibilidad al coste de preguntar y `cases.jsonl` enlaza cada rama con su evidencia. La auditoría opera sobre validación.

En las 40 historias de validación, la [auditoría con reglas públicas](../reports/tesis_valor_preguntas_v1/INFORME.md) encontró que aclarar el contexto ayudó al acierto actual en cinco historias y aclarar la duración ayudó al acierto futuro en ocho. En las 16 realizaciones de esas ocho historias, dar la réplica al escritor mejoró el acierto futuro manteniendo idéntica la respuesta actual. Los porcentajes de esa auditoría describen la oportunidad que ofrece el simulador al escritor de reglas.

## Verificación de escrituras

El verificador inspecciona la propuesta del modelo, su procedencia y el efecto de las operaciones sobre consultas protegidas. Ejecuta las alternativas en SQLite temporal. El modo `veto` conserva el mayor subconjunto de operaciones admisibles. El modo `repair` puede añadir una actualización confirmada que mejora el recuerdo sustentado y conserva las respuestas protegidas. Ambos reciben la observación pública de WRITE.

```bash
python -m research.tesis_memory_guard_cli \
  --input reports/tesis_verificador_memoria_v1/EJEMPLO_ENTRADA.json --mode repair

python -m research.tesis_memory_replay \
  --trajectories research/runs/weak_sft_natural_20261104_001/evaluation/B3/trajectories.jsonl \
  --output reports/mi_prueba_verificador
```

La reproducción emparejada mantiene fijas la aclaración y la respuesta actual. Se ejecutó en 30 episodios de SFT adicional y 30 de GRPO con crédito por fase. El acierto futuro sustentado subió **68 % → 72 % → 96 %** en el primero y **48 % → 60 % → 100 %** en el segundo al pasar de escritura original a veto y reparación. La reparación redujo de tres a cero y de seis a cero, respectivamente, los episodios con daño a alguna consulta protegida. Son pruebas retrospectivas sobre simulaciones. El verificador y el evaluador comparten la gramática y la comprobación de respaldo. [Informe SFT](../reports/tesis_verificador_memoria_v2/sft_adicional/INFORME.md) · [Informe GRPO](../reports/tesis_verificador_memoria_v2/grpo_credito/INFORME.md).

El mismo procedimiento acepta las trayectorias nativas del conjunto reservado. El análisis conserva los conteos de operaciones con y sin respaldo por familia y semilla. [Comando y flujo de la evaluación reservada](comparacion_tesis.md#verificación-de-escrituras-sobre-historias-reservadas)
