# SPEC PARA CODEX — Auditoría de viabilidad científica de clarificación de contradicciones en memoria de largo plazo

## Rol y objetivo
Actúa como investigador de LLMs, ingeniero de evaluación y auditor de reproducibilidad. Trabaja sobre el repositorio adjunto (`experimento.py`, `scenarios.py`) si está presente; si faltan archivos, solicítalos antes de afirmar que lo auditaste. **No entrenes modelos todavía**. Audita, corrige y ejecuta un piloto para establecer si hay un problema no trivial de clarificación que justifique posteriormente SFT y GRPO en asistentes alimentarios personalizados con memoria longitudinal.

**Pregunta de investigación:** ¿Cuándo y cómo debe solicitar aclaración un asistente cuando su memoria de largo plazo contiene afirmaciones directamente incompatibles sobre la misma preferencia, alcance y período, y en qué medida las aclaraciones mejoran la respuesta actual y la coherencia en sesiones posteriores frente a baselines de prompting y heurísticas?

El objetivo NO es minimizar artificialmente el desempeño del modelo ni forzar ASK ante toda contradicción histórica. Un conflicto histórico puede estar ya resuelto mediante corrección explícita, confirmación actual o restricciones de ámbito; si la consulta admite una respuesta condicional, preguntar podría ser innecesario.

## Entorno de inferencia (red VPN privada)
- Principal: `http://127.0.0.1:12555/v1`, modelo `zai-org/GLM-5.3-Flash`, API OpenAI-compatible servida con vLLM.
- Segundo endpoint: `http://127.0.0.1:12559/v1`, mismo modelo; **no es un evaluador independiente por ser otra instancia**.
- Embeddings opcionales: `http://127.0.0.1:11434/api/embed`, modelo `bge-m3:latest` por Ollama.
- Verificar conectividad mediante `/v1/models` y `/api/tags`. Los servicios solo son accesibles dentro de la VPN. No publicar estos endpoints ni registrar credenciales. Variables de entorno para credenciales, si existen.
- La API de inferencia no concede por sí misma acceso a pesos, entrenamiento ni recursos de Slurm. No asumir viabilidad de SFT/GRPO hasta comprobarla.
- GLM puede utilizar tokens para `reasoning`: registrar `finish_reason`, `content`, `usage.completion_tokens` y `usage.completion_tokens_details.reasoning_tokens`; reintentar límites agotados según protocolo fijo. Nunca calificar `content:null` como decisión errónea del agente: registrar error técnico por separado.

## Revisión obligatoria del prototipo actual
1. Leer `experimento.py`, `scenarios.py`, artefactos de resultados y pruebas. Explicar arquitectura real y elaborar `AUDITORIA.md` con hallazgos, evidencia y severidad (bloqueante/alta/media/baja). No inventar resultados contra la H200.
2. Investigar problemas de **validez de etiquetas**: que dos expresiones opuestas separadas por meses representen cambio de gusto y no conflicto vigente; distinguir `historical_conflict`, `unresolved_conflict`, `correction`, `scope_mismatch`, `source_conflict`, `unknown`.
3. Auditar **filtración de información**: `truth`, `gold_action`, `user_reply`, `rationale`, `option_A/B` no deben aparecer en ningún prompt del agente, recuperación, extractor ni reglas evaluadas. Jueces pueden ver claves de evaluación, pero deben mantenerse aislados y registrar el protocolo.
4. Auditar contaminación entre escenarios, memorias, políticas y sesiones: DB SQLite independiente por escenario; snapshots antes de ASK; solo persistir una corrección si fue comunicada de forma válida. Verificar que la consulta futura no contenga la aclaración textual y que el resultado provenga de memoria almacenada.
5. Auditar limitaciones del simulador actual: juez semántico automático del mismo modelo; preferencia latente revelada en cuanto el juez aprueba la pregunta; respuesta fija que quizá no contesta exactamente lo preguntado; historial sintético con relleno repetitivo; la memoria cruda almacenada no implica gestión adecuada mediante ADD/UPDATE/INACTIVATE; medidas calculadas solo sobre solicitudes válidas.
6. Escribir pruebas que **fallen** al introducir deliberadamente: pregunta irrelevante; respuesta oculta filtrada; contradicción aparente por contexto distinto; corrección que no se guardó; extracción que borra una negación; BGE que recupera solo una de dos memorias; repetición innecesaria de ASK; respuesta final que viola la aclaración.

## Diseño realista de datos longitudinales
- Empezar con 20 historias para smoke test y crear un holdout independiente de al menos 60–100 historias/personas para la evaluación de viabilidad. No confundir 100 variantes de 20 historias con 100 muestras independientes. Separar desarrollo/prueba por **historia o persona**, nunca por filas.
- Cada historia: 8–25 interacciones distribuidas en 3–18 meses, cada interacción con `session_id`, fecha/hora, `role`, canal, fuente, fragmento textual original y contexto conversacional mínimo. Las memorias deben ser generadas a partir de esas interacciones por una etapa explícita de extracción/almacenamiento; mantener además un modo oracle con todos los mensajes relevantes.
- Usar lenguaje natural de conversación cotidiana: alusiones («al final lo de siempre»), dudas, rectificaciones parciales, negaciones, erratas, preferencias expresadas casualmente y distractores semánticamente próximos; NO repetir plantillas obvias ni rotular `MEMORIA A/B`, `contradicción` o `elige A/B` dentro del prompt del agente.
- Contradicción directa = afirmaciones incompatibles para **la misma persona, propiedad, propósito, contexto y período aplicable**. Registrar evidencia textual exacta de ambos lados. No inferir conflicto simplemente de preferencias diferentes en contextos distintos ni asumir que lo más nuevo siempre gana. Tratar un cambio legítimo y claramente expresado como caso resuelto, no como error.
- Diseñar escenarios de incertidumbre genuina donde ni recencia, ni autoridad de la fuente, ni consulta actual permitan escoger una única opción. Incluir también controles difíciles: cambio explícito, excepciones temporales, otra persona hablando, cita de un tercero, planes hipotéticos, preferencias condicionales y memoria redundante consistente. Las declaraciones de alergia o restricciones médicas deben tratarse como restricciones de seguridad y no como gustos intercambiables.
- Pares mínimos contrafactuales: mismas sesiones históricas pero añadir (a) decisión actual, (b) corrección inequívoca en sesión anterior, (c) petición de alternativas, (d) una sola memoria, (e) diferente alcance. Las etiquetas deben ser justificadas semánticamente por anotadores y no deducidas solo del nombre de variante.
- Congelar fecha de referencia por historia; no depender implícitamente del reloj del entorno. Revisar consistencia cronológica y que consultas/follow-ups sean lingüísticamente naturales.

**Ejemplo orientativo (no copiar literalmente en todos los casos):**
- Sesión de mayo, hablando de cenas habituales: «Cuando organices mis cenas, evita ponerme queso. Últimamente no me sienta bien en esos platos».
- Sesión de agosto, hablando también de cenas habituales: «Por cierto, en los menús para casa pon queso normalmente, me encantan los gratinados».
- Ninguno de los dos mensajes indica de manera inequívoca si el segundo sustituye al primero, cuál de ellos es más estable o si existe un motivo contextual omitido.
- Consulta de octubre: «Voy a organizar las cenas de la semana y hacer la compra, ¿qué me recomiendas?».
- Si el modelo necesita fijar un menú único, una pregunta sobre **si desea incluir queso actualmente en las cenas** puede resolver la elección. Si puede proponer platos útiles sin queso, quizá ANSWER sea aceptable: la etiqueta debe considerar utilidad de la tarea y coste de interrumpir, no una regla automática ASK.

## Condiciones y ablaciones obligatorias
Evaluar sobre las mismas historias y presupuestos comparables:
1. `oracle_full_history` + prompt neutro: aislar razonamiento cuando la evidencia está presente.
2. `oracle_full_history` + prompt guiado: baseline de prompting fuerte.
3. `retrieval_bge_topk` + prompt guiado, al menos k=4,8,16: medir recuperación de **ambos lados** de la contradicción y vigencia.
4. `extracted_memory` + prompt guiado: extracción sin acceso a futuro; medir conservación de negación, procedencia, alcance y contradicciones.
5. Baselines de control: `always_ask`, `never_ask`/respuesta prudente, `recency_only`, reglas simples de conflicto; indicar límites.
6. Ablaciones de extracción/retrieval y memoria post-clarificación; evaluar en conjunto y por componente. No atribuir una falla de retrieval a la política ASK.
7. Si es viable, evaluar un modelo entrenable más pequeño (p. ej. 3B/7B) como posible sujeto futuro de SFT/GRPO. GLM grande puede ser benchmark de referencia, no necesariamente el modelo a entrenar.

## Simulador multi-turno correcto
- Estado privado por historia: preferencias verdaderas para fecha y alcance, contexto, modo de respuesta y posible incertidumbre genuina. **El agente nunca ve ese estado**.
- Ante ASK, el simulador interpreta la *pregunta exacta* y contesta solo la información solicitada con el nivel realista de precisión disponible. Una pregunta irrelevante recibe una respuesta irrelevante/no resolutiva, nunca la preferencia latente por defecto. Permitir «no estoy seguro/a» cuando corresponda.
- Evitar validar utilidad solo por keywords o por un juez GLM que comparte modelo con el agente. Juez automático auxiliar + rúbrica objetiva + evaluación humana ciega estratificada; registrar desacuerdos y `uncertain`. Mantener por separado `question_relevance`, `expected_information_gain`/reducción de hipótesis, `answer_grounded`, `response_task_success`.
- Después de ASK: comprobar si la respuesta del usuario resolvió el punto; permitir una segunda pregunta si aún falta información (máximo configurable), penalizar preguntas duplicadas y vueltas innecesarias.
- Guardar toda aclaración real en memoria persistente con timestamp, fuente y vínculo con memoria anterior. Probar política de conservación histórica: no sobrescribir silenciosamente hechos ni borrar procedencia; documentar cuándo se marca un dato como sustituido.
- Probar una **nueva sesión sin contexto de conversación reciente**; el agente solo puede utilizar memoria persistida y recuperación normal. También probar una sesión futura con distractores relevantes. Comprobar si repite la aclaración o contradice la preferencia confirmada.
- Reproducir roles `user`/`assistant` reales, no concatenar artificialmente todas las intervenciones como si fueran un mensaje de usuario.

## Métricas y análisis
Registrar denominadores y casos no evaluables; reportar media y desagregaciones por tipo de contradicción, historia/persona, longitud, ruido y modo de memoria:
- `historical_conflict_precision/recall`, `unresolved_conflict_precision/recall`.
- `action_accuracy` con conjunto de acciones aceptables, `ASK_recall`, `ASK_false_positive_rate`, `CONDITIONAL_appropriateness`.
- `question_utility_human`, `question_utility_auto`, cobertura del dato decisivo y penalización de preguntas reiteradas.
- `after_clarification_task_success`, `followup_memory_consistency`, `unnecessary_ask_after_clarification`.
- `retrieval_pair_recall`, extracción fiel de negaciones/alcance, tasa de errores de parseo, `finish_reason`, latencia, tokens, y costes por interacción.
- Comparaciones emparejadas sobre las mismas historias; intervalos de confianza (bootstrap agrupado por persona/historia). No hacer inferencia estadística considerando variantes de la misma historia como independientes. Incluir ejemplos cualitativos de fracaso con evidencia.
- No reportar 100% de «éxito end-to-end» solo porque el simulador entregó la respuesta oculta. Exigir: ASK útil -> aclaración real -> respuesta fiel -> memoria guardada -> consulta futura correcta.
- Anotación humana ciega de una muestra predefinida, equilibrada entre condiciones y errores (idealmente dos evaluadores en un subconjunto), con pautas de decisión y revisión de discrepancias.

## Reproducibilidad y calidad de software
- `seed` explícita, manifiesto JSON de parámetros, hashes/versiones de prompts y datasets, IDs de modelo y versión, configuración de `temperature`, `max_tokens`, tiempo de referencia y datos de tokens/latencia.
- CLI con `--dry-run`, `--limit`, `--seed`, `--two-turn`, `--memory-mode`, `--retrieval`, `--topk`, `--output`, `--resume`, `--timeout` y soporte `--judge-url` sin forzar acceso a red en tests.
- Persistencia incremental, reanudación idempotente y sin mezclar resultados de distintas configuraciones; no borrar ejecuciones previas sin confirmación.
- Test unitarios y de integración contra API falsa/local; smoke real de 5–10 casos; detener y registrar HTTP 4xx, timeouts y contenido vacío sin convertirlos en errores del agente.
- Nunca subir secretos, PII real ni exponer el endpoint de la VPN. Datos sintéticos y claramente etiquetados como tales.

## Entregables que debes producir
1. `AUDITORIA.md`: fallas detectadas en el simulador existente con archivo/línea, severidad y corrección.
2. `METODOLOGIA.md`: hipótesis falsables, anotación del gold, diseño por condiciones, protocolo de baselines, evaluadores y amenazas a la validez.
3. Código corregido y modular: `data_generation/`, `memory/`, `simulator/`, `agents/`, `evaluation/` o estructura equivalente; pruebas `pytest`.
4. `casos_gold.jsonl` PRIVADO y `casos_agent.jsonl` SIN gold ni atributos ocultos; pruebas automáticas de no filtración.
5. Resultados `per_case.jsonl`, `metrics.json`, `errors.csv`, `human_review_blind.csv`, tabla comparativa y muestras trazables.
6. `README.md` reproducible con comandos Fedora/VPN, pruebas sin red y ejecución contra la H200.
7. `INFORME_VIABILIDAD.md` con respuesta razonada: (a) si los conflictos son realistas y directos, (b) si funciona el simulador, (c) qué fallos son del razonador vs memoria, (d) margen frente al prompting fuerte, (e) si conviene SFT y GRPO o rediseñar la hipótesis. **Nunca afirmar una ganancia de entrenamiento si no se ha entrenado**.

## Criterios de aceptación / gates
- Antes de llamar a la H200: tests locales verdes, dataset auditado, cero filtración de gold, >= 5 variantes por historia y al menos 1 contradicción real validada por tema; declarar el tamaño real de historias independientes.
- Piloto: 5–10 escenarios de punta a punta con log completo y sin fallos técnicos silenciados.
- Viabilidad: al menos 60–100 historias independientes de test retenidas, preguntas anotadas, resultados por modalidad de memoria y comparación con prompting fuerte; en caso de resultados muy altos, incrementar realismo y evaluar modelo pequeño antes de recomendar RL.
- Indicadores orientativos (NO reglas estadísticas universales): errores reproducibles con diferencia relevante de éxito futuro (p. ej., >=10 puntos porcentuales entre métodos en un holdout), preguntas realmente informativas y fallos que persisten con prompting fuerte son buena motivación. Si guided se acerca al techo sin errores relevantes, reportarlo y NO justificar SFT/GRPO solo para mejorar el número.
- El resultado de la auditoría puede ser «todavía no viable con la evidencia disponible». Mantén esa posibilidad abierta.

## Orden de trabajo solicitado
Primero inspecciona el repo y da diagnóstico de problemas; segundo corrige pruebas/diseño y ejecuta `--dry-run`; tercero solicita los comandos mínimos para verificar servicios de la H200 y realiza un piloto si tienes acceso efectivo a VPN; cuarto evalúa y entrega el informe. **No inventes métricas ni presupongas acceso de red**. No pidas permiso para cambios no destructivos, pero nunca sobrescribas datos anteriores ni inicies entrenamiento de GPU sin aprobación.
