# Protocolo experimental y criterios de decisión

## Pregunta y estimandos

¿Cuándo mejora la utilidad actual y futura pedir una aclaración sobre preferencias
incompatibles para la misma persona, propiedad, propósito, contexto y período?
No se define éxito como preguntar siempre ni como adivinar una preferencia privada.

Hipótesis falsables:

1. Con toda la evidencia disponible, un prompt especializado puede bastar. Si no
   aparecen errores relevantes frente a la revisión humana, no se justifica RL.
2. Recuperar ambos lados y la corrección vigente explica parte del rendimiento.
   Si el fallo desaparece en oracle, atribuirlo primero a memoria/recuperación.
3. Una pregunta informativa aumenta fidelidad después de una respuesta real y en
   una sesión nueva; una pregunta irrelevante no puede obtener esa misma ventaja.
4. Los controles resueltos/contextuales penalizan preguntar innecesariamente.
5. Si persisten fallos de decisión con prompting fuerte en un modelo entrenable
   pequeño y holdout validado, procede comparar prompting vs SFT antes de GRPO.

Estimando principal futuro: diferencia emparejada de éxito de tarea en sesión
posterior, contando todas las historias elegibles y reportando no evaluables.
Secundarios: utilidad inmediata, coste de interrupción, detección, recuperación,
conservación de memoria, tokens y latencia. No hay medida de entrenamiento actual.

## Datos y unidad experimental

Se generan 20 biografías de desarrollo y 80 candidatas de evaluación, cada una con
12 variantes (1.200 filas). Las variantes no son sujetos independientes. Cada
trayectoria base tiene 9–11 sesiones, 8–12 tras controles, y 16–23 mensajes, a lo
largo de aproximadamente 4–17 meses. El tiempo de referencia es explícito.

El generador comparte 20 temas, frases y plantillas entre particiones. Estos son
100 perfiles sintéticos diferentes, no evidencia de 100 historias independientes
de población. La revisión humana debe descartar artefactos y ampliar/redactar
historias si la diversidad semántica resulta insuficiente. No se evalúa el test
retenido durante el piloto técnico ni se ajustan prompts a sus resultados.

El archivo público contiene mensajes/consultas y procedencia; el privado contiene
gold, justificaciones, evidencia exacta y estado del simulador. Los IDs públicos
no codifican variante, tema ni preferencia. Ninguna política recibe el objeto gold.

Variantes: abierto con respuesta condicional aceptable; necesidad de lista única;
decisión actual; corrección anterior; petición de alternativas; una sola nota;
otro ámbito/persona; fuente externa no confirmada; excepción temporal; hipótesis;
incertidumbre genuina; redundancia consistente.

## Anotación del gold

Todas las etiquetas iniciales son propuestas provisionales del generador. La
semántica de un caso no queda validada por su nombre ni por pasar el esquema.
La ausencia de una rectificación explícita tampoco demuestra que no cambió el gusto.

Los revisores reciben `gold_review_blind.csv` con historial y consulta, sin las
etiquetas propuestas. Deben indicar historical_conflict, unresolved_conflict,
tipo, conjunto de acciones aceptables, justificación y evidencia aplicable.
Acciones se separan con `|`. `true/false` se escriben explícitamente. Se conservan
propuesta original y desacuerdo al importar. Las restricciones médicas tienen
prioridad de seguridad; no se etiquetan como gustos que se pueden intercambiar.

Antes de evaluación científica: revisar todos los casos usados y al menos una
incompatibilidad directa por tema. Dos anotadores revisan inicialmente todos los
casos del piloto, y un subconjunto predefinido del holdout (al menos 20 personas).
Resolver discrepancias con adjudicación documentada, conservando uncertain cuando
la evidencia no determine el gold. Si se rechaza una historia, reemplazar la
biografía completa y conservar registro; nunca contar variantes adicionales como
nuevas personas. No autocompletar campos de revisor para superar un gate.

Para respuestas/preguntas se usa un paquete distinto, `human_review_blind.csv`,
que oculta política, modelo, variante y dictamen automático. Incluye evidencia,
pregunta exacta, respuesta del usuario, recomendación y sesión futura. Dos revisores
anotan relevancia, cobertura del dato decisivo, resolución, fidelidad y éxito de
tarea por separado. Publicar acuerdo, matriz de discrepancias y no evaluables;
el software prepara el paquete, no reemplaza el trabajo humano.

## Condiciones y presupuestos

Comparar las mismas historias y variantes bajo:

| Condición | Evidencia | Política |
|---|---|---|
| Oracle neutro | Todos los mensajes con roles reales | Prompt neutro |
| Oracle guiado | Todos los mensajes con roles reales | Prompt especializado |
| BGE k4/k8/k16 | Mensajes recuperados por consulta pública | Prompt especializado |
| Extracción completa | Memorias extraídas solo del pasado y con origen | Prompt especializado |
| Extracción+BGE | Memorias extraídas y recuperadas | Prompt especializado |
| Controles | La misma evidencia de cada comparación | always_ask, never_ask prudente, recency_only, simple_conflict |

`raw/full` y `oracle_full_history/full` son equivalentes como evidencia en esta
implementación; no contar su duplicación como ablación nueva. La diferencia
efectiva aparece al activar BGE o extracción. El baseline never_ask produce una
respuesta cauta/condicional: eso no garantiza utilidad y debe anotarse como tal.
Las heurísticas usan un léxico fijo público y reglas documentadas; no reciben gold.

Mantener modelo, temperatura, semilla, límite de salida, máximo de preguntas y
presupuestos por llamada comparables. Un reintento fijo duplica max_tokens hasta
16.384 solo ante contenido vacío o límite; contabilizar ambos intentos. No aumentar
presupuestos solo para un método tras mirar si acertó. Los jueces/extractores
consumen llamadas adicionales: informar coste por componente, no ocultarlo.

## Simulador y memoria

El respondedor local identifica conservadoramente intención, tema y ámbito. Una
pregunta sobre presupuesto recibe presupuesto; otro tema/ámbito/tercero o consulta
médica desconocida no obtiene la preferencia oculta. Una elección incierta recibe
«todavía no lo tengo claro». No hay un juez que habilite la respuesta privada.

Limitación crítica: este parser es de reglas, no comprensión semántica general.
`question_utility_auto` es un indicador provisional de cobertura/resolución, no
utilidad humana. `information_gain` vale 0/1 como proxy de resolución y no es una
estimación probabilística de entropía. Evaluar falsos positivos y negativos del
parser contra anotación humana antes de usarlo como señal de recompensa.

Se permiten dos preguntas por defecto; preguntas repetidas o paráfrasis del slot
ya resuelto tienen ganancia cero. Repetir una preferencia ya comunicada puede
producir una respuesta real, pero no se cuenta como nueva información útil.
Una pregunta acotada a «esta vez», hoy o una cena concreta solo confirma esa
ocasión. La respuesta no promete una preferencia futura. Esa aclaración puede
guardarse como evento, pero no acredita por sí sola resolución general o futura.
El detector temporal también es conservador y requiere validación humana.

SQLite es independiente por persona/variante/condición/intento. Antes de ASK hay
snapshot. Cada ADD conserva raw_text, procedencia, fechas, session_id y vínculos.
La consulta actual también es un evento real; una decisión comunicada allí puede
ser recuperada después. Las aclaraciones solo se guardan tras respuesta válida.
No se inactivan automáticamente todas las notas recuperadas. La operación explícita
INACTIVATE conserva el original y exige una corrección resuelta/comunicada; un
reconciliador semántico automático validado queda fuera del alcance actual.

En el follow-up se cierra y reabre SQLite. No se pasa `recent`; la consulta futura
no contiene la respuesta recién dada. Se prueban distractores y ablación sin guardar
aclaración. Una recomendación futura correcta sin recuperar evidencia suficiente
no basta para demostrar uso de memoria.

## Métricas y análisis

Registrar resultados por caso, no evaluables y denominadores. Separar:

- Detección histórica y vigente: precisión y recall.
- Acción: pertenencia al conjunto aceptable; ASK recall cuando ASK es obligatorio;
  falsos positivos cuando ASK no es aceptable. CONDITIONAL apropiado por etiqueta
  no implica que su contenido cumpla la tarea.
- Preguntas: relevancia, información solicitada, resolución, utilidad humana,
  duplicados y coste de interrupción.
- Respuesta: fidelidad y éxito de tarea separados; uncertain no es fail.
- Memoria: bilateralidad recuperada, aclaración guardada/recuperada, consistencia
  futura y ASK innecesario después de aclarar.
- Transporte: HTTP, timeouts, contenido vacío, parseo, finish_reason, todos los
  intentos, prompt/completion/reasoning tokens y latencia. Sin tarifas, coste monetario null.

La cota end-to-end exige pregunta resolutiva, respuesta fiel y útil, guardado,
recuperación y respuesta futura fiel/útil. Se informan por separado los eslabones
inciertos; la cota conservadora no se interpreta como tasa de fallos del modelo.
Comparaciones emparejadas por caso y bootstrap agrupado por persona (1.000 réplicas
para exploración). No hacer inferencia por filas independientes. Informar también
distribuciones por tipo, longitud y ruido, y pares excluidos por errores técnicos.

El juez GLM opcional es auxiliar del mismo modelo/familia; no es árbitro independiente.
La regla textual local de fidelidad es deliberadamente incompleta: muchos resultados
son uncertain. Nunca usar una cadena repetida del gold como prueba de éxito de tarea.

## Desviaciones observadas y estado de los gates

Los pilotos archivados full_v1/full_v2 y el diagnóstico BGE se realizaron antes de
completar la validación humana requerida por SPEC. El flag técnico había omitido
ese gate. Se reconoce la desviación, se detuvieron llamadas adicionales y se corrigió
la CLI: el piloto técnico y el diagnóstico remoto también exigen revisión previa.
No se presentan esas ejecuciones como evaluación confirmatoria ni como evidencia
suficiente de viabilidad. Los resultados previos y sus fuentes no se reescribieron.

La revisión interactiva de dos casos es parcial, posterior a observar resultados,
con un solo revisor; no constituye la muestra ciega predefinida de este protocolo.
Validó utilidad de una pregunta y una decisión aceptable; declaró insuficiencia
de información para fidelidad. Se conserva cada dimensión por separado.

La edad de una nota puede justificar confirmar una preferencia aun sin contradicción.
Por ello, falsos positivos de ASK derivados solo de la variante `single` son
provisionales: el gold debe considerar coste de interrumpir, antigüedad y utilidad
de responder. La sensibilidad posterior no sustituye una rúbrica predefinida.

## Decisión sobre entrenamiento

SFT se considerará después de observar errores reproducibles de política con
evidencia suficiente y prompting fuerte, y contar con demostraciones humanas
consistentes. Primero comparar SFT con prompting al mismo presupuesto.

GRPO solo tendrá justificación adicional si una recompensa validada refleja
utilidad multisesión y resiste atajos: ASK siempre, memorizar respuestas del
simulador, preguntas de palabras clave, respuestas evasivas o explotación del juez.
Comparar SFT+GRPO contra SFT, no solo contra prompt neutro. Un umbral orientativo
de diez puntos en éxito futuro no sustituye IC, relevancia práctica ni réplica.

Si GLM con prompt guiado queda cerca del techo, evaluar primero un modelo más
pequeño con pesos/recursos realmente disponibles. Que Ollama liste un 8B o que
vLLM exponga una API no prueba acceso a pesos, licencia, optimizador, Slurm ni GPU
para entrenamiento. No se ha comprobado esa viabilidad operativa.

## Referencias primarias y alcance

[LongMemEval](https://arxiv.org/abs/2410.10813) distingue extracción, razonamiento
multisesión/temporal, actualizaciones y abstención; motiva separar componentes.
[LoCoMo](https://arxiv.org/abs/2402.17753) describe generación con personas/eventos
y verificación/edición humana; fundamenta no equiparar plantillas con datos validados.
[DeepSeekMath](https://arxiv.org/abs/2402.03300) introduce GRPO para razonamiento
matemático. Transferir ese algoritmo a clarificación alimentaria es una hipótesis
de esta tesis, no un resultado demostrado por ese trabajo.
