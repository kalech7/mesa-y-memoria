# Protocolo previo a evaluación: SPEC-RL-002

Documento de diseño, no registro externo ni evidencia de resultados. Se redacta antes de examinar el nuevo test. Los artefactos de ejecución deben guardar su hash y configuración; una enmienda posterior debe identificarse como tal. SPEC-001 y sus 32 casos lingüísticos quedan intactos y no se usan para ajustar este piloto.

## Pregunta conservada y alcance de la adaptación

La pregunta documentada en `SPEC_CODEX.md:6` es: «¿Cuándo y cómo debe solicitar aclaración un asistente cuando su memoria de largo plazo contiene afirmaciones directamente incompatibles sobre la misma preferencia, alcance y período, y en qué medida las aclaraciones mejoran la respuesta actual y la coherencia en sesiones posteriores frente a baselines de prompting y heurísticas?» `METODOLOGIA.md:21-24` fija como estimando principal futuro el éxito en una sesión posterior, con utilidad inmediata y costo como secundarios. No hay aquí un documento institucional que certifique que este sea el objetivo aprobado por el tutor; es la formulación explícita encontrada en el repositorio.

SPEC-RL-002 pregunta por el efecto de entrenar la política mediante RL y amplía los conflictos a CPC, BOC y SCC. El piloto estudia decisión inmediata y adquisición de información durante un episodio, con memoria estructurada y extracción determinista como dos condiciones. No sustituye el objetivo anterior: no mide coherencia entre sesiones reales, gestión aprendida ADD/UPDATE/DELETE, recomendaciones abiertas, ni validación humana. Un resultado favorable solo justificaría continuar el estudio de la política de interacción; para responder la formulación previa completa falta una nueva sesión sin conversación reciente y recuperación desde persistencia.

## Modelo formal y criterio de secuencialidad

Se estudia un POMDP finito de adquisición de información, no un cambio físico de gustos provocado por preguntar. Su tupla es `(S,A,T,O,Z,R,H)`. El estado `s_t=(u,c,v,m,h_t,b_t)` contiene perfil latente `u`, contexto efectivo `c`, vigencia/estado actual `v`, memoria visible `m`, historial `h_t` y presupuesto `b_t`. La observación `o_t` contiene exclusivamente consulta, preferencias y restricciones comunicadas, contexto comunicado, mensajes observados, slots ya preguntados y presupuestos restantes. El agente decide `a_t~pi(.|o_0,a_0,...,o_t)`.

Las aclaraciones generan `o_(t+1)` condicionado a la pregunta concreta y al estado privado. La creencia se actualizaría como `b_(t+1)(s') proportional Z(o_(t+1)|s',a_t) sum_s T(s'|s,a_t)b_t(s)`. La política implementada puede usar una compresión de historial observable; no se afirmará que tal compresión es un estado suficiente ni que Q-learning tenga garantía de convergencia sobre ella. Objetivo: `J(pi)=E[sum_(t=0)^(H-1) r_t]`, descuento 1 por horizonte finito. Responder, preservar o abstenerse terminan; horizonte máximo 4 decisiones, como máximo 2 aclaraciones. Preguntar puede consumir presupuesto sin reducir incertidumbre.

El problema no se justifica como RL secuencial solo por tener `step` o una penalización por turno. Deben existir pares con idéntica observación inicial y estados latentes distintos, respuestas dependientes del slot, y una aclaración que permita adaptar una decisión posterior. Si las preguntas no cambian qué decisión fundamentada puede tomarse, el gate de secuencialidad falla. Un bandit contextual sirve como control de decisión inmediata, pero no representa el valor contingente de preguntar y luego actuar. Clasificación supervisada es una alternativa competitiva y obligatoria; superioridad RL no está presupuesta.

La formulación de estados parcialmente observables sigue a [Kaelbling, Littman y Cassandra, 1998](https://www.cassandra.org/arc/papers/aij98.pdf). La distinción entre control aprendido a partir de retornos, planificación y métodos tabulares sigue a [Sutton y Barto, segunda edición](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf). Son fundamentos metodológicos, no evidencia de validez humana del simulador.

## Contrato visible y acciones

`FoodPreferenceEnv(scenario, condition='ORACLE_STRUCTURED', reward_config=None, max_steps=4, max_clarifications=2)` ofrece `reset(seed=None,scenario_id=None)` y `step(action)`. `reset` retorna observación y diagnóstico vacío. `step` retorna observación, recompensa, terminación, truncamiento y diagnóstico seguro. IDs, semillas, categorías, etiquetas, respuesta futura y perfil latente no forman parte de observación o info accesible para la política.

Catálogo inicial: `ANSWER:ensalada`, `ANSWER:pizza`, `ANSWER:sopa`, `ANSWER:arroz`, `CLARIFY_DAY`, `CLARIFY_MEAL`, `CLARIFY_COMPANY`, `CLARIFY_PLACE`, `CLARIFY_CURRENT_PREFERENCE`, `PRESERVE_CONFLICT`, `ABSTAIN`. No existe VERIFY sin una fuente verificable, por lo que se excluye. Preservar conflicto no es borrar ni resolver memorias. La restricción clínica simulada se evalúa separadamente de gustos; ningún resultado de este catálogo constituye recomendación médica.

E1 ORACLE_STRUCTURED recibe memorias estructuradas correctas de evidencias visibles. No recibe la preferencia oculta efectiva ni el contexto no comunicado. E2 PIPELINE extrae de eventos usando `simulator/preferences.py`; cambia memoria accesible, no perfil latente, rewards ni test. Se debe separar degradación de extracción y error de política.

## Datos y presupuesto congelado

Propuesta inicial: 2.000 usuarios/episodios de entrenamiento, 400 validación, 600 test; tres semillas de entrenamiento independientes; mismas instancias test para todos los métodos. Una instancia por usuario permite bootstrap al nivel del episodio. Si existen variantes de una persona, se agrupan y remuestrean juntas. Presupuestos son límites de ingeniería, no cálculo de potencia ni garantía de generalización.

Particiones con usuarios, IDs y familias textuales diferentes. Catálogo semántico pequeño necesariamente compartido; se auditarán coincidencias de estados estructurales y se explicará si el test solo cambia superficie. Separación por ID por sí sola no demuestra generalización a combinaciones latentes nuevas. Un test adicional altera distribución de preferencias/contextos; no se ajustan parámetros sobre él. La mezcla debe contener casos resueltos, CPC de día/comida/compañía/lugar, BOC, SCC resoluble y no resoluble, evidencia insuficiente, preguntas informativas, parciales e inútiles y distractores. No se incrementa una categoría únicamente para favorecer RL.

Límite computacional: CPU local y biblioteca estándar; no instalaciones, APIs, GPUs ni LLM. Tres ejecuciones pequeñas con límite explícito de episodios y tiempo, checkpoints y curvas. Si una ejecución no concluye dentro del presupuesto se reporta incompleta. Python 3.13.15 y dependencias vacías en `pyproject.toml`; inspección local no encontró numpy, torch, gymnasium, matplotlib ni sklearn. Esto motiva un piloto de Q-learning sobre rasgos observables o algoritmo tabular equivalente. PPO se discute como extensión, no se etiqueta Q-learning como PPO. PPO no resuelve por sí mismo observabilidad parcial; requeriría historial, estado de creencia o recurrencia.

## Recompensas y separación de evaluación

Parámetros iniciales pactados para el adaptador: decisión correcta +1; incorrecta -1; afirmación sin fundamento penalización adicional 0,6; violación de restricción penalización adicional 2; preservación correcta de irresoluble +0,7; preservación incorrecta -0,2; abstención -0,35; aclaración costo 0,10. Bonus directo por información =0. Evita pagar por producir mensajes y permite que valor de información surja por decisión final. Las expectativas provienen del perfil sintético/evidencias, no de acción elegida por heurística.

No basta acertar por suerte la preferencia latente: fidelidad exige respaldo visible. Decisiones alternativas justificadas se admiten mediante un conjunto evaluable, no comparación exacta de texto. Una restricción es una condición de seguridad evaluada independientemente, aun si la recompensa compuesta parece favorable. Los pesos son preferencias del experimentador, no utilidades humanas estimadas.

Sensibilidad predefinida al costo de preguntar: 0,05 / 0,10 / 0,20. Evaluación sobre mismas trayectorias permite aislar utilidad compuesta; si se reentrena para esos pesos, se declara como entrenamiento adicional y se usa solo validación para selección. Se reporta también éxito, seguridad y fidelidad sin ponderar. Una mejora exclusiva de retorno que empeore calidad no se interpreta como mejora de razonamiento.

## Baselines y selección

NEVER_ASK responde o preserva con evidencia disponible; ALWAYS_ASK agota preguntas aplicables; CURRENT_HEURISTIC adapta `simulator/resolution.py` sin acceso a estado latente; SUPERVISED_POLICY aprende decisiones desde información comparable; STRONG_REFERENCE usa únicamente observaciones permitidas y una estrategia explícita de planificación/reducción de ambigüedad. Si este último consulta el perfil privado, deja de ser referencia permitida y se etiqueta solo cota privilegiada no alcanzable.

La fuente de etiquetas SFT, su cantidad y costo deben publicarse: si usa demostraciones de la referencia fuerte tiene una ventaja legítima documentada, no un baseline sin costo. RL recibe recompensas escalares a través de interacción. Todo rasgo usado por modelos o reglas se obtiene de observación. Categoría/ID/seed nunca son features.

La selección de hiperparámetros y del mejor baseline se realiza exclusivamente en validación y se congela antes de test. Se evalúan tres semillas, con hiperparámetros finitos definidos en manifiesto. No se extiende entrenamiento para lograr GO. Comparar contra política no entrenada, heurística y SFT permite distinguir aprendizaje de superioridad y necesidad de RL.

## Estimandos, intervalos y umbral práctico

Principal piloto: retorno medio y diferencia emparejada entre políticas en episodios test idénticos. IC95 mediante bootstrap de episodios independientes (usuarios agrupados si corresponde), con semilla y número de réplicas explícitos. Se mantiene la dependencia entre políticas en cada remuestreo. Se reportan también resultados por cada semilla de entrenamiento: 600 episodios repetidos en tres semillas no equivalen a 1.800 usuarios independientes.

Diferencia mínima de interés práctico predefinida: delta retorno=0,10 por episodio, un costo de aclaración (0,10) y equivalente a convertir el 5% de fallos -1 en aciertos +1 antes de penalizaciones. Delta calidad=5 puntos porcentuales. No son umbrales clínicos ni pruebas universales; si precisión estadística es insuficiente, se informa inconclusión. El headroom principal se estima en validación como retorno referencia fuerte menos retorno del mejor baseline convencional (incluida SUPERVISED_POLICY) seleccionado allí, además del margen frente a CURRENT_HEURISTIC. Test confirma el margen; no lo usa para decidir entrenar.

Métricas y denominadores:

- Retorno: suma de recompensas por episodio, promedio sobre todos los episodios elegibles, errores técnicos aparte.
- Exactitud final: decisión final aceptable según restricciones y evidencia; todos los episodios, truncamiento sin decisión cuenta fallo.
- Fidelidad: episodios cuya decisión no afirma alimentos sin respaldo visible / todos; abstención se informa aparte para evitar confundir silencio con utilidad.
- Calidad de clarificación: preguntas que comunican dato pertinente y cambian posibilidades fundamentadas / preguntas realizadas; no evaluable si cero preguntas. La mera longitud del mensaje no sirve.
- Clarificaciones innecesarias: pregunta cuando la evidencia ya permite respuesta fundamentada o cuando el slot no tiene utilidad marginal / preguntas realizadas, con subtotales por motivo.
- Necesarias omitidas: episodios con incertidumbre resoluble que terminan sin adquirir evidencia requerida / episodios con esa necesidad contrafactual, no todos los conflictos.
- Turnos y preguntas: media por todos los episodios; distribución de acciones y proporción que agota presupuesto.
- Afirmaciones injustificadas y violaciones: episodios con evento / todos, más denominador de episodios con restricciones.
- Preservación de irresolubles: preservación correcta / episodios genuinamente irresolubles; precisión preservaciones correctas / preservaciones emitidas.
- Categorías CPC/BOC/SCC/resuelto/ausencia, condición E1/E2 y OOD se muestran aparte. No se atribuye a RL el éxito histórico de SPEC-001.

## Gates y controles de invalidación

GATE 1: reproducibilidad reset/step, observaciones sin claves privadas, insensibilidad a cambios de IDs/semillas, estados latentes alternativos indistinguibles, preguntas con respuestas diferenciadas, presupuesto/terminación y restricciones comprobados. Análisis contrafactual de >=100 estados iniciales con acciones alternativas y misma aleatoriedad. Debe probarse utilidad contingente posterior; un costo fijo no es evidencia suficiente.

GATE 2: en validación, margen contra baseline apropiado >=0,10 y límite inferior IC95>0, además de casos reproducibles mejorables frente a heurística. La selección incluye SUPERVISED_POLICY. Si la referencia es igualada por SFT se declara margen de política menor/cero y necesidad RL no demostrada; en ese caso no se entrena RL porque el usuario condicionó el entrenamiento a superar los gates previos. Una brecha solo contra ALWAYS_ASK no basta.

GATE 3: tres semillas muestran aprendizaje no degenerado respecto a inicial/no entrenada y mejora reproducible frente a una referencia relevante en escenarios nuevos, con intervalos y calidad final. No exige vencer todos los métodos; debe distinguirse `aprendizaje demostrado`, `superioridad demostrada` y `RL necesario`. Si mejora retorno pero empeora seguridad o fidelidad, el gate no avala una política recomendable.

GATE 4: el reporte cuantifica mejora/igualdad/empeoramiento frente a convencionales con estimandos y límites. Puede aprobar capacidad de responder una pregunta acotada aunque el resultado sea negativo. No implica responder automáticamente la pregunta longitudinal oficial.

Se invalida un resultado con fuga, acceso futuro, uso de labels por política o recompensa circular derivada de heurística. Se conserva el artefacto fallido, se corrige y se repite con nueva versión. Controles: cambios de IDs/seed, contrafactual de estado privado, pregunta diferente mismo inicio, orden/distractor, restricción adversarial, coincidencias train/test, entropía/frecuencias de acciones, calidad vs retorno y cambio de distribución. Si categorías tienen correlaciones casi deterministas con acciones óptimas se documenta: aprender el catálogo sintético no demuestra habilidad general.

## Revisión y límites

Un conjunto manual escrito por un agente de código no constituye revisión humana independiente. Se puede entregar un conjunto manual sintético adicional y rúbrica, pero debe quedar pendiente adjudicación ciega por dos personas externas a implementación y generación. Los 32 casos originales se conservan como evidencia histórica; no se convierten en nuevo test RL. El dictamen final distingue viabilidad técnica, entorno secuencial, margen, aprendizaje y alcance científico; el resultado negativo es admisible.
