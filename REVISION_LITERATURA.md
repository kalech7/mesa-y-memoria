# Antecedentes y delimitación de la contribución

## Qué aporta Mesa y Memoria a noviembre de 2026

¿Puede un modelo pequeño aprender **en qué visita gastar una única pregunta**, cuando el pedido actual ya está resuelto? En dos historias gemelas la persona pide los mismos platos. Solo cambia qué comida prevé repetir. El asistente puede aclarar hoy o esperar, y la tercera visita permite comprobar el resultado en SQLite. El aporte que sostienen los datos es un diagnóstico reproducible en español que separa repartir preguntas entre historias de reconocer el momento correcto dentro de cada par. [Simulador y métrica](docs/momento_aclaracion_v6.md) · [Comparación v7](reports/tesis_timing_pares_a100_v7/INFORME.md)

### Qué han demostrado otros grupos

Búsqueda dirigida actualizada el **27 de noviembre de 2026** en ACL Anthology y arXiv, con términos sobre adquisición activa de preferencias, coste de preguntar, vigencia y memoria entre sesiones. Se priorizaron estudios primarios que comparten la decisión, el tipo de memoria o el presupuesto. Esta selección permite delimitar una contribución empírica y deja abierta la existencia de diseños similares.

| Trabajo | Qué prueba | Implicación para la tesis |
| --- | --- | --- |
| Wu et al., ATRBench, EMNLP 2026 | El agente debe obtener durante una interacción una preferencia oculta que solo será útil en una tarea posterior. Distinguen adquisición de aplicación. | Preguntar hoy para usar la respuesta después ya tiene un antecedente directo. Nuestro contraste invierte la agenda manteniendo iguales los pedidos y exige acertar ambos miembros del par. [Artículo](https://aclanthology.org/2026.emnlp-main.1505/) |
| Wang et al., ActiveWebPref, prepublicación, octubre de 2026 | Valoran la utilidad acumulada de una preferencia frente al coste de adquirirla y comparan políticas con presupuestos. Su presupuesto por interacción se renueva. | Aquí una pregunta usada en la primera visita se agota para la segunda. El coste de oportunidad de esperar se puede medir directamente. [Artículo, método y limitaciones](https://arxiv.org/html/2610.31107v2) |
| Li, Yao y Zheng, MCB, prepublicación revisada en octubre de 2026 | Comparan persistir, usar en el contexto actual, verificar o preguntar. Contrastan etiquetas y llamadas a herramientas y aportan anotación independiente de personas. | Conviene exigir una cita y leer el registro que efectivamente se reutilizó. La revisión de personas añade evidencia distinta del acuerdo entre agentes de IA. [Artículo](https://arxiv.org/abs/2608.19564) |
| Zhang et al., VagueBench y DialogGRPO, EMNLP 2026 | Entrenan la obtención de información faltante con recompensas por turno y valor futuro dentro de un diálogo. | Un resultado nulo de GRPO en Mesa y Memoria se limita al modelo, la tarea y las condiciones medidas. La recompensa y el objetivo de DialogGRPO son diferentes. [Artículo](https://aclanthology.org/2026.emnlp-main.1370/) |
| Zhang et al., *Ask Less, Ask Better*, prepublicación, octubre de 2026 | Separan la cantidad de interrupciones del esfuerzo de responder y comparan preguntas con juicios humanos. | Diez preguntas por veinte historias igualan el gasto experimental. Para hablar de experiencia de uso hay que medir qué preguntas resultan molestas, redundantes o difíciles. [Artículo](https://arxiv.org/abs/2610.24411) |
| Lu et al., HabitusMem, prepublicación del 19 de noviembre de 2026 | Entrenan un escritor de creencias que conserva evidencia, alcance y estado de cada preferencia. Lo evalúan en PersonaMem-v2 y VitaBench 2.0. | La vigencia y las citas ya se investigan. En nuestra tarea el escritor compartido permite aislar la decisión del modelo de adquirir evidencia. [Artículo](https://arxiv.org/abs/2611.25855) |
| Mehri et al., MultiSessionCollab, 2026 | Miden cómo el aprendizaje de preferencias influye en la colaboración entre sesiones e incluyen un estudio con personas. | Permite distinguir el acuerdo sobre las etiquetas de una historia de los beneficios que experimenta quien usa el asistente. [Artículo](https://arxiv.org/abs/2601.02702) |

### Qué añade evaluar peticiones con referencias de Ecuador

Placencia registró 171 interacciones en cinco tiendas de Quito y Manta y observó diferencias en cómo se atenúan las peticiones. Para la tesis importa una consecuencia práctica. Mencionar un plato local sitúa la conversación, pero la frecuencia de un giro y la forma de pedirlo dependen de quién habla, dónde y con quién. [Estudio de campo en Quito y Manta](https://benjamins.com/catalog/pbns.178.14pla)

Mayor-Rocher y colaboradores compararon nueve modelos en una prueba de opción múltiple sobre léxico y morfosintaxis de siete variedades del español. En esa prueba todos reconocieron mejor la peninsular. El banco agrupa Ecuador, Perú y Bolivia en una macrovariedad andina. Nuestro experimento observa otra conducta medible en español: ante dos pedidos idénticos, ¿cambia el asistente de visita cuando cambia qué preferencia podrá reutilizar? La redacción y revisión por IA permiten estudiar esa conducta en estos 30 pares. La naturalidad para hablantes de distintas regiones requiere evidencia de campo. [Artículo de evaluación](https://arxiv.org/abs/2504.20049) · [Descripción del banco](https://pmc.ncbi.nlm.nih.gov/articles/PMC12504998/)

### Evidencia que se puede defender

En v6, a diez preguntas por veinte historias, Qwen SFT alcanzó 62,21 % de acierto futuro y GRPO 62,04 %. El intervalo del contraste GRPO menos SFT fue de −0,67 a +0,25 puntos porcentuales. Al decidir libremente, el GRPO seleccionado resolvió cero pares completos en las tres semillas. Una regla que interpreta el calendario público resolvió los 80. El sistema concede también aciertos por abstenciones correctas, de ahí la importancia de presentar las dos mediciones juntas. [Informe v6](reports/tesis_timing_qwen_a100_v6/INFORME.md)

En v7 comparamos dos entrenamientos supervisados con los mismos ejemplos, tokens y actualizaciones. Añadir una penalización cuando el modelo ordena mal las dos agendas llevó los pares completos de 8,33 % a 12,92 % en promedio. La diferencia a cupo fijo fue de +0,25 puntos de acierto futuro. Una semilla empeoró y se repitieron ocho plantillas en el reservado. El resultado justifica investigar la sensibilidad a la agenda en este modelo concreto. La evidencia de generalización al lenguaje de los usuarios requiere otro banco. [Informe y resultados por semilla](reports/tesis_timing_pares_a100_v7/INFORME.md)

En el nuevo reservado sintético con referencias ecuatorianas, los mismos seis adaptadores pasaron del 10,00 % al 27,78 % de pares completos. El intervalo descriptivo del contraste fue de +8,89 a +28,89 puntos. A gasto fijo la ventaja fue de +0,56 puntos. En ambos reservados, mejorar la elección del momento tuvo una traducción pequeña al acierto futuro agregado. En Ecuador el planificador resolvió 28/30. [Materiales, ejecución y resultados](reports/tesis_ecuador_30_v1/INFORME.md)

### Qué cambia si la agenda suele fallar

Reejecutamos en CPU las ramas de SQLite de las 60 historias de validación v6, fijando una pregunta por historia. La regla se aprendió en entrenamiento. Después de fijar las decisiones variamos el peso de los mundos donde el plan anunciado se cumple. Las preferencias mantienen probabilidad 0,5 de ser habituales. Son expectativas del simulador: la columna de probabilidad es un supuesto que podemos cambiar, no una frecuencia observada en personas. [Reproducción en CPU](research/timing_agenda_sensitivity.py)

| Probabilidad de que se cumpla la agenda | Seguir la agenda | Preguntar siempre en la primera visita | Preguntar siempre en la segunda |
| ---: | ---: | ---: | ---: |
| 20 % | 65 % | 75 % | 75 % |
| 50 % | 75 % | 75 % | 75 % |
| 65 % | 80 % | 75 % | 75 % |
| 80 % | 85 % | 75 % | 75 % |
| 100 % | 91,67 % | 75 % | 75 % |

Con una agenda tan informativa como el azar desaparece la ventaja de seguirla. Si la agenda se cumple solo una vez de cada cinco, seguirla pierde diez puntos frente a preguntar en una visita fija. La simulación confirma así una condición que determina cuándo es útil la señal. Se reproduce desde la raíz con `python -m research.timing_agenda_sensitivity`. Estos porcentajes corresponden a una pregunta disponible por persona. La comparación entre modelos usa media pregunta por persona en cada paquete. [Dos regímenes de evaluación](docs/momento_aclaracion_v6.md)

### Qué puede hacer el autor con el entorno actual

1. **Recoger un reservado escrito por hablantes ecuatorianos.** El piloto sintético de 30 pares ya se redactó, revisó y evaluó. Una continuación viable es pedir a personas de dos localidades que expresen una misma agenda con sus palabras, y a otras personas que indiquen por separado en qué visita preguntarían y por qué. Registrar procedencia y contexto, y conservar casos ambiguos. Congelar este nuevo banco antes de abrir predicciones. El chat y los seis adaptadores permiten probar esas reformulaciones sin reentrenar. [Prueba de transferencia sintética](reports/tesis_ecuador_30_v1/INFORME.md)
2. **Separar comprensión y decisión.** Con los mismos casos y pesos, comparar la agenda original con una entrada estructurada que declare cuántas veces se espera usar cada comida, y con una versión donde se oculta esa pista. La entrada estructurada es un control diagnóstico. Si el modelo mejora solo allí, investigar cómo interpreta el español. Si también falla allí, centrar el trabajo en la decisión de esperar. Registrar la diferencia por par y por familia de redacción.
3. **Medir la utilidad con un presupuesto y con personas.** Primero completar la doble revisión ciega del banco de desarrollo. Después, en un estudio distinto, pedir a participantes que usen ambos momentos de aclaración en orden contrabalanceado y valoren la interrupción y la utilidad de la respuesta posterior. Documentar retiradas y errores de memoria. Los ZIP actuales sirven para la primera tarea; la segunda requiere un protocolo de interacción y el consentimiento que corresponda. [Materiales y procedimiento](docs/revision_humana_tesis.md)

Si solo hay tiempo para una ampliación, priorizaría el primer punto. El modelo, los adaptadores, la API y el resultado sintético ya existen. Lo que falta para sostener una afirmación más amplia es saber si el contraste sobrevive a redacciones que procedan de otras personas.

## Revisión de octubre de 2026

Búsqueda focalizada en fuentes primarias, 2026-10-10. No es una revisión sistemática
ni prueba exhaustiva de novedad. Los resultados de estos trabajos no son métricas
del presente proyecto.

| Trabajo | Qué ya estudia | Implicación para esta tesis |
|---|---|---|
| [CLAM, 2022/2023](https://arxiv.org/abs/2212.07769) | Clarificación selectiva por prompting y simulación con información privilegiada. | Pedir aclaraciones con un LLM no basta como contribución; incluir baseline fuerte y auditar simulación. |
| [Clarify When Necessary, 2023](https://arxiv.org/abs/2311.09469) | Cuándo preguntar, qué preguntar y cómo responder; utilidad de consultar según incertidumbre de intención. | La decisión ASK debe relacionarse con utilidad, no solo detección de conflicto. |
| [Modeling Future Conversation Turns, ICLR 2025](https://arxiv.org/abs/2410.13788) | Entrenamiento de clarificación con preferencias derivadas de turnos futuros simulados. | Optimizar aclaraciones por consecuencias posteriores tampoco es novedad por sí solo. |
| [LongMemEval, 2024/2025](https://arxiv.org/abs/2410.10813) | Extracción, razonamiento multisesión/temporal, actualización y abstención. | Separar fallos por componente y evaluar evidencia presente vs recuperada. |
| [LoCoMo, ACL 2024](https://arxiv.org/abs/2402.17753) | Conversaciones largas generadas y verificadas/editadas por humanos. | El realismo de una trayectoria no se obtiene solo aumentando filas sintéticas. |
| [Memory-R1, ACL 2026](https://aclanthology.org/2026.acl-long.583/) | Gestión/uso de memoria con agentes y RL, incluyendo PPO/GRPO. | Aplicar GRPO a memoria de agentes ya tiene antecedentes directos; debe diferenciarse la política de interacción con el usuario. |
| [MemConflict, preprint mayo 2026](https://arxiv.org/abs/2605.20926) | Conflictos dinámicos, estáticos y condicionales; diagnósticos de respuesta y recuperación/ranking. | Es el antecedente más próximo para validez de memoria; comparar taxonomía y protocolo, no presentar esos conceptos como nuevos. |
| [Knowing but Not Showing, preprint mayo 2026](https://arxiv.org/abs/2605.25284) | Diferencia entre reconocer ambigüedad y actuar preguntando; efecto del contexto recuperado. | Distinguir detección declarada de conducta real ASK. |
| [TANGLE, preprint agosto 2026](https://arxiv.org/abs/2608.13921) | Evalúa memoria personal bajo conflictos sin respuesta única y propone CAAP, una política de decisión y realización. | Antecedente directo del foco actual; CAAP debe considerarse como baseline fuerte. El estudio ya cubre la brecha entre reconocer conflicto y aclarar. |
| [CIGAsk, preprint septiembre 2026](https://arxiv.org/abs/2609.24290) | Entrena clarificación mediante GRPO, ganancia contrafactual bajo referencia congelada y señal de ambigüedad. | Aprender cuándo/cómo preguntar con GRPO ya tiene antecedente directo. Adaptarlo a memoria longitudinal exige evaluar un aporte adicional. |

MemConflict evalúa validez condicionada por consulta y separa resultados finales
de recuperación/ranking. Su protocolo descrito se centra en selección de memoria
y respuesta. La posible diferenciación de esta tesis es medir una intervención
de aclaración con coste, respuesta del usuario limitada a lo preguntado y efecto
persistente en una sesión nueva. Esto es una interpretación de alcance, no una
afirmación de exclusividad ni de que ningún otro trabajo cubra esa intersección.
[Texto completo consultado](https://arxiv.org/html/2605.20926v1).

Ampliación tras revisar los trabajos indicados por el usuario: TANGLE es más
directamente próximo al foco de acción frente a conflictos personales. Su CAAP
selecciona entre commit, conditionalize, clarify, verify, defer y reversible_trial;
no se debe atribuirle exactamente el espacio reducido ANSWER/CLARIFY/CONDITIONALIZE.
Esta reducción sería una decisión del presente proyecto.
[Sección 4.5 consultada](https://arxiv.org/html/2608.13921v1).

CIGAsk usa CIG para valorar el efecto de la respuesta de aclaración sobre la
probabilidad de la respuesta gold, además de otras señales. No basta trasladar
su nombre a un reward léxico. Para respuestas alimentarias libres, Exact Match
de toda la redacción no representa por sí solo cumplimiento semántico: una
posible adaptación es verificar campos estructurados y revisar utilidad aparte.
[Método consultado, secciones 3–4](https://arxiv.org/html/2609.24290v1).

La posible contribución se acota a la interacción con memoria longitudinal:
información obtenida mediante una pregunta, uso en la decisión actual, conservación
de lo realmente comunicado y beneficio posterior bajo coste de interacción.
Esto es una oportunidad por evaluar, no una garantía de novedad o de ventaja.

Una formulación defendible, pendiente de validar los datos y resultados, es:
**«Clarificación selectiva ante conflictos de memoria longitudinal: validez de
la evidencia, coste de interacción y consistencia multisesión»**.

El aporte podría ser un protocolo/dataset validado y una caracterización empírica
que muestre cuándo basta prompting y cuándo no. SFT y GRPO son posibles tratamientos
experimentales posteriores, no el resultado que el diseño debe garantizar.

La hipótesis «SFT+GRPO será mejor porque hay dos notas opuestas» es demasiado débil.
La hipótesis evaluable es «con evidencia igualmente disponible y presupuesto
comparable, una política entrenada mejora utilidad actual/futura y reduce
interrupciones innecesarias sobre prompting fuerte en historias retenidas y validadas».
También es científicamente válido encontrar que esa mejora no existe.
