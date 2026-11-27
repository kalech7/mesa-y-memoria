# Antecedentes y delimitación de la contribución

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
