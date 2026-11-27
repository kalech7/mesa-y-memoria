# Perfiles de usuario y aclaración entre visitas

La prueba reúne 24 historias de entrenamiento (264 mundos), 12 de validación (132 mundos) y 16 historias redactadas por separado para el reservado (22 mundos). Las historias son sintéticas. Dos revisores de IA corroboraron la respuesta, la escritura nueva y los recuerdos aplicables en las 48 visitas públicas reservadas. [Protocolo](../../docs/experimento_perfiles_usuarios.md) · [Adjudicación](../tesis_profiles_review/ADJUDICACION.md) · [Plan de la corrida](PLAN.json)

## Control tabular en validación

| Política | Utilidad | Acierto actual | Acierto futuro sustentado | Preguntas por historia | Preguntas en V3 |
|---|---:|---:|---:|---:|---:|
| Regla pública | 1,85289 | 86,52 % | 74,73 % | 1,667 | 0,000 |
| SFT con ramas y continuación fija | 1,99450 | 100,00 % | 74,81 % | 1,596 | 0,549 |
| RL con Bellman | 1,98994 | 100,00 % | 74,42 % | 1,580 | 0,549 |
| RL sin recompensa futura | 1,65115 | 100,00 % | 45,57 % | 0,404 | 0,404 |

Las tres tablas usaron las mismas 5.764 transiciones de TRAIN, de las cuales 2.064 cruzan visitas. Las tablas cubrieron todas sus decisiones no forzadas en validación. Para Bellman, retirar el término de recuperación futura redujo el acierto futuro en 28,86 puntos (IC 95 % de 24,28 a 33,80) y redujo el uso de preguntas en 1,176 por historia. El contraste describe el efecto de esa recompensa bajo este presupuesto. [Trazas, cobertura e intervalos](SEQUENTIAL_VALIDATION.json)

Bellman menos SFT con ramas dio −0,00456 de utilidad (IC 95 % de −0,00913 a 0). Las dos tablas longitudinales conservaron preguntas para la tercera visita y resolvieron mejor la consulta actual que la regla. Ambas entregaron cero recontactos en validación. Los perfiles de desarrollo combinan estilo de respuesta, disponibilidad, planes y ambigüedad posterior. Las diferencias entre familias describen los escenarios definidos y dejan abierto el efecto aislado de cada factor.

## Presupuesto de aprendizaje de Qwen

Se usó Qwen2.5-1.5B con LoRA a partir del mismo adaptador en ambos brazos. Los grupos se prepararon antes de optimizar. Los dos brazos reciben las mismas 346 decisiones de TRAIN, con retornos de cada acción bajo una continuación pública común. Ocho grupos de la primera visita incluyen un seguimiento tras una réplica parcial. El conteo de referencias de transición al computar los retornos es 8.532. Cada semilla comparte el orden de lotes, grupos, tokens y actualizaciones entre métodos. [Ramas compartidas](QWEN_BRANCHES.json)

El adaptador común obtuvo 86,52 % de acierto actual en validación y superó el mínimo de 84,52 %, con cero actualizaciones de preparación. Cada brazo y semilla ejecutó 176 actualizaciones sobre 1.316.636 tokens de entrada, con 728 retornos de acción registrados. Validación seleccionó SFT en épocas 1, 2 y 1 para las semillas 101, 202 y 303. Las seis candidaturas entrenadas de GRPO obtuvieron 95,71 % de acierto actual y reservaron en promedio 0,375 preguntas para la tercera visita. Su tasa de reutilización futura sin respaldo fue de 28,62 %, por encima del 25,27 % de la regla que se fijó como máximo. El procedimiento devolvió el adaptador inicial en las tres semillas y lo marcó como inelegible. [Validación y pesos elegidos](QWEN_VALIDATION.json) · [Control del adaptador inicial](shared-competence/PREFLIGHT.json) · [Huella de selección](SELECTION_LOCK.json)

## Reservado con selección congelada

| Política | Utilidad | Acierto actual | Acierto futuro sustentado | Preguntas por historia |
|---|---:|---:|---:|---:|
| Regla pública | 1,52083 | 73,33 % | 70,00 % | 0,583 |
| SFT tabular con continuación fija | 1,52083 | 73,33 % | 70,00 % | 0,583 |
| RL con Bellman | 1,52083 | 73,33 % | 70,00 % | 0,583 |
| RL sin recompensa futura | 1,48283 | 73,33 % | 66,67 % | 0,433 |
| Qwen inicial | 1,31750 | 73,33 % | 53,33 % | 0,000 |
| Qwen SFT por ramas | 1,51517 | 73,33 % | 69,44 % | 0,550 |
| Resultado del brazo Qwen GRPO tras selección | 1,31750 | 73,33 % | 53,33 % | 0,000 |

El Qwen seleccionado por SFT mejoró 16,11 puntos de acierto futuro frente al adaptador inicial (IC 95 % emparejado por historia de 11,67 a 20,28). Los dos tienen el mismo acierto actual. El brazo GRPO seleccionado corresponde al adaptador inicial. En tabular, Bellman y SFT recurrieron a la regla en 24 de las 33 decisiones evaluadas. Bellman ganó 3,33 puntos de acierto futuro frente al brazo sin recompensa futura (IC de 0,83 a 5,83) y usó 0,15 preguntas más, por encima del margen de 0,10 preguntas que se fijó en el plan. Las dos tablas longitudinales empataron con la regla. Los porcentajes son medias por familia y las unidades del intervalo son las historias, agrupando los mundos y semillas de cada una. [Resultados reservados](RESERVED_RESULTS.json)

## Diagnóstico de los pesos de la última época

Después de la evaluación principal se puntuó la época 2 de ambos brazos en las tres semillas para comparar pesos actualizados en condiciones simétricas. SFT obtuvo 69,72 % de acierto futuro, 0,567 preguntas por historia y 0,15 preguntas en la tercera visita. GRPO obtuvo 66,94 %, 0,500 y 0,20, respectivamente. Las dos políticas consiguieron 73,33 % de acierto actual. GRPO menos SFT dio −2,78 puntos de acierto futuro (IC 95 % de −5,56 a −0,54) y −0,03633 de utilidad (IC 95 % de −0,07267 a −0,00553). El diagnóstico es exploratorio. La selección principal conserva las decisiones fijadas antes de abrir el reservado. [Trazas del diagnóstico](LAST_EPOCH_DIAGNOSTIC.json)

## Alcance del menú de acciones

El analizador público habilitó la acción prevista en 10 de 12 oportunidades de aclaración del reservado al llegar con presupuesto. Una de las dos omisiones bloqueó una respuesta parcial útil. También se observaron dos aperturas adicionales sin una réplica prevista para esa pregunta. En una visita con lugar ambiguo, el sistema emite una respuesta condicional correcta según la adjudicación semántica. El indicador de acierto actual de la corrida exige una respuesta única sustentada. [Diagnóstico de apertura y sensibilidad](../tesis_profiles_review/DIAGNOSTICO_APERTURA.md)

## Reproducción y verificación

La corrida principal y su evaluación terminaron con código de salida 0 en una NVIDIA A100 de 80 GB del servidor `servidor-tesis`, desde una sesión `tmux` desacoplada. El servidor conserva el directorio `~/tesis_clarificacion_longitudinal/reports/tesis_profiles_qwen_a100_v4` con los pesos, optimizadores, trazas y selección. Pasaron 513 pruebas locales y se omitieron tres. [Comandos de reproducción](../../docs/experimento_perfiles_usuarios.md) · [Registro](EJECUCION.log)
