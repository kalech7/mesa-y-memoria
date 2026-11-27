# Qwen 1.5B con texto y comparación por ramas

La corrida en la A100 ajustó SFT y GRPO sobre el mismo Qwen2.5-1.5B-Instruct, adaptador inicial y banco de decisiones. Ambos métodos procesaron 400 grupos con 1.712 retornos de acción y completaron 400 actualizaciones con 2.338.648 tokens reales por método. La selección aplicó el mismo criterio de utilidad y las mismas guardas en 16 historias de validación. [Plan](PLAN.json) · [Comparación y contabilidad](RESULTS.json)

| Checkpoint publicado | Época | Acierto actual | Acierto futuro | Preguntas por historia | Utilidad |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen inicial | 0 | 17,42 % | 59,58 % | 1,00 | 0,9880 |
| SFT, retorno al inicial | 0 | 17,42 % | 59,58 % | 1,00 | 0,9880 |
| GRPO por ramas | 1 | 100 % | 69,17 % | 1,00 | 1,9575 |
| Regla pública | — | 100 % | 69,17 % | 1,00 | 1,9575 |
| Planificador | — | 100 % | 73,33 % | 0,75 | 2,0400 |

En SFT, la época 2 reservó la pregunta respondiendo provisionalmente en seis de ocho historias de poca disponibilidad. Llegó a 65,73 % de acierto futuro, con 34,27 % de recuperación sin respaldo frente a 30,83 % de la regla. Ninguna época SFT pasó las guardas. Por eso su resultado publicado corresponde al checkpoint inicial. El GRPO seleccionado preguntó al inicio de las dieciséis historias. La diferencia entre los métodos seleccionados incluye el retorno de SFT al inicial, por lo que conviene acompañarla con los resultados de sus checkpoints entrenados. [Épocas de SFT](SFT-ramas/checkpoint-004/LEDGER.json) · [Selección](LAB_MODELS.json)

El diagnóstico motivó una segunda comparación con señales públicas de la visita y de la ambigüedad en el prompt. Se mantuvo el protocolo de evaluación y el reservado quedó fuera de ambas comparaciones.
