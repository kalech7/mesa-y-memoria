# Qwen 1.5B con evidencia pública en la A100

Los dos brazos reciben el mismo resumen automático de evidencia visible. La nota extrae del texto indicios de disponibilidad, incertidumbre de planes y alcance aún pendiente. El simulador, el escritor, los retornos de las ramas y la selección se mantienen compartidos. Cada brazo evaluó cuatro épocas, desde el mismo adaptador inicial y con el mismo orden de entrenamiento. [Plan](PLAN.json) · [Código de la nota](../../research/tesis_shared_budget_qwen_context.py)

| Política publicada | Época | Acierto actual | Acierto futuro | Preguntas por historia | Utilidad |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen inicial | 0 | 14,06 % | 59,58 % | 1,00 | 0,9544 |
| SFT por ramas | 1 | 100 % | 73,33 % | 0,75 | 2,0400 |
| GRPO por ramas | 1 | 100 % | 69,17 % | 1,00 | 1,9575 |
| Regla pública | — | 100 % | 69,17 % | 1,00 | 1,9575 |
| Planificador público | — | 100 % | 73,33 % | 0,75 | 2,0400 |

El SFT seleccionado conservó la pregunta en las ocho historias con poca disponibilidad y preguntó en la primera visita de las ocho historias con disponibilidad. Aclaró el alcance en todas las segundas visitas en las que encontró ambigüedad y todavía tenía presupuesto. Los mundos ponderados con esa oportunidad representaron 0,25 por historia. El GRPO seleccionado gastó la pregunta en la primera visita de las 16 historias. La diferencia de SFT frente a GRPO fue de **4,17 puntos** de acierto futuro y **0,25 preguntas menos** por historia. [Resultados por familia y raíz](RESULTS.json)

Cada brazo completó **400 actualizaciones** y procesó **2.612.312 tokens** de entrada antes del padding. Compartieron **400 grupos públicos y 1.712 retornos de acción**, con las mismas huellas de datos y lotes. Los checkpoints publicados son de la primera época. Se conserva el entrenamiento hasta la cuarta época para inspeccionar el presupuesto completo. [Ramas](BRANCHES.json) · [Adaptadores publicados](LAB_MODELS.json)

La comparación emplea una semilla y 16 raíces sintéticas de validación. Los intervalos de anchura cero reflejan la construcción de los estratos. El costo de generalizar a historias y usuarios distintos queda pendiente de un reservado independiente. La prueba anterior con texto y la nueva variante del prompt describen resultados de desarrollo. [Primera comparación](../tesis_shared_budget_qwen_matched_a100_v1/INFORME.md) · [Justificación y evaluación pendiente](../../docs/justificacion_presupuesto_compartido.md)

La ejecución terminó con código 0 en una sesión `tmux` desacoplada en el servidor. El directorio conserva el optimizador y un adaptador por época. [Registro](EJECUCION.log) · [Dependencias](DEPENDENCIAS.txt)
