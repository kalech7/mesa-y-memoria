# Ajuste textual de Qwen en la A100

El experimento compara Qwen2.5-0.5B-Instruct congelado con el mismo modelo tras dos épocas de SFT con LoRA. Cada versión recibe el mismo diálogo en español, recuerdos visibles, presupuesto y menú de acciones legales. El ajuste imita al planificador aprendido en TRAIN. Qwen elige una acción y el escritor y lector de reglas la ejecutan. El reservado quedó fuera del ajuste y la evaluación.

| Política | Acierto actual sustentado | Acierto futuro sustentado | Preguntas por historia | Utilidad |
| --- | ---: | ---: | ---: | ---: |
| Qwen congelado | 100 % | 50,00 % | 0,00 | 1,7500 |
| Qwen con LoRA | 100 % | 69,17 % | 1,00 | 1,9575 |

El ajuste aumentó el acierto futuro en **19,17 puntos porcentuales** y la utilidad en **0,2075** frente al modelo congelado. El modelo ajustado usó una aclaración en la primera visita de cada historia. Igualó el resultado de la regla pública y quedó por debajo del planificador, que obtuvo **73,33 %** con **0,75** preguntas por historia. El resultado corresponde a una semilla y 16 historias sintéticas de validación. [Métricas y trayectorias agregadas por raíz](RESULTS.json) · [Comparación CPU](../tesis_shared_budget_pilot_v1/CPU.json)

El entrenamiento usó una A100 80 GB, 1.088 decisiones públicas únicas, **272 actualizaciones** y **1.544.808 tokens de entrada de entrenamiento** antes de padding, sumados entre las dos épocas. Cada época conservó el adaptador, el estado del optimizador y su registro de recursos. El trabajo terminó con código 0 dentro de una sesión `tmux` separada del cliente. [Plan congelado](PLAN.json) · [Dependencias](DEPENDENCIAS.txt) · [Registro](EJECUCION.log) · [Adaptador de la segunda época](checkpoint-002/adapter/)
