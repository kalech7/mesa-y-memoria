# Piloto GRPO del selector de aclaraciones

El piloto continúa el adaptador LoRA de Qwen2.5-0.5B-Instruct que ya pasó por SFT. GRPO actualiza los parámetros LoRA usados en la llamada SELECT. RESPOND y WRITE usan una segunda instancia del SFT congelado durante todo el ensayo. Ese control permite atribuir los cambios de conversación al selector y medir sus consecuencias sobre el recuerdo persistido. La comparación de los tres componentes entrenados juntos está en el [informe de GRPO por fase](../reports/reward_review_targeted_20261105_002/INFORME.md).

## Qué ejecuta cada grupo

1. Toma un episodio TRAIN. La observación del selector incluye el diálogo actual, la consulta y la memoria visible. El simulador conserva el alcance real y la consulta futura.
2. Muestrea cuatro decisiones de la distribución de Qwen sobre cinco acciones legales. Cada acción puede ocurrir varias veces. Cada rama distinta ejecuta una vez RESPOND y WRITE, guarda la memoria en su propia base SQLite y la reabre para la consulta futura. Los sorteos repetidos conservan su multiplicidad para calcular la pérdida.
3. El verificador local puntúa el acierto sustentado actual y futuro, la validez de la escritura y sus consecuencias. Para medir la contribución de WRITE vuelve a calcular el resultado futuro con la memoria inicial y la misma conversación. Una recompensa penaliza la pregunta, el formato inválido y la escritura sin sustento.
4. GRPO centra y normaliza los cuatro retornos. Ejecuta dos pasos de AdamW con la misma distribución anterior, clipping de 0,20 y KL categórica hacia el selector SFT. El diario registra por separado la variación por resultados de tarea y la variación total que incluye las penalizaciones.

La cohorte del piloto tiene dos raíces ambiguas vistas en sus dos mundos, permanente y de sesión. También incorpora cuatro controles de duración o contexto explícitos, corrección y preferencia estable. La selección es reproducible y usa los estratos del generador. La política recibe solo la información visible en cada turno.

La recompensa de entrenamiento es:

```text
tarea = 2 × éxito conjunto estricto
      + 0,5 × acierto actual sustentado
      + 0,5 × acierto futuro sustentado
      + 0,5 × efecto causal de la escritura

retorno = tarea
        − 0,10 × ASK
        − 0,50 × episodio con escritura sin sustento
        − 0,20 × episodio con formato inválido
```

## Ejecutar en local

Desde la raíz, usa un Python con PyTorch, Transformers y PEFT. En la instalación usada se verificaron Python 3.12.16, Torch 2.7.1 con CUDA 12.8, Transformers 4.49 y PEFT 0.14. El cargador usa la copia de Qwen en `models/Qwen2.5-0.5B-Instruct` y verifica el adaptador y la base con el manifiesto del padre SFT. Pasa `--model-path` si tienes esos mismos pesos en otra ruta. El verificador permite que una copia omita los archivos vacíos de bloqueo de descarga que el manifiesto histórico incluía. La [guía del entorno de investigación](../research/README.md) describe las dependencias.

```bash
python -u -m research.grpo_select_pilot \
  --bank research/data/grpo_select_20261114_001 \
  --output research/runs/grpo_select_reproduccion \
  --rounds 2 --budget-seconds 900 --device cuda
```

Usa un directorio de salida nuevo. La opción `--device cpu` ejecuta el mismo protocolo con el coste de inferencia de la CPU. El piloto comprueba el presupuesto entre grupos y mide aparte la validación y el guardado de checkpoints. El verificador y los pesos se cargan desde el equipo local.

El banco contiene 20 raíces TRAIN, 10 VALIDATION y 10 EVALUATION. Los dos mundos de una raíz ambigua comparten partición. El generador comprobó aislamiento semántico, consistencia de las transiciones y que modificar datos privados no cambia los prefijos públicos. La evidencia es sintética con una semilla y una plantilla de generación compartida. La evaluación reservada de este piloto se realiza después de seleccionar en validación y es exploratoria.

Antes de cargar pesos, el piloto tokeniza las cinco ramas posibles de todos los episodios. Excluye una raíz TRAIN completa si alguna rama pasa de 2048 tokens. Si ocurre en VALIDATION o EVALUATION, detiene el piloto y conserva el diagnóstico. El [intento 001](../research/runs/grpo_select_20261114_001/STOPPED.json) se detuvo tras cuatro grupos por un prefijo WRITE fuera del límite. En [002](../research/runs/grpo_select_20261114_002/TOKEN_PREFLIGHT.json) la comprobación encontró esa raíz TRAIN en ambos mundos, con un máximo de 2089 tokens, y dejó intactos los dos conjuntos de evaluación, con máximos de 2011 y 1829.

## Inspeccionar la corrida

- `PILOT_PLAN.json` congela datos, checkpoint, fuentes, cohorte, recompensa y criterio de selección.
- `TOKEN_PREFLIGHT.json` conserva los recuentos por fase y acción de todos los episodios, incluida la raíz excluida en TRAIN.
- `grpo_groups.jsonl` incluye las acciones sorteadas, retorno de tarea, retorno penalizado, ventaja, gradiente y enlace a cada traza inmutable de entrenamiento.
- `train_branches/` conserva las ramas efectivamente ejecutadas, el diálogo, las operaciones, el lector futuro y SQLite.
- `validation/` contiene las trazas del SFT y de los pasos intermedio y terminal. Las etiquetas de checkpoints cuentan grupos. Los grupos 4 y 8 corresponden a 8 y 16 pasos de AdamW. El paso cero puede ganar la selección.
- `SELECTION.json` liga la selección a los hashes de pesos y del banco de evaluación. El terminal se evalúa como diagnóstico predeclarado.
- `RESULT.json` agrega por realización, mundo, raíz y familia. Incluye acierto actual y futuro, éxito conjunto, preguntas informativas y no informativas, escrituras sin sustento, formato inválido y raíces que mejoran o empeoran.

## Resultado ejecutado

La corrida 002 terminó en la RTX 3050 Ti con 8 grupos, 32 decisiones muestreadas, 20 ramas únicas y 16 pasos de AdamW. Cambió el adaptador y se verificó que la base y el SFT de RESPOND/WRITE conservaran sus hashes. Seis grupos tuvieron diferencias en la recompensa de tarea, uno solo en las penalizaciones y uno obtuvo retornos idénticos. Nueve sorteos hicieron una pregunta informativa. [Grupos y actualizaciones](../research/runs/grpo_select_20261114_002/grpo_groups.jsonl).

En validación, los grupos 4 y 8 empataron al SFT en éxito conjunto estricto (60 %) y acierto futuro (75 %). Las salidas inválidas subieron de 30 % a 40 % y las preguntas no informativas de 50 % a 80 %. El criterio fijado previamente dejó el checkpoint SFT como modelo seleccionado, en el grupo 0. [Selección previa al reservado](../research/runs/grpo_select_20261114_002/SELECTION.json).

| Medida en la evaluación reservada | SFT y selección del grupo 0 | GRPO grupo 8, diagnóstico |
| --- | ---: | ---: |
| Acierto actual sustentado | 80 % | 80 % |
| Acierto futuro sustentado | 60 % | 60 % |
| Éxito conjunto estricto | 40 % | 35 % |
| Episodios con pregunta | 50 % | 100 % |
| Pregunta informativa | 10 % | 20 % |
| Pregunta no informativa | 40 % | 80 % |
| Escritura sin sustento | 40 % | 50 % |
| Salida inválida | 0 % | 5 % |
| Retorno longitudinal medio | 0,65 | 0,39 |

En 10 raíces y 12 episodios, el terminal mejoró el éxito estricto en una raíz, empeoró en otra y empató en ocho. La recompensa entregó señal útil en TRAIN, incluida la adquisición correcta de una preferencia permanente tras una aclaración. En los casos reservados el terminal preguntó en todos los episodios y el acierto futuro agregado se mantuvo en 60 %. [Resultados y pares por raíz](../research/runs/grpo_select_20261114_002/RESULT.json).
