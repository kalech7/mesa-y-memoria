# Aclaraciones en español y entrenamiento de GRPO

El estudio v5 prueba dos hipótesis. La primera es que el analizador público impide aclaraciones útiles cuando el usuario expresa la vigencia con otras palabras, menciona a otra persona o habla de la preparación de un plato. La segunda es que un modelo que empieza dando probabilidad muy baja a determinadas preguntas puede recibir poca señal para aprender a elegirlas, incluso si el simulador ya ha medido una buena consecuencia futura para esas preguntas.

## Diagnóstico de la corrida anterior

En v4 los seis checkpoints entrenados por GRPO alcanzaron 95,71 % de acierto actual y 71,38 % de acierto futuro en validación. Reservaron más preguntas para la última visita y registraron 28,62 % de reutilización sin respaldo. El máximo fijado con la regla pública era 25,27 %. Por eso las tres semillas volvieron a los pesos iniciales. La comparación posterior entre los pesos finales de SFT y GRPO también favoreció a SFT en 2,78 puntos de acierto futuro. La primera diferencia involucra la selección y la segunda compara directamente los dos aprendizajes. [Resultados v4](../reports/tesis_profiles_qwen_a100_v4/INFORME.md)

## Comparación emparejada

La política ve el diálogo original, los recuerdos, el presupuesto disponible y un resumen que el analizador deriva del texto público. El escritor recibe una representación normalizada y conserva el texto original en las citas. Se comparan las preguntas disponibles antes y después del cambio en las mismas visitas, con presupuesto disponible. Las historias reservadas que se abrieron en v4 sirven como regresión. La prueba nueva usa un corpus redactado por separado y sellado antes de ejecutar la comparación. [Analizador](../research/tesis_profiles_parser_v5.py) · [Manifiesto independiente](../research/data/tesis_profiles_heldout_v5/MANIFEST.json)

Se comparan SFT, GRPO con temperatura 1 y GRPO con temperatura 4 sobre Qwen2.5-1.5B con LoRA. La temperatura 4 suaviza durante el entrenamiento las probabilidades de las acciones, tanto para la actualización como para su distribución de referencia. En inferencia los tres eligen la acción legal con la mayor puntuación. Cada semilla usa el mismo punto de partida, acciones y retornos, orden de lotes, tokens y actualizaciones en los tres brazos. Se registran los empates, la probabilidad inicial de las acciones óptimas y la norma del gradiente de los logits para ver dónde cambia la señal. [Optimizador](../research/tesis_profiles_optimizer_v5.py)

El contraste principal compara la utilidad longitudinal de GRPO con temperatura 4 frente a SFT usando los pesos elegidos en validación. Una ventaja requiere checkpoints elegibles en las tres semillas, límite inferior positivo en el intervalo emparejado del 95 % por historia, pérdida de hasta dos puntos de acierto actual, como máximo 0,10 preguntas adicionales por historia y ninguna subida en reutilización sin respaldo. La comparación con el GRPO anterior permite aislar el efecto de la temperatura dentro del entorno v5. Se fija también la evaluación de la última época de todos los brazos. Así se puede observar lo que aprendieron los pesos aunque un brazo vuelva al padre durante la selección. [Corrida y evaluación](../research/tesis_profiles_v5.py)

## Aporte frente al trabajo publicado

ATRBench estudia cuándo obtener una preferencia que tendrá valor en una sesión posterior. AskBench evalúa cuándo aclarar información y usa aprendizaje por refuerzo con recompensas verificadas. En esta tesis la misma pregunta consume una oportunidad compartida con dos visitas posteriores. Algunas personas contestan parcialmente o retoman la conversación más tarde. Se puede medir el costo de gastar esa oportunidad, dejando idénticos el diálogo visible, el lector de memoria y las alternativas de entrenamiento para SFT y GRPO. [ATRBench, versión de septiembre de 2026](https://arxiv.org/abs/2605.28108) · [AskBench, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.845/)

PERMA estudia cambios de preferencias entre eventos y variación lingüística. MCB distingue decidir qué guardar de decidir qué hacer cuando ese recuerdo se podría reutilizar. Aquí se registran por separado la respuesta que admite el turno, la escritura que autoriza y la recuperación posterior. Esto permite rastrear hasta una frase concreta si el error nació al interpretar el español, al asignar vigencia, al recuperar el recuerdo o al decidir preguntar. Las historias independientes siguen siendo simuladas y fueron revisadas por IA. La generalización a personas reales requiere otra evaluación. [PERMA](https://arxiv.org/abs/2603.23231) · [Memory–Clarification Boundary](https://aclanthology.org/2026.nora-1.2/) · [Adjudicación de las historias en español](../reports/tesis_profiles_review_v5/ADJUDICACION.md)

## Reproducir en la A100

Desde VS Code conectado por SSH a `servidor-tesis`, abre `~/tesis_clarificacion_longitudinal`. El lanzador ejecuta las dos etapas en el servidor. Para repetir en un directorio nuevo desde una sesión desacoplada:

```bash
cd ~/tesis_clarificacion_longitudinal
tmux -L tesis-clarificacion new-session -d -s mi-estudio-v5 \
  'CUDA_VISIBLE_DEVICES=0 bash scripts/run_profiles_v5_a100.sh reports/mi_estudio_v5 > logs/mi_estudio_v5.log 2>&1'
```

Para consultar el progreso usa `tail -n 20 logs/mi_estudio_v5.log` o abre `STATUS.json` en el directorio elegido. Para recuperar un entrenamiento interrumpido, añade `--resume` después de ese directorio. [Lanzador](../scripts/run_profiles_v5_a100.sh)

## Resultado de la corrida universitaria

El ajuste y la apertura del reservado terminaron el 17 de noviembre de 2026 en la A100 de la universidad. El [informe de la comparación](../reports/tesis_profiles_qwen_a100_v5/INFORME.md) presenta los dos paneles, los intervalos y el costo de las preguntas. El [laboratorio de memoria](laboratorio_memoria.md) permite conversar con esos mismos checkpoints y comprobar la vigencia y las citas.
