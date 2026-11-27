# Exploración y recompensa del ajuste RL

La implementación v6 utiliza Qwen2.5-1.5B-Instruct con LoRA y un escritor compartido de memoria en SQLite. En las dos rondas de entrenamiento fuerza ASK y WAIT en cada estado de entrenamiento y usa ocho muestras de la continuación de Qwen por acción y mundo. La temperatura es 4. El objetivo de optimización es una adaptación de GRPO a dos acciones enumeradas, con clipping 0,2 y KL 0,02. La referencia es el adaptador común.

La recompensa longitudinal es el acierto actual más 1,5 veces el acierto futuro, menos 0,08 por pregunta y 0,35 veces la reutilización sin respaldo. El acierto futuro incluye las abstenciones correctas. Esta fórmula se calcula sobre las operaciones y lecturas de SQLite. El resultado v6 no mostró una ventaja de GRPO sobre SFT. Los resultados son verificables y describen este modelo y esta tarea. [Informe v6](../reports/tesis_timing_qwen_a100_v6/INFORME.md)

## Diagnóstico en entrenamiento

Hay cobertura de ambas acciones iniciales. La dificultad aparece en qué diferencias de recompensa aporta la continuación. En la primera ronda, la visita uno favorecía esperar en solo 8, 9 y 14 de los 30 pares que debían reservar la pregunta, por semilla. Con una continuación que apenas pregunta en la segunda visita, gastar la pregunta ahora puede tener mayor retorno bajo la política actual.

Al mezclar presupuestos uno y dos y ambas visitas, el 87,5 % del peso de los registros favorecía ASK. En la segunda ronda los estados favorables a esperar aportaban aproximadamente 2,92–3,36 % de la suma de diferencias absolutas de retorno ponderadas. Son proporciones de la señal de retorno. El gradiente también depende de las probabilidades, el clipping y la KL. La selección histórica se hacía por acierto futuro a cuota fija, que puede mejorar sin resolver las dos agendas de un par.

## Comparación de desarrollo v8

Se conserva el histórico y se escribe un experimento distinto. Las tres variantes parten del mismo adaptador v7 SFT por pares seleccionado en validación para cada semilla. Esto mide una continuación RL después de un ajuste supervisado común. Las tres usan exclusivamente los mismos 30 pares de entrenamiento, con presupuesto uno. Cada lote contiene dos pares completos y las cuatro observaciones públicas de cada par. El número de actualizaciones y los tokens se comparan entre variantes.

- SFT por pares de control continúa el objetivo supervisado anterior.
- GRPO exacto conserva el retorno de la política actual, pero integra exactamente las dos continuaciones posibles de la segunda visita.
- GRPO Bellman usa como control el mejor retorno esperado de la segunda visita. Primero promedia los mundos compatibles con la información pública y luego elige la acción. Este respaldo es una variante basada en el modelo del entorno.

Las dos variantes RL usan la diferencia de retorno centrada en la política anterior y dividida por la desviación entre acciones, con mínimo 0,05. Se evita que la magnitud de pocos estados domine el lote. Se mantienen temperatura 4 y clipping 0,2. Se prueban KL 0,005, entropía 0,01 y tasa de aprendizaje 0,00003 como configuración fijada. Se registran recompensa sin normalizar, entropía, KL, fracción recortada y norma de gradiente.

Se consideran checkpoints de las épocas 2, 4, 6 y 8. La selección prioriza pares completos en validación, después acierto futuro a cuota fija y finalmente menor época. El cambio conjunto de presupuesto, lotes, normalización y regularización compara configuraciones completas. La comparación de las dos variantes RL aísla el tipo de continuación bajo esa configuración.

Esta fase usa entrenamiento y validación. Un reservado nuevo evalúa después los checkpoints fijados, junto con el control supervisado y el planificador público. La mejora de RL es una hipótesis. Se informan los resultados que produzca el experimento.

## Reproducir el desarrollo

En el proyecto de la A100 con la base y los adaptadores ya disponibles:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.local/share/tesis-clarificacion/a100/bin/python -u \
  -m research.tesis_rl_v8
```

La implementación guarda su configuración y firmas de entrada en `PLAN.json`, los checkpoints y sus diagnósticos por semilla, `VALIDATION.json` y la selección final en `LOCK.json`. Si el proceso se interrumpe, las variantes ya terminadas se comprueban y las incompletas vuelven a entrenarse desde el inicio común.

GRPO procede de DeepSeekMath y usa un objetivo derivado de PPO. Los cambios de esta fase son adaptaciones explícitas a acciones enumerables y retornos del simulador. [DeepSeekMath](https://arxiv.org/abs/2402.03300) · [PPO](https://arxiv.org/abs/1707.06347)

## Ablación de desarrollo v8b

Tras observar la validación de v8, se añadió una variante híbrida de RL y comparación entre agendas. La pérdida adicional vale `softplus(signo de la diferencia de retornos × (margen de la segunda agenda − margen de la primera))`, con peso uno. El signo procede de las dos ramas de entrenamiento. Se compara con el mismo control supervisado, inicialización, datos y presupuesto. Esta prueba se registra en una corrida distinta y su selección sigue usando únicamente validación. Su nombre es `grpo-exacto-pares` para identificar la pérdida adicional. Las dos configuraciones se fijan antes de evaluar el reservado nuevo.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.local/share/tesis-clarificacion/a100/bin/python -u \
  -m research.tesis_rl_v8b
```
