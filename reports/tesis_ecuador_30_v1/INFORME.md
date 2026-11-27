# Cuándo gastar la única pregunta en 30 pares con referencias de Ecuador

**El entrenamiento por pares conservó la ventaja en los textos nuevos.** Al invertir la agenda, SFT convencional acertó ambas visitas en el 10,00 % de los pares y SFT por pares en el 27,78 %, en promedio sobre tres semillas. La diferencia fue de **+17,78 puntos porcentuales**. El intervalo descriptivo del 95 % por remuestreo de los 30 pares fue de **+8,89 a +28,89 puntos**. En el reservado anterior la diferencia había sido de +4,58 puntos. Los dos bancos tienen composiciones distintas. [Resultado anterior](../tesis_timing_pares_a100_v7/INFORME.md) · [Resultados completos](RESULTADOS.json)

| Semilla | SFT, pares completos | SFT por pares, pares completos | Acierto futuro a cupo fijo, SFT | Acierto futuro a cupo fijo, SFT por pares |
| --- | ---: | ---: | ---: | ---: |
| 101 | 3/30 | 9/30 | 63,17 % | 63,17 % |
| 202 | 6/30 | 8/30 | 62,50 % | 63,50 % |
| 303 | 0/30 | 8/30 | 62,50 % | 63,17 % |
| Media | **10,00 %** | **27,78 %** | **62,72 %** | **63,28 %** |

En cada par, una persona hace los mismos dos pedidos. Cambia únicamente qué comida o bebida prevé volver a elegir. Gastar la única pregunta en la cena impide aclarar en la segunda visita si el desayuno o la bebida son habituales. El acierto por par exige elegir bien en **las dos** versiones. Los seis adaptadores gastaron exactamente una pregunta por historia en su decisión libre. Para el acierto futuro a cupo fijo, el asignador impuso diez preguntas por cada veinte historias con los umbrales de v7. En ese régimen la mejora media fue de **+0,56 puntos**. El acierto futuro incluye abstenerse correctamente si el pedido era puntual. [Protocolo](PROTOCOLO.md) · [Asignador y métricas](../../research/tesis_timing_v6_eval.py)

### Qué cambió en las decisiones

En EC22, la persona pide **guatita** en la primera visita y **muchín de yuca** en la segunda. En una agenda elegirá cena cinco días y desayuno uno. En la otra será al revés. SFT por pares cambió la visita correctamente en las tres semillas. SFT convencional mantuvo el mismo momento al invertir la agenda en cada semilla. En EC01, con **locro de zapallo** y **pan de yuca**, las seis políticas preguntaron en la primera visita bajo ambas agendas, incluso cuando la cena solo se elegiría el jueves y el desayuno de lunes a viernes. [Textos de los pares](PARES.json) · [Márgenes de decisión](MARGENES.json)

SFT por pares asignó mayor margen a preguntar en la primera visita cuando se repetiría la cena en 64 de 90 combinaciones de par y semilla. SFT convencional lo hizo en 46 de 90. De esas 90 combinaciones, solo 25 y 9 respectivamente produjeron las dos decisiones correctas. Esto distingue un cambio en la puntuación del modelo de un cambio en lo que finalmente hace. Los 90 resultados reutilizan los mismos 30 pares. Por eso el intervalo agrupa las tres semillas al remuestrear cada par. [Decisiones y diagnóstico de márgenes](RESULTADOS.json)

El planificador con reglas de lectura del calendario y valores aprendidos solo en el entrenamiento histórico resolvió **28 de 30 pares** y alcanzó **66,83 %** a cupo fijo. La redacción controla parte de la dificultad y deja margen para mejorar el modelo pequeño. Sumando las tres semillas dentro de cada par, el objetivo por pares tuvo saldo positivo en diez pares y empate en veinte. Dentro de semillas individuales también hubo retrocesos. [Resultados del planificador y por par](RESULTADOS.json)

### Qué evidencia aporta

Dos subagentes redactaron 15 pares cada uno. Un tercer subagente leyó cada agenda con el orden y los lados reasignados, sin ver la clave ni las salidas de los modelos. En la primera ronda reconoció 58 de 60 visitas. Se aclaró si el texto hablaba de elegir una comida, abastecerse o elegir para otras personas. En la última ronda reconoció 60 de 60. Se conservaron las tres rondas y los 30 pares. El mismo revisor pudo reconocer redacciones de rondas anteriores. Son textos y juicios sintéticos. [Conciliación](CONCILIACION_REVISION.json) · [Valoraciones finales](JUICIOS.json)

Los 30 pares dieron 60 historias y 480 mundos con probabilidades conocidas. Las 60 agendas eran textualmente distintas de las agendas anteriores tras normalizar espacios y mayúsculas. En el simulador la agenda se cumple con probabilidad 0,8. Preguntar en la visita indicada da un 85 % esperado de acierto futuro, preguntar en la contraria un 65 % y esperar ambas un 50 %. Estas cifras describen el mecanismo del simulador. El banco sigue usando su estructura de cenas, desayunos y bebidas. El intervalo se refiere a estos materiales y semillas. La naturalidad percibida en Ecuador, el coste de interrumpir a una persona y el efecto aislado del dialecto siguen abiertos. [Auditoría de SQLite](SELLO.json)

La literatura ya muestra que los asistentes pueden perder preferencias futuras por no preguntar y que la utilidad esperada de reutilizar una respuesta puede justificar la interrupción. La tesis aporta un banco pareado en español y una prueba reproducible del momento elegido: con el mismo Qwen pequeño, datos y presupuesto de entrenamiento, añadir un objetivo por pares mejoró esa conducta en el reservado v7 y en este segundo banco sintético. La mejora en las consultas futuras a cupo fijo fue pequeña en ambos. Una continuación directa es evaluar las reformulaciones independientes de hablantes ecuatorianos con los seis pesos ya congelados. [ATRBench](https://aclanthology.org/2026.emnlp-main.1505/) · [ActiveWebPref](https://arxiv.org/abs/2610.31107) · [Literatura y trabajo propuesto](../../REVISION_LITERATURA.md)

### Reproducir

El corpus, la clave, los juicios, la auditoría, los umbrales históricos y el programa quedaron fijados en GitHub antes de ejecutar los adaptadores. Se usó una NVIDIA A100 de 80 GB con CUDA 12.8, PyTorch 2.7.1, Transformers 4.49.0 y PEFT 0.14.0. La entrada pública más larga tenía 243 tokens. Una segunda ejecución en la A100 reprodujo exactamente los márgenes archivados de los seis adaptadores y del planificador. [Commit previo a la inferencia](https://github.com/kalech7/mesa-y-memoria/commit/d421e93578820addca0332bbfcce2e63d9e66109) · [Adaptadores seleccionados](../tesis_timing_pares_a100_v7/ADAPTADORES.json)

Para comprobar el sello y recalcular en CPU todas las métricas desde los márgenes y las ramas de SQLite, basta con ejecutar desde la raíz:

```bash
python -m research.tesis_ecuador_eval verify
```

Para volver a obtener los márgenes de los seis adaptadores en una A100 y compararlos con los publicados, instala el modelo base y las dependencias GPU siguiendo la guía del proyecto. [Preparación del modelo](../../docs/modelos-publicos.md)

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -m research.tesis_ecuador_eval replay
```
