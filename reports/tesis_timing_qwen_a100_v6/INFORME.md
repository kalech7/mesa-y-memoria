# Preguntar hoy o reservar la aclaración para mañana

Se compararon cuatro ajustes LoRA de Qwen2.5-1.5B-Instruct en la NVIDIA A100 80GB PCIe de la USFQ. La prueba reservada contiene 80 pares dentro de 160 historias en 8 paquetes. Cada historia dura tres visitas. Los ocho mundos de cada historia representan combinaciones de alcance y de cumplimiento de la agenda. La misma población se evaluó para todos los métodos. [Plan fijado antes de la selección](PLAN.json) · [Selección en validación](LOCK.json) · [Resultados completos](RESERVED_RESULTS.json)

**Con el mismo gasto, GRPO alcanzó 62,04 % de acierto futuro y SFT 62,21 %.** La diferencia fue de −0,17 puntos porcentuales. Con libertad para elegir la visita, GRPO no resolvió ningún par completo en el reservado. En las semillas 101 y 303 preguntó en la primera visita de todas las historias. En la 202 esperó en ambas visitas de todas las historias. El planificador basado en la pista pública sí resolvió los 80 pares.

## Comparación a igual cantidad de preguntas

En cada paquete, ambos turnos de decisión comparten diez preguntas para veinte historias. Primero se comparan los márgenes que Qwen asigna a preguntar hoy en los diálogos ya recibidos. El cupo restante se reparte en la segunda visita entre quienes esperaron. El umbral se eligió en validación antes de abrir el reservado. Las preguntas que hubo que asignar al final incluso con un margen negativo se contabilizan por separado. [Asignación y medición](../../research/tesis_timing_v6_eval.py)

| Política | Acierto futuro total | Plato con evidencia¹ | Abstención correcta¹ | Preguntas en V1 por paquete | Preguntas en V2 por paquete |
|---|---|---|---|---|---|
| Qwen SFT | 62,21 % | 12,21 % | 50,00 % | 3,29 | 6,71 |
| Qwen GRPO con continuación actualizada | 62,04 % | 12,04 % | 50,00 % | 4,29 | 5,71 |
| Qwen GRPO con recompensa inmediata | 62,50 % | 12,50 % | 50,00 % | 3,46 | 6,54 |
| Qwen GRPO con continuación fija | 62,17 % | 12,17 % | 50,00 % | 1,17 | 8,83 |
| Planificador con la agenda pública | 67,50 % | 17,50 % | 50,00 % | 10,00 | 0,00 |
| Preguntar en cuanto haya cupo | 62,00 % | 12,00 % | 50,00 % | 10,00 | 0,00 |

¹ Porcentaje de todas las consultas futuras. Cada historia genera seis, cinco sobre la categoría que resulta más frecuente y una sobre la otra. Una abstención es correcta si la preferencia solo se autorizó para aquella sesión. Todos los métodos gastaron 0,50 preguntas por historia en promedio. Las dos primeras visitas ofrecían oportunidades de aclaración y cada persona podía recibir como máximo una pregunta en este análisis.

| Comparación de GRPO actualizado frente a | Diferencia de acierto futuro (pp) | IC 95 % (pp) | Cumple criterio |
|---|---|---|---|
| Qwen SFT | -0,17 | -0,67 a 0,25 | No |
| Qwen GRPO con recompensa inmediata | -0,46 | -0,92 a 0,00 | No |
| Qwen GRPO con continuación fija | -0,13 | -0,63 a 0,42 | No |
| Planificador con la agenda pública | -5,46 | -5,75 a -5,21 | No |
| Preguntar en cuanto haya cupo | 0,04 | -0,83 a 1,00 | No |

El contraste principal compara GRPO actualizado con SFT. Exige límite inferior positivo del intervalo emparejado del 95 %, costo idéntico, como máximo dos puntos de pérdida actual y ninguna subida en reutilización sin respaldo. Los 5.000 remuestreos toman paquetes completos para respetar la cuota compartida. El intervalo describe diferencias entre estos paquetes sintéticos bajo las tres semillas observadas. Las plantillas se reutilizan dentro de la partición.

## ¿Cambió Qwen de visita al cambiar la agenda?

En esta medición adicional, cada historia dispone de una pregunta propia. Para resolver un par completo, Qwen debe preguntar en la primera visita de una historia y, en su gemela, esperar y preguntar por la segunda preferencia en la visita siguiente. La columna de decisiones libres permite quedarse sin preguntar. En la última columna, si espera en la primera visita, el evaluador obliga a gastar la pregunta en la segunda.

| Política | Pares completos por semilla en decisiones libres (101 / 202 / 303) | Preguntas por historia en decisiones libres | Pares completos gastando una por historia |
|---|---|---|---|
| Qwen SFT | 0 / 0 / 1 | 1,00 | 0,42 % |
| Qwen GRPO con continuación actualizada | 0 / 0 / 0 | 0,67 | 0,00 % |
| Qwen GRPO con recompensa inmediata | 0 / 0 / 0 | 0,00 | 0,00 % |
| Qwen GRPO con continuación fija | 0 / 0 / 0 | 0,33 | 0,00 % |
| Planificador con la agenda pública | 80 / 80 / 80 | 1,00 | 100,00 % |
| Preguntar en cuanto haya cupo | 0 / 0 / 0 | 1,00 | 0,00 % |

El resultado de diez preguntas por paquete puede ser alto si el modelo sabe reconocer a diez personas para preguntarles hoy. Exigir las dos decisiones de cada par distingue esa estrategia de esperar y preguntar después en la misma historia.

## Qué ocurrió en la CPU y en las continuaciones de entrenamiento

En validación, con una pregunta por historia y la misma comida, los 30 pares cambiaron de decisión óptima al invertir la agenda anunciada. Seguirla dio 85 % de acierto futuro. Preguntar siempre hoy o siempre mañana dio 75 %. Elegir siempre la visita contraria a la agenda dio 65 %. Con dos preguntas el planificador pudo aclarar ambas preferencias y alcanzó 100 %. Estas puntuaciones miden cómo se comporta el simulador con una probabilidad de 0,80 de que el plan anunciado se cumpla. [Mediciones por historia y por presupuesto](CPU.json)

Se entrenó con 60 historias y se calibró con otras 60. En cada ronda se recorrieron 360 grupos de decisión. Se forzó tanto preguntar como esperar. Qwen decidió la continuación mediante ocho muestras estratificadas por mundo y acción. En la segunda ronda se usó el checkpoint GRPO obtenido en la primera. El control de continuación fija conservó las decisiones del padre. SFT y el GRPO principal recibieron las mismas trayectorias y retornos. El control inmediato recibió las mismas trayectorias y optimizó la recompensa de la visita presente. Cada brazo y semilla recibió 180 actualizaciones, 1440 decisiones de entrenamiento contando pasadas y 342.864 tokens de entrada sin relleno. [Contabilidad, huellas y pesos](TRAIN_RESULT.json) · [Entrenador](../../research/tesis_timing_v6_train.py)

| Semilla | Decisiones libres ASK en V2, ronda 1 | Decisiones libres ASK en V2, ronda 2 | Historias en que cambió la acción de mayor retorno en V1 (presupuesto 1) |
|---|---|---|---|
| 101 | 3695 | 5609 | 22 / 60 |
| 202 | 3753 | 5473 | 21 / 60 |
| 303 | 3705 | 5445 | 16 / 60 |

Cada recuento de ASK procede de 11.520 elecciones posteriores permitidas de Qwen por ronda y semilla. Otras 3.840 transiciones solo admitían esperar porque el presupuesto ya estaba agotado. En conjunto, actualizar la continuación cambió la acción que más convenía en 59 de 180 valoraciones de la primera visita con presupuesto uno. Los árboles se precalcularon con SQLite y Qwen eligió sobre las ramas permitidas. Las mediciones corresponden a personas simuladas. Aunque el control fijo y el GRPO actualizado usan los mismos ejemplos en la primera ronda, solo las semillas 101 y 202 produjeron pesos idénticos bit a bit en esa ronda. En la semilla 303 los pesos mostraron una diferencia numérica pequeña.

## Sensibilidad a la redacción de la agenda

Una vez terminada la evaluación, se midió en qué dirección cambia la diferencia de puntuaciones entre preguntar y esperar al intercambiar las agendas de un par. Se usaron los mismos pesos y la precisión de inferencia del experimento. De las 240 comparaciones de GRPO (80 pares por tres semillas), la puntuación de preguntar hoy fue mayor en la agenda correcta en 52, menor en 56 e idéntica en 132. En ninguna de las tres semillas hubo un solo par en que el margen pasara del lado correcto del umbral de decisión en ambos miembros. El resultado sugiere comprobar en TRAIN si una pérdida que compare explícitamente los dos diálogos produce mejor separación. La sensibilidad es un diagnóstico posterior que tampoco ajusta el modelo ni el umbral. [Resultados por semilla y partición](PAIR_MARGIN_DIAGNOSTIC.json) · [Reproducción del diagnóstico](../../research/tesis_timing_v6_margins.py)

## Alcance y reproducción

El banco utiliza platos y redacciones distintos entre entrenamiento, validación y reservado, con ocho plantillas de agenda reservadas, dos oportunidades de aclaración y una tercera visita de recuperación. Las dos preferencias tienen igual probabilidad de ser habituales. La respuesta a la aclaración es inequívoca y el escritor compartido impide guardar una preferencia sin cita. En este diseño, servir el plato de hoy ya está resuelto por el texto. Por eso conviene interpretar el acierto actual y la reutilización sin respaldo junto con el porcentaje de platos efectivamente recuperados. [Guía de VS Code y reproducción en la A100](../../docs/momento_aclaracion_v6.md) · [Ejemplo de un par de entrenamiento con las decisiones reales de Qwen](DEMO_PAR.json)
