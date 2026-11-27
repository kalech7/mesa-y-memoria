# Qwen en la A100 · analizador y aclaración entre visitas

Qwen2.5-1.5B-Instruct se ajustó con LoRA en la A100 de la universidad. Cada brazo recibió los mismos 346 grupos de decisión y sus 728 retornos por acción, con un padre común dentro de cada semilla. «Qwen inicial» designa el adaptador SFT heredado de v2 sobre Qwen2.5-1.5B-Instruct. Se entrenó con 24 historias y 264 mundos, se eligieron checkpoints en 12 historias de validación y se abrió un banco independiente de 18 historias, 54 visitas y 46 mundos. Las personas son ficticias y dos revisores de IA anotaron las visitas por separado. [Selección congelada](LOCK.json) · [Acta de revisión](../tesis_profiles_review_v5/ADJUDICACION.md)

## Resultado principal con los pesos elegidos en validación

GRPO T4 acertó más consultas futuras con evidencia y redujo la reutilización sin respaldo frente a SFT. Para lograrlo gastó 0,39 preguntas adicionales por historia, frente al máximo preestablecido de 0,10. SFT también necesitó volver al padre en una semilla. Por ambos motivos, la comparación queda por debajo del criterio formal de mejora. En la semilla 202 ambos obtuvieron idéntica utilidad. La diferencia fue de +0,246 en la 101 y de +0,019 en la 303. El 94 % de la ganancia agregada de acierto futuro procede de la semilla 101, donde SFT volvió al padre. En los pesos de la última época, la diferencia de utilidad entre los métodos cambió de signo según la semilla. Con los pesos seleccionados, la diferencia de acierto futuro fue de +10,49 puntos porcentuales con un intervalo emparejado del 95 % de +8,02 a +12,35.

| Modelo | Utilidad | Acierto futuro sustentado | Acierto actual sustentado | Preguntas por historia | Reutilización futura sin respaldo |
|---|---|---|---|---|---|
| Qwen inicial | 1,342 | 51,85 % | 77,78 % | 0,00 | 48,15 % |
| SFT | 1,499 | 70,99 % | 74,07 % | 0,67 | 29,01 % |
| GRPO T1 | 1,342 | 51,85 % | 77,78 % | 0,00 | 48,15 % |
| GRPO T4 | 1,588 | 81,48 % | 72,22 % | 1,06 | 18,52 % |

| GRPO T4 menos referencia | Diferencia de utilidad | IC del 95 % por historia | Diferencia de acierto futuro | Elegibilidad de ambos en tres semillas | Cumple el criterio conjunto |
|---|---|---|---|---|---|
| SFT | 0,088 | 0,056 a 0,118 | 10,49 pp | No | No |
| GRPO T1 | 0,246 | 0,157 a 0,334 | 29,63 pp | No | No |

## Diagnóstico de los pesos entrenados en la última época

| Modelo | Utilidad | Acierto futuro sustentado | Acierto actual sustentado | Preguntas por historia | Reutilización futura sin respaldo |
|---|---|---|---|---|---|
| SFT | 1,571 | 77,16 % | 74,07 % | 0,93 | 22,84 % |
| GRPO T1 | 1,515 | 66,67 % | 77,78 % | 0,61 | 33,33 % |
| GRPO T4 | 1,581 | 80,86 % | 72,22 % | 1,02 | 19,14 % |

| GRPO T4 menos referencia | Diferencia de utilidad | IC del 95 % por historia | Diferencia de acierto futuro | Elegibilidad de ambos en tres semillas | Cumple el criterio conjunto |
|---|---|---|---|---|---|
| SFT | 0,010 | -0,027 a 0,038 | 3,70 pp | No | No |
| GRPO T1 | 0,066 | -0,057 a 0,172 | 14,20 pp | No | No |

Los intervalos emplean 2.000 remuestreos por historia raíz dentro de su familia. Los mundos de cada historia se ponderan juntos y se promedian las tres semillas observadas. El contraste principal es el de los pesos seleccionados. El panel de última época permite inspeccionar qué cambió durante el aprendizaje. [Resultados por historia y semilla](RESERVED_RESULTS.json) · [Reglas fijadas antes del reservado](PLAN.json)

## Selección y costo de entrenamiento

| Semilla | SFT | GRPO T1 | GRPO T4 |
|---|---|---|---|
| 101 | Padre de reserva sin elegibilidad | Padre de reserva sin elegibilidad | Época 1 elegible |
| 202 | Época 2 elegible | Padre de reserva sin elegibilidad | Época 1 elegible |
| 303 | Época 1 elegible | Padre de reserva sin elegibilidad | Época 1 elegible |

Los candidatos debían alcanzar 84,52 % de acierto actual y un máximo de 25,27 % de reutilización sin respaldo en validación. La política de respaldo sirve el padre cuando ningún candidato cumple. En cada brazo y semilla hubo 176 actualizaciones y 1.186.628 tokens de entrada sin relleno. Los tres brazos utilizaron el mismo orden dentro de cada semilla. Se reutilizaron los retornos registrados en el simulador con una continuación fija. El ajuste realizó cero interacciones nuevas en línea. [Ramas y contabilidad](BRANCHES.json) · [Validación](VALIDATION.json)

## Qué corrigió el analizador

| Contra las anotaciones independientes | V4 | V5 |
|---|---|---|
| Respuesta exacta | 41/54 | 43/54 |
| Escrituras nuevas exactas | 47/54 | 47/54 |
| Recuerdos recuperados exactos | 49/54 | 49/54 |
| Visitas con los tres campos correctos | 40/54 | 41/54 |
| Aclaraciones previstas disponibles en todos los mundos | 15/16 | 16/16 |
| Aperturas adicionales a la anotación | 3 | 4 |

Para medir respuesta, escritura y recuperación se forzó continuar sin preguntar en cada visita. La cobertura de preguntas se calculó por separado con presupuesto disponible. Las métricas de política permiten la interacción y por eso tienen un denominador e interpretación diferentes.

El nuevo analizador abrió la aclaración prevista en las 16 oportunidades, a costa de una apertura adicional respecto de v4. En siete visitas faltó al menos una escritura autorizada. En una de ellas también se escribió «una rectificación para mí» como si fuera un plato tras una orden que empezaba con «Anota una rectificación para mí». Las citas de escritura pasaron la comprobación literal en todas las visitas. Esto muestra por qué conviene verificar también el significado del valor extraído. Tras abrir el reservado se corrigió este caso en la interfaz y se comprobó en la A100 que una visita posterior recupera la menestra con su cita. La tabla conserva la medición original del experimento. [Prueba de la interfaz en la A100](../tesis_profiles_review_v5/INTERFAZ_A100.json)

## Qué explica la diferencia entre los optimizadores

Reproducir el padre y las 346 decisiones de v4 reveló 174 grupos con retornos empatados. En otros 26 grupos el padre daba menos del 1 % de probabilidad a las mejores acciones. Con temperatura 4, la masa media ponderada que esas acciones reciben dentro del grupo subió de 0,69 % a 22,30 % y la norma del gradiente inicial de los logits pasó de 0,00255 a 0,01548. La intervención altera la señal de aprendizaje antes de la primera actualización. Tanto SFT como GRPO recibieron los retornos de todas las ramas. En v5, para cada uno de los 346 grupos de TRAIN hubo al menos una acción de máxima utilidad con la menor reutilización sin respaldo bajo la continuación fijada. [Diagnóstico reproducido de v4](../tesis_profiles_review_v5/V4_GRADIENT.json) · [Diagnóstico v5](BRANCHES.json)

## Qué aporta planificar

| Modelo | Utilidad | Acierto futuro sustentado | Acierto actual sustentado | Preguntas por historia | Reutilización futura sin respaldo |
|---|---|---|---|---|---|
| RL-Bellman | 1,588 | 81,48 % | 72,22 % | 1,06 | 18,52 % |
| RL-inmediato | 1,518 | 75,93 % | 72,22 % | 0,89 | 24,07 % |
| SFT-ramas-fijas | 1,588 | 81,48 % | 72,22 % | 1,06 | 18,52 % |
| regla-publica | 1,588 | 81,48 % | 72,22 % | 1,06 | 18,52 % |

Bellman utiliza la recompensa de las visitas posteriores y el control inmediato suprime ese componente del retorno. El control supervisado tabular aprende de ramas con continuación fija. En estados públicos ausentes de TRAIN, las tablas recurren a la regla pública. La cobertura de esas tablas figura junto a las métricas. Estos controles permiten comparar cuánto vale considerar las siguientes visitas y cuánto aporta el ajuste de Qwen. [Tablas de desarrollo](TABULAR.json) · [Evaluación y cobertura](RESERVED_RESULTS.json)

En el reservado, el planificador y el control supervisado recurrieron a la regla en 83 de 93 decisiones. Ambos y el GRPO T4 seleccionado igualaron el desempeño de esa regla en las 18 historias. GRPO T4 hizo 0,67 preguntas en la primera visita y 0,17 en la tercera, frente a 0,41 y 0,11 de SFT. En las ramas comparables se podían retirar 0,33 preguntas de T4 por historia y 0,22 de SFT sin empeorar los resultados que audita el simulador. La reutilización de recuerdos vencidos fue cero en ambos. La mejora observada del acierto futuro está acompañada por más preguntas tempranas. La mayor diferencia se observó en las historias de recontacto, con 100,00 % de acierto futuro para T4 y 70,37 % para SFT. En cambio, en la historia de calendario nuevo-07, T4 gastó dos preguntas adicionales frente al padre sin mejorar el acierto futuro. Las historias de atribución y citas dieron igual acierto futuro a SFT, T4 y al padre.

## Aporte de la tesis y siguiente prueba

El entorno permite dar dos decisiones distintas al mismo asistente desde el mismo turno, seguir el presupuesto común a través de tres visitas y medir el efecto en consultas posteriores con una memoria SQLite verificable. En esta corrida, cambiar la señal de entrenamiento consiguió que GRPO aprobara la selección en las tres semillas. El resultado aporta evidencia sobre el costo de adquirir preferencias y las condiciones en que una regla pública sigue siendo competitiva.

Para medir mejor cuándo conviene **reservar** una pregunta, la siguiente revisión de TRAIN debería incluir más pares con la misma preferencia y redacciones diversas. Por ejemplo, si ambas personas dicen «hoy me provoca sopa», puede que una esté cerrando el menú de toda la semana y otra anuncie que mañana podrá contestar sobre una restricción que afecta a varias comidas. Antes de usar la A100, conviene medir en CPU el resultado de preguntar ahora y de esperar en esos dos contextos con igual presupuesto, y contar cuántos estados del banco nuevo reconoce cada tabla. La siguiente evaluación debe usar otras historias independientes y comparar curvas de acierto futuro a igual número de preguntas. Los casos v5 ya abiertos sirven como regresiones.

## Uso y reproducción

La [guía de la interfaz](../../docs/laboratorio_memoria.md) explica cómo conversar, cambiar de visita e inspeccionar la cita de SQLite. La [metodología y los trabajos relacionados](../../docs/analizador_grpo_v5.md) documentan el protocolo y el comando para reproducir el estudio. Los mensajes que se escriben en la interfaz se guardan aparte del banco científico.
