# Auditoría de RL y prueba de las correcciones en la A100

**La variante GRPO con respaldo Bellman mejoró ambas medidas frente al control supervisado en un reservado sintético nuevo.** Acertó las dos versiones de 19 de 90 evaluaciones de par y semilla, frente a 14 de 90 del control. El acierto futuro a idéntico gasto pasó de **63,28 % a 63,72 %**. Son 30 pares compartidos por tres semillas. Este resultado describe las configuraciones y materiales de esta prueba. [Resultados crudos](RESERVE_RESULTS.json)

| Método | Pares completos por semilla 101 / 202 / 303 | Media de pares completos | Acierto futuro con diez preguntas por veinte historias |
| --- | ---: | ---: | ---: |
| SFT por pares de v7, punto de partida | 4 / 8 / 1 | 14,44 % | 62,83 % |
| SFT por pares, continuación de control | 5 / 4 / 5 | 15,56 % | 63,28 % |
| GRPO, continuación de política integrada exactamente | 5 / 6 / 6 | 18,89 % | 63,72 % |
| GRPO, respaldo Bellman esperado | **5 / 8 / 6** | **21,11 %** | **63,72 %** |
| RL con comparación adicional entre agendas, v8b | 5 / 7 / 4 | 17,78 % | 62,94 % |

El planificador público resolvió 27 de 30 pares. En la comparación emparejada, Bellman igualó el número de aciertos del control en la semilla 101 y lo mejoró en 202 y 303. El acierto futuro también igualó o mejoró el del control en cada semilla. La variante con pérdida adicional por pares tuvo una ganancia en validación que cambió de signo en el acierto futuro del reservado. Se conservan los resultados de todas las variantes. [Corrida v8b](../tesis_rl_credit_pares_a100_v8b/RESERVE_RESULTS.json)

Como diagnóstico posterior se ejecutaron también los seis checkpoints históricos v6 sobre este mismo reservado, sin recalibrarlos. El SFT histórico resolvió 0/90 pares y obtuvo 62,61 % de acierto futuro. GRPO histórico resolvió 0/90 y obtuvo 62,39 %. La configuración Bellman nueva alcanzó 21,11 % y 63,72 %. Entre el histórico y v8 cambiaron el inicio supervisado y varias condiciones de entrenamiento. El control supervisado nuevo permite evaluar qué añade RL desde ese mismo inicio. [Márgenes y resultados históricos en el banco nuevo](HISTORICAL_BASELINES.json)

Una segunda ejecución en la A100 reprodujo exactamente los márgenes archivados de todos los adaptadores nuevos seleccionados y de los tres puntos de partida v7. Las cuatro pruebas nuevas de regresión pasaron en CPU. La suite completa terminó con 1.453 pruebas aprobadas y 21 integraciones omitidas con razón registrada. [Verificador ejecutable](../../research/verify_rl_v8.py) · [Pruebas del retorno y del gradiente](../../research/tests/test_tesis_rl_v8_rewards.py)

## Qué se detectó en el entrenamiento anterior

V6 ya exploraba ASK y WAIT en cada estado inicial y valoraba ocho continuaciones por mundo y acción. La auditoría encontró un desequilibrio en la señal. Mezclando presupuestos uno y dos y ambas visitas, el 87,5 % del peso favorecía preguntar. En la segunda ronda, las filas que favorecían esperar aportaban aproximadamente entre 2,92 % y 3,36 % de la suma ponderada de diferencias absolutas de retorno. Estas proporciones describen los retornos antes de aplicar las probabilidades, el clipping y la KL. [Auditoría con firmas de los datos de entrenamiento](AUDITORIA_V6.json)

La recompensa v6 combina acierto actual, 1,5 veces acierto futuro, coste de 0,08 por pregunta y penalización de 0,35 por reutilización sin respaldo. Cuando la continuación apenas pregunta en la segunda visita, esperar puede tener poco retorno incluso si la agenda favorece esa comida. La selección histórica a cuota fija también admitía modelos que no cambiaban de visita al invertir la agenda. El histórico sigue publicado con su resultado de 62,04 % para GRPO y 62,21 % para SFT en aquel banco. [Informe histórico](../tesis_timing_qwen_a100_v6/INFORME.md)

## Qué cambió y cómo se controló

Cada semilla de esta fase parte del mismo adaptador v7 SFT por pares para los tres métodos. Cada uno recibe los mismos 30 pares de entrenamiento con presupuesto uno y lotes de dos pares completos. Se realizan 120 actualizaciones por variante y semilla. Los tokens reales, las decisiones, los forwards y las actualizaciones coinciden entre semillas. Dentro de cada semilla también coincide el relleno de los lotes.

Las variantes RL integran la continuación de la segunda visita en sus dos acciones posibles. Se comparan dos respaldos. Uno promedia las acciones con la probabilidad de la política actual. El otro elige la mejor acción después de promediar los mundos compatibles con la observación pública. Este segundo respaldo es una operación Bellman de la tarea conocida. Los valores terminales proceden de las lecturas y operaciones de SQLite, incluido el coste de preguntar. El entrenamiento no recibe una agenda privada como entrada. Las pruebas incluyen un caso donde elegir la mejor acción por mundo daría un valor mayor e indebido.

La ventaja se centra con la política anterior y se escala por la desviación de las dos recompensas, con mínimo 0,05. Se prueban temperatura 4, clipping 0,2, KL 0,005, entropía 0,01 y tasa de aprendizaje 0,00003. Se registran entropía, KL, fracción de ratios recortados y norma de gradiente. Son adaptaciones del objetivo de dos acciones del proyecto. La entropía termina baja incluso con estas correcciones. El colector exacto mantiene valoradas ambas acciones, pero eso por sí solo no impide que la política se vuelva determinista. [Configuración congelada](PLAN.json) · [Implementación](../../research/tesis_rl_v8.py)

Los estados contrafactuales de segunda visita reciben peso uniforme en los lotes, incluso si preguntar en la primera visita impediría llegar a ellos. Por tanto, la continuación se calcula exactamente y el entrenamiento conserva un objetivo por estados. Esta precisión describe el cálculo de Q y se distingue del gradiente del retorno de una trayectoria completa.

## Desarrollo y apertura del reservado

Se seleccionó entre las épocas 2, 4, 6 y 8 mediante pares completos en validación, seguido por acierto futuro a gasto fijo y menor época. En validación el control alcanzó 25/90 pares y 64,17 % de acierto futuro. GRPO exacto alcanzó 22/90 y 64,72 %. Bellman alcanzó 21/90 y 64,50 %. Después se añadió v8b, con un término de comparación entre márgenes derivado de los retornos. Esa variante alcanzó 23/90 y 65,17 % en validación. Su objetivo híbrido y su selección quedaron fijados antes de puntuar el reservado nuevo. [Validación v8](VALIDATION.json) · [Selección v8](LOCK.json) · [Validación v8b](../tesis_rl_credit_pares_a100_v8b/VALIDATION.json)

Una instancia IA redactó los 30 pares nuevos y otra revisó a ciegas sus dos agendas, con códigos y lados reasignados. La lectura coincidió en las 60 orientaciones. Se conservan dudas de naturalidad y matices sobre oportunidades ofrecidas y visitas confirmadas. El banco genera 480 mundos y usa 15 pares cena/desayuno y 15 cena/bebida. Los marcos lingüísticos se repiten. El autor de la tesis decidió trabajar con IA y su propia revisión, sin reclutar participantes. La lectura personal del autor del nuevo banco queda pendiente y se registra como tal. [Corpus y sello](../tesis_rl_ecuador_reserva_v8/SELLO.json) · [Juicios de IA](../tesis_rl_ecuador_reserva_v8/JUICIOS.json)

Se verificaron las firmas de la base Qwen, el adaptador padre, los programas de construcción, memoria, entrenamiento y evaluación. La prueba utilizó una NVIDIA A100 de 80 GB. Se guardaron los márgenes de todos los métodos y se recalcularon las tablas en CPU. [Entradas de ejecución](EXECUTION_INPUTS.json) · [Márgenes de la A100](RESERVE_MARGINS.json)

## Incertidumbre del contraste

| Diferencia frente al control supervisado | Pares completos, puntos porcentuales | Intervalo descriptivo del 95 % por pares | Acierto futuro, puntos porcentuales |
| --- | ---: | ---: | ---: |
| GRPO exacto | +3,33 | −1,11 a +7,78 | +0,44 |
| GRPO Bellman | +5,56 | +1,11 a +11,11 | +0,44 |
| RL con comparación entre agendas | +2,22 | −3,33 a +8,89 | −0,33 |

Se remuestrean los 30 pares con reemplazo 5.000 veces, manteniendo juntas las tres semillas. Los intervalos están condicionados a estas semillas y textos. Hay tres contrastes exploratorios y los intervalos no se ajustan por multiplicidad. Bellman fue la variante mejor observada en este reservado. Estos resultados aportan evidencia acotada de mejora y conservan los fallos del método híbrido. El rendimiento del planificador y la repetición de marcos limitan una conclusión general sobre agentes o español de Ecuador.

## Archivos y reproducción

Se publican los nueve adaptadores seleccionados de v8 y los seis de v8b. Los tres controles v8b reproducen el mismo entrenamiento del control v8 y no son repeticiones adicionales independientes. Los tensores se copian sin cambios. La configuración portátil utiliza el identificador público de Qwen y su correspondencia de firmas queda registrada. [Adaptadores v8](ADAPTADORES.json) · [Procedencia](PROCEDENCIA.json) · [Licencia de Qwen](LICENSE-Qwen.txt)

Desde la raíz del repositorio, estos comandos comprueban fuentes, corpus, pesos portátiles, métricas y contrastes en CPU:

```bash
python -m research.verify_rl_v8 --run reports/tesis_rl_credit_a100_v8 \
  --reserve reports/tesis_rl_ecuador_reserva_v8
python -m research.verify_rl_v8 --run reports/tesis_rl_credit_pares_a100_v8b \
  --reserve reports/tesis_rl_ecuador_reserva_v8
```

Con el modelo base y las dependencias CUDA preparadas, añade `--replay` para recomputar los márgenes de los adaptadores publicados y compararlos con los archivados. Las instrucciones de entrenamiento usan una carpeta de corrida nueva si se desea una repetición separada. [Entorno GPU](https://github.com/kalech7/mesa-y-memoria/blob/main/docs/modelos-publicos.md) · [Configuración y comandos de desarrollo](../../docs/rl_exploracion_recompensa_v8.md)
