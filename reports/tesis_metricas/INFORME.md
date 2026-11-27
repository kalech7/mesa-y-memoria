# Qué falla al aclarar, escribir y reutilizar preferencias

Se volvieron a puntuar las trayectorias archivadas de dos pilotos. Cada uno contiene 25 historias y 30 episodios por método, una semilla de entrenamiento y cinco familias. El acierto sustentado exige un valor correcto con evidencia admisible. El éxito conjunto exige acierto actual y futuro. El éxito conjunto estricto exige además salidas válidas y escrituras sustentadas.

## SFT débil y continuaciones con tiempo equiparado

| Indicador | SFT inicial | Más SFT | GRPO seleccionado |
| --- | ---: | ---: | ---: |
| Éxito conjunto, media por familia | 34 % | 64 % | 34 % |
| Éxito conjunto estricto, media por familia | 34 % | 62 % | 34 % |
| Cambio permanente sin resolver en la memoria | 10/10 | 3/10 | 10/10 |
| Cambio permanente omitido con evidencia visible para WRITE | 6/6 | 2/9 | 6/6 |
| Preferencia temporal guardada sin vencimiento | 0/13 | 3/13 | 0/13 |
| Episodio con daño en alguna consulta protegida | 0/30 | 3/30 | 0/30 |

En cambios duraderos explícitos (F1), más SFT pasó de omitir los 5 cambios a guardar los 5. En alcance ambiguo (F4), redujo de 8/10 a 2/10 las omisiones de la pregunta necesaria. Las 8 aclaraciones informativas de F4 resolvieron exclusivamente la decisión futura. En cambio, guardó sin vencimiento 2 cambios de una sola sesión en F4 y 1 excepción temporal en F2. Los tres alteraron al menos una respuesta protegida fuera de ese alcance. El GRPO seleccionado conserva el paso cero. Su terminal obtuvo 30 % de éxito conjunto y 8/10 omisiones de pregunta en F4. [Desglose y denominadores](sft_debil.json) · [Diseño del piloto](../weak_sft_natural_20261104_001/INFORME_COMPARACION.md)

## GRPO con crédito por fase y SFT congelado

| Indicador | SFT congelado | GRPO seleccionado |
| --- | ---: | ---: |
| Éxito conjunto estricto, media por familia | 22 % | 34 % |
| Cambios permanentes de F4 sin resolver en memoria | 1/5 | 0/5 |
| Omisiones de aclaración en F4 | 2/10 | 0/10 |
| Preguntas en episodios inicialmente suficientes | 10/15 | 12/15 |
| Episodio con daño en alguna consulta protegida | 11/30 | 6/30 |

El éxito conjunto estricto de F4 subió de 5/10 a 7/10. En F1, 4/5 cambios explícitos siguieron sin quedar guardados de forma funcional y sustentada, mientras las preguntas redundantes aumentaron de 2/5 a 4/5. En F3, ninguna de las cinco oportunidades de aclarar el contexto quedó resuelta. Estos casos separan dos mejoras pendientes del selector de duración: pedir el dato de contexto pertinente y ejecutar la actualización cuando el usuario ya fue explícito. [Desglose y denominadores](grpo_credito.json) · [Diseño del piloto dirigido](../reward_review_targeted_20261105_002/INFORME.md)

## Cómo se calculó

El JSON publica tasas, numeradores, denominadores y contrastes entre las mismas historias, mundos, redacciones y semillas. Las tasas macro pesan por igual las familias con oportunidades. Las fracciones de las tablas son conteos brutos. Los denominadores de omisiones de WRITE pueden cambiar entre métodos porque cada conversación entrega al escritor evidencia distinta. Las consultas protegidas recorren una cuadrícula finita de fechas, sesiones y contextos sobre preferencias previamente resolubles. La prueba excluye la región de un cambio autorizado y detecta daño si la respuesta se pierde, cambia o queda sustentada únicamente por registros infieles. Los dos bancos son sintéticos y cada piloto usa una semilla. El piloto dirigido compara GRPO con el padre SFT congelado.
