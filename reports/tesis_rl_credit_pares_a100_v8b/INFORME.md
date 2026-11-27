# Ablación RL con comparación de agendas

Desde los mismos pesos supervisados, se añadió al actor RL una pérdida que ordena los márgenes de las dos agendas según sus retornos de entrenamiento. En el reservado nuevo obtuvo 16/90 pares completos y 62,94 % de acierto futuro. El control supervisado obtuvo 14/90 y 63,28 %. La diferencia en acierto futuro fue de −0,33 puntos. En validación esa diferencia había sido de +1,00 punto. Se conservan los resultados como parte de la comparación exploratoria. [Informe conjunto y limitaciones](../tesis_rl_credit_a100_v8/INFORME.md) · [Resultados](RESERVE_RESULTS.json)

Los seis adaptadores seleccionados y sus firmas están publicados. Los controles supervisados corresponden al mismo entrenamiento realizado en v8. [Adaptadores](ADAPTADORES.json) · [Procedencia](PROCEDENCIA.json)
