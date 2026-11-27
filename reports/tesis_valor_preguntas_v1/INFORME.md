# Valor de preguntar antes de actualizar la memoria

Auditoría exploratoria sobre 40 historias de validación y 265 ramas legales. Cada rama se ejecutó en su propia base SQLite.

Para cada diálogo visible se compararon cuatro preguntas y continuar. La mejor acción se calculó promediando los mundos compatibles con el mismo diálogo. La ventaja y el coste se expresan en unidades de la recompensa registrada en `RESULT.json`.

| Familia | Historias | Prefijos con pregunta óptima al coste 0,10 | Δ acierto actual | Δ acierto futuro |
| --- | ---: | ---: | ---: | ---: |
| F0 | 8 | 0/8 | +0.0 pp | +0.0 pp |
| F1 | 8 | 0/8 | +0.0 pp | +0.0 pp |
| F2 | 8 | 0/8 | +0.0 pp | +0.0 pp |
| F3 | 8 | 5/8 | +62.5 pp | +0.0 pp |
| F4 | 8 | 8/8 | +0.0 pp | +100.0 pp |

En 16 de 16 ramas temporales informativas, el acierto futuro mejoró al permitir que la aclaración llegara al escritor. Empeoró en 0 ramas. La pregunta, la réplica y la respuesta actual fueron idénticas.

`COSTE_POR_FAMILIA.csv` contiene la sensibilidad al coste y `RESULT.json` el detalle por prefijo. `cases.jsonl` enlaza cada resultado con su traza y el estado persistido.

Los máximos estimados dependen del simulador, del escritor, del lector y de la recompensa fijada. Son referencias retrospectivas de desarrollo y la evaluación se limita a conversaciones simuladas.
