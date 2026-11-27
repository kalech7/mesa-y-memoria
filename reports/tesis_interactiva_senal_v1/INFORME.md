# Auditoría de la recompensa longitudinal

La auditoría compara ramificaciones reales del simulador en entrenamiento y validación. Cada política empieza con la misma observación y abre su propia base SQLite. La regla observable sirve de control del mecanismo. El informe registra resultados de la simulación sin entrenamiento.

| Partición | Episodios | Cambia el retorno de tarea | Cambia solo la penalización | Cambia el acierto futuro con acierto actual fijo | El horizonte futuro cambia la trayectoria preferida |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 80 | 42 | 19 | 26 | 26 |
| validation | 53 | 24 | 13 | 12 | 12 |

| Partición | Política | Acierto actual | Acierto futuro | Preguntas por episodio | Retorno longitudinal |
| --- | --- | ---: | ---: | ---: | ---: |
| train | proceed | 85.8% | 70.1% | 0.00 | 1.858 |
| train | observable_reference | 94.2% | 92.1% | 1.05 | 2.221 |
| train | generic_then_reference | 86.7% | 76.8% | 2.35 | 1.778 |
| validation | proceed | 85.0% | 76.2% | 0.00 | 1.932 |
| validation | observable_reference | 93.8% | 91.9% | 0.82 | 2.223 |
| validation | generic_then_reference | 90.0% | 80.0% | 2.40 | 1.855 |

La recompensa larga asigna 1 punto al promedio de aciertos actuales sustentados y 1,5 al promedio de aciertos sustentados en las consultas terminales. Cada pregunta cuesta 0,08 puntos. Las penalizaciones cuentan salidas inválidas, escrituras aplicadas sin respaldo y afirmaciones actuales sin evidencia. La variante inmediata elimina el término de consultas terminales. Las respuestas de las visitas intermedias pueden depender de lo que se guardó antes.

Una pregunta informativa, vaga, redundante o sin respuesta paga el mismo coste. Sus consecuencias pueden cambiar los aciertos actuales o futuros. El detalle JSON publica sensibilidad al coste, resultados por familia y conteos con varios turnos. Las trazas se exportan con los resultados.
