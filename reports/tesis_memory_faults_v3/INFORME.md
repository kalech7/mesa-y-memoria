# Auditoría factorial de la vigencia en SQLite

Se ejecutaron **288 trayectorias** de tres visitas. Cada una de las 36 raíces sintéticas se repitió bajo ocho combinaciones de escritor, lector y veto de escritura. Las semillas 101, 202 y 303 generan historias. Cada combinación conserva exactamente las mismas acciones, mensajes y réplicas. El lector actúa después de reabrir SQLite, sobre una instantánea idéntica para los dos niveles de lector a igualdad de escritor y verificador. Cada raíz aporta seis consultas futuras, incluidas el último día vigente, el día de vencimiento y una preferencia de control. [Plan](PLAN.json) · [Código](../../research/tesis_memory_faults_v3.py)

## Intervenciones y magnitud

En los casos temporales, el escritor defectuoso prolonga 30 días la fecha de fin de una propuesta válida. Conserva la cita y las demás condiciones de aplicación. El lector defectuoso omite el límite de fecha al realizar la consulta. Un control adicional del lector evalúa la respuesta candidata contra la caducidad registrada y se abstiene si cita al menos un recuerdo vencido.

| Escritor | Lector | Veto de escritura | Propuestas defectuosas persistidas | Consultas con cita vencida | Consultas con cita contaminada | Acierto futuro sustentado |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Fiel | Fiel | Activo | 0 | 0 | 0 | 216/216 |
| Fiel | Fiel | Desactivado | 0 | 0 | 0 | 216/216 |
| Fiel | Omite caducidad | Activo | 0 | 90 | 0 | 126/216 |
| Fiel | Omite caducidad | Desactivado | 0 | 90 | 0 | 126/216 |
| Prolonga vigencia | Fiel | Activo | 0 | 0 | 0 | 171/216 |
| Prolonga vigencia | Fiel | Desactivado | 27 | 0 | 99 | 117/216 |
| Prolonga vigencia | Omite caducidad | Activo | 0 | 36 | 0 | 135/216 |
| Prolonga vigencia | Omite caducidad | Desactivado | 27 | 36 | 99 | 81/216 |

- **El veto de escritura bloqueó las 27 propuestas alteradas de cada brazo.** Con lector fiel, recuperó el valor habitual en las 36 consultas del día de vencimiento, frente a 9 sin veto. Al descartar la propuesta completa, solo 9 de 36 consultas conservaron el valor sustentado del último día vigente. Una reparación de la fecha exigiría validar el alcance original contra la evidencia.
- **El acierto textual oculta citas caducadas.** Con escritor fiel, el lector defectuoso citó un recuerdo vencido en 90 consultas. En 12, el valor coincidía con el correcto. El control lector se abstuvo en las 90, reduciendo a cero las afirmaciones sin respaldo en ese brazo y conservando las 126 respuestas sustentadas.
- **Los controles actúan sobre fallos distintos.** Con los dos defectos y el veto de escritura desactivado, el filtro lector retiró 36 citas vencidas y quedaron 99 consultas que citaron una escritura con fecha alterada. El filtro usa la fecha que llega de SQLite. El cotejo con la evidencia que autorizó la escritura detecta esa alteración. La preferencia estable de otro atributo se recuperó en las 36 consultas de control de cada brazo.

Durante las tres visitas, almacenar la propuesta alterada conservó 100 % de acierto factual, con 50 % de acierto sustentado. La mitad de las respuestas dependió de un recuerdo con una fecha sin respaldo. Con el veto se obtuvo 50 % en ambos indicadores porque parte de la preferencia válida quedó fuera del almacén.

La tasa describe el banco de fallos deliberados y su composición por familia. Los efectos emparejados están calculados por raíz con detalle por semilla. Las trazas incluyen el texto recibido, las operaciones propuestas, el resultado del veto, los identificadores persistidos y la respuesta antes y después de aplicar la protección lectora. [Resultados por consulta](RESULTS.json) · [Trazas](TRACES.json)

Reproducción desde la raíz del proyecto con una carpeta nueva.

```bash
python -m research.tesis_memory_faults_v3 --output reports/mi_auditoria_de_vigencia
```
