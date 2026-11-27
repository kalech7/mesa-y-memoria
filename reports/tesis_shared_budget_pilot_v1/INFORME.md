# Una pregunta compartida entre dos sesiones

Cada historia dispone de una sola aclaración. El banco cruza la disponibilidad de la persona en la primera visita con el riesgo de que falte información en la segunda. Las pistas públicas informan probabilidades. En cada raíz, los mundos empiezan con la misma observación y difieren en la intención, la disponibilidad y la situación futura. El escritor de evidencia y el lector simbólico son comunes.

La auditoría enumeró 15.552 trayectorias terminales por partición del banco. TRAIN y validación tienen 16 raíces y 192 mundos cada uno. El planificador ajustó su tabla en TRAIN a partir de rasgos públicos comprimidos e historial de acciones. La política de un paso estimó el valor de la acción suponiendo que después siempre se continúa. Ambas usaron la misma función de utilidad.

## Resultado CPU

| Política | Acierto actual sustentado | Acierto futuro sustentado | Preguntas por historia | Preguntas innecesarias por historia |
| --- | ---: | ---: | ---: | ---: |
| Continuar | 100 % | 50,00 % | 0,00 | 0,000 |
| Regla pública | 100 % | 69,17 % | 1,00 | 0,425 |
| Valor de un paso | 100 % | 69,17 % | 1,00 | 0,425 |
| Planificador público | 100 % | 73,33 % | 0,75 | 0,050 |

Los mundos se ponderan por sus probabilidades dentro de cada historia. Consideramos innecesaria una pregunta si retirarla y reproducir las demás acciones mantiene o mejora los aciertos actual y futuro, sin aumentar el daño. Esta medida es retrospectiva dentro del simulador.

El planificador gastó la pregunta en la primera visita cuando la persona tenía más disponibilidad. Con poca disponibilidad, la reservó para la segunda y solo la usó cuando hizo falta. Además, en las situaciones con una elección temporal explícita pudo dar la misma recomendación de hoy sin guardar un registro de sesión innecesario. El planificador promedió 0,50 respuestas provisionales por historia.

El aumento fue **4,17 puntos porcentuales** frente a cada control principal. Quedó por debajo de los cinco puntos fijados como diferencia práctica. El remuestreo emparejado dio un intervalo de 4,17 a 4,17 puntos porque las raíces tienen el mismo efecto dentro de cada estrato sintético. Ese intervalo describe la composición del banco y no mide la variación esperable en usuarios reales.

Repetir las mismas acciones con el veto desactivado produjo el mismo resultado. En esas trayectorias, el escritor no propuso escrituras que el veto pudiera rechazar. No hubo citas vencidas. El planificador redujo el uso futuro de respuestas sin respaldo de 30,83 % a 26,67 %. Estos conteos reflejan la incertidumbre todavía compatible con el diálogo, incluso si la respuesta acertó por casualidad.

El servidor de la universidad reprodujo `CPU.json` byte por byte con Python 3.12. La versión local usó Python 3.14. El plan conserva las huellas y criterios previos a la validación. Los resultados CPU contienen las métricas por raíz y las comparaciones. [Plan](PLAN.json) · [Resultados CPU](CPU.json)

## Comparación exploratoria de los selectores compactos

| Política seleccionada en validación | Acierto futuro sustentado | Preguntas por historia | Utilidad |
| --- | ---: | ---: | ---: |
| SFT inicial | 69,17 % | 1,00 | 1,9575 |
| GRPO sin recompensa terminal | 69,17 % | 1,00 | 1,9575 |
| SFT con las mismas ramas | 73,33 % | 0,75 | 2,0400 |
| GRPO longitudinal | 73,75 % | 0,95 | 2,0303 |

GRPO longitudinal ganó 4,58 puntos frente a la variante sin recompensa terminal. En esa variante se seleccionó el SFT inicial en las tres semillas. Frente a SFT por ramas, GRPO ganó 0,42 puntos de acierto futuro y gastó 0,20 preguntas adicionales por historia. Con el costo fijado de 0,08 por pregunta, SFT por ramas obtuvo la mayor utilidad. Los cuatro brazos conservaron 100 % de acierto actual.

Los tres brazos de continuación compartieron por semilla 24 actualizaciones iniciales, 912 grupos públicos, 3.184 retornos de acción y 128 actualizaciones por brazo. Los grupos derivan de trayectorias TRAIN ya ejecutadas. El selector recibe rasgos numéricos y usa cero tokens de LLM. La selección y las cifras de esta sección usan la misma validación de desarrollo. [Entrenamiento y selección](TRAINING.json)
