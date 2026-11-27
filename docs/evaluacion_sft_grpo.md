# Auditar la comparación de SFT y GRPO

Desde la raíz del proyecto, usa Python local para volver a puntuar las trayectorias guardadas y comprobar sus artefactos.

```bash
python -m research.weak_sft_qa --archived research/runs/weak_sft_natural_20261104_001
```

Para obtener el detalle por familia, añade una ruta nueva al final del comando, por ejemplo `reports/mi_auditoria_sft_grpo.json`. El comando opera sobre datos ya recogidos. Comprueba los hashes de las etapas y checkpoints, el aislamiento de las particiones, la selección en validación, las trayectorias emparejadas y la integridad de los archivos SQLite. Vuelve a puntuar las respuestas y las escrituras y recalcula los retornos de las trayectorias de entrenamiento GRPO. Funciona sin cargar Qwen, hacer inferencias o usar GPU.

## Qué se comparó

- El piloto usa Qwen2.5-0.5B-Instruct con LoRA, una semilla de entrenamiento y el mismo banco de 25 historias y 30 episodios para cada método. Cada episodio exige decidir si preguntar, responder, gestionar la memoria y afrontar una consulta en una sesión posterior sin el diálogo anterior. [Diseño y resultados del piloto](../reports/weak_sft_natural_20261104_001/INFORME_COMPARACION.md).
- SFT de continuación y GRPO longitudinal parten del mismo SFT calibrado. Más SFT recibe como presupuesto el tiempo de optimización utilizado por GRPO, incluyendo la recolección de trayectorias. Obtiene 634 pasos frente a 12 de GRPO. Se selecciona por éxito conjunto en validación. Cuando hay empate se conserva el checkpoint inicial. [Plan guardado](../research/runs/weak_sft_natural_20261104_001/PILOT_PLAN.json) · [Selección](../research/runs/weak_sft_natural_20261104_001/SELECTION.json).
- C2 mide el acierto actual sustentado. FJS mide el acierto posterior sustentado por la memoria. J exige que ambos sean correctos en el mismo episodio. Primero se promedian realizaciones, luego mundos e historias, y las cinco familias reciben el mismo peso.

| Método | C2 | FJS | J | Preguntas por episodio | Escritura sin respaldo |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT calibrado | 80 % | 34 % | 34 % | 0,04 | 0 % |
| Más SFT | 96 % | 68 % | 64 % | 0,32 | 8 % |
| GRPO seleccionado, paso 0 | 80 % | 34 % | 34 % | 0,04 | 0 % |
| GRPO terminal, paso 12 | 84 % | 34 % | 30 % | 0,04 | 0 % |

En las trayectorias archivadas, el SFT calibrado y el GRPO terminal realizaron cero escrituras efectivas. En correcciones duraderas J fue 0 % en GRPO terminal y 80 % en más SFT. En excepciones temporales ambos obtuvieron 20 %. La pérdida de 4 puntos de GRPO terminal frente al SFT calibrado corresponde a una historia de control en la que el usuario habla de otro atributo. Son contrastes descriptivos con cinco historias por familia. [Trayectorias y agregados](../research/runs/weak_sft_natural_20261104_001/evaluation/).

## Qué evidencia aportó GRPO

El optimizador actualizó los pesos LoRA en 12 pasos con gradientes no nulos y 48 trayectorias. Ocho grupos de cuatro mostraron diferencias en utilidad de tarea entre rollouts y cuatro tuvieron diferencias debidas únicamente a costes. En validación el último paso empató con el inicial, y en la evaluación reservada la política del paso 12 empeoró el éxito conjunto en una de las 25 historias. El resultado es exploratorio, condicionado a una semilla, a las conversaciones simuladas y al presupuesto ejecutado. [Diagnósticos de grupos](../research/runs/weak_sft_natural_20261104_001/continuations/B2-L/optimizer/group_diagnostics.jsonl).

La continuación GRPO con crédito por fase de la campaña dirigida 002 parte de un checkpoint SFT limpio con 15 120 pasos supervisados previos. En su propio banco reservado de 25 historias, el paso 8 seleccionado sube el éxito conjunto **estricto** de 22 % a 34 % y el éxito futuro de 36 % a 48 % frente al padre congelado. Tres historias mejoran y 22 empatan. En un mundo ambiguo, GRPO pregunta por la duración y guarda una preferencia permanente nueva; en las otras mejoras intervienen la preservación de una preferencia ya registrada y la corrección de citas. La tasa de preguntas innecesarias sube de 40 % a 48 %. El contraste mide añadir GRPO frente a congelar y carece de un brazo de más SFT en esa misma configuración. Una semilla y un juicio automatizado de entrenamiento con un error documentado limitan el alcance. [Campaña dirigida 002](../reports/reward_review_targeted_20261105_002/INFORME.md) · [Auditoría independiente](../reports/reward_review_targeted_20261105_002/QA_INDEPENDIENTE_RUN002.md).

Para volver a puntuar también la campaña dirigida 002 en esta copia, ejecuta su auditoría con un nombre de salida nuevo junto a los informes originales.

```bash
PYTHONPATH=. python reports/reward_review_targeted_20261105_002/independent_runtime_qa.py \
  --run research/runs/reward_review_targeted_20261105_002 \
  --output reports/reward_review_targeted_20261105_002/MI_QA_REPRODUCIDA.json
```

El experimento posterior del controlador RL estable entrenó una cabeza contextual sobre un modelo congelado. Obtuvo 16 % por consulta, igual que su LM congelado, y 0 % de episodios con ambas consultas correctas. Su aumento de retorno provino de omitir una pregunta fija. Su arquitectura y banco corresponden a otro experimento. [Informe del controlador RL](../reports/reward_review_decision_20261105_003/INFORME.md).
