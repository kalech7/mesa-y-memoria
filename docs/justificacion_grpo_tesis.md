# Por qué evaluar aprendizaje por refuerzo para aclarar preferencias

Una pregunta cuesta un turno y puede cambiar la evidencia con la que el asistente guarda una preferencia. En un par ambiguo, «prefiero arroz con lentejas» permite responder lo mismo hoy tanto si el cambio es puntual como si es habitual. Preguntar por la duración puede decidir si la próxima conversación debe recuperar la preferencia anterior o la nueva. La respuesta inmediata aporta poca señal para aprender esa decisión. El entorno puede medir el efecto al reabrir la memoria y consultar de nuevo.

La primera campaña entrena solo SELECT. El modelo elige entre continuar y cuatro preguntas. RESPOND y WRITE se mantienen congelados y el lector reutiliza SQLite. La formulación corresponde a una decisión contextual con retorno diferido. Se adapta GRPO para comparar cuatro acciones muestreadas del mismo estado y optimizar LoRA con ventajas relativas al grupo, recorte y regularización respecto al SFT inicial. El SFT adicional parte de los mismos pesos y usa supervisión categórica con la misma asignación temporal. La formulación del problema y el origen de GRPO se documentan en [Contextual Bandit Learning with Predictable Rewards](https://proceedings.mlr.press/v22/agarwal12.html) y [DeepSeekMath](https://arxiv.org/abs/2402.03300).

## Qué señal recibió GRPO

Se auditaron 60 grupos de entrenamiento y 120 pasos de optimización. El retorno varió en 35 grupos. En 16 varió el componente de tarea y en 19 solo las penalizaciones. En 11 grupos cambió el acierto futuro entre las ramas y el acierto actual quedó fijo. En 6 grupos, incluir los términos futuros cambió el conjunto de ramas con mayor recompensa frente a puntuar solo el acierto actual y los mismos costes.

En la semilla 101, grupo 2, preguntar por la duración y continuar obtuvieron el mismo acierto actual. Preguntar permitió un acierto futuro sustentado, pese a que el episodio aún incluyó una escritura sin respaldo. El retorno fue 0,40 al preguntar y 0 al continuar. Al eliminar los términos futuros, quedó 0,10 por debajo de continuar debido al coste de la pregunta. Es evidencia del mecanismo de recompensa sobre ramas observadas. [Grupos auditados](../reports/tesis_comparacion_completa_v1/SENAL_GRPO.json).

## Qué resultado obtuvo la política

En las 50 historias reservadas, validación seleccionó el SFT inicial en las tres semillas de GRPO. Los tres selectores seleccionados obtuvieron 49,7 % de acierto futuro sustentado y 3,0 % de éxito conjunto estricto. El experimento muestra que la recompensa pudo distinguir algunas decisiones con valor futuro y que la política seleccionada no mejoró con la receta y el presupuesto probados. Los resultados de GRPO corresponden a los mismos pesos que la referencia SFT. [Selección](../research/runs/tesis_comparison_full_v1/SELECTION.json) · [Comparación](comparacion_tesis.md#resultado-en-50-historias-reservadas).

## El papel de la escritura

Al reproducir las mismas ramas TRAIN con escritura verificada, en los 20 grupos de duración ambigua los casos con variación de acierto futuro subieron de 6 a 13. En esa familia, incluir el horizonte futuro cambió las ramas preferidas en 5 grupos con el escritor original y en 12 con reparación. Se trata de recalcular retornos para acciones ya muestreadas. Los pesos se mantienen fijos en este diagnóstico.

En el conjunto reservado, la reparación elevó el acierto futuro de SFT y GRPO de 49,7 % a 93,3 %. Los 20 fallos residuales de esos dos selectores ocurrieron en casos de duración ambigua de la semilla 202, todos sin pregunta. Los 18 fallos residuales de SFT adicional tienen el mismo patrón. La escritura puede usar la evidencia disponible y la política de aclaración decide si llega a obtenerla. El verificador comparte la gramática del evaluador y las trayectorias corresponden a usuarios simulados. [Reproducción reservada](../reports/tesis_verificador_reservado_full_v1/INFORME.md).

## Qué limitaba al entorno original

En los 200 episodios hay cuatro mensajes de apertura por episodio. La memoria inicial tiene un registro en 15 episodios, dos en 120 y tres en 65. En 100 episodios ninguna pregunta tiene réplica informativa y en los otros 100 hay exactamente una. Estas decisiones permiten aislar el alcance temporal, pero no representan intentos sucesivos de aclaración, cambios de tema o el efecto de usar y actualizar la memoria en una conversación posterior.

## Contraste con varias decisiones por historia

La extensión conserva el entorno de preferencias alimentarias y el motor SQLite. Una historia incluye dos conversaciones en las que el asistente puede preguntar, insistir si la respuesta fue parcial o continuar. La segunda conversación carga la memoria de la primera. Al terminal se mide qué preferencia se recupera y si se mantuvo una preferencia de otro atributo.

La recompensa longitudinal asigna 1 punto al promedio de acierto sustentado en las conversaciones y 1,5 al promedio de acierto sustentado en las consultas terminales. Cada pregunta cuesta 0,08. Se penalizan las salidas inválidas, las escrituras aplicadas sin respaldo y las afirmaciones que exceden la evidencia. La condición inmediata elimina únicamente el término terminal. Puede conservar dependencia de la memoria en el acierto de la segunda conversación.

El peso terminal hace que una aclaración pueda compensar su coste por una mejora posterior. Con cuatro consultas terminales, acertar sustentadamente una más aporta 0,375 al retorno. Dos preguntas cuestan 0,16. Preguntar dos veces por una respuesta que sigue sin resolver el alcance consume ese coste y no produce crédito por el dato. El coste de preguntar es una decisión de modelado. La auditoría prueba varios costes y registra cuándo se invierte la acción preferida.

El protocolo compara GRPO con recompensa longitudinal, GRPO con recompensa inmediata, SFT inicial y SFT adicional. Las continuaciones arrancan del mismo selector supervisado por semilla. Comparten orden de historias y número de actualizaciones. GRPO consume más interacciones que SFT porque recoge varias trayectorias por grupo. Una regla observable y una política que siempre continúa dan referencias adicionales. La familia, la intención privada y las respuestas futuras se quedan en el simulador y el evaluador. El selector recibe rasgos de la evidencia pública extraídos mediante el analizador del proyecto.

El contraste principal se fija antes de la evaluación reservada. Exige que el acierto futuro de GRPO longitudinal supere al de GRPO inmediato por al menos 5 puntos porcentuales, con intervalo emparejado del 95 % por encima de cero, pérdida actual de como máximo 2 puntos y sin aumentar escrituras aplicadas sin respaldo. Los contrastes frente a SFT se informan por separado. El protocolo también conserva una comparación diagnóstica de las dos políticas terminales con los mismos pasos de entrenamiento.

## Resultado en conversaciones de varios turnos

El banco conserva 60 historias de entrenamiento, 40 de validación y 50 reservadas. Se verificaron 200 episodios, 50 pares de mundos con observaciones iniciales idénticas, y la procedencia de los recuerdos iniciales. Cada episodio comienza con cinco o seis recuerdos sobre distintos atributos. La segunda conversación puede recuperar una preferencia, introducir una excepción o corregirla. Las consultas de control comprueban otro atributo y, en los casos pertinentes, los límites de vigencia. El simulador incluye respuestas directas, parciales y negativas a aclarar. Una réplica parcial puede permitir una segunda pregunta. Son comportamientos simulados con una gramática observable y plantillas, evaluados por código. [Auditoría del banco](../reports/tesis_interactive_bank/AUDITORIA.json).

En 26 de 80 episodios de entrenamiento y 12 de 53 de validación, preguntar cambió el acierto futuro con el actual fijo. En esos pares, la recompensa inmediata prefirió continuar y la longitudinal prefirió preguntar. El efecto se mantuvo al variar el coste por pregunta entre 0,02 y 0,32. La auditoría usa una referencia observable y un control que siempre continúa. [Auditoría de recompensa](../reports/tesis_interactiva_senal_v1/INFORME.md).

| Política | Acierto actual sustentado | Acierto futuro sustentado | Éxito conjunto estricto | Preguntas por episodio |
| --- | ---: | ---: | ---: | ---: |
| SFT y continuaciones seleccionadas en validación | 97,0 % | 93,5 % | 86,0 % | 1,00 |
| GRPO inmediato, último checkpoint | 88,0 % | 72,2 % | 36,0 % | 0,00 |
| GRPO longitudinal, último checkpoint | 96,3 % | 91,1 % | 82,7 % | 0,94 |

En los checkpoints terminales, tras 200 pasos por método y semilla, añadir el término futuro aportó **18,9 puntos porcentuales de acierto futuro** frente a GRPO inmediato, con IC 95 % emparejado **de 14,2 a 23,7 puntos**. El éxito conjunto pasó de 36,0 % a 82,7 %, y el acierto futuro en duración ambigua de 50 % a 95 %. El intervalo remuestrea historias con las tres semillas de entrenamiento observadas. En el contraste principal, validación conservó el SFT inicial para las dos continuaciones GRPO en las tres semillas, así que el incremento de acierto futuro fue de 0 puntos. El selector entrenado es una red compacta con rasgos públicos del analizador. El escritor compartido comprueba el respaldo antes de persistir. [Plan](../research/runs/tesis_interactive_v1/PLAN.json) · [Selección](../research/runs/tesis_interactive_v1/SELECTION.json) · [Resultados](../research/runs/tesis_interactive_v1/RESULTS.json) · [Auditoría](../research/runs/tesis_interactive_v1/AUDIT.json).

### Sensibilidad a un inicio con menos supervisión

En un segundo banco certificado, los métodos partieron de una sola actualización SFT, con un lote de 16 estados muestreados. En sus 50 historias reservadas, SFT inicial obtuvo 76,5 %, SFT adicional 88,8 % y ambas variantes GRPO seleccionadas 75,5 % de acierto futuro sustentado. El contraste terminal entre las dos recompensas fue de 0 puntos. Solo 3 de 300 grupos de GRPO longitudinal presentaron variación del retorno de tarea. En el banco principal fueron 67 de 300. Los bancos y la cantidad de supervisión inicial difieren entre corridas, por lo que los contrastes entre métodos se hacen dentro de cada banco. [Separación del banco](../reports/tesis_interactive_bank/SENSIBILIDAD.json) · [Resultados](../research/runs/tesis_interactive_low_supervision_v2/RESULTS.json) · [Auditoría](../research/runs/tesis_interactive_low_supervision_v2/AUDIT.json).

### Verificación de recuerdos en las historias nuevas

En los 201 episodios reservados del checkpoint longitudinal terminal, se reprodujeron las 402 observaciones públicas de escritura. El verificador conservó las 261 operaciones respaldadas. En un control con omisiones recuperó las 261. Al sustituir valores y ampliar alcances como controles, rechazó 261 de 261 valores inventados y 192 de 192 alcances fuera de lo confirmado. Cada prueba conserva la conversación y el estado previo de SQLite. El escritor y el verificador comparten el analizador de evidencia, por lo que el resultado mide preservación y rechazo dentro de esa gramática. [Trazabilidad de la prueba](../reports/tesis_interactive_bank/VERIFICADOR.json).

## Ajuste de exploración y crédito de GRPO

En la comparación anterior, los seis episodios que perdieron aciertos futuros con GRPO terminal correspondieron a excepciones temporales de una semilla. En ellos se omitió preguntar por la fecha de vencimiento. Con poca supervisión inicial, solo tres de los 96 grupos que podían resolver la primera aclaración llegaron a explorar la secuencia pertinente. En el banco principal, las diferencias medias de retorno por cambiar solo el coste de preguntar fueron de 0,094 frente a 0,920 cuando cambiaba algún acierto. La normalización por desviación dentro de cada grupo dejó magnitudes medias de ventaja de 0,80 y 0,86. [Diagnóstico reproducible de trayectorias](../reports/tesis_interactive_bank/DIAGNOSTICO_GRPO.json).

La variante por ramas toma un turno de una trayectoria de entrenamiento y ejecuta cada acción disponible, incluida la decisión de continuar. Completa cada rama con la misma regla pública y calcula el retorno usando el simulador existente. Ajusta solo la decisión donde se abren las ramas. Centra la ventaja con las probabilidades del selector y conserva la magnitud original de las diferencias de retorno. Mantiene el recorte de GRPO y la regularización respecto al SFT inicial. El retorno de cada acción queda condicionado a la regla usada para completar la conversación. [Implementación](../research/tesis_grpo_branches.py) · [GRPO original](https://arxiv.org/abs/2402.03300) · [Análisis del escalado por grupo](https://arxiv.org/abs/2503.20783).

Todos los métodos mantienen la misma política de usuario simulado, el escritor, las reglas de vigencia, SQLite y la recompensa longitudinal. Comparten arquitectura, rasgos públicos, pesos iniciales y orden de historias. SFT adicional imita la regla pública completa. GRPO longitudinal muestrea cinco trayectorias por grupo. SFT con las mejores ramas recibe las mismas trayectorias y puntuaciones que GRPO por ramas, y ajusta por máxima verosimilitud hacia las acciones con mayor retorno. Ambos métodos por ramas usan el mismo número de actualizaciones. SFT adicional utiliza menos trayectorias por grupo. La evaluación mide el acierto futuro, el actual, las escrituras sin respaldo y las preguntas que se le hacen al usuario. [Plan previo a la evaluación](../research/runs/tesis_grpo_ramificado_v1/PLAN.json) · [Auditoría de separación del banco](../reports/tesis_interactive_bank/COMPARACION_GRPO.json).

En las **50 historias reservadas** del banco nuevo, las cuatro continuaciones conservaron el SFT inicial en las tres semillas. Todas las políticas seleccionadas y la regla pública obtuvieron **88,3 % de acierto futuro**, **99,0 % de acierto actual**, **78,0 % de éxito conjunto estricto** y **0,98 preguntas por episodio**. Frente a SFT adicional, GRPO por ramas obtuvo una diferencia de **0 puntos** en acierto futuro, con IC emparejado del 95 % de **0 a 0** bajo las semillas observadas. El contraste no alcanzó el umbral de superioridad de dos puntos. Los 54 fallos de consultas futuras por método ocurrieron en episodios donde el usuario simulado rechazó una aclaración. [Resultados reservados](../research/runs/tesis_grpo_ramificado_v1/RESULTS.json) · [Auditoría](../research/runs/tesis_grpo_ramificado_v1/AUDIT.json).

Durante el entrenamiento, las ramas elevaron de 69 a 100 sobre 300 el número de grupos con variación en el retorno de tarea. En validación, GRPO por ramas conservó el **93,1 %** de acierto futuro inicial al final de las tres semillas. El GRPO longitudinal original terminó con **93,1 %**, **79,8 %** y **86,9 %**. Como diagnóstico previo, la regla ya alcanzaba el máximo de acierto en los 53 episodios de la validación histórica al explorar continuar o hacer la pregunta que podía aportar información. Esta comparación estudia el entrenamiento del selector compacto en el entorno existente. La conducta humana queda representada por réplicas simuladas y la escritura sigue dependiendo del verificador compartido. [Selección y evidencia por checkpoint](../research/runs/tesis_grpo_ramificado_v1/SELECTION.json) · [Diagnóstico histórico](../reports/tesis_interactive_bank/DIAGNOSTICO_GRPO.json).

## Recomendaciones con exclusiones en Qwen

Se adaptaron los pesos LoRA del Qwen2.5-0.5B del laboratorio con casos de rechazo explícito, erratas, múltiples exclusiones, peticiones de variar lo comido, rectificaciones, referencias ambiguas y escenarios sin un plato disponible. El texto de la captura entró en entrenamiento. El banco incluyó 96 casos de entrenamiento, 24 de validación y 40 reservados. Validación y evaluación usan redacciones y preparaciones propias. Los conjuntos comparten el vocabulario de ingredientes y el formato del menú.

Las dos continuaciones partieron del mismo ajuste común de 144 pasos supervisados. Recibieron los mismos 96 casos en el mismo orden durante dos pasadas y ejecutaron 96 actualizaciones cada una. SFT maximiza la probabilidad de las opciones compatibles en todo el vocabulario. GRPO categórico muestrea seis respuestas por caso dentro del menú, centra sus ventajas por grupo y usa regularización hacia el ajuste común. La recompensa da 1 a una opción válida, −1 a un conflicto o una aclaración omitida y −0,35 a otras decisiones inadecuadas. El entorno ofrece seis acciones, cinco platos con ingredientes y la opción de aclarar. La métrica principal usa la elección de Qwen en ese espacio antes de aplicar el verificador del chat. El diagnóstico adicional deja competir a todo el vocabulario en el primer token.

La evaluación de casos nuevos se hizo tras dejar fijados los checkpoints por validación. El intervalo empareja métodos en los mismos diálogos y agrupa las realizaciones del mismo texto con menús distintos. Esta corrida explora una semilla de entrenamiento. [Plan](../research/runs/tesis_meal_lora_v1/PLAN.json) · [LoRA](https://arxiv.org/abs/2106.09685) · [GRPO](https://arxiv.org/abs/2402.03300).

| Adaptador | Acierto antes del verificador | Conflicto con lo excluido |
| --- | ---: | ---: |
| SFT original | 47,5 % | 32,5 % |
| SFT común después de 144 pasos | 47,5 % | 30,0 % |
| SFT adicional, paso 48 | **57,5 %** | **17,5 %** |
| SFT + GRPO, paso 48 | 55,0 % | 22,5 % |

En los 31 diálogos distintos del banco reservado, la estimación emparejada de GRPO frente a SFT fue de **−3,9 puntos porcentuales**, con IC 95 % de **−19,2 a 10,3**. GRPO recibió recompensas distintas en 90 de 192 grupos. Al rotar las seis opciones de cada caso, el promedio bajó a **48,8 % para SFT** y **48,3 % para GRPO**, cerca del **47,9 %** de elegir al azar. El orden del menú influyó en ambos adaptadores. La corrida documenta una mejora en el menú original y un límite de generalización al cambiar las posiciones. [Resultados](../research/runs/tesis_meal_lora_v1/RESULTS.json) · [Auditoría](../research/runs/tesis_meal_lora_v1/AUDIT.json) · [Prueba de rotación](../research/runs/tesis_meal_lora_v1/POSITION.json).

En la segunda comparación, Qwen puntuó cada plato por separado a partir del diálogo y los ingredientes. Así, la posición de los platos quedó fuera de la puntuación. Los dos métodos recibieron las mismas 144 historias de ajuste y 144 actualizaciones por continuación. En **48 casos nuevos**, el ajuste SFT común y GRPO seleccionado obtuvieron **58,3 %**. SFT adicional obtuvo **39,6 %** y pidió aclaraciones innecesarias en **60,4 %** de los casos. La diferencia de **18,8 puntos** frente a SFT adicional se explica por esta degradación. Validación conservó para GRPO los pesos SFT del paso cero. La segunda comparación queda disponible como diagnóstico en Resultados de tesis, y los adaptadores interactivos corresponden a la primera comparación. [Resultados](../research/runs/tesis_meal_lora_v2/RESULTS.json) · [Auditoría](../research/runs/tesis_meal_lora_v2/AUDIT.json).

```bash
python -m research.tesis_meal_train --device cuda --output research/runs/mi_comparacion_comidas
python -m research.tesis_meal_audit research/runs/mi_comparacion_comidas
python -m research.tesis_meal_position research/runs/mi_comparacion_comidas
python -m research.tesis_meal_compat --output research/runs/mi_comparacion_por_ingredientes
python -m research.tesis_meal_compat_audit research/runs/mi_comparacion_por_ingredientes
```

### Entrenamiento con el menú reordenado

Se ampliaron las conversaciones a 530 casos de entrenamiento, 100 de validación y 150 reservados, con 403, 88 y 125 textos distintos tras normalizar. Los subagentes escribieron por separado las plantillas de entrenamiento y las de validación y evaluación. Las conversaciones son sintéticas de estilo cotidiano. El mensaje original del usuario se conservó en entrenamiento. La evaluación contiene platos propios y no comparte textos normalizados con los bancos anteriores. Las familias incluyen menciones de comidas pasadas, preferencias de terceros, follow-ups que confirman una exclusión y cambios de idea.

El ajuste supervisado común y las dos continuaciones vieron cada caso en las seis rotaciones, con el mismo orden de exposiciones entre SFT adicional y GRPO. Las dos continuaciones actualizan el mismo tipo de adaptador LoRA y optimizan sobre las mismas seis opciones. GRPO toma ocho elecciones por caso y hace dos pasos de actualización con las probabilidades de muestreo congeladas. Recibe recompensa positiva por una opción compatible o una aclaración necesaria, y una penalización por un conflicto o por aclarar de más. Validación fija los checkpoints antes de puntuar las elecciones de Qwen.

La evaluación conserva la elección directa de Qwen sobre cada menú completo. También mide el acierto en los seis órdenes de un mismo caso y la proporción de pares en la que ambas elecciones son correctas. Los pares tienen el mismo menú y una restricción distinta que cambia cuál es el único plato válido. El intervalo emparejado agrupa las realizaciones del mismo texto. Una misma familia de redacción puede dar lugar a varios diálogos de un banco. [Plan archivado](../research/runs/tesis_meal_lora_varied_v1/PLAN.json) · [GRPO](https://arxiv.org/abs/2402.03300).

| Checkpoint | Acierto en 900 elecciones | Conflicto con lo excluido | Acierto en los seis órdenes | Pares resueltos |
| --- | ---: | ---: | ---: | ---: |
| Adaptador original | 42,1 % | 37,4 % | 0 % | 0 % |
| SFT común | 90,8 % | 6,6 % | 77,3 % | 100 % |
| Selección de SFT adicional | 90,8 % | 6,6 % | 77,3 % | 100 % |
| Selección de GRPO | 90,8 % | 6,6 % | 77,3 % | 100 % |

Las dos continuaciones seleccionaron el SFT común, que obtuvo **86,2 %** en validación. Los checkpoints posteriores de SFT obtuvieron **81,8 %** y **72,5 %**; los dos de GRPO obtuvieron **44,5 %** y colapsaron a una letra fija. GRPO vio retornos distintos en **627 de 3.180 grupos**. La ganancia en los casos reservados procede del ajuste supervisado común con más datos y órdenes. Las exclusiones múltiples y las erratas siguen siendo las familias con menor acierto, con **70,8 %** y **68,1 %**. Resultados condicionados a una semilla, al menú de seis opciones y a redacciones sintéticas cotidianas. [Comparación reservada](../research/runs/tesis_meal_lora_varied_v1/RESULTS.json) · [Selección](../research/runs/tesis_meal_lora_varied_v1/SELECTION.json) · [Auditoría](../research/runs/tesis_meal_lora_varied_v1/AUDIT.json).

En tres permutaciones fuera de las rotaciones por caso, el ajuste elegido obtuvo **89,6 %** de acierto y **8,0 %** de conflicto. Los adaptadores anteriores, cada uno con su prompt de entrenamiento, obtuvieron **43,6 %** y **42,4 %** en esos mismos menús. Al ocultar el diálogo y conservar los platos de la prueba, el resultado del ajuste nuevo bajó a **16,2 %**. Este control prueba cuánto depende la decisión del texto visible en el banco. En los 15 pares reservados eligió correctamente en ambos diálogos para cada uno de los seis órdenes. [Permutaciones y control sin diálogo](../research/runs/tesis_meal_lora_varied_v1/SHUFFLE.json).

```bash
python -m research.tesis_meal_varied --output research/runs/mi_comparacion_menu_variado
python -m research.tesis_meal_varied_audit research/runs/mi_comparacion_menu_variado
python -m research.tesis_meal_shuffle_check research/runs/mi_comparacion_menu_variado
```

### QA con erratas y rectificaciones

Dos agentes probaron los perfiles previos en 14 conversaciones nuevas de tres o cuatro turnos, a través de la API del chat y con el verificador apagado. Los dos perfiles usaban el mismo adaptador SFT, seleccionado antes de continuar el entrenamiento. Cada perfil acertó 23 de 45 turnos y recomendó ingredientes descartados en 21. Se observaron fallos al acumular vetos, retirar un veto previo, distinguir instrucciones para comensales distintos y reconocer que se habían agotado los platos. [Trazas de erratas](../research/data/meal_qa_typos.json) · [Trazas de rectificaciones](../research/data/meal_qa_corrections.json).

El nuevo entrenamiento incluyó todos los prefijos de las 11 conversaciones con fallos y de 22 conversaciones adicionales escritas por los agentes. Cada prefijo se etiqueta según los ingredientes compatibles con las exclusiones vigentes. La aclaración cuenta como válida cuando falta un ingrediente indispensable o el menú se agota. Se mezclaron 107 prefijos de QA y 107 casos de entrenamiento del banco anterior. Los menús rotaron y ambas ramas recibieron las mismas presentaciones y pasos de optimización desde los mismos pesos LoRA. SFT maximiza la probabilidad conjunta de las letras válidas, normalizada sobre el vocabulario completo. GRPO muestrea ocho elecciones entre las seis acciones del menú y compara sus recompensas dentro del grupo. Ambas ramas se regularizan respecto al mismo SFT inicial. [Plan](../research/runs/tesis_meal_qa_v1/PLAN.json) · [Implementación](../research/tesis_meal_qa.py).

Validación eligió entre el inicio y las tres épocas usando ocho conversaciones independientes y una muestra de validación del banco previo. La regla exige conservar el acierto histórico dentro de tres puntos porcentuales del SFT inicial. Entre los candidatos que cumplen, prioriza el acierto de QA en seis órdenes, los conflictos, las aclaraciones omitidas y el acierto histórico. Después de congelar los pesos, se evaluaron 14 conversaciones reservadas de un tercer agente, con 43 turnos. Para el banco nuevo, el catálogo de cinco platos es el mismo que se usa en el chat. El banco previo aporta un control de regresión con otros platos. Las seis rotaciones permiten inspeccionar sensibilidad al orden y cada conversación sigue siendo una sola unidad. [Selección](../research/runs/tesis_meal_qa_v1/SELECTION.json) · [Conversaciones reservadas](../research/data/meal_qa_reserved.json).

La prueba usa una semilla y conversaciones sintéticas en español. Mide elección de platos y aclaración dentro de la misma conversación, con las respuestas anteriores del usuario como contexto. La memoria persistente se evalúa en los experimentos longitudinales. Los diálogos que motivaron el ajuste sirven para medir si reaparecen los fallos observados. Los diálogos reservados permiten medir transferencia a formulaciones nuevas con el menú del laboratorio.

| Adaptador | Aciertos en el orden del chat | Turnos con ingredientes descartados | Conversaciones completas | Aciertos en seis órdenes | Aciertos en el banco previo |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT inicial | 20/43 | 22/43 | 1/14 | 126/258 | 817/900 |
| SFT adicional, época 1 | 29/43 | 11/43 | 6/14 | 184/258 | 857/900 |
| SFT + GRPO, época 3 | **33/43** | **9/43** | 6/14 | **202/258** | **870/900** |

La tabla usa inferencia en CUDA. GRPO superó a SFT adicional en 9,3 puntos porcentuales en el orden del chat y 7,0 puntos contando los seis órdenes. Ambos resolvieron 9 de los 14 turnos finales. SFT resolvió 4 conversaciones en todos sus turnos y órdenes; GRPO resolvió 2. Cuando hacía falta aclarar, SFT omitió 5 de 5 aclaraciones en el orden del chat y GRPO omitió 4. En el banco previo, SFT tuvo 22 conflictos y GRPO 27, aunque GRPO obtuvo más aciertos totales. El remuestreo emparejado de la auditoría agrupa por conversación y promedia familias. La incertidumbre de las 14 historias queda condicionada a la semilla seleccionada. [Resultados](../research/runs/tesis_meal_qa_v1/RESULTS.json) · [Auditoría](../research/runs/tesis_meal_qa_v1/AUDIT.json).

La repetición por la API del chat en CPU produjo 129 decisiones directas con el verificador apagado. El SFT previo acertó 22/43 turnos, SFT adicional 29/43 y GRPO 33/43. Los conflictos fueron 20, 11 y 9 y las conversaciones completas 2, 6 y 6. La diferencia de GRPO fue de +9,3 puntos frente a SFT adicional, con un intervalo exploratorio del 95 % de −7,0 a +27,3 puntos al remuestrear 10.000 veces las conversaciones emparejadas con semilla 417 y recalcular la proporción de turnos correctos. CPU y CUDA usaron el mismo orden de menú en cada turno. La ejecución de CPU usa FP32 sobre el mismo modelo base redondeado a BF16 y procesa un turno a la vez. Algunas elecciones cambiaron entre CPU y CUDA. [Traza inicial en CPU](../research/runs/tesis_meal_qa_v1/reserved-http-baseline.jsonl) · [Traza de las dos continuaciones en CPU](../research/runs/tesis_meal_qa_v1/reserved-http-adapted.jsonl).

GRPO permite penalizar más un plato incompatible o una aclaración omitida (−1) que una aclaración innecesaria (−0,35). Una elección válida recibe +1. La ventaja se calcula dentro de las ocho opciones muestreadas, se reutiliza durante dos actualizaciones con recorte de la razón de probabilidades y se regulariza hacia el SFT congelado. Hubo retornos distintos en 390 de 963 grupos. Esta prueba optimiza una decisión por prefijo y mide el efecto de ese crédito inmediato. En una rectificación que volvía a permitir tomate y mantenía la exclusión de papa y arroz, solo GRPO eligió la ensalada compatible. Ante «sin papa, sin arros y sin pepno» con el menú agotado, ambas continuaciones recomendaron un plato incompatible. [Conversaciones y trazas](../research/data/meal_qa_reserved.json) · [Decisiones de SFT](../research/runs/tesis_meal_qa_v1/sft-reserved.jsonl) · [Decisiones de GRPO](../research/runs/tesis_meal_qa_v1/grpo-reserved.jsonl).

En una prueba exploratoria por la API en CPU, con tres conversaciones adicionales preparadas tras iniciar el entrenamiento, el SFT inicial acertó 4 de 10 turnos, SFT adicional 6 y GRPO 5. Ambos ajustes acertaron los tres turnos de la conversación con autocorrecciones. Ambos fallaron las dos ocasiones en que se agotó el menú. Los tres modelos recibieron el mismo orden de opciones en cada turno. [Casos exploratorios](../research/data/meal_qa_exploratory.json) · [Traza inicial](../research/runs/tesis_meal_qa_v1/exploratory-http-baseline.jsonl) · [Traza de las dos continuaciones](../research/runs/tesis_meal_qa_v1/exploratory-http-adapted.jsonl).

```bash
python -m research.tesis_meal_qa --device cuda --output research/runs/mi_comparacion_erratas
python -m research.tesis_meal_qa_audit research/runs/mi_comparacion_erratas
```

Con los perfiles publicados en el servidor local, puedes volver a medir el recorrido del chat y guardar la traza en un archivo nuevo.

```bash
python -m research.tesis_meal_qa_http \
  --model comidas-variado-sft --model comidas-variado-qa-sft --model comidas-variado-qa-grpo \
  --output research/runs/tesis_meal_qa_v1/reserved-http-repeticion.jsonl
```

### Aclaración cuando se agota el menú

La regresión de la captura incluía «ayer comi pollo no quiero pollo comer hoy recomiendame cosas para comer» y «solo por hoy». La expresión «recomiéndame cosas para comer» ahora entra en el flujo de recomendaciones. La conversación de la captura se incluyó en entrenamiento y se verifica como regresión. Otros agentes redactaron por separado 14 conversaciones de validación y 16 reservadas. El banco reservado tiene 52 turnos, 24 sin platos compatibles, 24 con opciones compatibles y 4 que piden identificar un ingrediente desconocido. Cada historia tiene entre tres y cuatro turnos.

Cada rama recibe los mismos 82 prefijos conversacionales y 48 ejemplos del banco anterior por semilla. SFT, GRPO con recompensa revisada y GRPO con recompensa anterior parten del mismo LoRA SFT. Se prueban las semillas 101, 202 y 303. Validación permite un deterioro máximo de cuatro puntos sobre el banco anterior y luego prioriza el promedio entre aclarar cuando se agota el menú y acertar cuando queda un plato. La semilla 101 quedó fijada en el plan para los dos perfiles del chat.

La recompensa anterior daba +1 a una elección válida, −1 al plato incompatible o a la aclaración omitida y −0,35 a la pregunta innecesaria. La revisada conserva +1 para una elección válida, usa −2 al proponer un plato con el menú agotado, −1,5 al omitir una aclaración por un ingrediente desconocido, −1,25 al preguntar de más y −1 al proponer un plato incompatible cuando hay opciones. Se probaron estos costes en exclusiones alimentarias simuladas. Los dos brazos GRPO muestrean doce acciones por estado y centran los retornos del grupo con el mismo divisor fijo. Esta escala conserva las diferencias de magnitud entre las penalizaciones. Los grupos que nunca prueban la aclaración ofrecen poca señal para aprender a elegirla en un menú agotado. [Plan](../research/runs/tesis_meal_clarification_v1/PLAN.json) · [Entrenamiento](../research/tesis_meal_clarification.py).

| Método · tres semillas | Aclaró sin plato | Acertó con plato disponible | Aclaró de más | Aciertos totales |
| --- | ---: | ---: | ---: | ---: |
| SFT | **46/72** | 35/72 | 35/72 | **90/156** |
| GRPO con recompensa revisada | 14/72 | 47/72 | 4/72 | 69/156 |
| GRPO con recompensa anterior | 20/72 | **53/72** | 3/72 | 80/156 |

La tabla usa el orden del chat en CUDA. Los adaptadores previos acertaron 0/24 aclaraciones por agotamiento cada uno. SFT logró la mayor tasa de aclaración en las tres semillas y también preguntó de más con mayor frecuencia. La recompensa revisada de GRPO perdió frente al control en las tres semillas para el objetivo de aclarar cuando se agota el menú. Con las seis rotaciones, SFT logró 560/936 aciertos, GRPO revisado 352/936 y el control 441/936. La diferencia entre variantes de GRPO en las rotaciones fue de −8,1 puntos, con IC 95 % de −11,7 a −4,8 al remuestrear conversaciones y semillas y promediar por familia. Las rotaciones y semillas son mediciones repetidas de las mismas 16 conversaciones. [Auditoría y contrastes por semilla](../research/runs/tesis_meal_clarification_v1/AUDIT.json).

En los checkpoints elegidos de GRPO revisado, 42 de los 145 grupos de menú agotado no muestrearon la aclaración. Además, en 34 grupos sí se muestreó aclarar y todas las opciones muestreadas obtuvieron la misma recompensa. Aumentar la penalización deja estos grupos sin ventaja de tarea. En los estados con algún plato compatible, el coste revisado de la pregunta innecesaria también queda por debajo del de elegir un plato incompatible (−1,25 frente a −1). La comparación mide el efecto de cambiar los costes en conjunto. [Grupos y recompensas](../research/runs/tesis_meal_clarification_v1/AUDIT.json).

En tres conversaciones exploratorias por la API, ambas con el menú del laboratorio y las decisiones directas, SFT de la semilla 202 acertó las cuatro aclaraciones por agotamiento y 2/6 turnos viables. GRPO acertó una aclaración por agotamiento y 4/6 turnos viables. La versión anterior de SFT acertó 1/4 y la de GRPO 0/4 cuando se agotó el menú. [Trazas de QA exploratorias](../research/runs/tesis_meal_clarification_v1/exploratory-http-adapted.jsonl).

La prueba por la API en CPU registró ocho perfiles con el verificador apagado. En las tres semillas, SFT acertó 91/156 turnos, 47/72 con el menú agotado y 35/72 con un plato viable. GRPO revisado acertó 69/156 turnos, 14/72 con el menú agotado y 47/72 con un plato viable. SFT preguntó de más en 34 turnos y GRPO en 4. Los adaptadores previos acertaron 0/24 aclaraciones por agotamiento cada uno. Antes del arreglo de enrutamiento, cada perfil desvió un turno al controlador de memoria al recibir «¿qué cena me propones?». Se contó como fallo. Tras ampliar el reconocimiento de las solicitudes de menú, los ocho perfiles tomaron una decisión directa y todos siguieron omitiendo la aclaración en ese turno. La regresión de la captura pasó en los dos perfiles del chat, con y sin verificar. [Repeticiones de enrutamiento y QA](../research/runs/tesis_meal_clarification_v1/HTTP_SUMMARY.json) · [Captura probada en navegador](../research/runs/tesis_meal_clarification_v1/capture-browser-101.jsonl).

```bash
python -m research.tesis_meal_clarification train --output research/runs/mi_prueba_aclaracion
python -m research.tesis_meal_clarification test --output research/runs/mi_prueba_aclaracion
python -m research.tesis_meal_clarification_audit research/runs/mi_prueba_aclaracion
```

### GRPO con exploración de las seis acciones

En el experimento de aclaración anterior, GRPO revisado llegó a muestrear la aclaración en 103 de 145 grupos de menú agotado de los checkpoints seleccionados. En 69 grupos hubo retornos distintos. La nueva variante de GRPO evalúa las seis acciones de cada menú durante el entrenamiento, incluida la aclaración, con un estado por prefijo de conversación.

Las tres ramas parten del mismo SFT previo dentro de cada semilla. Las continuaciones reciben los mismos diálogos, órdenes de menú y pasos de optimización. GRPO con exploración exhaustiva suma el objetivo recortado sobre las seis acciones y las pondera con la política de comportamiento congelada. El control estima el mismo objetivo con ocho acciones muestreadas. Ambos centran los retornos con la misma esperanza, usan la misma regularización hacia SFT y mezclan un 25 % de probabilidad uniforme durante el entrenamiento. En el chat, Qwen elige la letra con mayor puntuación entre las seis. [GRPO](https://arxiv.org/abs/2402.03300) · [Gradientes de política esperados](https://www.jmlr.org/papers/v21/18-012.html).

La recompensa es +1,5 por aclarar un menú agotado, +1 por un plato compatible o una aclaración sobre un ingrediente desconocido, −2,5 por un plato incompatible, −1,75 por omitir una aclaración sobre un ingrediente desconocido y −0,65 por preguntar cuando queda algún plato. La preferencia por una elección compatible se comprueba también en turnos con un único plato disponible. Los agentes escribieron 20 historias de entrenamiento con pares que cambian cuál ingrediente se permite, 12 de validación y 18 reservadas. Las 18 historias reservadas suman 60 turnos, con 28 menús agotados, 27 viables y 5 con un ingrediente desconocido. Los pesos de las semillas 101, 202 y 303 se congelan con validación antes de evaluar las historias reservadas. [Plan](../research/runs/tesis_meal_exploration_v1/PLAN.json) · [Banco reservado](../research/data/meal_exploration_reserved.json).

| Método · orden del chat en CUDA | Aciertos | Aclaró el menú agotado | Acertó con plato disponible | Platos incompatibles |
| --- | ---: | ---: | ---: | ---: |
| SFT adicional | 95/180 | 49/84 | **34/81** | 47 |
| GRPO con seis acciones por grupo | 88/180 | 46/84 | 30/81 | 53 |
| GRPO con mezcla uniforme y muestreo | **99/180** | **59/84** | 28/81 | **38** |
| SFT previo | 98/180 | 67/84 | 19/81 | 27 |
| GRPO previo | 62/180 | 21/84 | 30/81 | 98 |

La enumeración incluyó la aclaración y obtuvo recompensas distintas en los 1.206 grupos de entrenamiento. El muestreo incluyó la aclaración en 953 grupos y obtuvo recompensas distintas en 1.008. En estados de menú agotado, incluyó la aclaración en 583 de 591. El GRPO exhaustivo mejoró frente al GRPO anterior, cuyo entrenamiento, recompensa e inicio eran distintos. Con la comparación emparejada, el exhaustivo quedó 3,8 puntos por debajo de SFT en acierto por turno en el orden del chat, con un IC del 95 % de −10,8 a +1,0 puntos al remuestrear historias y semillas y promediar familias. SFT acertó 598/1.080 en los seis órdenes, el GRPO muestreado acertó 593 y el exhaustivo 565. En la semilla 101, los tres métodos retuvieron los mismos pesos iniciales. [Resultados](../research/runs/tesis_meal_exploration_v1/RESULTS.json) · [Auditoría](../research/runs/tesis_meal_exploration_v1/AUDIT.json).

Los agentes probaron los mismos turnos a través de la API con el verificador apagado. En CPU, SFT acertó 93/180 y GRPO exhaustivo 84/180, con 48 y 56 conflictos de ingredientes. En CUDA, la comparación por la API de SFT con GRPO muestreado dio 93/180 y 99/180 aciertos, respectivamente, con 49 y 37 conflictos. Al usar solo los checkpoints actualizados de las semillas 202 y 303, GRPO con muestreo acertó 65/120 y SFT 61/120, con 25 y 34 conflictos. El GRPO con muestreo acertó 39/56 menús agotados y SFT 29/56 en esas dos semillas. La semilla 101 tuvo los mismos pesos en los dos brazos y produjo pequeñas diferencias en CUDA; esas diferencias son de inferencia. Las comparaciones CPU y CUDA usan el mismo orden y se reportan con sus cifras propias. [QA de GRPO exhaustivo en CPU](../research/runs/tesis_meal_exploration_v1/HTTP_SUMMARY.json) · [QA de GRPO con muestreo en CUDA](../research/runs/tesis_meal_exploration_v1/SAMPLED_HTTP_CUDA_SUMMARY.json).

El chat ofrece el GRPO con muestreo y el SFT de la semilla 202, la primera con un checkpoint GRPO posterior al inicial según validación. Se eligió ese método para el chat por sus resultados en el orden de menú usado allí, con ambos brazos probados sobre el banco reservado. En la prueba del navegador con el verificador apagado, los dos modelos preguntaron al agotarse el menú y volvieron a preguntar cuando ya quedaba un plato. [QA en la interfaz](../research/runs/tesis_meal_exploration_v1/browser-sampled-regression.jsonl).

```bash
python -m research.tesis_meal_exploration train --output research/runs/mi_prueba_exploracion
python -m research.tesis_meal_exploration test --output research/runs/mi_prueba_exploracion
python -m research.tesis_meal_exploration_audit research/runs/mi_prueba_exploracion
```

## Reproducción

Usa un Python local con PyTorch y NumPy. El entrenamiento del selector se ejecuta en CPU y conserva las tres semillas, los retornos por grupo, las trayectorias y los pesos seleccionados en validación.

```bash
python -m research.tesis_interactive_train --output research/runs/mi_comparacion_interactiva
python -m research.tesis_interactive_train --output research/runs/mi_comparacion_interactiva --audit
python -m research.tesis_interactive_reward_audit --output reports/mi_auditoria_recompensa
python -m research.tesis_interactive_train --bank-seed 684023 --warmup-steps 1 \
  --output research/runs/mi_comparacion_supervision_reducida
python -m research.tesis_interactive_guard_audit \
  --run research/runs/tesis_interactive_v1 \
  --output reports/mi_auditoria_verificador_interactivo.json
python -m research.tesis_grpo_branches --output research/runs/mi_comparacion_por_ramas
python -m research.tesis_grpo_branches --output research/runs/mi_comparacion_por_ramas --audit
```

```bash
python -m research.tesis_rl_signal \
  --run research/runs/tesis_comparison_full_v1 \
  --output reports/mi_auditoria_senal_grpo.json
```
