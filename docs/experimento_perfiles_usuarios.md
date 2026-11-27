# Aclaración con perfiles de usuario

Cada historia representa tres visitas de una persona ficticia. Hay seis familias de comportamiento que cruzan disponibilidad, tipo de respuesta y estabilidad de los planes. Un usuario puede contestar en el momento, dejar un dato pendiente, negarse a precisarlo o enviar la respuesta en una visita posterior. La respuesta diferida solo aparece si la política hizo antes la pregunta correspondiente. Preguntar consume una de las una o dos preguntas de que dispone la historia, incluso cuando la respuesta aún no ha llegado.

El entorno distingue una excepción de sesión, un intervalo con fecha de fin, una preferencia restringida a un lugar y un cambio habitual. Las historias incluyen rectificaciones y citas de otras personas. La política elige cuándo aclarar o cerrar. El escritor y el lector de SQLite son comunes a todos los brazos. El asistente ve el diálogo, el presupuesto, la memoria respaldada y los indicios estimados a partir del texto. El simulador conserva los tipos de perfil, los planes futuros y las respuestas latentes para reproducir las ramas. [Banco de desarrollo](../research/tesis_profiles_bank.py) · [Entorno y observación pública](../research/tesis_profiles_env.py)

## Aporte frente a trabajos relacionados

ATRBench ya mide la adquisición de preferencias para tareas futuras. PERMA ya estudia cómo evolucionan las preferencias por eventos y usa variación lingüística. MCB evalúa si se debe persistir, limitar al contexto, verificar o aclarar un recuerdo cuando se conoce el contexto de reutilización. [ATRBench](https://arxiv.org/html/2605.28108v3) · [PERMA](https://arxiv.org/abs/2603.23231) · [MCB](https://aclanthology.org/2026.nora-1.2/)

Este estudio mide el costo de oportunidad de la aclaración. Una pregunta gasta parte de un presupuesto compartido y puede cambiar qué evidencia se recibe en días posteriores. Al reproducir las alternativas desde el mismo prefijo y ejecutar sus efectos sobre SQLite, puede atribuirse el resultado a decisiones distintas de aclaración bajo la misma memoria. La comparación con SFT y con una recompensa sin recuperación futura permite medir por separado el valor de optimizar las decisiones siguientes y el de incentivar el acierto posterior.

## Dos ramas que se pueden reproducir

Dos mundos de TRAIN empiezan con el mismo diálogo. La persona tiene prisa y expresa que prefiere menestra de fréjol para la cena. Si el asistente pregunta por el alcance, la persona retoma el chat y confirma que es un hábito. La escritura pasa a SQLite con esa cita. El plan que se revela más tarde determina si se necesita otra pregunta sobre la bebida en la tercera visita. En ambas ramas, después de la primera decisión se aplica la misma regla pública.

| Plan que se revela en la segunda visita | Preguntar en la primera visita | Reservar la pregunta |
|---|---|---|
| Se cancela el encuentro | 4/4 consultas futuras sustentadas con 2 preguntas | 3/4 con 1 pregunta |
| Cambia el encuentro | 2/4 consultas futuras sustentadas con 2 preguntas | 3/4 con 2 preguntas |

El acierto actual fue 3/3 en las cuatro ejecuciones. Al ponderar el conjunto de mundos de esa historia, reservar la pregunta funciona mejor bajo la continuación fijada.

En otro mundo, la primera pregunta obtiene una negativa. Si se reserva, el asistente puede preguntar en la tercera visita en qué lugar se debe aplicar la recomendación y responder con la preferencia para casa. Con la misma continuación pública se aciertan 3/3 consultas actuales al reservar y 2/3 al preguntar primero. Ambas ramas consumen dos preguntas y aciertan 3/4 recuperaciones futuras. Son contrastes del simulador, seleccionados para ilustrar el mecanismo.

Los mundos de recontacto son `vw47492fef56ad9b380bc3613d` y `vw6f3e808ce5fa5c15643ff0e2`. El ejemplo de lugar usa `vwbfe68d56c289044f4a9cd866`. Pueden reproducirse con `build_development()` y `enumerate_world(story)`, forzando `PROCEED` o `TEMPORAL_SCOPE` como primera acción y usando `rule_action` en `traverse` para la continuación. [Enumerador](../research/tesis_profiles_env.py) · [Recorrido de ramas](../research/tesis_shared_budget_planner.py)

## Particiones y revisión

El banco de desarrollo usa 24 historias de TRAIN y 12 de validación, con identificadores y redacciones de apertura separados. Los mundos de una historia representan posibilidades que el asistente todavía desconoce. Comparten la observación inicial y se ponderan dentro de su historia. Un agente separado redactó 16 historias reservadas con 48 visitas y 22 mundos ejecutables. Dos agentes revisaron por separado la respuesta que permite el diálogo, la escritura nueva que autoriza y qué recuerdos previos aplican. La población y las anotaciones son sintéticas. [Paquete reservado](../research/data/tesis_profiles_heldout_v1/README.md) · [Sello del corpus](../reports/tesis_profiles_review/SEAL.json) · [Adjudicación](../reports/tesis_profiles_review/ADJUDICACION.md)

## Qué permite probar RL

La pregunta inicial cambia tanto el presupuesto de mañana como la evidencia que podría aparecer mañana. La decisión depende de la respuesta que se espera de la persona, de las dudas que compiten por la pregunta y de qué puede resolver el asistente ahora. El contraste tabular reutiliza exactamente las mismas transiciones de TRAIN en tres brazos. SFT supervisa la mejor acción si las visitas posteriores siguen una regla fija. RL con Bellman estima el valor futuro de las transiciones públicas y vuelve a optimizar las acciones de las visitas siguientes. El control inmediato conserva esa optimización y elimina la recompensa por recuperación futura. En estados públicos nuevos se aplica la misma regla de respaldo y se registra su frecuencia. [Aprendizaje tabular](../research/tesis_profiles_learning.py)

La comparación de Qwen2.5-1.5B usa un adaptador inicial compartido y verifica su acierto actual. SFT y GRPO por ramas ven los mismos prefijos, alternativas, retornos exactos, tokens y actualizaciones en las semillas 101, 202 y 303. El retorno de cada alternativa usa una continuación fija común basada en la política tabular. SFT imita las mejores alternativas y GRPO pondera las alternativas por sus ventajas relativas. Es una comparación con retornos registrados. El modelo selecciona acciones de un menú público y la memoria común ejecuta las operaciones. [Entrenamiento emparejado](../research/tesis_profiles_study.py) · [Objetivos de optimización](../research/tesis_shared_budget_qwen_matched.py)

La selección usa validación con guardas de acierto actual y reutilización sin respaldo. El criterio primario es la utilidad longitudinal. Para sostener una ventaja en el reservado, el límite inferior del intervalo emparejado por historia debe quedar por encima de cero, con hasta dos puntos de pérdida en acierto actual, hasta 0,10 preguntas adicionales en promedio y la misma tasa o menor de reutilización sin respaldo. Se publican también el acierto futuro sustentado y las preguntas por visita. Se fijan hashes del código, del banco y de los checkpoints elegidos antes de evaluar el reservado. [Evaluación reservada](../research/tesis_profiles_score.py)

## Reproducción en el servidor

Desde VS Code conectado por SSH a `servidor-tesis`, abre `~/tesis_clarificacion_longitudinal`. La ejecución principal terminó en la sesión desacoplada `qwen-profiles-v4`. [Informe de resultados](../reports/tesis_profiles_qwen_a100_v4/INFORME.md)

```bash
cd ~/tesis_clarificacion_longitudinal
cat reports/tesis_profiles_qwen_a100_v4/STATUS.json
tail -n 25 logs/qwen_profiles_v4.log
```

Para reproducir en una carpeta nueva, abre una sesión `tmux` en el servidor y ejecuta los comandos con una A100 disponible. Si el entrenamiento se interrumpe, recupera la misma carpeta con los mismos argumentos y añade `--resume`. Evalúa el reservado después de que exista `SELECTION_LOCK.json`.

```bash
cd ~/tesis_clarificacion_longitudinal
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
tesis_python="$HOME/.local/share/tesis-clarificacion/a100/bin/python"
tesis_run=reports/mi_qwen_perfiles
"$tesis_python" -u -m research.tesis_profiles_study --output "$tesis_run" \
  --reserve-seal reports/tesis_profiles_review/SEAL.json \
  --workers 16 --epochs 2 --batch-size 8
"$tesis_python" -u -m research.tesis_profiles_score --run "$tesis_run" \
  --reserved research/data/tesis_profiles_heldout_v1 --workers 12
```

Para reproducir el diagnóstico exploratorio que compara los pesos entrenados en la última época de ambos brazos, ejecuta después `research.tesis_profiles_last_epoch` con los mismos argumentos `--run` y `--reserved`.
