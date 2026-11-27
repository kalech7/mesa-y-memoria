# Adquisición de memoria en tres visitas

La extensión permite usar hasta dos preguntas entre tres visitas. Una réplica parcial puede consumir la primera pregunta y dejar el alcance sin resolver. En la segunda visita se revela si el plan cambió. Conservar presupuesto permite volver a aclarar en la tercera, cuando el recuerdo puede afectar dos consultas con fechas distintas. El programa ejecuta las escrituras en SQLite y vuelve a abrir la memoria para recuperar los datos. El analizador público de texto estima qué dato falta y acota las acciones disponibles a preguntas pertinentes y cierres. [Entorno](../research/tesis_extended_env.py) · [Banco de desarrollo](../research/tesis_extended_bank.py)

## Reservado y criterio de selección

Un autor por IA separado redactó **12 raíces y 36 visitas**, con **16 mundos ejecutables**. Dos instancias de IA revisaron los paquetes públicos por separado. La rúbrica distingue la respuesta actual, una escritura autorizada en esa visita y los recuerdos previos aplicables. La adjudicación fija los extremos inclusivos de los intervalos, los ámbitos contextuales y el límite de las excepciones de sesión. La validez del corpus en usuarios reales queda fuera de esta revisión sintética. [Adjudicación](../reports/tesis_extended_reserve_review/ADJUDICACION.md) · [Sello](../reports/tesis_extended_reserve_review/SEAL.json)

En desarrollo, una política de un paso decide valorando la acción actual seguida de cierres sin más preguntas. El planificador estima el valor de conservar opciones en visitas posteriores. Ambos ajustan su tabla en TRAIN. Los retornos que reciben SFT y GRPO se calculan para acciones alternativas desde el mismo prefijo público, con los mismos mundos compatibles y una continuación fija. El modelo recibe el diálogo y un resumen derivado de la evidencia pública. La comparación por ramas evalúa aprendizaje sobre esos retornos registrados. [Preparación del currículo](../research/tesis_extended_analysis.py) · [Entrenador emparejado](../research/tesis_extended_experiment.py)

El punto de partida hereda el adaptador Qwen2.5-1.5B que resolvía la visita actual en el piloto de dos sesiones. Antes de las tres parejas de entrenamiento se mide su acierto actual en el nuevo entorno. Un ajuste común con retornos de acierto actual de TRAIN está previsto si queda más de dos puntos por debajo de la regla. Las semillas 101, 202 y 303 cambian el orden de los lotes. Cada pareja comparte adaptador, ramas, tokens, lotes y actualizaciones. Se elige por utilidad en validación, con guardas de acierto actual y reutilización sin respaldo. [Entrenamiento multisemilla](../research/tesis_extended_experiment.py)

La evaluación reservada verifica primero las huellas del plan, el código, el banco, el planificador y todos los checkpoints seleccionados. Después abre los mundos reservados y puntúa los modelos sin entrenamiento adicional. Se promedian los mundos dentro de cada raíz. Los intervalos emparejados por raíz dependen de las semillas observadas y del tamaño del banco. [Evaluación bloqueada](../research/tesis_extended_score_reserved.py)

## Margen observado y punto de partida

En las 12 raíces de validación, el planificador obtuvo 78,98 % de acierto futuro con 1,75 preguntas por historia. La política de un paso obtuvo 61,86 % con 2,00 preguntas. Ambas obtuvieron 100 % de acierto actual. La diferencia de utilidad fue +0,2771, con intervalo emparejado del 95 % de +0,2668 a +0,2874. En promedio, la política de un paso gastó todo el presupuesto antes de la tercera visita. El planificador hizo 0,55 preguntas en esa última visita. El diseño asigna dos consultas futuras a la tercera preferencia, lo que hace explícito el costo de agotar antes el presupuesto. [Diagnóstico CPU](../reports/tesis_extended_qwen_a100_v3/CPU_MARGIN.json)

El adaptador heredado pasó el control de acierto actual con 100 %, frente al mínimo de 98 %. Las tres semillas reutilizaron ese adaptador sin otra fase de ajuste común. El banco compartido de optimización contiene 240 grupos públicos y da peso propio a las decisiones después de una réplica parcial. [Control de competencia](../reports/tesis_extended_qwen_a100_v3/shared-competence/PREFLIGHT.json) · [Ramas](../reports/tesis_extended_qwen_a100_v3/BRANCHES.json)

SFT obtuvo 75,00 % y GRPO 72,50 % de acierto futuro en validación. En el reservado, ambos empataron en 57,41 %, con la regla en 61,11 % y el adaptador inicial en 47,22 %. El entorno de tres visitas ofrece un margen medible para planificar. Con redacciones nuevas, el menú público habilitó 7 de las 9 oportunidades de aclaración anotadas. [Informe final](../reports/tesis_extended_qwen_a100_v3/INFORME.md) · [Cobertura de acciones](../reports/tesis_extended_reserve_review/MASK_AUDIT.json)

## Separación entre política y memoria

La auditoría fija las acciones y cruza un escritor fiel o con caducidad extendida, un lector fiel o que omite el límite de vigencia y el veto de escritura activado o desactivado. El control completo dio 216 de 216 recuperaciones sustentadas. El escritor defectuoso sin veto obtuvo 117 de 216 con un lector fiel. El veto elevó esa cifra a 171 de 216 al bloquear las 27 propuestas alteradas. Con escritor fiel, el lector defectuoso citó un recuerdo vencido en 90 de 216 consultas. El filtro de vigencia se abstuvo ante esas citas, conservando las 126 respuestas sustentadas. [Informe del experimento factorial](../reports/tesis_memory_faults_v3/INFORME.md)

El replay semántico del reservado ejecuta tres cierres sin aclarar. Obtuvo 26/36 respuestas exactas, 31/36 escrituras exactas y 32/36 recuperaciones exactas. La respuesta condicional correcta del caso con lugar ambiguo se cuenta en el indicador semántico. El acierto sustentado estricto conserva su definición de respuesta directa y dio 25/36. [Trazas y etiquetas](../reports/tesis_extended_reserve_review/SEMANTIC.json)

## Ejecución en la A100

El trabajo terminó en el servidor `servidor-tesis`, en una sesión `tmux` desacoplada. Se pueden inspeccionar el estado y el registro desde VS Code por SSH.

```bash
cd ~/tesis_clarificacion_longitudinal
cat reports/tesis_extended_qwen_a100_v3/STATUS.json
tail -n 25 logs/qwen_extended_v3.log
```

Para reproducir en una carpeta nueva, ejecuta lo siguiente dentro de una sesión del servidor. Tras una interrupción, conserva la carpeta y añade `--resume` al comando de entrenamiento. El comando de evaluación se lanza tras guardar `SELECTION_LOCK.json`.

```bash
cd ~/tesis_clarificacion_longitudinal
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PYTHON="$HOME/.local/share/tesis-clarificacion/a100/bin/python"
RUN=reports/mi_qwen_tres_visitas
"$PYTHON" -u -m research.tesis_extended_experiment --output "$RUN" \
  --reserve-seal reports/tesis_extended_reserve_review/SEAL.json \
  --workers 16 --epochs 2 --batch-size 8
"$PYTHON" -u -m research.tesis_extended_score_reserved --run "$RUN" \
  --reserved research/data/tesis_extended_heldout_v1 --workers 12
```
