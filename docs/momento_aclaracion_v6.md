# Cuándo gastar la pregunta

El estudio v6 pregunta si Qwen aprende a **elegir la visita en la que una aclaración ayuda más**. Para comprobarlo, dos personas simuladas hacen exactamente los mismos pedidos y anuncian agendas distintas. Se ejecutan ambas decisiones posibles desde el mismo punto del diálogo y se leen los recuerdos que realmente quedan en SQLite. [Simulador](../research/tesis_timing_v6_env.py) · [Informe de la corrida en la A100](../reports/tesis_timing_qwen_a100_v6/INFORME.md)

## Probar el modelo y ver los recuerdos

Con VS Code conectado por Remote SSH a `servidor-tesis`, abre `~/tesis_clarificacion_longitudinal` y reenvía el puerto **8767** en la pestaña **Puertos**. En el chat puedes elegir SFT o RL (GRPO), enviar tus propios mensajes y abrir **Memoria de esta conversación** para revisar una preferencia y sus citas. También puedes llegar desde el marcador anterior si reenvías **8766**. [Chat de alimentación](http://127.0.0.1:8767/chat) · [Marcador anterior](http://127.0.0.1:8766/#perfiles)

En **Simulación guiada**, ejecuta una prueba y selecciona la visita para ver los mensajes, la pregunta que se gastó y el contenido de la memoria SQLite en ese punto. Abre la evidencia del recuerdo para leer la declaración y la aclaración originales. La tercera visita muestra las consultas y qué recuerdo respaldó cada respuesta. [Probar la simulación](http://127.0.0.1:8767/)

Para ver la diferencia entre un hábito y una excepción, fuerza la pregunta en la primera visita, elige que la persona solo quiere sopa en esa visita y avanza a la segunda. El registro conserva la cita y su vigencia termina al cambiar de visita. Después prueba la misma agenda dejando que Qwen elija. Cada ejecución simula su propia historia. Para recuperar una conversación de v5, abre el archivo desde el mismo navegador y puerto local que utilizaste antes. [Archivo de conversaciones v5](http://127.0.0.1:8766/archivo#perfiles)

El servicio v6 escucha en el servidor y usa la sesión tmux `laboratorio-momento-v6`. Al cerrar el equipo se cierra el túnel de ese equipo. Puedes volver a abrirlo desde VS Code. Si reinicias **el servidor** y necesitas levantar el proceso desde la terminal remota:

```bash
cd ~/tesis_clarificacion_longitudinal
mkdir -p logs
tmux -L tesis-clarificacion has-session -t laboratorio-momento-v6 2>/dev/null ||
  tmux -L tesis-clarificacion new-session -d -s laboratorio-momento-v6 \
    'cd ~/tesis_clarificacion_longitudinal && CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false ~/.local/share/tesis-clarificacion/a100/bin/python -u -m web_demo.timing_service --port 8767 >> logs/laboratorio_momento_v6.log 2>&1'
```

## Dos personas que hoy quieren sopa

En un par de entrenamiento, ambas personas dicen «Hoy quiero sopa para cenar». La primera añade «Solo tengo una casilla para el desayuno. Para la cena estoy armando el menú de toda la semana». La segunda añade «Solo tengo una casilla para la cena. Para el desayuno estoy armando el menú de toda la semana». En la visita siguiente ambas dicen «Anota pan con queso para el desayuno de esta vez». Hoy se puede servir sopa en los dos casos. Lo que falta saber es si la persona autoriza recordarla en otras visitas.

Con una sola pregunta, el asistente puede aclarar hoy el alcance de la sopa o esperar y aclarar mañana el del pan con queso. Si la persona confirma un hábito, su frase se guarda con la preferencia. Si limita su elección a esa visita, el recuerdo caduca. En la tercera visita se confirma el calendario. Después se consultan seis servicios, cinco de una categoría y uno de la otra. En el simulador, la previsión de agenda se cumple con probabilidad 0,80. El alcance de cada una de las dos preferencias es independiente de la agenda y tiene probabilidad 0,50 de ser habitual. [Banco de pares y pruebas de calendario](../research/tesis_timing_v6_bank.py) · [Pruebas de los pares](../research/tests/test_tesis_timing_v6_bank.py)

## Lo que comprueba la CPU

En validación se comparan las dos ramas de cada historia, ponderando los ocho mundos posibles. La tabla expresa resultados esperados con **una pregunta por historia**. El acierto futuro incluye dos resultados distintos. Uno es contestar con el plato correcto y una cita que lo respalda. El otro es abstenerse cuando el usuario solo había elegido algo para aquella ocasión.

| Elección de visita | Respuesta futura con plato respaldado | Abstención correcta | Acierto futuro total |
| --- | ---: | ---: | ---: |
| Aclarar el menú que se prevé repetir | 35 % | 50 % | 85 % |
| Aclarar siempre en la primera visita o siempre en la segunda | 25 % | 50 % | 75 % |
| Aclarar el menú que se prevé usar una sola vez | 15 % | 50 % | 65 % |

El 35 % de la primera fila surge de cinco ocasiones útiles si se cumple el plan y una si cambia, con un hábito confirmado en la mitad de los mundos. En ausencia de hábito, abstenerse es la respuesta correcta. El planificador que usa la agenda pública de validación alcanza el 85 %. Siempre preguntar en la primera visita resuelve bien una de las dos historias del par. Con dos preguntas disponibles se puede aclarar en ambas visitas y comprobar por separado el efecto de la escasez. Estas cifras describen el simulador y los controles de CPU. El informe de la A100 contiene las mediciones del modelo. [Auditoría CPU](../research/tesis_timing_v6_env.py) · [Condiciones para iniciar el entrenamiento](../research/tesis_timing_v6_experiment.py)

## Cómo se compara a Qwen

Cada método recibe **diez preguntas para veinte historias simultáneas**, con un máximo de una pregunta por historia. En la primera visita la asignación considera los veinte primeros mensajes. El umbral y la puntuación de cada petición determinan cuáles obtienen pregunta. Las preguntas restantes se asignan en la segunda visita entre las historias que esperaron. El gasto se comprueba por paquete. Por eso en una comparación con cupo exacto puede gastarse la última pregunta incluso cuando el modelo tenía poca confianza en preguntar.

El diagnóstico por par responde otra pregunta. Con una oportunidad por persona, ¿eligió el modelo la primera visita para la agenda de cenas y la segunda para la agenda de desayunos? Se registran tanto la elección con una pregunta exigida como la secuencia que el modelo ejecutaría libremente. En esta última, esperar dos veces cuenta como haber terminado sin preguntar. Una puntuación alta en el paquete también puede lograrse seleccionando muy bien qué diez historias atender en la primera visita. El diagnóstico individual permite distinguir ese resultado de esperar y preguntar después. [Asignación y métricas](../research/tesis_timing_v6_eval.py)

Se entrena Qwen2.5-1.5B-Instruct con LoRA, tres semillas y dos rondas. Se prueban ambas acciones en los puntos de entrenamiento. Las decisiones siguientes se muestrean con el adaptador común en la primera ronda y con el GRPO longitudinal de la ronda anterior en la segunda. SFT recibe esas mismas ramas y aprende la acción con mayor retorno. GRPO longitudinal utiliza también los resultados futuros para actualizar su política. Otros dos ajustes de GRPO permiten interpretar la comparación. Uno recibe solo la recompensa de la visita actual y el costo de preguntar. El otro conserva las continuaciones generadas con el adaptador inicial. [Entrenamiento de las cuatro políticas](../research/tesis_timing_v6_train.py)

Entrenamiento y validación tienen 30 pares cada uno. El reservado contiene 80 pares en ocho paquetes de 20 historias. Las particiones tienen platos y redacciones diferentes. Se congelan los pesos y el umbral elegidos en validación antes de evaluar el reservado. La comparación principal usa el acierto futuro al mismo gasto y comprueba el acierto actual y la reutilización sin respaldo. El intervalo emparejado del 95 % remuestrea paquetes completos y corresponde a las tres semillas observadas. Los casos proceden de plantillas sintéticas en español y las probabilidades de respuesta son supuestos del entorno. La tarea mide dos oportunidades de aclaración y seis consultas sobre preferencias alimentarias. [Protocolo de selección y evaluación](../research/tesis_timing_v6_experiment.py)

## Examinar la decisión sobre sopa en la corrida

Desde la carpeta del proyecto en el servidor, el siguiente comando permite examinar el par de entrenamiento con el modelo ajustado. El artefacto `DEMO_PAR.json` queda en el directorio de la corrida y permite revisar las acciones junto con las frases que recibió Qwen.

```bash
~/.local/share/tesis-clarificacion/a100/bin/python -m research.tesis_timing_v6_demo \
  --run reports/tesis_timing_qwen_a100_v6
```

## Reproducir desde VS Code en la USFQ

Conecta VS Code por Remote SSH a `servidor-tesis`, abre `~/tesis_clarificacion_longitudinal` e inicia una terminal remota. El siguiente ejemplo crea un nombre de sesión y un directorio nuevos con la fecha y hora del servidor. Usa la GPU 0 con el entorno de Python instalado para la tesis. El lanzador prepara el estudio, entrena, selecciona y evalúa dentro de tmux.

```bash
cd ~/tesis_clarificacion_longitudinal
mkdir -p logs
timing_id="v6-$(date +%Y%m%d-%H%M%S)"
tmux -L tesis-clarificacion new-session -d -s "$timing_id" \
  "CUDA_VISIBLE_DEVICES=0 bash scripts/run_timing_v6_a100.sh reports/$timing_id > logs/$timing_id.log 2>&1"
echo "$timing_id"
```

Consulta el progreso con estos comandos. Después de reconectar, asigna a `timing_id` el nombre que imprimió la terminal. El proceso de entrenamiento continúa en el servidor al cerrar VS Code o apagar el PC. La carpeta de la corrida conserva la auditoría CPU, las rondas de entrenamiento, la selección y las mediciones reservadas.

```bash
tmux -L tesis-clarificacion list-sessions
tail -n 20 "logs/$timing_id.log"
ls "reports/$timing_id"
```

Al terminar, `RESERVED_RESULTS.json` contiene las mediciones y la huella de la selección congelada en `LOCK.json`. Si una corrida se interrumpe, vuelve a ejecutar el lanzador con **el mismo directorio** para recuperar los checkpoints y las rondas completadas. [Lanzador de la A100](../scripts/run_timing_v6_a100.sh)
