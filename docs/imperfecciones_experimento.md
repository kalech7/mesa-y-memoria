# Experimento de demostraciones imperfectas y fallos de información

La recomendación del tutor se implementa en dos experimentos separados. El
primero estudia SFT seguido de RL cuando algunas demostraciones son imperfectas.
El segundo mide cómo una política fija responde a información que no llega o a
usuarios que aclaran parcialmente. Las frecuencias son decisiones experimentales;
su parecido con usuarios reales necesita validación independiente.

## Qué se introduce en SFT

`research.imperfect_sft` produce siete ejemplos por episodio, como el productor
existente. Selecciona por semilla una proporción exacta de raíces de TRAIN:
0 %, 10 % o 30 % por defecto, con redondeo al entero más cercano. Todos los
mundos de una raíz permanecen juntos. La proporción indica **historias con una
demostración natural imperfecta**, y no la tasa de error esperada del modelo ni
la proporción de todos los targets modificados. El registro publica ambas tasas.

Los errores posibles son omitir una aclaración, preguntar innecesariamente,
guardar un alcance temporal/contextual como permanente o no guardar una
actualización sustentada. El formato estructurado sigue siendo válido. Cuando
cambia SELECT, se ejecuta otra vez la interacción para obtener observaciones
posteriores coherentes; no se conserva una respuesta a una pregunta inexistente.
Las cuatro demostraciones procedentes de rutas forzadas conservan sus targets
correctos. La variante de tasa cero coincide con el productor anterior.

Cada paquete conserva `clean_examples.jsonl`, `examples.jsonl`, `mutations.json`
y un manifiesto con hashes. Las copias opcionales de validación y evaluación
son idénticas byte por byte. Se rechazan historias equivalentes entre particiones,
incluso con IDs diferentes. Los golds y el evaluador conservan su semántica.

## Qué se introduce en el entorno

`ImperfectionConfig` permite configurar:

| Parámetro | Efecto |
|---|---|
| `memory_omission_rate` | Algunos recuerdos almacenados no se entregan en S2. |
| `unavailable_reply_rate` | El usuario devuelve una falta explícita de información. |
| `partial_reply_rate` | Se intenta entregar solo un hecho contextual seguro y literal. |
| `seed` | Fija los fallos compartidos por todos los métodos. |

Las respuestas parciales se limitan a hechos autónomos de contexto respaldados
por la tabla de respuestas. Una fuente incierta o marcada como no disponible no
puede convertirse en afirmación. Cuando no hay un fragmento seguro, se registra
`no_safe_partial_fragment` y se devuelve falta de información. Los informes
separan intentos de respuestas parciales realmente entregadas y estos fallback.

Las omisiones afectan la entrega de memoria en S2; no borran SQLite. En S3 el
lector recibe todos los registros que realmente quedaron persistidos, incluidos
los errores de escritura del actor. La asignación de fallos depende de raíz,
semilla y recuerdo/slot: no depende del método, mundo oculto, orden de ejecución
ni semilla de generación. Las auditorías privadas nunca entran al prompt.
La configuración cero mantiene el comportamiento anterior.

Los perfiles de diagnóstico son `clean`, `unavailable20`, `partial30`,
`memorydrop20` y `combined`. Este último usa .2/.3/.2 respectivamente. Los datos
variados existentes ya contienen distractores, otros atributos y correcciones;
se conservan. Las paráfrasis libres y conversaciones de personas requieren
revisión semántica para ampliar la gramática limitada del evaluador.

## Verificación local sin cargar pesos

Desde la raíz del proyecto, ejecutar con el Python que tenga pytest:

```sh
PYTHONPATH=.:tests python -m pytest -q tests research/tests
python -m research.imperfection_experiment smoke \
  --output reports/imperfecciones_mi_smoke --roots 20 --seed 7331
```

El smoke crea los tres paquetes SFT y ejecuta B4, C-NEVER y C-ALWAYS con los cinco
perfiles. Usa fixtures sintéticas y registra cero entrenamiento y cero llamadas
al LLM. `SMOKE.json`, `environment/summary.json`, `metrics.csv`, las trayectorias
y las bases SQLite son evidencia del circuito, no del rendimiento de Qwen.
Todos los destinos deben ser nuevos.

Para diagnosticar una política ya seleccionada, sin cambiar sus pesos:

```sh
python -m research.imperfection_experiment evaluate \
  --bank RUTA_DATOS/validation.jsonl --checkpoint RUTA_CHECKPOINT \
  --output reports/imperfecciones_modelo_fijo
```

Se usan los prompts comunes `conversation_v3`. El comando admite desarrollo o
validación; los bancos retenidos de evaluación/test no se abren por esta vía.

## Preparar y ejecutar la comparación de aprendizaje

Primero generar un corpus nuevo con las fuentes actuales; las fuentes de un
corpus auditado quedan congeladas. La preparación comprueba exclusión respecto
a bancos anteriores y no consulta rendimiento del modelo:

```sh
python -m research.competence_data \
  --output research/data/imperfecciones_nuevo_001 \
  --seed 20261101 --partition-seed 20261102
python -m research.imperfection_experiment prepare \
  --data research/data/imperfecciones_nuevo_001 \
  --output research/runs/imperfecciones_nuevo_001 --rates 0 .1 .3 --seed 7331
```

La siguiente etapa requiere el JUnit real de pruebas y el informe CUDA real de
SFT/GRPO/capacidad del entorno existente. No se crean comprobantes ficticios:

```sh
python -m research.imperfection_experiment prepare-campaigns \
  --experiment research/runs/imperfecciones_nuevo_001 \
  --tests-report RUTA_JUNIT_REAL.xml --technical-report RUTA_SMOKE_CUDA_REAL.json
python -m research.imperfection_experiment execute \
  --experiment research/runs/imperfecciones_nuevo_001
```

`execute` sí carga modelos y entrena. Cada tasa comparte los padres históricos
seleccionados por semilla 101/202/303; dentro de cada tasa, B1c sin continuación,
más SFT (B3), RL inmediato (B2-I) y RL longitudinal (B2-L) comparten el B1c
seleccionado. B1c y B3 usan exactamente el mismo paquete de demostraciones de
su tasa. Las recompensas RL siguen correctas y el entorno de entrenamiento
permanece limpio para aislar el efecto de las demostraciones.

Se reutilizan los presupuestos y gates de competencia/señal existentes. Si la
competencia o la señal de tarea no bastan, RL queda sin ejecutar y se informa
el motivo; no se cambian umbrales después de observar resultados. El plan
verifica fuentes, hashes y targets antes de cargar pesos. **Las tres tasas
terminan su selección antes de la primera evaluación retenida.** Se sellan los
checkpoints y los archivos que registran esas selecciones. Un bloqueo exclusivo
evita dos ejecuciones sobre el mismo experimento.

`COMPARISON.json` conserva los resultados de todas las tasas y los informes por
campaña. C2, FJS, J, preguntas, escrituras sin sustento y formato se publican por
separado. La robustez es otro diagnóstico: no se mezcla con el contraste de
aprendizaje para atribuir una mejora a varios cambios simultáneos.

La revisión humana y la superioridad de RL se informan únicamente cuando exista
evidencia correspondiente. Una prueba aprobada verifica la implementación; no
demuestra por sí sola que RL mejore el modelo.
