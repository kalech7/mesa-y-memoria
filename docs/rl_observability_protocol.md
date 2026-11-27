# SPEC-RL-004 — Protocolo del piloto mínimo

Este protocolo añade `rl_observability/` y no modifica los gates, módulos,
pruebas ni resultados de SPEC-RL-003. La autorización del piloto procede de
SPEC-RL-004: estudiar entrenabilidad no requiere presumir superioridad.

## Estimandos y condiciones

1. Observabilidad: B2 fijo, diferencias O2−O1 y O3−O2 sobre usuarios emparejados.
2. Aprendizaje: B3/B4 frente a B1/B2 con la misma observación. Inicialización
   sin entrenamiento frente a checkpoints seleccionados de B4.
3. Extracción: B2 PIPELINE frente a B2 VISIBLE_STRUCTURED, como diagnóstico
   separado, sin atribuirlo al aprendizaje.

Se utilizan cinco familias y tres sesiones. S1 registra preferencia inicial y
restricción explícita; la política actúa en S2 y S3. Una pregunta como máximo en
S2; ninguna en S3. Los conflictos son declaraciones opuestas sobre el mismo
alimento y alcance sin un marcador explícito de sustitución. Su respuesta
privada sólo se convierte en evidencia después de preguntar por la preferencia.
Las preguntas sobre otros slots no revelan esa respuesta.

O1 conserva un hecho reciente; O2 todo el historial relevante; O3 añade
interpretaciones trazables de evidencia visible. Las restricciones se conservan
en las tres. El alimento objetivo, la fecha y el contexto de consulta se
suministran estructurados. O3 puede empatar con O2 si el decisor obtiene las
mismas características por sí mismo. No se degradan las reglas para evitarlo.

## Políticas y recompensa

B1 aplica última escritura y se declara ablación. B2 interpreta vigencia,
excepciones, sustitución explícita, conflictos y aclaraciones. B3 clona
demostraciones de B2 de entrenamiento. B4 conserva Q-learning tabular. B3/B4
comparten exactamente `feature_key`, catálogo de acciones y capacidad tabular;
ninguno utiliza reglas como fallback. La interpretación simbólica compartida no
se atribuye a RL. Los IDs, fechas absolutas, familia, gold, respuestas futuras y
`required_decision` no forman parte de las características aprendidas.

La acción combina conversación, polaridad o slot y operación de memoria.
Las operaciones son ADD_MEMORY, UPDATE_WITH_HISTORY, PRESERVE_CONFLICT y NOOP.
Las conversaciones son ANSWER, CLARIFY(slot), PRESERVE_CONFLICT,
CONDITIONAL_ANSWER y NOOP. Son decisiones simbólicas; no se evalúa generación
de respuestas naturales ni fluidez. No se fuerza la decisión correcta mediante
el contrato. Sólo seguridad y presupuesto pueden bloquear una propuesta.
Se registran propuesta y ejecución por separado.

Coeficientes: +1 decisión correcta y fundamentada, −1 incorrecta, −0,1 pregunta,
−0,5 por actualización injustificada, −0,6 respuesta sin soporte, −2 intento
inseguro, +0,7 preservación justificada. Una coincidencia con el estado privado
no basta: la respuesta o la incertidumbre deben estar fundamentadas en evidencia
expuesta. Sin resolución comunicada, preservar un conflicto observable es
correcto aunque no maximice la utilidad de resolverlo. Preguntar sin obtener
respuesta sólo añade coste. En las familias resoluble e irresoluble no hay una
pista previa que revele qué contestará el usuario; no se exige anticiparlo.

## Partición, selección e incertidumbre

Ejecución principal: 100 usuarios de entrenamiento, 50 de validación y 100 de
test; semillas 101, 202 y 303. Presupuesto Q: hasta 600 episodios por semilla y
condición, o 60 segundos, lo que ocurra primero; bloques de validación de 100.
α=0,3, γ=1, ε lineal 1→0,05. Los usuarios de entrenamiento se remuestrean y esas
visitas no se contabilizan como nuevas personas independientes.

Se selecciona observación por retorno B2 de validación; empates prefieren
O1/O2/O3 en ese orden. Checkpoints y mejor baseline se seleccionan por retorno
de validación; los empates siguen el orden determinista del catálogo. No se
cambian recompensa ni parámetros tras inspeccionar el test.

El bootstrap percentil 95% usa 1.000 réplicas y remuestrea usuarios completos,
con sus dos decisiones y ambas condiciones. En la media por semillas primero
se promedian semillas dentro de cada usuario. Las tasas de componentes se
calculan como suma de numeradores / suma de denominadores, también en cada
remuestreo; denominador cero se reporta no evaluable. No se interpreta un cero
por ausencia de conflictos como sensibilidad cero.

Se mantienen perfiles, fechas y plantillas textuales disjuntos, pero se
comparten los cinco mecanismos y estados abstractos entre particiones. Esta
evaluación estudia transferencia entre usuarios/redacciones del mismo soporte
simbólico; **no demuestra generalización a plantillas abstractas nuevas**.
Los IC sólo reflejan variación de este generador, no diversidad poblacional.

## Persistencia y reproducción

Se reutilizan `TemporalMemoryStore`, `TemporalFact`, el extractor existente y
las utilidades estadísticas. Cada trayectoria usa un SQLite propio y recarga
la misma base en S3. Las bases temporales usan journal en memoria y desactivan
fsync; conservan las mismas transacciones y semántica, pero no garantizan
durabilidad ante caída del proceso. Las bases solicitadas mediante `db_path`
mantienen el modo normal. El historial y el archivo bruto no se borran.

Los JSON privados se usan sólo en simulación/evaluación. Los registros de
diagnóstico, aunque contienen las observaciones exactas, también incluyen
etiquetas privadas y no son entradas públicas de entrenamiento. Las trazas
registran evidencia disponible/recuperada, observación, propuestas, ejecución,
recompensa y resultado. Las rutas temporales del SQLite se conservan como
traza, pero esas bases se eliminan al cerrar; el historial completo está
serializado en los artefactos. Para conservar una base física usar `db_path`.

```bash
uv run --offline python -m unittest discover -s tests -v
uv run --offline python -m rl_observability.diagnostics \
  --output reports/rl_observability/diagnosis_reproduccion
uv run --offline python -m rl_observability.run \
  --output reports/rl_observability/reproduccion \
  --train-users 100 --validation-users 50 --test-users 100 \
  --episodes 600 --max-seconds 60 --block-size 100 --bootstrap 1000 \
  --seed 41004 --seeds 101 202 303
```

El destino debe ser nuevo. Se guardan configuración, snapshot y hashes de
fuentes, datasets, checkpoints, curvas y resultados por usuario/semilla.
Matplotlib es opcional para figuras (`uv pip install matplotlib`); las curvas
CSV siempre se generan. No se necesita GPU. B5 requiere acceso efectivo a un
LLM: la consulta de disponibilidad no es una evaluación de prompting.

La revisión humana de etiquetas y la comparación con un LLM accesible quedan
pendientes. H3 puede sostenerse sin H4; ninguna mejora de observabilidad se
atribuye a aprendizaje.
