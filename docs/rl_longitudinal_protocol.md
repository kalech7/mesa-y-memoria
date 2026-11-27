# SPEC-RL-003: protocolo previo a la evaluación

Fecha: 2026-10-19. Este protocolo se fija antes de ejecutar el test de esta revisión.
Los resultados y las fuentes de SPEC-001 y SPEC-RL-002 se conservan.

## Pregunta y alcance

La pregunta oficial en `SPEC_CODEX.md:6` trata de cuándo y cómo aclarar
afirmaciones incompatibles de la misma preferencia, alcance y período, y del
efecto actual y en sesiones posteriores frente a prompting y heurísticas.
`METODOLOGIA.md:21` establece como estimando futuro principal el éxito de tarea
en sesión posterior. SPEC-RL-003 agrega operaciones temporales de memoria y una
comparación condicional con RL. No sustituye la pregunta oficial ni demuestra
resultados contra prompting de LLM: aquí se estudian políticas simbólicas pequeñas.

## Auditoría previa

| Función | Estado anterior | Evidencia | Extensión necesaria |
|---|---|---|---|
| Evidencia alimentaria | Implementada, extracción de reglas | `simulator/preferences.py` | Adaptación temporal, sin afirmar comprensión general |
| Memoria SQLite por persona | Implementada con origen y operaciones | `memory/store.py` | Nuevas tablas append-only y auditoría de consulta |
| Vigencia/contexto | Campos existentes y resolución finita | `simulator/resolution.py` | Separar observación, registro y tiempo consultado |
| Interacción RL | Adaptador de una consulta | `rl_feasibility/environment.py` | Cierre/reapertura entre 3–10 sesiones |
| Baselines | Reglas y clonación supervisada | `rl_feasibility/baselines.py` | Historia temporal y operaciones de memoria |
| Entrenamiento RL previo | No ejecutado: sin margen medido | informe SPEC-RL-002 | Volver a medir margen; no asumir que cambió |
| Revisión humana | No validada para estos nuevos casos | no hay anotaciones de esta revisión | Preparar paquete ciego; declarar pendiente |

## Modelo y estimandos

POMDP de horizonte finito: estado privado = preferencia sintética vigente,
disposición a contestar y futuros acontecimientos; estado controlado = SQLite.
La observación contiene sólo consulta, contexto comunicado, evidencias ya
comunicadas y memoria recuperada hasta ese instante. Las operaciones actuales
cambian lo que una sesión futura puede recuperar. La política no recibe
cronograma futuro, etiquetas, semillas ni estado latente.

Formalmente \(\mathcal P=(S,A,T,O,Z,R,\gamma,H)\), con
\(a_t=(a_t^{conv},a_t^{mem})\), \(o_t\sim Z(s_t)\),
\(s_{t+1}\sim T(s_t,a_t)\). La política es
\(\pi(a_t\mid o_{1:t},a_{1:t-1})\). Se usa \(\gamma=1\) para el
retorno finito; la abstracción tabular no se presume estado de Markov suficiente.
La historia latente no basta para justificar RL: debe haber consecuencias de
acciones sobre observaciones futuras y ventaja frente a métodos simples.

Unidad de comparación: usuario sintético completo, no cada sesión como muestra
independiente. Estimandos: retorno por usuario y por sesión; éxito en consultas
posteriores; exactitud de decisión fundada; operaciones injustificadas;
clarificaciones requeridas, irrelevantes y repetidas; seguridad. Las métricas se
desagregan por categoría, cantidad de sesiones y modalidad de extracción.

## Datos y congelación

Presupuesto inicial: 600 usuarios de entrenamiento, 120 de validación, 240 de
test y 120 de cambio de distribución; parámetros configurables. Semillas
42/43/44/45 y entre 3 y 10 sesiones por usuario. Se auditan usuarios, plantillas,
fechas y firmas semánticas entre particiones. Compartir patrones abstractos se
declara expresamente: usuarios nuevos no equivalen a familias temporales nuevas.
Un conjunto de familias retenidas adicional debe medir esta limitación.
El modo ORACLE_STRUCTURED sólo ofrece la extracción correcta de lo ya comunicado.
PIPELINE emplea el extractor existente, con sus fallos, sin ajustar sus reglas al test.

Las expectativas se escriben en el generador/evaluador a partir del diseño
latente, nunca llamando al reconciliador o a una política. Las etiquetas son
propuestas sintéticas, no adjudicación humana. No se puede completar una revisión
externa mediante otro subagente: se prepara un CSV ciego y se informa pendiente.

## Invariantes, comparadores y seguridad

Todas las decisiones consultan memoria, incluido resultado vacío. Se conserva
historial, procedencia y fechas. No se reemplazan alergias con una elección
reciente. Las políticas protegidas cumplen clarificación material/no repetición.
LAST_WRITE_WINS y la condición sin clarificación son ablaciones explícitas:
no se presentan como políticas conformes a MEM-02/MEM-05/MEM-06. Su incumplimiento
se mide, no se oculta. El filtro de restricciones permanece en todas las condiciones.
Se registran acciones propuestas y ejecutadas para distinguir competencia de la
política del efecto del contrato obligatorio.

Baselines: recencia, reglas históricas, reglas de clarificación, supervisión
con la misma información que RL y planificación visible con supuestos explícitos.
La supervisión aprende de demostraciones del planificador/reglas: esto mide
imitación, no anotación humana. No se debilita para abrir margen a RL.

## Gates y criterio de relevancia

A: todos los tests críticos pasan, incluida recuperación histórica, vigencia,
persistencia real y ausencia de información futura.
B: preguntas específicas ante incertidumbre material resoluble, sin preguntas
irrelevantes/repetidas en controles; informar fallos incluso si el resultado final
es correcto.
C: contrafactuales pareados que cambian exclusivamente una operación de memoria
y afectan recuperación y decisiones posteriores tras cerrar y reabrir SQLite.
D: margen del planificador frente al mejor baseline no-RL (incluida supervisión),
seleccionado en validación. Diferencia mínima: 0,10 unidades de retorno medio por
sesión, equivalente al coste inicial de una pregunta; IC95% pareado inferior > 0.
Si el IC superior es menor que 0,10: no hay margen práctico demostrado. Cero
margen contra una referencia no demuestra optimalidad universal.
E: sólo si A–D lo permiten, Q-learning pequeño, semillas 101/202/303, una pasada
por entrenamiento por semilla, máximo 120 s por semilla, checkpoint seleccionado
en validación y test una vez congelado. Sin entrenamiento de LLM/GPU ni APIs.
Una mejora frente a recencia sola no habilita RL.

Recompensa: acierto fundado +1, error -1; preguntar cuesta 0,10 incluso si es
obligatorio; abstención sin fundamento suficiente cuesta 0,35; preservación
justificada puede recibir crédito parcial; violaciones de restricción y
actualizaciones injustificadas reciben penalización adicional. Coeficientes
exactos y descomposición quedan en `reward.py` y manifiesto antes del test.
No hay bonus por preguntar o por copiar la acción de la heurística.
El retorno longitudinal suma resultados presentes y futuros. La ablación sin
recompensa diferida afecta entrenamiento si éste se ejecuta; sin entrenamiento,
sólo puede presentarse como reponderación descriptiva, no como política aprendida.

## Estadística, robustez y límites

Bootstrap pareado de usuarios completos, 1.000 remuestreos y semilla 819;
denominadores separados, métricas no evaluables marcadas con `null`. Análisis
con coste de pregunta 0,05/0,10/0,20 sobre trayectorias congeladas, sin seleccionar
el coeficiente más favorable. Comparaciones por familias y horizonte; no ocultar
fallos bajo un promedio global. Gráficos de rendimiento; curva de aprendizaje
únicamente si existió entrenamiento.

No adjudicación externa, simulación de reglas, catálogo limitado y ausencia de
comparación LLM deben limitar cualquier conclusión de tesis. Los invariantes
implementados por una protección obligatoria no son aprendizaje demostrado.
