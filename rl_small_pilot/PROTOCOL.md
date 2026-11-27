# SPEC-RL-005 — Protocolo previo al entrenamiento principal

La pregunta oficial de tesis se conserva: cuándo y cómo aclarar afirmaciones
incompatibles para la misma preferencia, alcance y período, y qué efectos tienen
las aclaraciones sobre la respuesta actual y la coherencia posterior frente a
prompting y heurísticas. La política neuronal externa evalúa un componente de
decisión; no es un LLM y no demuestra comprensión libre del lenguaje.

## Selección técnica y recursos

El usuario autorizó instalar PyTorch y aprobó el plan de pruebas unittest.
macOS Intel con Python 3.13 no tiene wheel PyTorch compatible. Se creó un entorno
aislado Python 3.12.13 / PyTorch 2.2.2, conservando el entorno del proyecto.
No se entrena GLM ni otro LLM. La API USFQ previamente autorizada se usa sólo
para M4, con credenciales de entorno y sin escribirlas en artefactos.

Se elige REINFORCE porque actualizar pesos neuronales requiere un estimador de
gradiente apropiado: Q-learning tabular no actualiza esos pesos. La elección es
la alternativa básica expresamente contemplada por SPEC-RL-005 y no responde a
considerar POMDP el problema. Red 51→64 Tanh→64 Tanh→32, CPU con un hilo,
9.568 parámetros. Los tipos, conflictos y aplicabilidad se derivan mediante un
intérprete simbólico compartido; no se atribuyen al aprendizaje de lenguaje.

## Datos mínimos y límites

300 usuarios TRAIN, 60 VALIDATION, 120 TEST por defecto, tres sesiones, cinco
familias. Se reduce a la mitad la sugerencia de 600/120/240 para acotar el coste
de replicar trayectorias SQLite, tres semillas y dos condiciones de extracción.
Las cantidades son configurables y quedan registradas.

Se balancean ambas direcciones de cambio LIKE/DISLIKE; los IDs y los años son
disjuntos. Se reservan combinaciones de meal/place/company: cuatro en TRAIN,
dos en VALIDATION y otras dos en TEST. Las frases son distintas por partición.
Los cinco mecanismos y parte del soporte simbólico siguen compartidos: esto
no es una prueba de generalización conceptual. El vector codifica aplicabilidad
y presencia de ámbitos, no las identidades concretas de los contextos.

Se reutilizan generador, SQLite, extracción y recompensa anteriores mediante
adaptadores. Se corrigen antes de entrenar dos atajos: signo de preferencia
asociado a familia y evaluación de respuestas de aclaración fijada a LIKE.
El día se deriva públicamente de query_at para todos los métodos. No se cambia
el historial experimental ni se atribuyen estas correcciones convencionales a RL.

## Selección y comparaciones

1. M3 fijo compara O1/O2/O3 en VALIDATION. Se elige entre O2/O3 por retorno;
   empate favorece O2. Se conservan esas condiciones para la ablación de TEST.
2. Antes de entrenar: diagnóstico ORACLE/PIPELINE, prueba de recompensa con
   políticas constantes y suite completa, incluidos cambios reales de pesos.
3. M0 aprende de demostraciones M3 sólo de TRAIN: Adam, lr=0,003, 20 épocas,
   lotes de 20 trayectorias; validación cada cuatro épocas.
4. M1 y M2 parten de idéntico M0 por semilla. M1 REINFORCE tiene γ=1 y baseline
   escalar calculado sólo con lotes anteriores; no usa crítico ni bonus de
   entropía. M2 recibe supervisión adicional con igual número de actualizaciones
   y usuarios por lote. Se reportan sus tiempos y pases de acciones diferentes.
5. Único ajuste: learning rate 0,0001 versus 0,0003. Cada configuración usa tres
   semillas 101/202/303, 30 pasos de optimizador y 20 trayectorias por lote.
   Los checkpoints se evalúan en pasos 0/10/20/30. Se elige lr por retorno medio
   de VALIDATION de las tres semillas. Cada checkpoint empata al más temprano.
6. Se congelan observación, recompensa, lr y checkpoints antes de TEST. Se
   evalúan M0/M1/M2/M3 en los mismos usuarios ORACLE y PIPELINE. M4 usa la misma
   observación y episodios; si hay fallos técnicos o un subconjunto, se reporta
   cobertura exacta y comparaciones emparejadas únicamente donde se completaron.

Se guardan checkpoint seleccionado y último. Un seleccionado idéntico a M0 no
acredita mejora RL aunque el último tenga cambios de pesos. Se verifican hashes,
gradientes no nulos y distancia L2 para no confundir entrenamiento ejecutado
con aprendizaje útil. No se degrada M0 para fabricar margen.

## Recompensa y estimandos

+1 acierto fundamentado, −1 error, −0,1 pregunta, +0,7 preservación justificada;
penalizaciones adicionales −0,5 actualización injustificada, −0,6 respuesta sin
respaldo, −2 intento inseguro. No se modifican coeficientes tras inspeccionar TEST.
S1 inicializa; S2/S3 contribuyen al retorno. Preservar incertidumbre puede ser
correcto y obtener menos retorno que resolverla. Una pregunta en S2 y ninguna
en S3 impiden atribuir el cero de repeticiones a aprendizaje.

Estimandos: O2−O1, O3−O2 con M3 fijo; M1−M0, M1−M2 y M1−M3 con la misma
observación; PIPELINE−ORACLE sobre los mismos usuarios. Bootstrap percentil
95%, 1.000 réplicas, usuarios completos emparejados. Las semillas se promedian
dentro de usuario sin inflar n; también se informa cada semilla.

Para afirmar superioridad de RL se exige mejora reproducible de retorno frente
a M0, M2 y M3, con IC positivo y al menos +0,10 por episodio como margen práctico
predefinido. Un cambio de pesos no satisface ese criterio. Un empate no demuestra
inutilidad general de RL y un techo de M0 no autoriza empeorarlo.

## Validez y siguiente paso

Las etiquetas son del generador, sin adjudicación humana independiente. Los
errores de extracción de campos no se equiparan a fallos de decisión; se registra
si cambian aplicabilidad. Las causas del diagnóstico son probables, incluyendo
causas múltiples e indeterminadas. La seguridad se comprueba técnicamente y no
se presenta como un benchmark amplio de alergias. Sólo una ventaja independiente
y replicada, acompañada de validez externa y recursos reales de entrenamiento,
permitiría planificar RL sobre los pesos de un LLM.
