# Enmienda002: ejecución exploratoria completa autorizada

Fecha:2026-10-23. Versión operativa1.2. Ocurre antes de abrir test.

El usuario pidió ejecutar todo para completar el spec y eligió explícitamente «Autorizar ejecución exploratoria con auditoría automática». Se conserva esta autorización en manifests/exploratory_authorization.json. La auditoría automática valida programación, particiones, evidencia y semántica del banco; no se presenta como revisión humana ni como confirmación independiente. T07 humano permanece not_executed_human. La ejecución y sus conclusiones se etiquetan exploratory.

Se ejecutarán E0–E6 en los componentes automatizables, SFT de dos épocas completas en600raíces por101/202/303 y las tres continuaciones desde B1: B2-L, B3 y B2-I. Cada continuación conserva el límite3600segundos, máximo120actualizaciones y la misma secuencia de grupos RL. Una actualización en curso y la evaluación final pueden exceder el límite: se registra el exceso, sin ocultarlo.

La evaluación usa inferencia greedy por lotes de8 para reducir tiempo. Conserva los prefijos de cada fase sin contexto futuro y reconstruye su ejecución con igualdad de bytes de observación. El batch de entrenamiento sigue siendo microbatch1 y8raíces×4rollouts. Los checkpoints SFT elegibles son paso0 y cada época completa; GRPO mantiene paso0 y cada10actualizaciones, además del final. Frecuencia y uso compartido de métricas B1 iniciales quedan registrados antes de entrenar.

La referencia B1 se almacena en CPU y se mueve temporalmente a GPU para calcular KL; regresa a CPU ANTES del backward del actor. La sonda real2048+512tokens pasó con gradientes finitos, referencia inalterada y pico reservado3200MiB en estaGPU4GB (reports/reference_offload_capacity_001.json). El batch8de inferencia pasó con pico reservado1516MiB (reports/batch_timing_001.json). Se mantienen BF16, LoRA y los límites de la versión1.1.

Los prompts comunes añaden ejemplos de formato durante desarrollo, tras observar fallos JSON en una sonda de desarrollo, antes de test. Se documentan dos formulaciones de desarrollo; no se usa test para afinarlos ni se reintenta una salida malformada.

OOD incluye composiciones reservadas y redacción escrita por código independiente del generador principal. El conjunto externo será un proxy explícitamente escrito por IA, con etiqueta AI_AUTHORED_PROXY y condición humana no ejecutada. No es evidencia de historias humanas independientes. Los resultados conservan denominadores ID, OOD y proxy separados.

El tamaño de test se fija una vez tras planificación exclusivamente en validación. Se prepara un tamaño factible300raíces ID: si la simulación sugiere más para la potencia objetivo, se registra sensibilidad limitada según §15.1, sin aumentar el tamaño después de mirar test. Las100raíces de robustez se fijan por familia antes de abrir test.

No se cambian recompensas para fabricar señal ni se exige que RL gane. Se conservan paso0, fallos, resultados negativos y comparaciones inconclusas. Las trazas y los informes finales se escriben desde ejecuciones reales. La investigación confirmatoria original no se declara completada: estos resultados pertenecen al alcance exploratorio autorizado y enumeran los requisitos humanos no ejecutados.

Revisión IA adicional solicitada expresamente: el agente principal revisó100historias visibles y Astra high hizo una segunda revisión ciega sin generador, etiquetas latentes ni test. Ambos coinciden en500/500campos de decisión comparados. Se conservan textos y hashes en reports/ai_reviews; la revisión principal no es independiente del autor y no se calcula acuerdo humano. Seis cautelas semánticas fueron adjudicadas/documentadas antes de congelar sin cambiar respuestas para favorecer modelos.
