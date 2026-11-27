# SPEC V4 — Cierre de evaluación: entorno, simulación, memoria y RL

**Proyecto:** tesis sobre aclaración selectiva y persistencia de preferencias.  
**Fecha:** 24 de octubre de 2026.  
**Estado:** especificación para implementar y ejecutar localmente; no certificación de que las pruebas ya hayan pasado.  
**Modelo de referencia:** Qwen2.5-0.5B-Instruct + LoRA.  
**Recursos declarados:** RTX 3050 Ti Laptop, 4 GB; semillas 101, 202 y 303.  
**Prioridad:** validar el circuito experimental y responder PI1, PI2 y PI3 antes de ampliar RL.  
**Base documental:** informes de la campaña completa y auditoría V3; no solamente el análisis parcial. [F1–F8]

> **Decisión recomendada:** conservar inicialmente el entorno y GRPO existentes, corregir fallos demostrados, instrumentar lo que falta y ejecutar un piloto pequeño. No cambiar simultáneamente modelo, recompensa, escritor, lector y algoritmo. Si el problema demostrado es falta de cobertura de acciones, utilizar la alternativa discreta del anexo; no introducirla silenciosamente como si fuera el mismo GRPO.

## 0. Qué significa cerrar esta evaluación

“Definitiva” significa cerrar una versión documentada del experimento: contratos verificados, datos y evaluación congelados, comparaciones interpretables y limitaciones explícitas. No significa demostrar ausencia universal de errores ni garantizar que RL mejore.

“Rápida” significa reutilizar resultados válidos, ejecutar primero pruebas deterministas, inspeccionar pocas trayectorias y detener ramas cuya interpretación esté bloqueada. No repetir toda la campaña de 75.357 filas antes de resolver los defectos conocidos. [F8]

Este documento especifica trabajo futuro. Los valores marcados **PROPUESTA** son decisiones de ingeniería para el piloto, no hiperparámetros óptimos encontrados en tus datos. Los valores **REPORTADOS** proceden de los archivos o de la descripción aportada; necesitan contraste con la configuración local antes de llamarlos verificados en ejecución.

### 0.1 Resultado que debe producir el agente ejecutor

Entregar un dictamen con cuatro estados separados:

| Dimensión | Estados permitidos |
|---|---|
| Infraestructura y contratos | `passed`, `failed`, `blocked_missing_evidence` |
| Competencia del modelo | Métricas y errores observados, no un sello genérico de corrección |
| Evidencia de aprendizaje RL | `positive`, `negative_with_precision`, `inconclusive`, `not_executed` |
| Validez externa | Alcance sintético documentado; evidencia humana solo cuando exista |

Una infraestructura correcta puede medir un modelo malo. Un porcentaje alto puede provenir de una infraestructura incorrecta. No confundir ambas situaciones.

### 0.2 Orden obligatorio

1. Inventariar código, configuraciones, checkpoints y trazas.
2. Corregir exportación y verificar evaluador, simulador, contratos y aislamiento.
3. Localizar errores de adquisición, interpretación, escritura y lectura sin entrenar.
4. Validar máscaras, gradientes, referencia y guardado de entrenamiento.
5. Hacer un piloto de RL y decidir si la señal permite continuar.
6. Congelar protocolo; ejecutar únicamente los brazos necesarios y un test nuevo.
7. Redactar respuestas a Vicent, incluidos resultados negativos y bloqueos.

No pasar de una etapa a la siguiente con un fallo crítico que invalide sus resultados. Un componente secundario bloqueado puede quedar fuera del alcance, pero debe declararse y no sostener una conclusión sobre él.

---

## 1. Preguntas de Vicent y afirmaciones que deben identificarse

El archivo `RESPUESTAS_DIRECTOR.md` documenta estas tres preguntas. No contiene un mensaje literal de Vicent; no atribuirle otras exigencias. [F6]

| Pregunta | Qué debe aislarse | Comparación necesaria | Qué no demuestra por sí sola |
|---|---|---|---|
| **PI1. Observabilidad** | Efecto de la evidencia disponible | Mismo caso/modelo, O1 frente a O2, contrato de formato común | Que un resultado nulo con B0 incapaz implique que la evidencia no importa |
| **PI2. Adquisición y persistencia** | Efecto de hacer disponible la aclaración para escribir | I1 frente a I2, misma evidencia previa e interacción controlada | Que el agente decida autónomamente preguntar bien o que toda persistencia sea beneficiosa |
| **PI3. Aprendizaje** | Valor añadido de la continuación y del horizonte de recompensa | B1, más SFT, RL inmediato y RL longitudinal comparables | Que tener un adaptador RL o seleccionar paso cero demuestre aprendizaje |

**Aporte que puede sostener esta ejecución:** identificar cuándo una aclaración contribuye al éxito futuro, cuándo su beneficio se pierde al construir o consumir memoria y si RL longitudinal añade algo sobre controles pertinentes. No exigir que RL gane para aceptar la tesis.

---

## 2. Punto de partida: evidencia existente y verificaciones pendientes

### 2.1 Hallazgos que motivan esta SPEC

| ID | Evidencia existente | Acción obligatoria |
|---|---|---|
| H01 | `RESULTADOS.csv` omite `nonresponse_rate`; la auditoría identifica 4.509 claves repetidas, 9.018 filas afectadas. [F8] | Reexportar desde trazas; no deduplicar ni inferir tasas por orden de filas. |
| H02 | C-ALWAYS en ID registra 0,20 preguntas y aproximadamente 0,8008 salidas inválidas por episodio. [F1, F8] | Verificar si fuerza intención, slot o interacción realmente ejecutada. |
| H03 | En `primary_F4`, B1-I2 supera a I1 en 97,78 puntos de FJS. [F4] | Verificar que la disponibilidad de aclaración para escritura sea el único cambio pertinente. |
| H04 | En casos claros, I2 reduce FJS en 5,83 puntos para B1 y B2-L; aumentan escrituras no sustentadas. [F4, F8] | Revisar respuestas del simulador y alcance de escritura; no ocultar el daño. |
| H05 | `unaffected_attribute` tiene FJS=1 y C2=0 en todas las ramas. [F4] | Inspeccionar consulta actual, gold y significado del control. |
| H06 | Con lector LLM, E4-ID obtiene FJS≈49,31% para B1 y 50,33% para B4. [F1, F8] | Separar serialización, recuperación —si existe—, interpretación y scoring. |
| H07 | E5-EXTRACT obtiene FJS=0 para B1 y 30% para B4. [F1, F8] | Inspeccionar memoria inicial extraída y evidencias perdidas. |
| H08 | E6 obtiene FJS=0 en todos los métodos. [F1, F8] | Probar equivalencia con E4 al desactivar la interacción posterior. |
| H09 | B3-303 se deteriora; alrededor del paso 3502 se registra una pérdida de 10,0646. [F7, F8] | Reproducir la región 3400–3600 con ejemplos, máscaras y optimizador instrumentados. |
| H10 | Las ramas RL ejecutaron 13–15 actualizaciones; faltan diagnósticos completos de señal por grupo. [F3, F7, F8] | Instrumentar antes de aumentar duración o culpar a la capacidad. |
| H11 | Éxito conjunto OOD: B1 y B2-L≈16,44%; B3≈18,33%. Análisis añadido en V3. [F8] | Reportar interacción completa, además de FJS. No convertirlo retrospectivamente en endpoint original. |

H02–H10 son anomalías o limitaciones que requieren explicación, no diagnósticos causales ya demostrados. H01 sí es un defecto comprobado de identificación en la exportación.

### 2.2 Lo que los archivos no permiten verificar todavía

No están incluidos aquí los JSONL originales, los textos completos de todas las memorias, el generador, el código del evaluador ni los pesos. Los paths que aparecen en `ANALISIS.json` apuntan al equipo local del autor. Un hash declarado no prueba que se haya comprobado el archivo correspondiente. [F1, F8]

No están documentados de manera suficiente en los informes: LR histórico efectivo, rango y módulos LoRA, máscaras exactas, longitud máxima real, definición de cada slot y semántica completa de O1/O2/I0/I1/I2. Recuperarlos del repositorio; no inventarlos a partir de F0–F4.

---

## 3. Inventario local y conservación de evidencia

### 3.1 Descubrimiento, sin asumir nombres de módulos

La ruta histórica reportada es:

```text

research/runs/full_exploratory_001/
```

Localizar en ese repositorio o en su ubicación actual:

- Generador, splits, raíces y plantillas de entrenamiento/validación/test.
- Simulador de usuario, funciones de observación O1/O2 y contratos I0/I1/I2.
- Extractor, esquema de memoria, operaciones de actualización y lector.
- Parser, evaluador y componentes de recompensa.
- Entrenadores SFT/RL, configuración efectiva y selección de checkpoints.
- Checkpoints B1 y continuaciones; `adapter_config.json`, tokenizer y revisiones.
- Trazas indicadas en `ANALISIS.json`; manifiestos SFT/continuaciones.

**No implementar un entorno nuevo para sustituir el existente sin haber inspeccionado su código.** Si falta un artefacto, registrar `blocked_missing_evidence` para la comprobación afectada y continuar solo con lo que no dependa de él.

### 3.2 Congelación y manifiesto

Crear un directorio nuevo, por ejemplo `research/runs/closure_v4_<run_id>/`. No sobrescribir `full_exploratory_001`.

Registrar:

```yaml
run_id: unique_id
protocol_version: closure_v4
parent_run: full_exploratory_001
code_commit: actual_commit_or_unversioned
code_diff_sha256: actual_hash
model_id: Qwen/Qwen2.5-0.5B-Instruct
model_revision: 7ae557604adf67be50417f59c2c2f167def9a775
tokenizer_revision: 7ae557604adf67be50417f59c2c2f167def9a775
actor_checkpoint_sha256: null
reference_checkpoint_sha256: null
reader_checkpoint_sha256: null
dataset_manifest_sha256: null
prompt_manifest_sha256: null
evaluator_version: null
simulator_version: null
precision: bfloat16
quantization: none
training_seeds: [101, 202, 303]
```

La revisión, precisión y ausencia de cuantización son valores reportados por la campaña, no comprobaciones hechas sobre el dispositivo en esta SPEC. Completar los campos `null` antes de una corrida que dependa de ellos. [F5]

Guardar versiones reales de Python, PyTorch, Transformers, PEFT, TRL —si se usa—, CUDA y controlador. No actualizar bibliotecas como parte de la auditoría. Si se requiere una actualización, crear un entorno separado y repetir las pruebas afectadas. Las APIs y defaults no deben inferirse de documentación de otra versión. [T1–T5]

---

## 4. Matriz de variables verificadas y no verificadas

Crear `MATRIZ_VERIFICACION.csv` con:

```text
check_id,variable,expected,observed,status,evidence_path,evidence_sha256,
code_location,affected_claim,severity,action
```

Estados: `reported_only`, `verified`, `failed`, `not_executed`, `not_applicable`, `not_identifiable_from_current_artifacts`.

| Variable o propiedad | Situación inicial | Cómo verificar | Consecuencia si falta |
|---|---|---|---|
| Modelo/revisión/tokenizer | Reportados | Configuración y archivos cargados | No asegurar reproducibilidad exacta |
| Identidad real de B1 por semilla | No comprobada por pesos aquí | Hash y prueba de logits/salidas | Bloquea atribución de continuación |
| Referencia KL igual a B1 | Booleano reportado | Pesos, adaptador activo y logits | Bloquea RL interpretable |
| LR/scheduler/optimizador históricos | No disponibles completos | Config y estado guardado | Bloquea réplica exacta; no impide un piloto nuevo explícito |
| Parámetros entrenables LoRA | No verificados | Nombres, número, `requires_grad`, gradientes | Bloquea entrenamiento |
| Mapping F0–F4 y slots | Insuficientemente documentado | Generador y casos auditados | Bloquea interpretación por familia |
| Identidad de condiciones de no respuesta | Falla en CSV | Reexportación desde trazas | Bloquea comparación 20%/40% |
| Verdad latente sintética | Verificable dentro del diseño | Generador y fixtures independientes | No equivale a preferencias humanas reales |
| Evidencia observable previa a ASK | No verificada | Serializador y hashes por mundos compatibles | Bloquea ausencia de filtración |
| Necesidad de aclarar | No medida adecuadamente | Mundos compatibles + tarea + slots resolutivos | No inferirla del número de preguntas |
| Pregunta ejecutada y slot | Parcialmente agregados | Log completo de transición | Bloquea “siempre preguntar” |
| Respuesta del simulador | No inspeccionada aquí | Gold sintético, slot solicitado y texto revelado | Bloquea utilidad causal de interacción |
| Valor/alcance de memoria | No verificable con agregados | Memoria antes/después y evidencia | Bloquea atribuir el fallo al escritor |
| Estado oculto entre sesiones | No verificado | Reinicio, pruebas de aislamiento y cachés | Bloquea persistencia real |
| Recuperación selectiva | Existencia pendiente | Código y IDs recuperados | Usar `not_applicable` si se suministra toda la memoria |
| Correctitud semántica del evaluador | No certificada | Fixtures y mutaciones independientes | Bloquea todas las conclusiones de desempeño |
| Recompensa de tarea/total por grupo | Falta telemetría | Retornos desglosados y ventajas | No diagnosticar exploración/saturación |
| Tamaño efectivo de muestra | Raíces conocidas en informes | Descendencia y solapamientos de cohortes | No contar semillas o mundos como usuarios |
| Solapamiento train/test por estructura | No verificado | Manifiestos, plantillas, huellas semánticas | Bloquea afirmaciones fuertes de generalización |
| VRAM real y soporte numérico | Hardware declarado | Perfil de forward/backward/generación | No garantizar que una configuración cabe |
| Validez de datos sintéticos en humanos | No identificada | Revisión independiente y, para eficacia humana, estudio adicional | Limitar las conclusiones al entorno |
| Estabilidad de preferencias más allá del horizonte | No evaluada | Experimentos adicionales | No afirmar memoria indefinida o de largo plazo general |

**Importante:** `verified` significa “pasó esta prueba bajo estas condiciones”, no “verdad universal”. No convertir un estado desconocido en un valor favorable.

---

## 5. Contrato de información: actor, simulador y evaluador

### 5.1 Separación de objetos

Mantener tres objetos diferentes, incluso si el código original usa diccionarios:

| Objeto | Contenido | Acceso |
|---|---|---|
| `latent_world` | Preferencia real sintética, duración y contexto ocultos | Simulador y evaluador |
| `observed_evidence` | Memoria visible, mensajes recibidos, preguntas y respuestas efectivamente entregadas | Actor y escritor según intervención |
| `evaluation_metadata` | Golds, familia, IDs internos, consultas contrafactuales y etiquetas | Evaluación y logs, nunca prompt del actor |

El actor puede conocer los nombres públicos de slots y el esquema de salida. No puede recibir etiquetas que resuelvan su decisión: familia, necesidad gold, valores ocultos, gold de memoria, puntuación de alternativas o consulta futura concreta.

### 5.2 Pruebas de no filtración

**INF-01:** dos mundos compatibles con exactamente la misma evidencia antes de ASK deben producir los mismos bytes/token IDs de prompt. La igualdad se exige dentro de la misma realización lingüística, no entre paráfrasis distintas.

**INF-02:** cambiar solo el mundo oculto no cambia el prompt, la lista legal de acciones, la recuperación previa ni la serialización de memoria.

**INF-03:** cambiar `world_id`, `root_id` o familia en metadatos no cambia la salida determinista cuando el input observable es idéntico.

**INF-04:** la máscara de acciones usa reglas de interfaz, no si un slot es gold o necesario. No retirar preguntas redundantes únicamente porque el evaluador sabe que sobran.

**INF-05:** logs y `info` del entorno no se concatenan al prompt accidentalmente.

Un fallo en estas pruebas bloquea cualquier conclusión sobre aprendizaje de aclaración.

### 5.3 Simulador de usuario

Primero verificar el simulador existente. Para fixtures, usar un respondedor determinista de slots como referencia, no un segundo LLM que pueda compartir errores.

El contrato debe especificar qué devuelve para cada slot, qué puede revelar adicionalmente y cuándo no responde. La respuesta no debe resolver otros atributos ocultos por iniciativa propia.

Pruebas obligatorias:

- Misma pregunta, mundo y estado: respuesta semánticamente idéntica en modo determinista.
- Pregunta irrelevante: no revela el atributo objetivo por accidente.
- Pregunta redundante: confirma el mismo dato y alcance; no introduce una preferencia nueva.
- Negación y alcance: “solo hoy” no significa cambio duradero; “desde ahora” no significa únicamente esta sesión.
- No respuesta: genera un estado explícito `unavailable`, no una respuesta gold encubierta.
- Pregunta inválida: no devuelve una aclaración útil usando la intención que el evaluador cree que tenía el actor.
- Respuesta natural y canónica: deben revelar el mismo contenido; la canónica no puede añadir contexto o duración ausentes.

Para tasas 0,20/0,40, generar un uniforme estable por episodio mediante una función hash y declarar no respuesta si `u < p`. Así se acoplan escenarios sin depender del orden de ejecución. La identidad de métodos no debe entrar en el sorteo compartido. Si se permite más de una pregunta, incluir índice y slot según el mecanismo declarado.

No exigir que cada muestra pequeña alcance exactamente 20% o 40% observado. Comprobar la regla determinista de asignación y reportar la tasa efectivamente observada entre preguntas entregadas.

---

## 6. Contrato de memoria y temporalidad

### 6.1 Dos fuentes de verdad distintas

El estado del mundo decide si una respuesta coincide con la realidad sintética. La evidencia recibida decide si esa respuesta o escritura está sustentada. Coincidir por azar con un dato oculto no es éxito justificado.

Construir una memoria de referencia para diagnóstico **solo desde la evidencia entregada a esa rama**. Un escritor de referencia no puede arreglar una pregunta omitida leyendo el mundo oculto.

### 6.2 Campos mínimos a inspeccionar

No imponer un formato nuevo al actor antes de comprobar el actual. Crear una representación de auditoría normalizada con campos equivalentes a:

```yaml
memory_id: stable_identifier
attribute: actual_attribute
value: observed_value
context_scope: explicit_context_or_unknown
valid_from: explicit_anchor_or_unknown
valid_until: explicit_boundary_or_unknown
duration_mode: unknown  # ejemplo; otras categorías se fijan en el esquema
source_evidence_ids: []
operation: actual_operation
supersedes: []
```

`valid_until: null` no puede significar simultáneamente “desconocido” y “vigente hasta nuevo cambio”. Separar ambas semánticas. La existencia de un ID de evidencia tampoco demuestra que esa evidencia sustente el contenido.

### 6.3 Casos que deben representarse correctamente

| Situación | Conducta de memoria válida | Error a detectar |
|---|---|---|
| Preferencia estable no contradicha | Conservarla | Borrado injustificado |
| Cambio explícito duradero | Actualizar desde el momento declarado | Mantener la preferencia anterior como única vigente |
| Excepción temporal | Conservar excepción con alcance y la base aplicable fuera de él | Convertirla en regla permanente |
| Preferencia contextual | Mantener las asociaciones contexto–valor | Aplicar el valor a todos los contextos |
| Duración desconocida | Representar incertidumbre o limitar la escritura a lo sustentado | Inventar caducidad o permanencia |
| No respuesta | No añadir el dato no adquirido | Interpretar silencio como confirmación |
| Dato ajeno al atributo actualizado | Preservarlo | Daño colateral por sobrescritura |

### 6.4 Pruebas del almacén, no del modelo

**MEM-01:** guardar y recargar conserva el contenido semántico y sus evidencias.

**MEM-02:** NOOP deja intacto el estado.

**MEM-03:** las operaciones permitidas producen el mismo resultado en el motor real y en fixtures escritos de forma independiente.

**MEM-04:** repetir una operación solo es idempotente cuando el contrato lo exige; no imponer idempotencia a operaciones acumulativas legítimas.

**MEM-05:** una excepción vencida deja de ser aplicable, sin eliminar indebidamente la preferencia base. Un registro puede permanecer archivado aunque ya no aplique.

**MEM-06:** la prioridad de contexto y tiempo está documentada. No asumir “lo más reciente gana” cuando existe una excepción o un contexto distinto.

**MEM-07:** la sesión 3 reinicia contexto, KV cache, mensajes y estado transitorio. Solo recibe memoria persistida y su consulta permitida.

**MEM-08:** alterar el diálogo de sesión 2 en memoria de proceso después de guardar no cambia la sesión 3 si el almacenamiento persistido permanece igual.

**MEM-09:** ejecutar ramas en distinto orden no contamina sus memorias ni resultados deterministas.

**MEM-10:** modificar la respuesta latente sin modificar memoria persistida no cambia la predicción del lector; puede cambiar el scoring. Detecta lectores que consultan accidentalmente el mundo.

---

## 7. Evaluador y formato: puerta técnica obligatoria

### 7.1 Fixtures independientes

Construir **30 casos mínimos de infraestructura, PROPUESTA**, cubriendo las situaciones de la sección 6, preguntas irrelevantes, no respuesta y datos insuficientes. Cada caso debe tener al menos una salida válida y una mutación incorrecta.

El expected debe ser escrito o revisado independientemente del código que se prueba. No generar gold y score con la misma función y declarar independencia. Una mutación incorrecta debe modificar realmente la semántica relevante: cambiar una fecha que no afecta a ninguna consulta no exige que cambie C2/FJS, aunque sí puede cambiar una métrica de alcance.

En casos donde la verdad no es identificable desde la evidencia, una abstención o respuesta condicional fiel puede ser válida pero no resolver la tarea. Mantener ambas dimensiones.

Pruebas:

| ID | Mutación | Resultado esperado |
|---|---|---|
| EVA-01 | Respuesta correcta y sustentada | Éxito; recompensa de respuesta +1 |
| EVA-02 | Mismo valor, pero evidencia eliminada | No éxito justificado |
| EVA-03 | Valor equivocado | No éxito |
| EVA-04 | Alcance temporal equivocado | Error de alcance; fallo en consulta que lo distingue |
| EVA-05 | Contexto equivocado | Error contextual |
| EVA-06 | Abstención/condicional fiel | Tarea no resuelta, recompensa 0, no confundir con invención |
| EVA-07 | JSON sintácticamente incorrecto | Error de parseo, no excepción silenciosa ni score favorable |
| EVA-08 | JSON válido con slot/operación ilegal | Error de esquema/acción |
| EVA-09 | Salida truncada | Estado explícito; no reparar usando gold |
| EVA-10 | Misma semántica con orden de claves distinto | Mismo score |
| EVA-11 | Paráfrasis admitida por el contrato | Mismo score; fuera de cobertura, marcar limitación |
| EVA-12 | Evidencia citada que no sustenta la afirmación | No éxito sustentado |

**Aceptación:** todos los fixtures deterministas deben pasar. Esta exigencia se aplica al software y a salidas prefijadas, no a que el LLM acierte todos los ejemplos naturales.

### 7.2 No esconder problemas de interfaz

Guardar `raw_output`, `parsed_output`, `parse_status`, `schema_status`, `truncated`, `fallback_used` y `fallback_reason`.

Si hay normalización puramente sintáctica, guardar antes/después y demostrar que no introduce contenido. No usar un reparador que lee el gold.

La decodificación restringida es otra condición experimental: puede cambiar la elección de contenido, no solo el parseo. Debe aplicarse igual a los métodos comparados y recibir otro `condition_id`. No presentar la diferencia como efecto puro de RL.

---

## 8. Contratos de intervenciones y controles

### 8.1 Recuperar las definiciones originales

Exportar del código una tabla de O1/O2/I0/I1/I2: mensajes visibles al selector, respondedor, escritor y lector; acceso a la aclaración; presupuesto de interacción; memoria inicial.

O1/O2 no deben cambiar simultáneamente la consulta gold, el formato de salida o el evaluador. Si en realidad lo hacen, describir el contraste como compuesto y repararlo en una versión nueva.

No reinterpretar retrospectivamente I0/I1/I2. Conservar la ejecución histórica y crear una reproducción corregida con nuevo ID cuando sea necesario.

### 8.2 Controles de acción

- **C-NEVER:** ejecuta cero preguntas en todos los episodios del dominio declarado.
- **C-ALWAYS:** entrega exactamente una pregunta válida por episodio elegible. La regla de slot usa solo evidencia observable.
- **ASK-REFERENCE:** selecciona un slot de referencia que resuelve la variable pertinente. Es una intervención diagnóstica, no una política aprendida.
- **KEEP-INITIAL:** no actualiza memoria. Cuantifica cuánto éxito futuro proviene de conservar lo previo.
- **REFERENCE-WRITER:** escribe solo lo revelado a su rama. Aísla el potencial de mejorar escritura.

Reportar `ask_requested`, `ask_emitted`, `ask_delivered`, `slot`, `user_response_received` y `response_informative`. Preguntar sin obtener respuesta sigue teniendo coste de interacción.

**No exigir que C-ALWAYS sea bueno:** exigir que cumpla su contrato. Tampoco llamar a un controlador oracle que conoce el mundo una línea base justa del actor.

### 8.3 Control positivo para PI1

Repetir O1/O2 con B1 congelado y una interfaz validada, conservando B0 como antecedente de efecto suelo. No sustituir el contraste histórico en las tablas: nombrar el nuevo `PI1_v4`.

El controlador B4 sirve como referencia de resolubilidad bajo sus reglas. Inspeccionar si utiliza verdad latente; si la utiliza, separar `B4_observable` y `oracle_hidden` y no usar el segundo para afirmar que la tarea es resoluble por un actor con menos información.

---

## 9. Diagnóstico rápido sin reentrenar

### 9.1 Cohortes y límites

**PROPUESTA operativa:**

- `fixtures`: 30 casos sintéticos mínimos; no sostienen generalización.
- `smoke`: 10 raíces de desarrollo, cubriendo las familias existentes tras verificar su mapping.
- `diagnostic`: 50 raíces de desarrollo, con mezcla de necesidad actual, futura, casos claros y controles. Mantener todos los mundos/realizaciones asociados cuando la comparación lo requiera.
- Primera inspección con semilla 101 por regla fija, no por ser la más favorable. Repetir las comprobaciones sensibles a modelo en 20 raíces prefijadas con 202 y 303.
- Tope inicial: 600 llamadas nuevas al LLM. Es un límite de ejecución, no una estimación de duración ni de potencia. Antes de lanzar, calcular las llamadas pendientes. Lo que exceda el tope se marca `not_executed_budget` y no se convierte en `passed`.

Reutilizar las mismas escrituras al cambiar el lector; no regenerar sesión 2 innecesariamente. El banco ya observado puede usarse para localizar fallos, pero no será test independiente del método reparado.

### 9.2 Matriz mínima de adquisición

| Rama | Intervención | Componentes no intervenidos |
|---|---|---|
| D0 | Política natural | B1 congelado |
| D1 | PROCEED forzado | Respondedor/escritor B1 |
| D2 | ASK de referencia + respuesta fiel | Respondedor/escritor B1 |
| D3 | Misma pregunta + respuesta neutra explícita | Respondedor/escritor B1 |
| D4 | Misma evidencia de D2 + escritor de referencia | Lector común |

D2−D1 estima el efecto de la interacción completa. D2−D3 estima el efecto del contenido informativo bajo ese control neutro. D4−D2 estima el potencial de mejorar escritura con información ya adquirida. No sumarlas como causas exhaustivas: pueden interactuar.

El slot de referencia se decide a partir del estado observable y la estructura del problema, no eligiendo una pregunta diferente según la respuesta oculta. Con varios slots válidos, fijar desempate sin mirar el test.

Para casos claros, la respuesta redundante no debe inventar cambios. Incluir un atributo no afectado y una consulta posterior fuera del alcance de una excepción.

D3 no elimina todas las diferencias distribucionales: una respuesta neutra puede ser menos natural. Guardar esa limitación y probar una segunda formulación neutra solo como sensibilidad declarada.

### 9.3 Matriz de memoria y lector

Cruzar en una cohorte común:

| Memoria | Lector simbólico | Lector LLM común congelado |
|---|---|---|
| Referencia sustentada | Control de representación/evaluador | Capacidad de consumo con evidencia correcta |
| Producida por B1 | Calidad funcional de escritura | Circuito real escritor–lector |
| Inicial extraída | Efecto de extracción | Efecto de extracción más consumo |

Si hay un recuperador, agregar comparación entre selección natural e inyección del subconjunto relevante. Si toda la memoria entra directamente, registrar recuperación como `not_applicable`; no afirmar recuperación aprendida.

### 9.4 Depuración específica de E6

1. Usar las mismas raíces, memoria, consulta, lector y parser de E4.
2. Desactivar la interacción de sesión 3: `max_questions_s3=0`.
3. Exigir equivalencia de inputs y resultados deterministas con E4.
4. Activar una pregunta y registrar el turno recibido por el simulador.
5. Entregar una respuesta prefijada correcta; comprobar que llega al lector con el rol y orden adecuados.
6. Probar una respuesta no informativa, sin inyectar gold.
7. Separar errores de decisión, entrega, interpretación, formato y score final.

Si el lector sigue fallando incluso con evidencia correcta, no usar E6 como prueba de inutilidad de la información. Puede quedar como limitación del consumidor, sin bloquear una conclusión específicamente circunscrita al lector simbólico.

### 9.5 Resultado diagnóstico

Por cada caso inspeccionado, emitir: evidencia disponible, acción, respuesta recibida, memoria antes/después, consulta futura, respuesta futura, score y primer punto donde se viola el contrato o se pierde la información.

Una explicación de error sin esa trazabilidad se etiqueta `hypothesis`, no `confirmed_cause`.

---

## 10. Métricas que debe exportar la versión de cierre

### 10.1 Resultados de tarea

- `C2`: éxito actual correcto y sustentado.
- `FJS_primary`: éxito futuro en la consulta primaria predefinida.
- `J_primary = C2 * FJS_primary`, calculado por episodio antes de agregar.
- `R_long`, `R_short` y todos sus componentes.
- Categoría de respuesta: correcta sustentada, abstención/condicional fiel, incorrecta, no sustentada, inválida.

Si existen varios futuros, reportar por separado `FJS_mean`, `J_mean = C2 * FJS_mean` y `J_all = C2 * min_j(FJS_j)`. No reemplazar una métrica por otra sin cambiar su nombre.

### 10.2 Aclaración

Definir `need_now`, `need_future` y `resolvable_by_available_slots` desde mundos compatibles y tareas relevantes. No etiquetar toda incertidumbre como necesidad de preguntar. Distinguir información estructuralmente necesaria de utilidad neta para el consumidor concreto.

Métricas con numerador/denominador explícitos:

```text
necessary_omission = P(ask_delivered = 0 | need_any = 1, resolvable = 1)
unnecessary_ask = P(ask_delivered = 1 | need_any = 0)
slot_accuracy = P(slot pertenece a valid_slots | ask_delivered = 1, need_any = 1)
complete_acquisition = P(pregunta válida y slot pertinente y respuesta informativa | need_any = 1)
```

Reportar por separado decisión de ASK y ejecución de ASK para no atribuir fallos del transporte al selector. Si no hay denominador, usar `N/A`, no cero.

No comparar causalmente los episodios que preguntaron naturalmente con los que no: la política elige subconjuntos diferentes.

### 10.3 Memoria y alcance

Medir precisión del contenido escrito, actualizaciones sustentadas omitidas, alcance correcto, sobrepersistencia temporal, pérdida prematura, contaminación entre contextos y preservación de atributos ajenos.

Reportar tanto `unsupported_write_count` como `unsupported_write_episode`. El segundo no sustituye al primero al reconstruir una recompensa que penaliza cada escritura.

El estado “no se escribió nada” puede evitar invenciones y a la vez omitir una actualización necesaria. No interpretarlo automáticamente como memoria correcta.

### 10.4 Formato, recursos y errores técnicos

Separar `invalid_output_count`, `any_invalid_output` y validez por fase. Registrar truncamiento, reintentos, fallback, uso máximo de VRAM, tokens de entrada/salida/entrenamiento y tiempo por generación, simulación, optimización, lectura y validación.

No eliminar errores técnicos del denominador silenciosamente. Para la estimación del sistema operativo usar un tratamiento conservador preregistrado y presentar también análisis sobre ejecución completada. Una infraestructura que falla masivamente no ofrece una comparación de capacidad fiable.

---

## 11. Recompensa: conservar primero, verificar después

### 11.1 Definición inicial

Conservar inicialmente la fórmula declarada:

```text
R_long = r_actual + r_futuro
         - 0.10 * N_preguntas
         - 0.50 * N_escrituras_no_sustentadas
         - 0.20 * N_salidas_invalidas

R_short = r_actual
          - 0.10 * N_preguntas
          - 0.50 * N_escrituras_no_sustentadas
          - 0.20 * N_salidas_invalidas
```

Cada respuesta recibe +1 si es correcta y sustentada, 0 si es una abstención/condicional fiel que no resuelve, y −1 si es incorrecta, no sustentada o inválida. La implementación exacta y los denominadores deben recuperarse del código; no reconstruirlos suponiendo que todo fallo es −1.

### 11.2 Pruebas de recompensa

**REW-01:** recomputar desde trazas y comparar con el retorno almacenado, tolerancia numérica declarada.

**REW-02:** una pregunta redundante no obtiene un bono positivo por existir. Su beneficio solo puede venir de resultados reales, descontado su coste.

**REW-03:** no respuesta no crea evidencia ni convierte una escritura en sustentada.

**REW-04:** cambiar únicamente la respuesta/consulta futura no debe modificar la recompensa de entrenamiento definida como puramente inmediata, manteniendo constantes interacción actual y escritura. Si `N_salidas_invalidas` incluye errores del lector futuro, R_short ya usa una señal futura: corregir con nueva versión o dejar de llamarlo puramente inmediato.

**REW-05:** mantener separados el retorno del entorno y el término KL del optimizador. La varianza de penalizaciones no equivale a información sobre éxito de tarea.

**REW-06:** confirmar si una salida inválida recibe −1 en su respuesta y además −0,20. Si ese doble coste es deliberado, documentarlo; no corregirlo retrospectivamente sin versionar.

### 11.3 Utilidad contrafactual para diagnóstico

Calcular en estados de desarrollo:

```text
Q(s,a) = E[U(s,z,a) | evidencia observable s]
Delta(s,q) = Q(s,ASK(q)) - Q(s,PROCEED)
```

Promediar mundos compatibles antes de elegir la mejor acción. No usar `E[max_a U(s,z,a)]` como si fuera una decisión sin acceso al mundo oculto.

Comparar alternativas sobre el mismo mundo y aleatoriedad del simulador. Congelar el consumidor durante este diagnóstico para no mezclar cambios de actor y de evaluador.

**No introducir de entrada un bono por preguntar, por familia o por etiqueta de ambigüedad.** Sería una señal nueva que requiere su propia comparación.

Restar el mismo baseline a todos los retornos de un grupo no soluciona señal nula en GRPO:

```text
(R_i - b - mean(R - b)) / (std(R - b) + eps)
= (R_i - mean(R)) / (std(R) + eps)
```

La información adicional debe provenir de alternativas nuevas, de mejores consecuencias evaluadas o de una asignación de crédito distinta, no del nombre “contrafactual”.

---

## 12. Validación del entrenamiento antes del piloto RL

### 12.1 Identidad de modelos

- Cargar B1 de la semilla correspondiente y reproducir su evaluación paso cero sobre inputs idénticos.
- Verificar modelo base, tokenizer, plantilla, adaptador activo y configuración de generación.
- Comprobar que la referencia KL permanezca congelada y corresponda a ese B1.
- Desactivar LoRA no equivale a B1 si B1 reside en el adaptador: puede recuperar el modelo base sin ajustar. [T2]
- No fusionar dos veces ni cargar la continuación encima de un B1 ya fusionado de manera incompatible.
- Guardar y recargar el adaptador debe conservar logits/salidas dentro de la tolerancia previamente fijada para el mismo hardware. Diferencias numéricas entre entornos no deben confundirse con identidad exacta. [T2, T5]

### 12.2 Máscaras y optimizador

La loss solo debe incluir tokens generados por el actor que realmente se entrenan. Enmascarar sistema, usuario, simulador, padding y fases congeladas. Comprobar el desplazamiento de labels y log-probabilidades token a token.

En una prueba pequeña con señal conocida, verificar gradientes en parámetros LoRA, ejecución real de `optimizer.step`, cambio de pesos y cambio esperado de probabilidades. Un hash que cambia no basta para demostrar aprendizaje útil.

Registrar acumulación, clipping, pasos omitidos por precisión mixta y estado del scheduler. No usar un número llamado `step` para mezclar microbatches, grupos y actualizaciones del optimizador.

### 12.3 Fixtures GRPO

| Prueba | Criterio |
|---|---|
| Recompensas `[1,1,1,1]` | Ventajas de tarea cero, sin NaN |
| Recompensas diferentes, por ejemplo `[1,0,-1,0]` | Signos y normalización esperados según la convención declarada |
| Orden de trayectorias permutado | Mismo objetivo agregado; ventajas permutadas coherentemente |
| Actor y referencia idénticos | KL próxima a cero con el estimador y precisión declarados |
| Política old sin actualizar | Ratios coherentes con 1 antes de actualizar, bajo el mismo procesamiento |
| Tokens del simulador alterados en labels | No contribuyen como targets; sí pueden cambiar el contexto observable legítimo |
| Guardado/recarga | Se conserva la política efectiva |

Declarar la convención de desviación estándar y epsilon. No exigir números incompatibles entre `ddof=0` y `ddof=1`.

La distribución que genera trayectorias y las log-probabilidades empleadas deben tener un tratamiento consistente de temperatura, truncamiento de vocabulario y restricciones. No forzar ASK/PROCEED y tratarlos luego como muestras on-policy sin derivación o corrección.

---

## 13. Parámetros: qué tocar y en qué orden

### 13.1 Reglas

Primero recuperar los hiperparámetros históricos. Para reproducir una anomalía, mantenerlos. Para el piloto corregido, registrar todos los cambios y aplicar la misma configuración común a B2-L y B2-I.

No cambiar el rango LoRA al continuar un checkpoint existente: eso cambia la parametrización y requiere una transición explícita. No adoptar el rango 8 de una propuesta anterior como si fuera el rango histórico.

### 13.2 Configuración propuesta para un piloto nuevo

Estos valores son un punto de partida limitado, no una garantía de mejora ni de cabida en 4 GB.

| Parámetro | PROPUESTA inicial | Cuándo cambiar |
|---|---|---|
| Modelo/revisión/tokenizer | Conservar los reportados | Solo por problema demostrado; nueva condición |
| Rango, alpha y módulos LoRA | Heredar exactamente B1 | No cambiar en la comparación principal |
| Precisión | Mantener BF16 si la prueba real funciona | Verificar soporte y estabilidad; FP16/otra precisión es otra condición [T4] |
| Cuantización | No introducir inicialmente | Solo si el perfil de memoria lo exige; repetir baseline |
| Microbatch de forward/backward | 1 trayectoria | Aumentar solo con VRAM medida |
| Grupos por actualización | 2 estados observables/mundos, PROPUESTA | Mantener idéntico en RL-L/RL-I |
| Trayectorias por grupo `G` | 4, como se declaró históricamente | No aumentarlo antes de medir diversidad |
| Acumulación efectiva | 8 trayectorias por optimizer step | Implementar respetando grupos completos y normalización |
| LR RL | `5e-6` para el piloto nuevo | Probar `1e-6` si hay inestabilidad; no aumentar por señal cero |
| LR continuación SFT | `1e-5`; conservar una histórica menor si corresponde | En reproducción, usar la histórica; luego probar reducción aislada |
| Optimizador | Heredar tipo/betas/eps y registrar | No inventar los históricos |
| Scheduler piloto nuevo | Constante, sin warmup, declarado | No confundir con la receta histórica; mantener entre RL-L/RL-I |
| Norma máxima de gradiente | `1.0` | Medir norma anterior y posterior; no interpretar la anterior como fallo [T3] |
| Clipping de ratio | `0.20` | Mantener inicialmente |
| Coeficiente KL | `0.02` respecto a B1 | Mantener inicialmente; estudiar aparte si su contribución domina |
| Reutilización de cada rollout | Una iteración de optimización | Más reutilización exige validar old log-probs y clipping |
| Muestreo del piloto nuevo | Temperatura `1.0`, top-p `1.0`, sin top-k | Igual en las dos ramas; eval greedy |
| Decodificación de evaluación | Greedy, parámetros explícitos | No comparar con sampling por accidente |
| Contexto máximo | Conservar el necesario para toda la evidencia | Nunca truncar la cláusula temporal para ahorrar memoria |
| Salidas por fase | Medir y fijar límites que cubran el esquema | Reducir verbosidad, no borrar campos ni cortar JSON |
| Preguntas en sesión 2 | Presupuesto máximo común de 1 para la ruta mínima | Si el código histórico difiere, versionar el cambio |

Para un entrenador basado en TRL, estos son requisitos lógicos, no una copia literal de nombres de argumentos. La compatibilidad entre batch, generación y acumulación depende de la versión; comprobar el número real de grupos y muestras. No sustituir el loss original por un default distinto de la biblioteca. [T1]

En muestreo on-policy con una sola actualización y ratios inicialmente iguales a 1, un clipping observado de cero puede ser normal. Cambiar epsilon no crea exploración.

### 13.3 Perfil de memoria para 4 GB

Usar un solo modelo base residente cuando sea viable, con adaptadores actor/referencia separados o cálculo secuencial de referencia. Verificar que intercambiar adaptadores no altere la referencia ni active accidentalmente sus gradientes.

Generar grupos secuencialmente, conservar tokens y rewards fuera de GPU y calcular la loss con acumulación correcta. No mantener todos los grafos de generación. Guardar `logp_old` antes de actualizar; no recalcularla después con el actor nuevo.

Usar gradient checkpointing solo tras comprobar corrección. Separar cache de inferencia y configuración de entrenamiento. Evitar instalar un servidor de generación adicional como primer intento en una GPU de 4 GB.

Antes de la campaña, ejecutar forward, generación y backward en las longitudes máxima y típica. Registrar VRAM asignada/reservada y memoria libre. Si hay OOM, reducir concurrencia y longitud superflua primero. Cambiar cuantización o contexto semántico requiere nueva condición; no es una reparación invisible.

### 13.4 Regla de cambio según síntoma

| Síntoma comprobado | Acción prioritaria | Qué no hacer |
|---|---|---|
| GRPO recibe recompensas iguales y todas las acciones realmente son equivalentes | Reconocer saturación/no oportunidad en esos estados | Subir LR o premiar preguntar para fabricar señal |
| Recompensas muestreadas iguales, pero acciones legales tienen valores diferentes | Revisar cobertura; piloto de exploración o anexo discreto | Declarar incapacidad del modelo sin comparar alternativas |
| La aclaración correcta no ayuda al consumidor | Corregir interpretación/escritura/lector | Entrenar solo al selector durante más horas |
| JSON inválido/truncado | Corregir interfaz y cobertura de formato | Reducir penalización para esconder errores |
| Inestabilidad en B3-303 | Reproducir ejemplos 3400–3600 y revisar LR/máscaras/optimizador | Afirmar sobreajuste sin prueba |
| KL domina y la señal de tarea es útil | Medir gradientes de ambos términos; probar un único ajuste en desarrollo | Quitar KL y cambiar otras cinco cosas a la vez |
| Normas grandes antes del clipping | Medir después del clipping y cambio de pesos | Suponer que clipping no se ejecutó |
| Rendimiento bajo solo con extracción | Reparar la evidencia inicial | Culpar al horizonte de RL |

---

## 14. Telemetría, piloto y decisión de continuar RL

### 14.1 Campos obligatorios por grupo/actualización

Guardar raíz, mundo, hash de estado observable, acciones/slots, retornos por componente, `std_task`, `std_total`, ventajas, proporción de grupos con señal, diversidad semántica y entropía disponible.

Registrar loss de política y KL por separado; gradiente de tarea y de regularización en una muestra diagnóstica; norma pre/post clipping; ratios; clip fraction; LR; tokens entrenados; pasos efectivos; parámetros con gradiente; tiempo y VRAM. La documentación oficial de GRPO distingue estas métricas; su mera presencia no valida tu implementación. [T1]

Si se usa una restricción de acciones, medir también entropía sobre acciones legales. La entropía de tokens no equivale a diversidad de decisiones.

### 14.2 Piloto mínimo

**PROPUESTA:** semilla 101, máximo 5 actualizaciones reales por rama RL-L/RL-I con grupos pequeños, después de pasar las pruebas técnicas. Evaluar paso 0, 1 y 5 en el mismo smoke de desarrollo. No usar este piloto para una conclusión estadística de eficacia.

En paralelo, un diagnóstico sin aprendizaje de 20 estados de desarrollo compara acciones legales bajo el continuador congelado. Cuando la enumeración completa sea costosa, comenzar por PROCEED y slots candidatos declarados, marcándolo como enumeración parcial.

No mezclar en un grupo trayectorias de mundos distintos para el GRPO histórico. Cada grupo comparte estado y mundo; el muestreo del conjunto de entrenamiento debe cubrir los mundos compatibles sin que el actor los observe.

### 14.3 Semáforo técnico y de señal

- **Bloqueo inmediato:** filtración de gold, referencias incorrectas, NaN/Inf, máscaras erróneas, serialización defectuosa o intervención que no cumple su contrato.
- **Diagnóstico obligatorio:** más del 90% de grupos sin variación de recompensa de tarea sobre al menos 40 grupos observados. El 90% y los 40 grupos son umbrales operativos propuestos, no leyes estadísticas.
- **Continuación razonable:** contratos correctos, cambios reales de parámetros, ejemplos con señal de tarea y consumidor capaz de distinguir consecuencias pertinentes. No exige una mejora significativa con cinco pasos.

Contar grupos hasta completar la ventana mediante diagnóstico sin actualizaciones cuando sea necesario. Diez grupos del piloto de cinco pasos no satisfacen por sí solos el mínimo de cuarenta.

Si no hay señal, detener gasto y clasificar: acciones equivalentes, exploración insuficiente, consumidor incapaz, recompensa no discriminativa o fallo técnico. Con varias causas compatibles, declarar diagnóstico no identificado.

No descartar grupos iguales para re-muestrear hasta encontrar unos favorables sin registrar ese procedimiento: cambia la distribución de entrenamiento y el coste.

### 14.4 Límite de continuación

Después del piloto, fijar para las tres continuaciones **PROPUESTA: 3600 segundos de entrenamiento/rollout por rama y semilla**, como límite de recurso, no como promesa de duración total. La validación y el guardado tienen un presupuesto separado, fijado antes de lanzar y contabilizado; no ocultarlos como coste cero. Reducir el límite común es válido si se decide antes de ver el nuevo test.

La ruta por defecto compara recetas a presupuesto de tiempo. Sus checkpoints elegibles son el paso cero y los correspondientes a 1/3, 2/3 y al final del presupuesto, para SFT y las dos ramas RL. Completar la operación atómica que cruza el umbral, guardar y registrar tiempo real y sobrepaso; no iniciar otra operación de entrenamiento cuando el presupuesto se agotó. Registrar el paso real de cada checkpoint. Las evaluaciones técnicas tempranas de los pasos 1 y 5 del piloto no son candidatos adicionales de selección de la campaña final.

Si una rama se detiene antes por una regla técnica predeclarada, usar solo sus checkpoints válidos y declarar que tuvo menos oportunidades. Un checkpoint final adicional por interrupción se conserva para diagnóstico, no se vuelve elegible retroactivamente. No completar el presupuesto cambiando hiperparámetros a mitad de una rama.

No exigir un número concreto de pasos como garantía de suficiencia: registrar grupos con señal y actualizaciones efectivas. Si se prefiere un límite por pasos o tokens, fijarlo como otro diseño antes de abrir el test, no mezclarlo con la afirmación de presupuesto temporal común. No afirmar igualdad simultánea de pasos, tiempo y datos.

Al llegar al límite, exportar resultado incluso si no hubo mejora. No repetir hasta obtener una semilla favorable. Un ensayo corto sin señal suficiente se cierra como `inconclusive`, no como refutación general de RL.

---

## 15. Campaña mínima después de pasar las puertas

### 15.1 Brazos principales

| Brazo nuevo | Inicialización | Actualización |
|---|---|---|
| `B1-v4` | B1 seleccionado por semilla | Ninguna; evaluación con infraestructura v4 |
| `B3-v4` | Mismo B1 | Continuación SFT con datos permitidos |
| `B2-I-v4` | Mismo B1 | GRPO con recompensa inmediata verificada |
| `B2-L-v4` | Mismo B1 | Mismo GRPO más componente futuro |

Compartir modelo, slots, interfaz, simulador, escritor/lector según arquitectura, datos disponibles, semillas, presupuesto declarado y regla de selección. Mantener la misma arquitectura de fases entrenables en RL-I/RL-L. Si solo entrenas selector, nombrar `selector-only`; no afirmar que has aprendido también el escritor.

Si se añade un conjunto correctivo común de formato o memoria, debe utilizarse antes de bifurcar los brazos; denominar el nuevo inicio `B1c` y conservar B1 como antecedente. Si se proporcionan nuevos pares contrafactuales a RL, dar un control SFT sobre esos mismos pares.

No comparar únicamente un método nuevo con el RL histórico y atribuir toda diferencia a un hiperparámetro.

### 15.2 Selección

Usar únicamente validación de selección. Mantener elegible el inicio. Seleccionar máxima utilidad longitudinal bajo guardas de validez fijadas; empate dentro de tolerancia numérica, elegir el paso más temprano. La tolerancia es numérica, no un margen estadístico para preferir el resultado deseado.

RL-I también se seleccionará con el mismo criterio longitudinal para aislar el objetivo de entrenamiento. Explicar que no está libre de toda información futura. Una selección puramente inmediata sería otro experimento.

Evaluar inicio, seleccionado y final por transparencia, pero no elegir después el que funcionó mejor en test.

---

## 16. Datos nuevos, generalización y estadística

### 16.1 Separar desarrollo y confirmación

Los tests existentes ya observados conservan su papel histórico y sirven ahora para diagnóstico exploratorio. No reutilizarlos como confirmación independiente de v4.

Propuesta inicial, ajustable antes del bloqueo por precisión y recursos:

| Partición | Raíces propuestas | Uso |
|---|---:|---|
| Desarrollo diagnóstico | 50 | Detectar y reparar fallos |
| Validación nueva | 100 | Selección e hiperparámetros |
| Test nuevo ID | 100 | Repetición en distribución |
| Test nuevo de composición | 200 | Contraste principal de aprendizaje |
| Dominio reservado | Opcional, separado | Solo sostener transferencia de dominio si se ejecuta |

Estos tamaños son límites logísticos iniciales, no cálculo de potencia. Antes de generar/abrir el test final, estimar en desarrollo la precisión de diferencias pareadas y fijar el tamaño. Si el presupuesto no permite distinguir el efecto relevante, declararlo prospectivamente.

Separar todas las realizaciones, mundos, consultas y paráfrasis de una raíz. Reservar además plantillas/composiciones para el test de composición. Una nueva semilla del mismo generador no acredita diversidad estructural.

Mantener campos `root_id`, `template_id`, `composition_id`, `generator_version`, `domain` y `observable_state_id`. Evaluar duplicados exactos y equivalencias semánticas. Si se reutilizan plantillas, aclarar que la inferencia está condicionada a ellas o agrupar a ese nivel cuando sea el nivel independiente pertinente.

### 16.2 Consultas futuras de alcance

Conservar una consulta primaria por episodio para comparabilidad y éxito conjunto. Agregar un paquete diagnóstico de consultas:

- Dentro del período/contexto donde la nueva información aplica.
- Fuera del período/contexto, cuando existe una diferencia semántica real.
- Sobre un atributo no afectado.

No inventar una caducidad donde no fue expresada. Los probes no aplicables se marcan `N/A`. Su distribución se congela antes de evaluar métodos y no se elige según sus respuestas.

Inicialmente no añadir estos probes a la recompensa histórica. Si el diagnóstico demuestra que la consulta de entrenamiento no distingue alcance correcto e incorrecto, crear una variante longitudinal con futuros muestreados de una distribución fija y un control inmediato con el mismo resto del protocolo. Promediar pesos que sumen uno; no aumentar el premio duplicando preguntas favorables.

### 16.3 Contrastes nuevos

Propuesta de cinco contrastes predeclarados, con nombres distintos de los históricos:

| Contraste v4 | Métrica y cohorte |
|---|---|
| PI1: B1 O2−O1 | C2, test ID nuevo pareado |
| PI2: I2−I1 con B1 | FJS, cohorte nueva elegible de duración/alcance ambiguo |
| PI3a: RL-L−más SFT | J_primary, test de composición |
| PI3b: RL-L−RL-I | J_primary, mismo test |
| PI3c: RL-L−B1 | J_primary, mismo test |

La familia objetivo de PI2 debe definirse por el generador, no seleccionando los casos donde la intervención mejoró. Registrar F4 como tal solo después de verificar su semántica.

Añadir tres guardas para PI3a y PI3b: respuesta actual incorrecta/no sustentada, escritura no sustentada por episodio y error de alcance. **PROPUESTA de margen práctico:** no deterioro mayor a 0,02 en proporción, sujeto a justificación previa. No cambiarlo tras ver resultados.

Son once afirmaciones planeadas: cinco contrastes y seis guardas. Para control familiar conservador de 0,05, usar intervalos individuales con confianza `1 - 0.05/11` y declarar el método. Si cambia el número de afirmaciones, fijar de nuevo la familia antes de abrir test. No presentar intervalos individuales del 99% como cobertura familiar del 99%.

### 16.4 Agregación e incertidumbre

Calcular métricas por episodio; promediar realizaciones dentro de mundo, mundos dentro de raíz y raíces dentro de familia, con familias igualmente ponderadas como en la campaña, salvo cambio predeclarado. Mantener el mismo orden y pesos entre métodos.

Bootstrap pareado de raíces, estratificado por familia, 20.000 remuestreos y semilla de análisis fija. Conservar íntegra la estructura interna de cada raíz y el emparejamiento. Las semillas de entrenamiento no son nuevos usuarios: promediarlas para el contraste condicionado y mostrar además efectos por semilla y sensibilidad entre semillas.

No sumar grupos de control solapados como raíces distintas. En E3 histórico, 60+20+20 no corresponden a 100 raíces únicas; la auditoría registra 89. [F8]

Con cero discordancias, no declarar equivalencia a partir de un bootstrap `[0,0]`. Una cota binomial sobre eventos raíz requiere definir ese evento; no aplicar Clopper–Pearson a una media continua sin justificarlo. Reportar sensibilidad conservadora y limitaciones.

No confundir ausencia de significación con igualdad. Para concluir equivalencia o ausencia de una mejora relevante se necesita un margen predefinido y precisión suficiente. Para una guarda donde un valor mayor es peor, el límite superior del intervalo de la diferencia candidato−control debe quedar por debajo del margen de daño. Que el intervalo simplemente incluya cero no demuestra no inferioridad. Reportar también diferencias absolutas por familia y semilla, sin multiplicar afirmaciones confirmatorias a posteriori.

### 16.5 Revisión humana

Preparar un subconjunto ciego de 30 raíces, PROPUESTA logística, con evidencia, necesidad, slot, respuesta del simulador y alcance. Procurar dos revisores independientes y adjudicación de desacuerdos. Reportar qué se revisó y el acuerdo, sin convertirlo en garantía global.

Revisar etiquetas y semántica del banco antes de bloquearlo. Una revisión cualitativa posterior de errores de test no permite cambiar el método, elegir otro checkpoint o corregir selectivamente sus puntuaciones. Un defecto nuevo del evaluador exige versionado y reevaluación simétrica, con la limitación de reutilización del test declarada.

Si no se realiza, usar `human_audit: not_executed`. Una revisión por IA no sustituye evidencia humana independiente. La evaluación puede cerrarse como exploratoria rigurosa, no como validación con usuarios reales. [F5]

---

## 17. Exportación, trazas y reproducibilidad

### 17.1 Clave única

Usar un identificador de condición que codifique todos los factores de intervención. Clave primaria propuesta:

```text
run_id,condition_id,method,checkpoint_id,training_seed,root_id,
latent_world_id,realization_id,future_probe_id
```

Incluir además como columnas legibles: experimento, split, observación, intervención, control, lector, extracción, tasa de no respuesta, generación seed, réplica estocástica, versión de parser/evaluador/simulador y estado técnico. Las réplicas estocásticas tienen su propio ID cuando existen.

No deduplicar resultados legítimos de tasas 20% y 40%. No convertir un `null` de “no aplica” en una tasa 0 sin registrar la semántica.

### 17.2 Traza mínima por episodio

```text
IDs y versiones
estado latente solo en la parte privada del log
prompt exacto/token IDs o referencia hash verificable
memoria inicial y evidencia visible
acción solicitada, emitida y ejecutada
slot y pregunta entregada
respuesta del simulador y hechos efectivamente revelados
respuesta actual, parseo y score desglosado
operaciones de memoria, evidencia, antes/después
memoria serializada y recargada para sesión 3
consulta futura, evidencia recuperada si procede, respuesta y score
componentes de recompensa
latencia, tokens, truncamiento, fallos y reintentos
```

La verdad latente en logs es para auditoría, no para construir el prompt. Mantener serializadores distintos para actor y auditoría.

### 17.3 Caché y ahorro

Reutilizar únicamente cuando coinciden modelo efectivo, pesos, tokenizer, prompt, estado visible, configuración de generación y semilla relevante. Incluir el hash de memoria y del continuador cuando corresponda.

Una caché de trayectorias de un actor anterior no es muestreo on-policy de un actor actualizado. No reutilizarla para entrenar GRPO como si fuera nueva. Los resultados greedy idénticos de un baseline sin entrenamiento pueden compartirse entre rótulos de semilla, pero no contarse como entrenamientos independientes.

### 17.4 Archivos de salida

| Archivo | Contenido |
|---|---|
| `MANIFIESTO_V4.json` | Código, datos, modelos, versiones y hashes |
| `CONTRATOS_ENTORNO.md` | Semántica de familias, slots, observaciones, intervenciones, memoria y scoring |
| `MATRIZ_VERIFICACION.csv` | Cada propiedad y su evidencia |
| `PRUEBAS_ENTORNO.json` | Fixtures y resultados, incluidos fallos |
| `DIAGNOSTICO_CAUSAL.md` | Casos trazables, explicaciones confirmadas e hipótesis |
| `RESULTADOS_V4.csv` | Episodios con clave completa y denominadores |
| `CURVAS_RL_V4.csv` | Señal por grupo, KL, gradientes, acciones y recursos |
| `MEMORIAS_AUDITADAS.jsonl` | Estados de memoria y evidencia, antes/después |
| `PRESUPUESTOS_V4.csv` | Coste total y por componente |
| `DECISION_RL.json` | Continuar, reparar componente, cerrar inconcluso o cambio de ruta |
| `ANALISIS_V4.json` | Contrastes, guardas, incertidumbre y cobertura |
| `RESPUESTA_VICENT.md` | Respuesta directa PI1/PI2/PI3, sustentada en esta ejecución |
| `LIMITACIONES_V4.md` | Qué quedó sin ejecutar o sin identificar |

Cada tarea debe dejar un estado explícito. No generar archivos de “éxito” vacíos ni rellenar métricas ausentes con estimaciones.

---

## 18. Reglas de decisión y cierre

### 18.1 Puertas

| Puerta | Condición para pasar | Si falla |
|---|---|---|
| G0: procedencia | Artefactos críticos y versiones identificados | Bloquear réplica exacta afectada |
| G1: infraestructura | Fixtures deterministas, no filtración, aislamiento, scoring e intervenciones correctos | Reparar antes de interpretar resultados |
| G2: diagnóstico | Anomalías conocidas explicadas o delimitadas con evidencia | Restringir alcance; no llamar válido al componente desconocido |
| G3: entrenamiento | Identidad, máscaras, referencia, gradientes y recarga correctos | No lanzar continuación |
| G4: señal | Se ha caracterizado la oportunidad de mejora y el piloto no tiene fallos técnicos | Continuar, cambiar componente o cerrar sin más gasto |
| G5: evaluación final | Protocolo y test congelados, contraste pareado y guardas informadas | Mantener carácter exploratorio o inconcluso |

E4/E5/E6 no tienen que obtener puntuaciones altas para pasar una auditoría: tienen que medir correctamente y tener entradas/contratos verificables. Un lector incapaz no es necesariamente un entorno defectuoso.

### 18.2 Cuándo declarar mejora

Declarar ventaja de RL-L sobre un comparador solo si el contraste predefinido la respalda y las guardas de daño cumplen su criterio. Mostrar tamaño de efecto, incertidumbre, semillas, coste y checkpoint seleccionado.

Si selecciona paso cero, informar que la receta conservó B1. No atribuir su calidad a las actualizaciones.

Si mejora FJS pero empeora respuestas actuales o alcance, describir el intercambio; no llamarlo mejora global.

### 18.3 Cierres válidos

- Entorno auditado y mejora específica respaldada.
- Entorno auditado, pero ninguna ventaja adicional de RL identificada con precisión suficiente para el margen acordado.
- Entorno auditado, ensayo de aprendizaje inconcluso por presupuesto/señal/precisión.
- Fallo localizado en extracción, escritura o lector; conclusión de aprendizaje restringida a los componentes válidos.
- Infraestructura todavía bloqueada, con una causa reproducible y requisitos concretos pendientes.

No prolongar entrenamientos o cambiar el test hasta lograr un resultado favorable.

---

## 19. Plantilla de respuesta a Vicent

Rellenar únicamente con resultados realmente ejecutados. Mantener huecos como `no ejecutado` cuando corresponda.

### PI1 — Evidencia disponible

> Comparé O1 y O2 manteniendo constantes modelo, consulta y evaluación. El contraste histórico con B0 quedó limitado por su efecto suelo de formato/rendimiento. En la versión auditada, [método] obtiene una diferencia de [valor e intervalo] en C2. Esto [respalda/no identifica] un efecto de la evidencia disponible bajo estas condiciones. El controlador de reglas es una referencia de resolubilidad, no una prueba de aprendizaje del LLM.

### PI2 — Aclaración y persistencia

> Verifiqué qué información recibe el escritor en I1 e I2 y que la sesión posterior usa únicamente memoria persistida. La diferencia futura fue [valor e intervalo] en los casos elegibles. En casos claros y fuera de alcance observé [resultados de guardas]. Esto identifica el efecto de disponibilizar la aclaración para escribir, pero no demuestra por sí solo selección autónoma óptima de preguntas.

### PI3 — Qué añade RL

> Partí del mismo B1 por semilla y comparé más SFT, RL inmediato y RL longitudinal bajo [presupuesto y selección]. RL longitudinal frente a más SFT dio [resultado]; frente a RL inmediato, [resultado]. Los pasos seleccionados fueron [pasos]. Por tanto, [conclusión limitada]. La selección de RL inmediato también utilizó validación longitudinal; el contraste se refiere al objetivo de entrenamiento, no a ausencia total de información futura.

### Estado del sistema y aporte

> El entorno pasó [pruebas] y quedaron pendientes [limitaciones]. La evaluación distingue errores de adquisición, interpretación, escritura y consumo de memoria. El aporte de esta ejecución es [hallazgo demostrado], no asegurar que RL deba ganar ni que el comportamiento se generalice a usuarios reales.

El informe histórico ya contiene un efecto PI2 fuerte y no demuestra superioridad de RL frente a más SFT ni frente al horizonte inmediato. Preservar esos resultados aunque cambie la versión de evaluación. [F3, F4, F6]

---

## Anexo A. Alternativa acotada si se demuestra falta de exploración

**Activar solo si:** el circuito técnico pasa, un consumidor congelado aprovecha la información y enumerar acciones descubre diferencias que el muestreo no visita. No es un paso obligatorio para terminar la evaluación actual.

Como los slots ya son finitos, definir una política sobre `PROCEED` y `ASK(slot)` con continuador B1 y lector congelados. Precalcular retornos de entrenamiento por estado y acción con mundos compatibles, sin datos del nuevo test.

```text
J(theta) = mean_s [ sum_a pi_theta(a|s) Q_long(s,a)
                   - beta * KL(pi_theta(.|s) || pi_B1(.|s)) ]
```

Las probabilidades deben normalizarse sobre el mismo conjunto legal usado en ejecución. Si se usan secuencias para representar acciones, especificar su probabilidad completa y la distribución inducida; no promediar log-probabilidades por longitud arbitrariamente.

Comparar, sobre los mismos estados y datos de valor:

| Brazo | Propósito |
|---|---|
| B1 selector congelado | Referencia |
| SFT-CF | Imitación de acciones de alto valor con los mismos pares |
| RL-CF-I | Optimización de retorno inmediato |
| RL-CF-L | Optimización de retorno longitudinal |

Los empates no deben favorecer ASK por defecto. Usar una regla predeclarada, por ejemplo menor coste, y documentar cómo se tratan valores inciertos.

Esto es una política discreta de retorno esperado bajo continuaciones congeladas. No es el GRPO original de trayectoria completa, no aprende generación libre de preguntas y no demuestra aprendizaje de escritura si el escritor está congelado.

Coste de precálculo, llamadas al simulador y futuros evaluados forman parte del presupuesto. El ahorro proviene de reutilizar continuaciones congeladas, no de fingir que ese cómputo es gratuito.

Si la escritura sigue siendo el cuello de botella, no añadir otro ciclo de RL al selector. Corregir o estudiar el escritor bajo la misma evidencia y evaluar su contribución por separado.

---

## Anexo B. Instrucción para un agente de implementación local

> Trabaja sobre el repositorio existente y conserva la campaña histórica. Implementa esta SPEC por etapas. Primero descubre los módulos reales y documenta sus contratos; no crees un simulador sustituto sin justificarlo. Ejecuta fixtures y diagnósticos pequeños antes de entrenar. Para cada afirmación usa estados verificado, fallido o pendiente con evidencia trazable. Corrige exportación y controles rotos; no modifiques etiquetas, recompensas o denominadores para elevar las puntuaciones. Mantén aislada la información latente del actor. Reproduce B3-303 y valida la referencia/máscaras de GRPO. Ejecuta el piloto solo cuando sus dependencias pasen. Respeta los presupuestos, conserva resultados negativos y genera RESPUESTA_VICENT.md según PI1, PI2 y PI3. No afirmes revisión humana si no existió. No declares superioridad por existencia de un adaptador ni por seleccionar paso cero. Si faltan archivos, entrega un diagnóstico parcial con las dependencias bloqueadas, sin inventar resultados.

### Interfaz de ejecución que debe implementarse o mapearse

Los siguientes nombres son una **interfaz propuesta**, no comandos existentes verificados:

```text
closure_v4 inventory
closure_v4 test-environment
closure_v4 diagnose --cohort diagnostic
closure_v4 test-training
closure_v4 pilot-rl
closure_v4 lock-protocol
closure_v4 train-final
closure_v4 evaluate-locked-test
closure_v4 report
```

Crear un `RUNBOOK.md` con los comandos reales del repositorio y las dependencias de cada etapa. `evaluate-locked-test` debe rechazar ejecución sin manifiesto bloqueado y checkpoints seleccionados en validación.

---

## Anexo C. Fuentes y separación entre evidencia y propuesta

### Archivos de la campaña

- **[F1]** `ANALISIS.json`: `groups`, `primary_contrasts`, `metadata`, `source_traces` y `training_artifacts`. Estado exploratorio; hashes de trazas referidas, no acceso automático a ellas.
- **[F2]** `INFORME_OBSERVABILIDAD.md`: tablas E1 y contraste PI1.
- **[F3]** `INFORME_SFT_RL.md`: tablas E2 y “Entrenamiento realmente ejecutado”.
- **[F4]** `INFORME_VALOR_PERSISTENCIA.md`: ramas I0/I1/I2 y controles.
- **[F5]** `LIMITACIONES_Y_DESVIACIONES.md`: revisión/modelo, semillas, lector simbólico, ausencia de revisión humana y presupuesto.
- **[F6]** `RESPUESTAS_DIRECTOR.md`: PI1 como evidencia disponible, PI2 como disponibilidad para escritura y PI3 como recetas bajo presupuesto.
- **[F7]** `RESULTADOS.csv`, `CURVAS_ENTRENAMIENTO.csv`, `PRESUPUESTOS.csv`: episodios, curvas y recursos.
- **[F8]** `auditoria_evaluaciones_v3/LEEME_Y_DICTAMEN.md`, `DIAGNOSTICOS_PAREADOS.csv` y `EXITO_CONJUNTO.csv`: auditoría anterior generada en esta conversación, con sus límites. No es validación humana independiente.

La SPEC V2 precede al cierre de E3–E6. Esta versión incorpora la campaña completa y prioriza auditar el entorno antes de ejecutar su propuesta alternativa de política discreta.

### Documentación técnica consultada el 24 de octubre de 2026

- **[T1]** Hugging Face, GRPO Trainer: objetivo, diferencias de implementación y métricas de señal, entropía, KL y clipping. `https://huggingface.co/docs/trl/grpo_trainer`
- **[T2]** Hugging Face, PEFT checkpoint format: separación entre adaptador, configuración y modelo base. `https://huggingface.co/docs/peft/main/developer_guides/checkpoint`
- **[T3]** PyTorch, `torch.nn.utils.clip_grad_norm_`: clipping y norma retornada. `https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html`
- **[T4]** PyTorch, `torch.cuda.is_bf16_supported`: comprobación de soporte; no sustituye al perfil real. `https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html`
- **[T5]** PyTorch, Reproducibility: límites de reproducibilidad entre versiones/plataformas y control de aleatoriedad. `https://docs.pytorch.org/docs/stable/notes/randomness.html`

Estas fuentes orientan las pruebas técnicas. No demuestran que una LR concreta, una ventana de contexto o un número de pasos sean óptimos para esta tesis. Los tamaños de cohortes, límites, márgenes y configuraciones propuestos deben registrarse antes de la ejecución correspondiente.

**Criterio final:** terminar con un resultado verificable, aunque sea negativo o inconcluso, es preferible a una mejora numérica cuya causa no puede identificarse.
