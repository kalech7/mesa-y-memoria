# SPEC-TESIS-CLARIFICACION-MEMORIA-001

## Aclaración selectiva y memoria temporal: valor presente, persistencia y utilidad futura en un agente LLM

**Versión:** 1.0  
**Fecha:** 23 de octubre de 2026  
**Tipo:** protocolo de investigación y especificación de implementación  
**Estado:** listo para implementar; la ejecución experimental exige superar las puertas de validación de este documento.  
**Sustituye como siguiente etapa:** SPEC-LLM-LOCAL-001. No invalida ni reemplaza los resultados históricos.  
**Idioma del banco principal:** español.  
**Dominio:** preferencias alimentarias no clínicas, sintéticas y cambiantes.  
**Ejecución:** local; sin APIs de inferencia durante generación experimental, entrenamiento ni evaluación.

> **Decisión central.** La investigación no intentará demostrar por obligación que RL supera a SFT. Estudiará si una aclaración selectiva aporta información útil, si esa información debe persistirse con alcance temporal o contextual, y qué ventaja incremental aporta optimizar las consecuencias futuras. La contribución principal será un protocolo longitudinal intervenible y evidencia reproducible; una superioridad de RL será un resultado posible, no un requisito de éxito.

> **Límite de la palabra “definitivo”.** Esta versión fija las preguntas, comparaciones, contratos y reglas de análisis. El modelo exacto, sus revisiones, las dependencias y el presupuesto computacional se cerrarán mediante la inspección técnica G0, antes de abrir el test. No se presupone hardware que no se ha inspeccionado, código ya implementado ni resultados todavía inexistentes.

---

## 0. Guía de ejecución y procedencia

### 0.1. Orden obligatorio

Implementar y validar en este orden:

1. Inventario del repositorio y cierre técnico G0.
2. Entorno, memoria, simulador, evaluador y pruebas de no filtración.
3. Banco, particiones, auditoría independiente y congelación del protocolo.
4. Baselines y experimento de observabilidad.
5. SFT y, después, sus tres continuaciones: RL longitudinal, SFT adicional y RL inmediato.
6. Evaluación cerrada, intervenciones sobre persistencia y pruebas de robustez.
7. Análisis estadístico y redacción de conclusiones, incluidas las negativas.

**No comenzar entrenamiento largo antes de verificar el evaluador y los límites de información. No ampliar el banco mirando resultados de test.**

### 0.2. Cómo interpretar las afirmaciones

- **[P0]** identifica antecedentes reportados por el resumen del proyecto, no una auditoría del código o de sus registros.
- **[R1]–[R5]** identifican trabajos relacionados. Las diferencias propuestas se limitan a las versiones revisadas.
- **[D1]–[D3]** identifican documentación técnica oficial consultada.
- Todo umbral, tamaño, fórmula de utilidad, esquema, experimento e instrucción de implementación no atribuido expresamente a una fuente es una **decisión propuesta en esta SPEC**, no un resultado de los artículos.

Las referencias completas están al final. No se asume acceso a la SPEC anterior íntegra, al repositorio, a los mensajes originales del director ni a los informes que el resumen solamente menciona.

### 0.3. Antecedentes que justifican la nueva etapa

El resumen [P0] reporta que O1 obtuvo 60 % de error frente a 0 % de O2 y O3 en 240 decisiones, y que el piloto supervisado, RL, supervisión adicional y reglas empataron en 0/240 errores. También reporta que el mejor checkpoint de RL fue el inicial en las tres semillas. Estos resultados motivan controlar observabilidad, saturación y generalización; **no se reutilizarán como evidencia de una ventaja de RL**.

El resumen señala persistencia con SQLite e historial temporal. Antes de modificar el proyecto, verificar que esos componentes existen y cuáles de sus pruebas continúan funcionando. No asumir que los nombres de módulos o comandos propuestos aquí ya están implementados.

---

## 1. Problema, vacío delimitado y contribución

### 1.1. Problema

Un agente puede responder correctamente a la solicitud presente y aun así deteriorar su memoria. Por ejemplo, puede convertir una preferencia válida solo hoy en una regla permanente. También puede hacer una buena pregunta, usar la respuesta inmediatamente y perderla al terminar la sesión.

Se estudiará el ciclo:

**evidencia disponible → decisión de aclarar → respuesta del usuario → actualización de memoria → reutilización posterior.**

El objetivo no es eliminar toda contradicción. Algunas diferencias corresponden a contextos compatibles, excepciones o información insuficiente. La memoria debe conservar esas distinciones.

### 1.2. Lo que ya cubren los artículos de partida

| Fuente | Capacidad que ya estudia | Diferencia concreta de este protocolo |
|---|---|---|
| [R1] Memory-R1, §3 y figura 2, pp. 3–4 | Aprendizaje de operaciones de memoria y utilización de recuerdos mediante recompensas de QA; emplea un Answer Agent congelado al entrenar al Memory Manager. | Añadir adquisición activa de evidencia del usuario y aislar el efecto de hacerla persistente en una sesión posterior. No reclamar como nueva la optimización de memoria por resultados posteriores. |
| [R2] CIGAsk, §3 y figura 2, pp. 2–5 | Decidir cuándo aclarar y obtener información útil dentro de un rollout de QA, con CIG y una bonificación basada en ambigüedad. | Evaluar una frontera entre sesiones: escribir lo aclarado con alcance correcto y medir su reutilización tras eliminar el contexto transitorio. No reclamar como nueva la aclaración entrenada con RL. |
| [R3] TANGLE, §3.1, §3.4 y §4.5 | Conflictos sin respuesta única, separación entre memoria de referencia y pipeline, evaluación conductual y CAAP. | Ejecutar la interacción con el usuario, aplicar escrituras y observar consecuencias posteriores; no limitarse a valorar una respuesta que propone preguntar. No llamar “resolución” a imponer una respuesta única. |

### 1.3. Trabajos adicionales que obligan a acotar la novedad

La revisión externa identificó dos antecedentes particularmente próximos:

**MCB / Remember, Verify, or Ask? [R4].** Ya estudia la frontera entre persistir, usar localmente, verificar y aclarar, y evalúa decisiones y selección estructurada de herramientas. Por tanto, “decidir si preguntar antes de guardar” no se presentará como idea inédita. Nuestra distinción propuesta es ejecutar el ciclo interactivo, observar memoria realmente persistida y realizar intervenciones emparejadas sobre la disponibilidad de la respuesta en la fase de escritura.

**CAPTURE [R5].** Ya combina cambios de preferencias, memoria con distintas escalas temporales, aclaración y auditoría contrafactual, con énfasis en autenticidad de las preferencias y contaminación de memoria. También incluye evaluaciones longitudinales. Por tanto, no se afirmará que combinar aclaración, memoria y tiempo sea nuevo. Aquí se delimita un estudio no adversarial del valor diferido de la aclaración y del entrenamiento del controlador LLM, con comparación SFT/RL, horizonte de recompensa y una intervención explícita que separa uso inmediato de persistencia.

**Conclusión de posicionamiento:** hay una oportunidad de investigación específica, no evidencia suficiente para afirmar que nadie haya estudiado esta combinación. La búsqueda realizada es focalizada, no una revisión sistemática exhaustiva.

### 1.4. Aportación que se intentará demostrar

> Un protocolo controlado para estimar cuándo una aclaración sobre el alcance de una preferencia mejora decisiones futuras porque su respuesta se incorpora correctamente a memoria, y para distinguir ese efecto del acceso a mejor historial, del entrenamiento supervisado y de la optimización mediante RL.

Los entregables científicos serán: un banco auditable de episodios; un entorno con acciones y persistencia efectivas; estimaciones emparejadas del valor de adquirir y conservar información; y una comparación de métodos con presupuesto documentado.

**La aplicación a alimentos, la elección de LoRA y el uso de GRPO no constituyen, por sí solos, la novedad.**

---

## 2. Preguntas de investigación e hipótesis

| ID | Pregunta | Contraste principal | Resultado que podría refutar la expectativa |
|---|---|---|---|
| PI1 | ¿Cuánto ayudan el historial y sus relaciones temporales cuando se mantiene fija la política? | O2 frente a O1 sobre los mismos estados previos a actuar. | No hay mejora detectable o el efecto depende de señales filtradas. |
| PI2 | ¿Qué beneficio inmediato y futuro tiene una aclaración, y qué parte del beneficio futuro depende de permitir que su respuesta llegue a la memoria persistente? | Intervenciones I0, I1 e I2, definidas en §12. | Preguntar no ayuda, o conservar la información no mejora la reutilización. |
| PI3 | ¿Qué beneficio incremental aporta RL longitudinal frente a SFT, SFT adicional y RL con recompensa inmediata? | B2-L frente a B1, B3 y B2-I. | Las mejoras son nulas, imprecisas, explicadas por mayor coste o acompañadas de daño en memoria. |

Hipótesis direccionales, no supuestos del evaluador:

**H1.** O2 reducirá errores atribuibles a información histórica ausente, pero no eliminará la ambigüedad cuyo dato decisivo nunca fue expresado.

**H2.** En casos de ambigüedad de alcance, permitir que la respuesta aclaratoria llegue al escritor mejorará la decisión futura, aun cuando la respuesta actual sea idéntica.

**H3.** La recompensa longitudinal puede inducir mejores decisiones de adquisición y escritura que una recompensa inmediata. Que esa diferencia supere a SFT adicional es una cuestión empírica independiente.

**H4, secundaria.** Las mejoras bajo evidencia de referencia pueden disminuir cuando la extracción pierde relaciones relevantes.

Estas preguntas operacionalizan las recomendaciones de mejorar observabilidad y reducir el experimento reportadas en [P0]. No se atribuyen al director como citas literales.

---

## 3. Alcance cerrado

### 3.1. Núcleo obligatorio

Un solo backbone local; adaptadores LoRA sobre ese backbone; tres sesiones por episodio; cinco familias de escenarios; decisiones estructuradas; simulador determinista; evaluación principal verificable sin juez LLM; tres semillas de entrenamiento, **101, 202 y 303**.

Se entrena la política lingüística que selecciona la aclaración, responde en la sesión de adquisición y decide la escritura. **No se sustituye por una red pequeña sobre características diseñadas manualmente.**

### 3.2. Separación deliberada entre controlador y lector

Para aislar adquisición y escritura, la lectura de la tercera sesión se mantiene fija:

- **Lector principal:** función determinista documentada que opera sobre la memoria estructurada producida por cada política. No conoce el estado verdadero.
- **Lector de robustez:** el mismo backbone base, congelado, con un prompt común, recibe esa misma memoria y una consulta posterior. No recibe el diálogo anterior ni entrenamiento adicional.

El controlador y el lector pueden ejecutarse secuencialmente en el mismo equipo. No es necesario mantener dos copias en GPU simultáneamente.

**Qué permite afirmar este diseño:** efecto de la política LLM de adquisición/escritura sobre una reutilización controlada, y su transferencia a un lector LLM fijo.

**Qué no permite afirmar:** que se haya entrenado de extremo a extremo todo un asistente o que se haya optimizado también su capacidad de lectura. Una extensión con lector entrenable requiere otra SPEC.

### 3.3. Fuera de alcance

No incluir nutrición clínica, alergias, medicación, recomendaciones médicas, contaminación adversarial, herramientas web, conflictos de autoridad entre terceros, diálogos humanos en vivo ni generación abierta de preguntas como contribución principal. No reproducir las arquitecturas completas de los cinco trabajos relacionados.

Las preguntas se seleccionan por variable y se verbalizan mediante plantillas comunes. La tesis evaluará **cuándo preguntar y qué variable solicitar**, no la calidad literaria de una pregunta libre.

---

## 4. Episodio, estado oculto y límites de información

### 4.1. Unidad experimental

La unidad básica es una **historia raíz** `root_id`. Sus paráfrasis, mundos latentes alternativos y condiciones de intervención pertenecen al mismo grupo estadístico y a la misma partición.

Un episodio tiene:

| Sesión | Función | Acceso de la política |
|---|---|---|
| S1 | Establecer antecedentes y preferencias previas. | Memoria inicial construida de forma común a todas las variantes. |
| S2 | Recibir una declaración o consulta; decidir si aclarar; responder; escribir memoria. | Memoria visible según O1/O2, mensajes actuales y, solo si pregunta, respuesta del simulador. |
| S3 | Consultar nuevamente después del cierre de S2. | Exclusivamente memoria persistida, fecha/contexto actual explícito y nueva consulta. |

La inicialización de S1 es común y no se aprende en el experimento principal. El entrenamiento se concentra en S2. La consecuencia de S3 puede recompensar las acciones de S2.

### 4.2. Frontera real entre sesiones

Al terminar S2 se deben eliminar: mensajes de trabajo, respuesta inmediata del asistente, respuesta del usuario no guardada, variables de Python con contenido conversacional accesible al agente, caché KV y resúmenes transitorios.

S3 abre una nueva instancia de lectura desde SQLite. El registro de auditoría existe para evaluar, pero **no es una memoria recuperable por el agente**.

El modelo no actualiza sus pesos durante evaluación. No hay aprendizaje entre usuarios de test.

### 4.3. Reutilización sin canal de aclaración

En la evaluación principal, S3 es una reutilización autónoma sin posibilidad de consultar nuevamente al usuario. Esta condición se informa de forma general al controlador desde el inicio: habrá una consulta posterior del mismo atributo, pero no se revelan su texto, respuesta ni contexto futuro oculto.

Esto hace identificable el valor de adquirir información antes de perder el canal de interacción. Es una condición experimental, no una afirmación de que todos los asistentes funcionen así.

Una evaluación secundaria permite una pregunta en S3 y mide la carga adicional de interacción. No se mezclan ambos regímenes en una única exactitud.

### 4.4. Estado oculto

El simulador conserva privadamente: intención, alcance temporal real, condiciones contextuales y guion de eventos. Solo el simulador y el evaluador tienen acceso a estos campos.

El agente no recibe: `scenario_family`, `needs_clarification`, `gold_action`, `latent_scope`, `gold_preference`, mensajes futuros, etiquetas de split, etiquetas de dificultad o nombres de archivo que codifiquen la respuesta.

Los identificadores internos de experimento no se incluyen en el prompt. Los IDs visibles de memoria se generan sin correlación con tipo o valor verdadero.

### 4.5. Supuesto de evolución del mundo

En el núcleo, las preferencias continúan según el alcance comunicado y los cambios se expresan en los eventos del episodio. No se exige adivinar cambios futuros silenciosos.

La prueba de incertidumbre/no respuesta relaja la posibilidad de recuperar el dato, no introduce arbitrariamente cambios invisibles que después se puntúen como errores evitables.

---

## 5. Familias y pares contrafactuales

### 5.1. Cinco familias obligatorias

| ID | Familia | Qué se comprueba |
|---|---|---|
| F0 | Preferencia estable | Reutilizar información suficiente, no preguntar ni reescribir por rutina. |
| F1 | Cambio duradero explícito | Actualizar la preferencia vigente sin convertir la historia anterior en falsa. |
| F2 | Excepción temporal explícita | Guardar o utilizar una excepción con vencimiento, sin reemplazar indebidamente la regla general. |
| F3 | Preferencia dependiente del contexto | Usar una relación contexto–preferencia; aclarar el contexto actual cuando falte, sin suponer que ese contexto persistirá. |
| F4 | Alcance de cambio ambiguo | Distinguir respuesta presente de autorización para persistir una preferencia. |

F3 tendrá consultas con contexto explícito y omitido. F4 tendrá distintas formulaciones de ambigüedad y controles donde el alcance sí se comunica. No exponer estos subtipos al modelo.

### 5.2. Caso decisivo F4: beneficio exclusivamente futuro

Ejemplo ilustrativo, que debe revisarse por ambigüedad lingüística antes de convertirse en dato:

**S1:** «En general prefiero ensalada para cenar».

**S2:** «Ahora prefiero pasta. ¿Qué me recomiendas para la cena de hoy?»

En dos mundos compatibles con ese mismo mensaje, la selección actual es pasta. Lo desconocido es el alcance:

- Mundo A: el cambio es solo para hoy.
- Mundo B: es la nueva preferencia habitual hasta que se comunique otro cambio.

La pregunta `TEMPORAL_SCOPE` obtiene, respectivamente, una respuesta equivalente a «Solo para hoy» o «Como preferencia habitual desde ahora».

**S3, otra fecha:** se solicita una nueva sugerencia sin reabrir la aclaración. A requiere preservar la preferencia general anterior; B requiere aplicar el cambio duradero.

Antes de la aclaración, los prompts de A y B deben ser idénticos byte a byte. La distribución entre ambos mundos será equilibrada. El profesor automático no puede etiquetar una escritura duradera antes de observar la respuesta.

**Interpretación:** una mejora futura no debe confundirse con mejora en la selección actual, que ya estaba determinada.

### 5.3. Controles que evitan una conclusión trivial

F0/F1/F2 deben contener suficiente evidencia para actuar sin preguntar. F3 debe incluir casos donde aclarar ayuda ahora pero no autoriza trasladar el contexto actual a otra sesión. En parte de F4, la información aclarada dejará de ser aplicable por una fecha de expiración explícita; conservarla no significa seguir aplicándola indefinidamente.

También se crearán consultas de control sobre un atributo no afectado. La aclaración no debería cambiar esas respuestas.

### 5.4. Estructura narrativa

Cada historia tendrá antecedentes breves, de cuatro a ocho registros, y como máximo tres distractores no decisivos. Variar orden narrativo, referencias temporales, negación, elipsis y formulaciones naturales. No aumentar la longitud hasta exceder el contexto como mecanismo artificial de dificultad.

Un mismo mundo puede admitir varias operaciones de memoria equivalentes. El evaluador no exigirá coincidir con una secuencia textual única de acciones.

---

## 6. Observaciones y condiciones de extracción

### 6.1. O1 y O2

**O1 — última memoria.** Entregar solamente el registro más reciente del atributo consultado, con sus campos realmente disponibles, además del mensaje actual. No seleccionar la entrada usando relevancia gold.

**O2 — historial temporal.** Entregar todos los registros del atributo y los distractores previstos, con fechas, fuentes y alcances efectivamente documentados. Mantener un orden canónico común a todas las políticas.

El contraste O2–O1 mide el efecto del conjunto de evidencia disponible, no una eficiencia de tokens equivalente. Reportar longitud de entrada y truncamientos. El núcleo se diseña para que ninguna condición se trunque.

### 6.2. Ablación temporal secundaria

Se puede eliminar el campo de fecha de registro conservando el texto para medir el valor de ese metadato. Se llamará **“sin timestamp estructurado”**, no “sin información temporal”, porque el texto podría seguir conteniendo «hoy», «desde enero» o «solo esta semana».

No implementar O3 por defecto. Solo añadir un campo enriquecido cuando exista una hipótesis específica y pueda obtenerse sin conocer el futuro.

### 6.3. REF y EXTRACT

**REF:** memoria inicial de S1 curada a partir de evidencia explícita. Es una condición de referencia, no acceso a la intención latente de S2.

**EXTRACT:** un extractor local congelado procesa S1 cronológicamente y produce el mismo esquema. Sus salidas se cachean y comparten entre todas las políticas. S2 continúa siendo un mensaje visible de usuario, no una etiqueta perfecta de su significado.

REF es el núcleo de entrenamiento y comparación. EXTRACT es una prueba secundaria sin reentrenamiento. Se registrará si conserva toda, parte o ninguna de las relaciones decisivas. Esta separación se inspira en [R3], pero no reproduce su benchmark.

---

## 7. Contrato de acciones y memoria

### 7.1. Tres fases de política en S2

El mismo LLM, con el mismo adaptador, produce tres salidas separadas:

1. **SELECT:** `PROCEED` o `ASK`, indicando una variable.
2. **RESPOND:** respuesta estructurada para la consulta actual.
3. **WRITE:** operaciones sobre la memoria persistente.

Cada fase se construye con un prompt explícito. WRITE recibe evidencia del usuario, no la respuesta generada por RESPOND. Esto impide convertir una invención del asistente en una fuente y permite las intervenciones de §12.

No se requieren cadenas de razonamiento extensas. Se registran decisiones, referencias a evidencia y operaciones verificables.

### 7.2. Selección de pregunta

```json
{"action":"ASK","slot":"TEMPORAL_SCOPE"}
```

Valores de `slot`: `TEMPORAL_SCOPE`, `CURRENT_CONTEXT`, `VALID_UNTIL` y `GENERIC`. `PROCEED` exige `slot: null`.

Plantillas comunes:

| Slot | Pregunta |
|---|---|
| TEMPORAL_SCOPE | «¿Lo dices solo para esta ocasión o como preferencia habitual?» |
| CURRENT_CONTEXT | «¿En qué contexto quieres que aplique la recomendación?» |
| VALID_UNTIL | «¿Hasta qué fecha se aplica esa excepción?» |
| GENERIC | «¿Puedes darme más detalles?» |

Cada historia principal se construye de modo que como máximo una pregunta específica sea suficiente. No permitir preguntas compuestas ni solicitudes de la respuesta de S3. El presupuesto principal es una pregunta en S2.

`GENERIC` sirve para detectar preguntas que no solicitan la variable decisiva. No obtiene automáticamente todos los datos faltantes.

### 7.3. Respuesta actual

```json
{
  "mode": "ANSWER",
  "value": "pasta",
  "branches": [],
  "evidence_ids": ["u2"]
}
```

`mode` admite `ANSWER`, `CONDITIONALIZE` y `DEFER`. `CONDITIONALIZE` contiene ramas explícitas con condición y valor, sin elegir una condición no observada. `DEFER` usa `value: null`.

La verbalización final se genera mediante plantillas deterministas comunes. Una evaluación posterior de prosa libre sería una extensión distinta.

### 7.4. Memoria persistente

```json
{
  "memory_id": "m7",
  "attribute": "dinner_preference",
  "value": "pasta",
  "scope": {
    "kind": "ongoing",
    "session_id": null,
    "context": null,
    "valid_from": "2026-01-12",
    "valid_until_exclusive": null
  },
  "status": "confirmed",
  "source_turn_ids": ["u2", "u3"],
  "evidence_quotes": [
    {"turn_id":"u2","text":"Ahora prefiero pasta"},
    {"turn_id":"u3","text":"Como preferencia habitual desde ahora"}
  ],
  "supersedes": ["m1"]
}
```

El ejemplo representa una escritura **posterior** a la aclaración del mundo B, no una etiqueta disponible antes de preguntar.

`scope.kind` admite `session`, `interval`, `contextual`, `ongoing` y `unknown`. Un final nulo no implica duración indefinida si `kind` es `unknown`. Las fechas se normalizan mediante reglas públicas y pruebas de calendario, nunca a partir de eventos futuros.

### 7.5. Operaciones

```json
{
  "operations": [
    {
      "op": "SUPERSEDE",
      "target_ids": ["m1"],
      "record": {
        "attribute": "dinner_preference",
        "value": "pasta",
        "scope": {
          "kind": "ongoing",
          "session_id": null,
          "context": null,
          "valid_from": "2026-01-12",
          "valid_until_exclusive": null
        },
        "status": "confirmed",
        "source_turn_ids": ["u2", "u3"],
        "evidence_quotes": [
          {"turn_id":"u2","text":"Ahora prefiero pasta"},
          {"turn_id":"u3","text":"Como preferencia habitual desde ahora"}
        ]
      }
    }
  ]
}
```

Operaciones permitidas:

| Operación | Semántica |
|---|---|
| NOOP | No cambiar la memoria. |
| ADD | Incorporar un registro sin invalidar automáticamente otros. |
| SUPERSEDE | Incorporar una versión y cerrar la aplicabilidad de registros indicados en el alcance declarado; conservar versiones históricas. |
| MARK_UNRESOLVED | Incorporar evidencia cuyo alcance sigue desconocido y mantener abiertas las alternativas relevantes. |

Máximo dos operaciones por fase WRITE. NOOP será la única operación de su lote. Los IDs nuevos los asigna el ejecutor; el modelo referencia IDs existentes.

No hay borrado físico. Una excepción se añade con alcance propio, no reemplaza automáticamente la preferencia general.

### 7.6. Ejecutor y validación

El ejecutor valida sintaxis, tipos, límites y existencia de referencias. No usa el estado verdadero para corregir una elección semántica.

Una operación estructuralmente válida pero injustificada se aplica y se penaliza. Una operación estructuralmente inválida se rechaza, se registra y cuenta como fallo; el ejecutor no genera la operación correcta en su lugar.

Si SELECT falla, no se pregunta. Si RESPOND falla, se registra una respuesta inválida. Si WRITE falla, la memoria queda sin ese lote. Estas salidas administrativas no se puntúan como decisiones acertadas.

### 7.7. Lector determinista

Opera únicamente sobre registros persistidos y la consulta actual:

- Filtra por atributo y alcance temporal/contextual declarado.
- Aplica las relaciones de sustitución declaradas, preservando el acceso histórico.
- Una excepción específica válida tiene prioridad sobre la regla general dentro de su alcance.
- Un conflicto entre alternativas todavía aplicables, o un registro `unknown` que pueda alterar la respuesta, impide un compromiso único.
- Si hay un único valor sustentado por el estado de memoria, devuelve ese valor; de lo contrario devuelve alternativas o indeterminación.

No repara registros consultando el diálogo de auditoría. La fidelidad al usuario la comprueba el evaluador, separado del lector. Así puede detectarse una memoria internamente coherente pero falsa.

---

## 8. Simulador y evaluación de evidencia

### 8.1. Simulador fijo

Implementar una tabla de respuestas por episodio y slot. La selección de la respuesta depende del estado oculto y del slot solicitado, no de la identidad del método, su recompensa o sus errores.

La respuesta revela solo el dato preguntado. No devuelve `gold_action`, el JSON de memoria esperado ni una explicación de cómo ganar la recompensa. Las preguntas irrelevantes reciben una respuesta no informativa. Una pregunta sobre un dato ya expresado puede repetirlo, pero incurre en el mismo coste.

El uso de un simulador restringido se relaciona con la precaución del protocolo de [R2, apéndice A.3]; las plantillas y reglas concretas aquí son nuevas decisiones de diseño.

### 8.2. Dos conjuntos de evidencia, no una única etiqueta omnisciente

El evaluador mantiene:

**Evidencia accesible al decisor:** la observación efectivamente presentada en cada fase. Permite juzgar si una afirmación estaba justificada.

**Registro verdadero de interacción:** los mensajes del usuario realmente emitidos hasta ese punto. Permite medir pérdidas de memoria y comprobar si una escritura les es fiel. No contiene mensajes futuros aún no emitidos.

El estado latente se utiliza para la tarea factual y para producir respuestas del simulador, pero no convierte una suposición en una afirmación justificada.

### 8.3. Compatibilidad y respuestas admisibles

Representar un conjunto finito de mundos compatibles con la evidencia. Una respuesta única está justificada si el mismo valor es válido en todos esos mundos para la consulta correspondiente.

Una respuesta que coincide por azar con el mundo latente, pero no está respaldada por la evidencia disponible, se registra como **acierto factual no justificado**, no como éxito completo.

El conjunto de mundos usado para juzgar al decisor no puede basarse en datos ocultos excluidos de O1. El conjunto usado para juzgar fidelidad de escritura tampoco puede incluir una aclaración aún no recibida.

### 8.4. Evaluación de memoria

Verificar valor, alcance, fechas, contexto, fuente, sustituciones y preservación de alternativas. No exigir igualdad textual con un registro gold si dos representaciones tienen la misma semántica.

Comparar la memoria mediante consultas sonda: presente, fecha posterior, fecha histórica y contexto alternativo. Estas sondas son herramientas del evaluador, nunca observaciones del controlador.

NOOP evita escrituras falsas, pero no recibe crédito por conservar información que nunca se guardó. Reportar por separado fidelidad y cobertura para evitar premiar una memoria vacía.

---

## 9. Banco, particiones y revisión independiente

### 9.1. Tamaño operativo inicial

| Partición | Historias raíz | Uso |
|---|---:|---|
| `dev_smoke` | 40 | Pruebas de implementación; nunca resultados finales. |
| `train` | 600 | SFT, continuaciones y rollouts RL. |
| `validation` | 150 | Selección de checkpoints, prompts y diagnóstico previo al test. |
| `test_id` | 300 como mínimo | Generalización a nuevas historias del generador fijado. |
| `test_ood` | 150 | Redacción independiente y combinaciones reservadas. |
| `test_external` | 50 | Historias escritas por una persona distinta del autor del generador. |

Mantener balance por las cinco familias en los conjuntos sintéticos. Los pares de F4 y las paráfrasis aumentan el número de episodios, **no el número de unidades independientes**. Promediar primero dentro de cada raíz.

Estos tamaños son decisiones operativas, no una afirmación de potencia estadística suficiente. La regla de ampliación previa al test se especifica en §15.

### 9.2. Particiones antes de verbalizar

Asignar `root_id`, mundos asociados y variantes a una partición antes de generar paráfrasis. Una raíz no puede aparecer en más de una partición.

Distinguir claramente:

- `test_id`: comparte la gramática y mecanismos generales de entrenamiento, pero no las historias raíz.
- `test_ood`: reserva realizadores lingüísticos y combinaciones de mecanismos, por ejemplo una excepción dentro de una preferencia contextual. No se limita a cambiar nombres o alimentos.

La normalización de duplicados elimina nombres, IDs y cambios puramente cosméticos. Registrar las coincidencias restantes de estados canónicos. Un estado repetido no es automáticamente una fuga; sí limita lo que puede afirmarse como novedad estructural.

### 9.3. Reglas antisesgo

Equilibrar valores, duración, longitud y orden de opciones. No correlacionar un alimento, nombre, plantilla o ID con `ASK` o con una respuesta latente concreta.

Comprobar que cambiar únicamente el mundo latente de un par ambiguo no cambia el prompt anterior a preguntar. Comprobar también que sí cambia la respuesta del simulador cuando se solicita la variable decisiva.

No eliminar historias porque un método falle. Solo excluir defectos de construcción definidos antes del test, con registro del motivo y evaluación común para todas las variantes.

### 9.4. Revisión humana

Antes del entrenamiento definitivo, dos revisores examinarán independientemente 100 raíces de desarrollo/entrenamiento/validación, balanceadas por familia, sin etiquetas del generador. Deben identificar evidencia suficiente, variable faltante, respuestas admisibles y alcance que puede escribirse.

La revisión debe incluir al menos 20 pares F4. Ambos mundos deben ser plausibles bajo el mismo prefijo. Adjudicar desacuerdos y conservar las decisiones originales.

Umbral de liberación propuesto: al menos 90 % de acuerdo inicial en suficiencia de información y variable decisiva; ningún caso liberado con contradicción no adjudicada entre texto y semántica. Reportar acuerdo exacto y una medida corregida por azar cuando corresponda.

Las 50 historias externas se entregan al evaluador cerrado, no al proceso de ajuste. No se inventará que esta revisión fue realizada: mientras falte, el estado es `pending_human_audit`.

### 9.5. Congelación

Guardar hashes de raíces, realizadores, splits, prompts, evaluador y configuración. Las historias vistas en los pilotos anteriores se consideran desarrollo; no test independiente de esta etapa.

Después de abrir el test, cualquier cambio que altere la semántica, la recompensa o la selección de modelos genera otra versión del protocolo y otro conjunto confirmatorio.

---

## 10. Variantes, datos de supervisión y equidad

### 10.1. Matriz de políticas

| ID | Controlador de S2 | Finalidad |
|---|---|---|
| B0 | Backbone instruct sin ajuste para esta tarea, con instrucciones completas. | Baseline competente sin entrenamiento específico. |
| B0-P | B0 con prompt estructurado de diagnóstico, decisión y escritura. | Controlar cuánto se obtiene solo con prompting. |
| B1 | Backbone + SFT. | Punto de partida supervisado. |
| B2-L | B1 + GRPO con recompensa actual y futura. | Método longitudinal estudiado. |
| B3 | B1 + SFT adicional. | Controlar el beneficio de continuar supervisión con presupuesto comparable. |
| B2-I | B1 + GRPO con recompensa inmediata; el futuro se calcula pero no recompensa. | Aislar el horizonte de recompensa. |
| B4 | Reglas temporales/contextuales sobre exactamente la misma evidencia visible. | Baseline convencional fuerte. |
| C-NEVER | B1 forzado a no preguntar; RESPOND y WRITE se mantienen. | Ablación de adquisición activa. |
| C-ALWAYS | B1 forzado a preguntar una vez; el slot lo elige el modelo mediante un prompt común. | Control de exceso de interacción. |

B0-P puede inspirarse en la separación entre decisión y realización de CAAP [R3], pero se etiquetará como **adaptación de prompting**, no reproducción de CAAP ni comparación directa con sus resultados publicados.

B4 no recibe familia, variable gold o estado oculto. Una regla que opere sobre campos semánticos perfectos no disponibles para las demás variantes se reportaría aparte como referencia privilegiada.

Todos usan el mismo simulador, ejecutor, memoria, presupuesto de preguntas, lectores y casos. La tabla compara **controladores**, no sistemas con lectores diferentes.

### 10.2. SFT no debilitado artificialmente

El corpus de SFT debe incluir trayectorias correctas de las cinco familias: no preguntar, preguntar por la variable pertinente, interpretar la respuesta, conservar incertidumbre y escribir con alcance adecuado. Incluir estados posteriores a una pregunta irrelevante o no informativa y ejemplos de no escritura.

El profesor de SFT utiliza la evidencia disponible y un solucionador del pequeño espacio de hipótesis. No usa el valor latente para decidir una escritura antes de que sea observable. Después de preguntar puede utilizar la respuesta efectivamente recibida.

Las alternativas equivalentes se canonicalizan para entrenamiento, pero el evaluador acepta equivalencias semánticas. Registrar el número de ejemplos por fase, tipo de acción y familia.

### 10.3. Comparaciones de presupuesto

**B1 → B2-L/B3/B2-I:** para cada semilla, copiar exactamente el checkpoint B1 seleccionado. Reiniciar los estados de optimizador de las continuaciones. Mantener idéntico conjunto de parámetros entrenables.

**B2-L frente a B3:** comparar el mejor checkpoint de validación dentro del mismo límite de tiempo de acelerador asignado posterior a SFT. Medir ese límite con una prueba técnica y congelarlo antes de los runs finales. B3 puede repetir el corpus supervisado, con barajado registrado, hasta consumir su límite. No imponerle menos entrenamiento para facilitar una ventaja de RL.

**B2-L frente a B2-I:** usar el mismo número máximo de actualizaciones, la misma secuencia de raíces y el mismo número de rollouts por grupo. Las trayectorias pueden diferir porque las políticas divergen. Reportar el coste real; este contraste iguala el máximo de oportunidades de interacción, no garantiza FLOPs idénticos. Una parada anticipada puede reducir las oportunidades consumidas; en ese caso, añadir la comparación de checkpoints en el mayor prefijo de actualizaciones común y declarar que el resultado del mejor checkpoint incluye la regla de parada.

Registrar por variante: segundos de acelerador asignado, tokens de entrada/generados/entrenados, pasos, episodios, preguntas al simulador, evaluaciones de validación y coste de construcción de datos. Diferenciar presupuesto máximo de consumo efectivo.

Estas comparaciones estiman el efecto de **recetas de aprendizaje bajo recursos declarados**. No se afirmará que son una comparación pura de optimizadores con ejemplos, señales y cómputo exactamente idénticos.

### 10.4. Selección común

Todos los checkpoints se seleccionan en validación por la misma utilidad longitudinal de §11. Los desempates favorecen menos escrituras injustificadas, menos preguntas y finalmente el checkpoint más temprano.

Incluir el checkpoint de paso cero. Si es el mejor, conservarlo y declararlo: el resultado no prueba una mejora causada por RL.

B2-I también utiliza la validación longitudinal para selección. Esto da a todas las variantes el mismo objetivo de selección; la diferencia está en la recompensa de entrenamiento. Publicar además los resultados del checkpoint final, para no ocultar deterioro durante la continuación.

B0/B0-P tienen como máximo cuatro formulaciones de prompt probadas en validación. Congelar la ganadora antes del test. No afinar prompts con errores de test.

---

## 11. Recompensa, algoritmo y cierre técnico

### 11.1. Resultado de una sesión

Definir `C_s = 1` cuando hay una respuesta única correcta, respaldada por la evidencia realmente disponible, y `0` en otro caso.

Definir una recompensa factual/conductual separada:

| Resultado | `r_s` |
|---|---:|
| Compromiso correcto y justificado | +1 |
| Condicionalización fiel o abstención sin afirmaciones falsas, pero sin resolver la tarea | 0 |
| Compromiso equivocado o no justificado | -1 |
| Salida inválida o fallo técnico sin respuesta | -1 |

La abstención evita penalizar como falsedad una incertidumbre real, pero no obtiene el mismo crédito de tarea que resolverla. En el conjunto de no respuesta se reporta adicionalmente adecuación de la abstención; no se confunde con exactitud factual.

Para S3, un valor respaldado solo por una escritura infiel no cuenta como justificado aunque coincida con la intención latente. La evaluación debe auditar la procedencia del registro utilizado y su compatibilidad con todos los mensajes del usuario efectivamente obtenidos hasta el cierre de S2. Olvidar una declaración incompatible no convierte la memoria antigua en evidencia suficiente: el evaluador conserva esa declaración en su registro privado y detecta la omisión. No exige al agente datos que nunca tuvo oportunidad de observar, pero sí mide las pérdidas de la evidencia que recibió.

### 11.2. Recompensas fijadas

Sean `Q2` el número de preguntas realizadas en S2, `W_bad` el número de operaciones estructuralmente ejecutadas que introducen o suprimen aplicabilidad sin respaldo, e `I_bad` el número de salidas de fase inválidas.

```text
R_long  = r2 + 1.00 * r3 - 0.10 * Q2 - 0.50 * W_bad - 0.20 * I_bad
R_short = r2             - 0.10 * Q2 - 0.50 * W_bad - 0.20 * I_bad
```

`W_bad` se cuenta una vez por operación, no una vez por cada campo incorrecto. Su máximo es dos. `I_bad` se cuenta por fase; no se repiten inferencias hasta obtener una salida válida.

La penalización de memoria emplea validación semántica del entorno. Por ello el método **no se describirá como aprendizaje exclusivamente con recompensa de respuesta final**.

No hay recompensa positiva por emitir `ASK`, por generar más tokens ni por escribir más registros. No se implementa CIG en el núcleo: ni esta recompensa ni la intervención de persistencia se presentarán como reproducción de CIGAsk.

Los pesos son elecciones de utilidad de esta SPEC. Reportar también una reevaluación de las políticas congeladas con costes de pregunta `0.00`, `0.05` y `0.20`. Esa reevaluación no equivale a haber entrenado políticas óptimas para esos costes.

### 11.3. Qué recibe gradiente

En B2-L, el retorno completo de S2+S3 se asigna a las salidas SELECT, RESPOND y WRITE de S2. En B2-I, esas mismas salidas reciben solo `R_short`.

El lector de S3, el simulador, el extractor y el evaluador permanecen congelados. Los tokens de usuario, prompts y respuestas del simulador no forman parte de la pérdida del actor.

**Prueba obligatoria de crédito diferido:** manteniendo las salidas y recompensas de S2, alterar una consecuencia válida de S3 debe alterar las ventajas de B2-L y no las de B2-I.

### 11.4. GRPO episódico

Para una misma raíz y el mismo mundo latente inicial, generar `G = 4` trayectorias independientes del actor. Cada rollout tiene una memoria y un generador aleatorio propios; no compartir SQLite mutable entre rollouts.

Para cada grupo:

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + 1e-6)
```

Aplicar una pérdida recortada tipo GRPO a los tokens del actor, con `clip_epsilon = 0.20` y regularización KL frente a una copia congelada del checkpoint SFT B1 de la misma semilla. Promediar por trayectoria y normalizar por sus tokens de actor, para no ponderar más una trayectoria solo por ser larga.

Cada salida de fase se evalúa bajo el prompt que realmente vio al generarse. **No concatenar retrospectivamente todas las sesiones para calcular probabilidades:** eso podría introducir información futura en las probabilidades de acciones pasadas.

Guardar por fase: prompt tokenizado, tokens generados, máscara del actor, log-probabilidades antiguas, identificador de grupo, retorno y ventaja. Recalcular log-probabilidades nuevas sobre esos mismos prefijos observados.

Si la desviación de recompensas de un grupo es cero, su ventaja de tarea será cero. Registrar la proporción de grupos sin señal. Diez actualizaciones consecutivas sin señal de tarea activan una parada diagnóstica; no añadir una recompensa improvisada para fabricar aprendizaje. Un movimiento producido solo por KL no demuestra aprendizaje de la tarea.

### 11.5. Configuración propuesta

| Parámetro | Valor inicial fijado |
|---|---|
| Backbone preferido | `Qwen/Qwen2.5-3B-Instruct` |
| Fallback técnico | `Qwen/Qwen2.5-1.5B-Instruct`, solo si el preferido no supera la prueba de recursos |
| Adaptación | LoRA, rango 16, alpha 32, dropout 0 |
| Módulos candidatos | `q_proj`, `k_proj`, `v_proj`, `o_proj`; verificar nombres reales |
| Aprendizaje SFT | `2e-4` |
| Épocas SFT | 2; selección entre checkpoints intermedios y final |
| Aprendizaje RL | `5e-6` |
| KL de RL | `0.02`, referencia B1 congelada |
| Batch efectivo RL | 8 raíces × 4 rollouts |
| Máximo de actualizaciones RL | 120 por continuación y semilla |
| Checkpoints RL | Paso 0 y cada 10 actualizaciones |
| Muestreo en entrenamiento RL | Temperatura 1.0, `top_p = 1.0` |
| Evaluación | Greedy, misma configuración en todas las variantes |
| Entrada máxima por fase | 4096 tokens, sin truncamiento silencioso |
| Salida máxima SELECT / RESPOND / WRITE | 64 / 256 / 512 tokens |
| Preguntas | Una en S2; ninguna en S3 principal |
| Semillas de entrenamiento | 101, 202, 303 |

Estas cantidades no provienen de una optimización ya realizada. Se permite corregir incompatibilidades o inestabilidad numérica en `dev_smoke`; cualquier cambio se documenta y congela antes de las ejecuciones comparativas. No se cambia de modelo porque otro gane más en test.

Las tarjetas oficiales de los dos candidatos están disponibles [D1–D2]. Registrar licencia y revisión exactas; no asumir que comparten licencia ni que cualquier hardware puede entrenarlos.

### 11.6. Librerías y adaptación del trainer

Usar PyTorch, Transformers y PEFT con versiones fijadas. Se puede reutilizar TRL si la versión instalada permite representar correctamente los rollouts y las máscaras. Su documentación contempla entrenamiento de agentes y funciones de rollout, pero las interfaces evolucionan [D3].

**No basta con llamar a un trainer estándar sobre respuestas independientes.** Antes de usarlo, demostrar que respeta grupos de episodios, estado separado, prefijos por fase y crédito de S3 sobre decisiones de S2. Si no lo permite, implementar un adaptador episódico explícito y probarlo; no rebautizar una pérdida SFT como RL.

En modo cuantizado, mantener la misma cuantización para todas las variantes comparables. Preferir BF16 con LoRA si la prueba de recursos lo permite; usar QLoRA solo si la combinación de hardware y librerías supera las pruebas. No prometer un consumo de VRAM no medido.

La referencia de KL debe incluir el adaptador SFT. Desactivar todos los adaptadores y obtener el backbone original **no** produce esa referencia.

### 11.7. Auditoría de entrenamiento

Antes y después de una actualización, comprobar normas de gradiente, variación de parámetros entrenables, integridad de los pesos congelados y correspondencia entre actualización y acción generada.

Guardar el checkpoint inicial, el mejor de validación y el final, junto con el número efectivo de pasos. Los pesos del LLM o de sus adaptadores deben cambiar; actualizar únicamente un selector externo no satisface esta SPEC.

---

## 12. Experimentos y estimandos

### E0. Validación del mecanismo

Antes de comparar modelos, ejecutar historias manuales con un solucionador de referencia, escrituras deliberadamente incorrectas, preguntas irrelevantes y desaparición del contexto entre sesiones.

Se exige verificar que la preferencia temporal no se vuelve duradera por defecto, que una aclaración puede cambiar S3 y que una respuesta presente correcta no garantiza memoria correcta.

El solucionador puede inspeccionar la semántica para validar el entorno, pero sus resultados no se incluyen como un competidor no privilegiado.

### E1. Observabilidad aislada — PI1

Evaluar B0 y B4 sobre los mismos snapshots de S1/S2 bajo O1 y O2. Para este contraste inicial, desactivar la interacción y no aplicar escrituras antes de medir la decisión presente. Así la diferencia no queda mediada por preguntar o por modificaciones previas de memoria.

Medidas: `C2`, compromisos no justificados, abstenciones y coste de contexto, por familia. O2 no está obligado a resolver F4: allí falta información que no existe en el historial.

La condición principal para los restantes experimentos será O2-REF, independientemente de cuál obtenga mejor puntuación en E1.

### E2. Comparación longitudinal de controladores — PI3

Evaluar B0, B0-P, B1, B2-L, B3, B2-I y B4 en episodios completos O2-REF. C-NEVER y C-ALWAYS se ejecutan como ablaciones de B1.

Todos empiezan con la misma memoria inicial. Sus decisiones generan memorias distintas, que realmente se utilizan en S3. No sustituir la memoria del método por la memoria gold antes de evaluar.

Comparaciones de entrenamiento obligatorias: B2-L–B3, B2-L–B1 y B2-L–B2-I. No comparar resultados propios con cifras publicadas de otros benchmarks como si compartieran protocolo.

### E3. Intervención sobre adquisición y persistencia — PI2

Usar un conjunto de oportunidades predefinido por la construcción del banco, no seleccionar solo episodios donde el modelo decidió preguntar. La intervención primaria se realiza con B1; repetir con B2-L como análisis de mecanismo.

Para cada raíz elegible, clonar estado, pesos, mundo latente y semilla. La pregunta específica se fija para el experimento; es un estímulo controlado, no una decisión aprendida. Ejecutar:

| Brazo | RESPOND recibe la respuesta aclaratoria | WRITE recibe la respuesta aclaratoria | S3 conserva contexto transitorio |
|---|---|---|---|
| I0: sin adquisición | No | No | No |
| I1: uso inmediato | Sí | No | No |
| I2: uso inmediato y escritura | Sí | Sí | No |

Los prompts de RESPOND son idénticos en I1 e I2. Los de WRITE son idénticos en I0 e I1. WRITE no recibe respuestas del asistente, resúmenes o acciones intermedias que puedan revelar indirectamente la aclaración.

**No borrar solo un campo de la memoria final.** La respuesta podría haber quedado parafraseada en otro registro. La separación se realiza antes de escribir, construyendo explícitamente el conjunto de evidencia visible en cada brazo.

Estimandos emparejados:

```text
Delta_actual      = mean(C2_I1 - C2_I0)
Delta_persistencia = mean(C3_I2 - C3_I1)
Delta_total_futuro = mean(C3_I2 - C3_I0)
Delta_utilidad    = mean(R_long_I2 - R_long_I0)
```

En F4 de beneficio exclusivamente futuro se espera `Delta_actual` próxima a cero. Un `Delta_persistencia` positivo identifica el efecto de permitir el acceso de la respuesta al escritor, bajo el lector y entorno fijados.

**Límite causal:** es el efecto de una intervención de disponibilidad de información en este simulador. No es una estimación del efecto de preguntar a cualquier usuario real, ni prueba que el selector aprendido detecte todas las oportunidades. Esa capacidad se mide en E2.

Controles negativos: ejecutar versiones equivalentes en casos claros y consultas sobre atributos no afectados. No deberían adquirir una mejora futura grande por una respuesta redundante o irrelevante.

### E4. Generalización y lector LLM

Sin ajustar ningún componente, evaluar los checkpoints seleccionados en `test_ood` y `test_external`. Reportar las diferencias respecto a ID sin agruparlas como una única distribución.

Releer las mismas memorias con el lector LLM base congelado. Comparar el resultado del lector simbólico y el lingüístico. Una memoria correcta puede ser mal utilizada por el lector; una respuesta acertada no demuestra que la memoria sea fiel.

No entrenar otro lector por método. No escoger el lector que maximice la ventaja de B2-L.

### E5. Robustez de extracción y respuesta del usuario

Realizar dos controles secundarios sobre una lista de 100 raíces ID fijada antes del test y balanceada por familia:

**Extracción:** sustituir la memoria inicial REF por EXTRACT, dejando igual todo lo demás. Reportar observabilidad de relaciones, errores de extracción y resultado final. El alcance es degradación de la memoria inicial, no una evaluación completa de todos los sistemas de memoria existentes.

**No respuesta:** en 20 % y 40 % de las oportunidades, el simulador devuelve una respuesta no resolutiva preasignada por raíz y slot. No revela de todos modos la intención oculta. Reportar compromiso injustificado, conservación de incertidumbre y éxito de tarea por separado.

La asignación no depende del método ni de si este acierta. Mantener el simulador principal determinista y cooperativo para el contraste causal limpio; no mezclar estas condiciones en su resultado principal.

### E6. Reutilización con interacción permitida

Evaluación secundaria sobre las mismas memorias, permitiendo una pregunta en S3 con un lector/controlador congelado común. Reportar preguntas totales, preguntas repetidas sobre el mismo alcance ya comunicado y éxito tras la interacción.

Una pregunta sobre un contexto que cambió no es una repetición innecesaria. Esta prueba comprueba si el beneficio principal se mantiene como reducción de carga de interacción cuando preguntar nuevamente sí es posible.

---

## 13. Métricas y denominadores

### 13.1. Variable primaria

**Éxito futuro justificado (`FJS`):** proporción de historias raíz donde S3 produce una selección correcta y respaldada por memoria fiel a la evidencia efectivamente obtenida.

Promediar primero episodios y mundos dentro de cada raíz; después raíces dentro de familia; finalmente las cinco familias con igual peso. Promediar las semillas sin tratarlas como nuevos usuarios.

Publicar también la media micro, pero no sustituir la macro por ella después de mirar resultados. En F4 publicar expresamente el subconjunto donde la respuesta actual ya estaba determinada.

### 13.2. Métricas secundarias obligatorias

| Métrica | Definición y denominador |
|---|---|
| Éxito presente `C2` | Respuestas actuales correctas y justificadas / decisiones actuales programadas. |
| Compromiso no justificado | Compromisos únicos sin evidencia suficiente / decisiones programadas. |
| Escritura injustificada por episodio | Raíces con al menos una operación injustificada / raíces evaluadas. |
| Precisión de escritura | Operaciones fieles / operaciones ejecutadas; N/A si no escribe. |
| Cobertura de memoria | Hechos decisivos correctamente disponibles después de S2 / hechos decisivos observados que deberían seguir siendo aplicables. |
| Error de alcance | Uso o escritura fuera de la sesión, intervalo o contexto autorizado / oportunidades de alcance. |
| Preguntas por episodio | Preguntas efectivamente realizadas / episodios programados. |
| Preguntas innecesarias | Preguntas sobre variables ya resueltas o incapaces de mejorar la tarea presente o futura / oportunidades inicialmente clasificadas como suficientes. |
| Adquisición útil | Preguntas que recuperan la variable decisiva / preguntas realizadas; desglosar necesidad actual y futura. |
| Retorno | Media de `R_long` y sus componentes, no solo el escalar total. |
| Reutilización sin repetir | S3 correcta sin nueva pregunta / oportunidades donde el dato ya había sido aclarado y sigue aplicable. |
| Fallo técnico | Episodios con fallo persistente de infraestructura / episodios intentados. |
| Formato inválido | Salidas de fase no válidas / salidas programadas. |

Una etiqueta de utilidad futura no se define simplemente como “es ambiguo”. Debe existir una variable recuperable, relevante para una decisión futura prevista, cuya respuesta pueda cambiar la memoria aplicable. Estas etiquetas son diagnósticas y para construir SFT; no son entradas del actor.

### 13.3. Fallos y reintentos

Reportar métricas sobre todos los episodios programados y, por separado, sobre los técnicamente completados. Los primeros constituyen el análisis operativo principal.

Un error de infraestructura permite un único reintento con idéntica configuración y semilla. Guardar ambos intentos. Un JSON incorrecto del modelo no es un error de infraestructura y no autoriza reintentar.

Un fallo persistente cuenta como `C_s = 0` y se marca aparte. No excluirlo silenciosamente ni sustituirlo por una historia más fácil.

---

## 14. Estadística y reglas de interpretación

### 14.1. Análisis emparejado

Comparar métodos sobre las mismas raíces, mundos y condiciones. El registro de análisis debe contener una fila por combinación `method × seed × root × condition`, después de promediar sus realizaciones.

No usar cada turno, paráfrasis, mundo o rollout como una observación independiente. El bootstrap remuestrea raíces completas dentro de familia y utiliza los mismos índices para todos los métodos del contraste.

### 14.2. Intervalos y semillas

Realizar 20 000 remuestreos, con semilla de análisis **7331**. Informar intervalos del 95 % para descripción y resultados individuales de las tres semillas.

El intervalo principal de raíces describe incertidumbre sobre nuevas historias de esa distribución, condicionada a las tres ejecuciones entrenadas. Añadir un análisis de sensibilidad que remuestree también semillas completas y emparejadas. Con tres semillas, esa estimación entre entrenamientos es limitada; no presentar miles de turnos como sustituto de más entrenamientos.

Para tasas raras o ausencia total de eventos, no reportar un intervalo degenerado `[0,0]` obtenido por bootstrap. Añadir límites binomiales conservadores a nivel de raíz cuando corresponda y advertir la limitada precisión.

### 14.3. Familia de contrastes principales

Fijar cinco contrastes antes del test:

1. E1: O2–O1 sobre `C2` de B0.
2. E3: I2–I1 sobre `FJS` de B1, en el conjunto F4 elegible.
3. E2: B2-L–B3 sobre `FJS` global.
4. E2: B2-L–B1 sobre `FJS` global.
5. E2: B2-L–B2-I sobre `FJS` global.

Para afirmaciones confirmatorias, usar intervalos del **99 % por contraste**, como corrección Bonferroni para esta familia de cinco. Los intervalos del 95 % y análisis por subtipo son descriptivos salvo preregistro adicional. Publicar todos los contrastes, no solo los favorables.

### 14.4. Relevancia práctica y daños

Umbral de mejora futura relevante propuesto: **5 puntos porcentuales de FJS**. No confundir “estimación ≥ 5 puntos e intervalo que excluye cero” con “el límite inferior demuestra al menos 5 puntos”. La segunda afirmación exige que el límite inferior también supere 5 puntos.

Una mejora limpia debe acompañarse de una variación en errores actuales y escrituras injustificadas compatible con los márgenes fijados: máximo **2 puntos porcentuales de deterioro** en cada una. Evaluar esos márgenes con intervalos, no solo con medias.

Si aumenta la exactitud futura a costa de daño presente o de escrituras infieles, reportar un compromiso entre objetivos, no una mejora global. El coste de preguntas debe aparecer tanto desagregado como en utilidad neta.

### 14.5. Conclusiones permitidas

| Patrón | Conclusión autorizada |
|---|---|
| O2 mejora y los métodos aprendidos no se separan. | La observabilidad explica la mayor parte de la mejora observada en este banco. |
| I2 supera a I1 y sus respuestas actuales son iguales. | Hay evidencia del valor de permitir que la aclaración informe la escritura persistente bajo el protocolo. |
| B2-L supera a B1, pero no a B3. | No se aisló una ventaja frente a continuar SFT. |
| B2-L supera a B2-I, pero no a B3. | El horizonte de recompensa influye, sin demostrar superioridad frente a supervisión adicional. |
| B2-L supera a B1 y B3 con guardas satisfechas. | Beneficio incremental de la receta RL longitudinal en la distribución y presupuesto evaluados. |
| El mejor checkpoint de B2-L es el inicial. | No hay mejora atribuible a sus actualizaciones. |
| El intervalo es amplio e incluye cero. | Resultado inconcluso; no demuestra equivalencia. |
| Las reglas igualan o superan al método. | Las reglas son una solución competitiva en este alcance; no ocultar ese resultado. |
| La ventaja desaparece en OOD o EXTRACT. | El beneficio no se generaliza a esa condición; localizar la pérdida. |

El éxito de la tesis no exige que B2-L gane. Sí exige que el diseño discrimine entre estas explicaciones.

---

## 15. Tamaño, saturación y paradas

### 15.1. Precisión antes de abrir el test

Antes del test, usar exclusivamente pilotos de validación para estimar la variabilidad de diferencias emparejadas. Simular tamaños entre 300 y 1200 raíces ID, manteniendo balance y agrupación, para un efecto de 5 puntos, nivel corregido de 0.01 y potencia objetivo de 0.80.

La simulación debe modelar pares y semillas; no aplicar una fórmula de dos proporciones independientes ignorando el emparejamiento. La estimación de variabilidad del piloto es incierta y debe incluir un escenario conservador.

Fijar el tamaño final una sola vez, antes de obtener resultados de test. Si los recursos impiden la precisión prevista, mantener el tamaño factible y declarar la comparación como un estudio de estimación con sensibilidad limitada. No extender el test hasta conseguir significación.

La precisión del contraste de persistencia se calcula con el número de raíces F4, no con el total de decisiones del banco.

### 15.2. Saturación

Si B1 alcanza prácticamente el techo en validación y no deja margen en utilidad, aclaraciones o memoria, registrar `ceiling_detected`. No debilitar SFT ni alterar los casos para hacer ganar a RL.

Se puede completar un piloto RL predefinido o detener entrenamiento adicional por falta de margen. En ese caso, la tesis conserva PI1 y PI2 y documenta que PI3 no permite establecer una ganancia en el alcance probado. No afirmar que se ejecutó una comparación completa si no se hizo.

### 15.3. Paradas técnicas

Detener ante filtración de etiquetas, desalineación de máscaras, contaminación entre rollouts, gradientes no finitos, divergencia no explicada, escritura fuera del espacio permitido o fallo sistemático del evaluador.

La corrección técnica exige volver a ejecutar todas las condiciones afectadas. Conservar los logs originales. No mezclar resultados previos y posteriores al arreglo bajo la misma versión.

---

## 16. Implementación, contratos y trazabilidad

### 16.1. Reutilización del proyecto

Inspeccionar primero el repositorio. Conservar SQLite, historial, pruebas y simuladores existentes cuando satisfagan los contratos. Crear una rama o versión separada y migraciones explícitas. No sobrescribir resultados de SPEC-RL-005.

Los siguientes nombres son una **estructura objetivo**, no una descripción de archivos ya existentes:

```text
research/
  specs/SPEC-TESIS-CLARIFICACION-MEMORIA-001.md
  configs/protocol.yaml
  configs/environment.lock.yaml
  manifests/data_manifest.json
  manifests/run_manifest.json
  prompts/select.txt
  prompts/respond.txt
  prompts/write.txt
  prompts/reader.txt
  prompts/extractor.txt
  schemas/episode.schema.json
  schemas/action.schema.json
  schemas/memory.schema.json
  src/data/build_bank.py
  src/data/audit_splits.py
  src/env/episode.py
  src/env/user_simulator.py
  src/memory/store.py
  src/memory/executor.py
  src/policy/llm_controller.py
  src/policy/rule_controller.py
  src/readers/symbolic.py
  src/readers/frozen_llm.py
  src/scoring/evidence.py
  src/scoring/reward.py
  src/training/sft.py
  src/training/episodic_grpo.py
  src/evaluation/run_closed_test.py
  src/evaluation/persistence_interventions.py
  src/analysis/paired_statistics.py
  tests/
  reports/
```

Mantener compatibilidad con la arquitectura real del proyecto; no crear dos implementaciones incompatibles del mismo almacén sin justificación.

### 16.2. Contratos mínimos

```python
class EpisodeEnvironment:
    def reset(self, episode_id: str, rollout_seed: int): ...
    def observe(self, phase: str, condition: str): ...
    def ask(self, slot: str): ...
    def apply_memory_operations(self, operations: list[dict]): ...
    def close_session(self): ...
    def reopen_for_reuse(self): ...

class Policy:
    def select(self, observation: dict) -> dict: ...
    def respond(self, observation: dict) -> dict: ...
    def write(self, observation: dict) -> dict: ...

class Evaluator:
    def score_current(self, episode_trace: dict) -> dict: ...
    def score_memory(self, episode_trace: dict) -> dict: ...
    def score_future(self, episode_trace: dict) -> dict: ...
```

Son interfaces a implementar, no código ejecutable entregado con este documento. El evaluador y los atributos ocultos del entorno no se exponen como herramientas invocables por el modelo.

### 16.3. Identidad de ejecución

Cada resultado debe incluir:

```text
protocol_version, git_commit, model_id, model_revision,
tokenizer_revision, adapter_hash, parent_checkpoint_hash,
method, seed, root_id, latent_world_id, realization_id,
split, observation_condition, extraction_track, intervention,
reader_id, decoding_config_hash, evaluator_hash,
initial_memory_hash, final_memory_hash,
prompt_hashes, raw_outputs, parsed_actions,
user_reply, executed_operations, rewards_by_component,
metrics, technical_status, retry_count, resource_counters
```

Los campos privados como `latent_world_id` aparecen solo en logs de evaluación. No entran en los prompts. Los resultados deben reconstruirse desde trazas, no únicamente desde una tabla agregada.

### 16.4. Cierre técnico G0

`environment.lock.yaml` debe fijar: hardware, VRAM observada, CPU/RAM, driver, CUDA, versiones de Python/PyTorch/Transformers/PEFT/trainer, modelo y revisión, tokenizer, precisión/cuantización, módulos LoRA, presupuesto de acelerador, batch/microbatch/acumulación y límites de tokens.

Ejecutar una carga, una generación, un paso SFT y una actualización episódica RL de humo. Verificar que ambas caben y que la referencia congelada corresponde al checkpoint correcto. La elección entre modelo preferido y fallback se hace por estas comprobaciones, no por puntuaciones de test.

Hasta completar el lock, el sistema debe impedir comandos de entrenamiento final y emitir `TECHNICAL_LOCK_REQUIRED`.

---

## 17. Pruebas de aceptación

### 17.1. Datos y evidencia

| ID | Prueba | Condición de aceptación |
|---|---|---|
| T01 | Separación de raíces | Ningún `root_id` atraviesa particiones. |
| T02 | Pares observacionalmente idénticos | Mismo prompt previo a la pregunta para mundos latentes alternativos. |
| T03 | Revelación selectiva | La respuesta cambia al pedir la variable pertinente, no al inspeccionar IDs. |
| T04 | Sin visión del futuro | Modificar un evento futuro no modifica observaciones anteriores. |
| T05 | Sin etiqueta semántica oculta | Ningún campo gold llega al prompt o a herramientas del actor. |
| T06 | Profesor observable | SFT no contiene decisiones incompatibles con la evidencia de su prefijo. |
| T07 | Auditoría humana | Informe real, adjudicación y estado explícito; no aprobación inventada. |

### 17.2. Memoria, simulador e intervención

| ID | Prueba | Condición de aceptación |
|---|---|---|
| T08 | Persistencia real | S3 reconstruye su entrada leyendo la base cerrada y reabierta. |
| T09 | Ausencia de contexto oculto | Eliminar la memoria relevante reduce la información disponible aunque exista el log privado. |
| T10 | Excepción no duradera | Una excepción vence sin borrar la preferencia general. |
| T11 | Sustitución histórica | Un cambio duradero afecta al futuro sin alterar lo que se sabía de una fecha anterior. |
| T12 | Contexto no trasladable | Aclarar el contexto actual no lo convierte en contexto de la sesión siguiente. |
| T13 | Escritor sin autoevidencia | RESPOND no se reintroduce como afirmación del usuario en WRITE. |
| T14 | I1/I2 | RESPOND ve exactamente los mismos bytes en ambos brazos. |
| T15 | I0/I1 | WRITE ve exactamente los mismos bytes en ambos brazos. |
| T16 | Simulador no complaciente | GENERIC o un slot irrelevante no revelan toda la intención. |
| T17 | Ejecutor no corrector | Una operación semánticamente falsa no se transforma en la operación gold. |
| T18 | Acierto por azar | Un valor latente acertado sin evidencia no obtiene éxito justificado. |

### 17.3. Aprendizaje y evaluación

| ID | Prueba | Condición de aceptación |
|---|---|---|
| T19 | Parámetros correctos | Cambia LoRA; el backbone congelado y la referencia no cambian. |
| T20 | Máscara de pérdida | Solo tokens del actor reciben pérdida; usuario/simulador quedan excluidos. |
| T21 | Referencia B1 | La KL se calcula frente a B1 congelado, no frente al backbone sin adaptador. |
| T22 | Crédito diferido | Una diferencia de S3 altera B2-L, no B2-I. |
| T23 | Aislamiento de rollouts | Escribir en un rollout no modifica ningún otro. |
| T24 | Prefijos reales | No se recalculan acciones pasadas con mensajes futuros. |
| T25 | Variación nula | Un grupo con retornos iguales no produce una ventaja de tarea ficticia. |
| T26 | Paso cero elegible | El selector puede escoger el checkpoint inicial y lo reporta. |
| T27 | Test cerrado | No hay gradientes, selección de prompts ni cambios de recompensa durante test. |
| T28 | Denominadores | Todos los intentos, fallos y reintentos aparecen en el informe. |
| T29 | Agrupación estadística | Duplicar una paráfrasis no duplica el tamaño independiente de muestra. |
| T30 | Lectores comunes | Todos los controladores se evalúan con exactamente los mismos lectores. |

Las pruebas T01–T06, T08–T30 son automatizables. T07 requiere personas. Registrar salidas de pruebas; no basta con declarar que existen.

---

## 18. Puertas de avance y entregables

| Puerta | Qué debe existir | Qué habilita |
|---|---|---|
| G0 | Inventario y `environment.lock.yaml`; prueba real de actualización del LLM. | Implementación experimental en el modelo fijado. |
| G1 | Entorno, lector, evaluador y pruebas críticas sin filtraciones. | Construcción del banco completo. |
| G2 | Particiones auditadas, revisión humana, manifiesto y protocolo congelados. | Baselines y entrenamiento comparativo. |
| G3 | E1, B0/B0-P/B4 y SFT; diagnóstico de techo y presupuesto cerrado. | Continuaciones B2-L, B3 y B2-I. |
| G4 | Checkpoints seleccionados solo en validación; tamaño final de test fijado. | Apertura única del test. |
| G5 | E2–E6 ejecutados en su alcance declarado; trazas y fallos completos. | Informe de resultados y conclusiones. |

Entregables mínimos:

**`INFORME_OBSERVABILIDAD.md`** con E1, longitudes de contexto y análisis por familia.

**`INFORME_VALOR_PERSISTENCIA.md`** con I0/I1/I2, comprobación de invariantes y efectos presente/futuro.

**`INFORME_SFT_RL.md`** con curvas, selección de checkpoints, presupuestos, semillas, ablation de horizonte y resultados negativos.

**`INFORME_GENERALIZACION.md`** con ID/OOD/externo, lector LLM, extracción y no respuesta, sin mezclar sus denominadores.

**`RESULTADOS.csv` y `TRAZAS.jsonl`** suficientes para regenerar todas las tablas y reconstruir ejemplos.

**`RESPUESTAS_DIRECTOR.md`** organizado por PI1, PI2 y PI3: evidencia disponible, limitaciones, decisión y siguiente trabajo realmente justificado.

**`LIMITACIONES_Y_DESVIACIONES.md`** con toda desviación del protocolo, su fecha y si ocurrió antes o después de conocer resultados de test.

Los informes deben separar claramente `executed`, `not_executed`, `failed` y `inconclusive`. No rellenar tablas con estimaciones inventadas para que el experimento parezca completo.

---

## 19. Configuración de referencia

Este YAML es una especificación legible de los parámetros; el software debe implementar y validar sus campos. Los valores nulos corresponden a decisiones técnicas que debe cerrar G0 y al tamaño final de test que debe cerrar G4. El software bloqueará cada etapa mientras sus campos obligatorios sigan pendientes.

```yaml
spec_id: SPEC-TESIS-CLARIFICACION-MEMORIA-001
protocol_version: "1.0"
language: es
domain: non_clinical_food_preferences
execution: local_only

model:
  preferred: Qwen/Qwen2.5-3B-Instruct
  technical_fallback: Qwen/Qwen2.5-1.5B-Instruct
  selected_id: null
  revision: null
  tokenizer_revision: null
  precision: null
  quantization: null

adaptation:
  method: lora
  rank: 16
  alpha: 32
  dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj]

seeds:
  training: [101, 202, 303]
  data_partition: 4107
  analysis: 7331

environment:
  sessions: 3
  learned_session: 2
  initialize_session_1_identically: true
  max_questions_session_2: 1
  max_questions_session_3_primary: 0
  max_write_operations: 2
  reset_transient_context_between_sessions: true
  observation_primary: O2
  extraction_primary: REF
  reader_primary: fixed_symbolic
  reader_secondary: frozen_base_llm
  show_future_query_to_actor: false
  show_future_reuse_regime_to_actor: true

reward:
  current_weight: 1.0
  future_weight_long: 1.0
  future_weight_short: 0.0
  question_cost: 0.10
  unsupported_operation_cost: 0.50
  invalid_output_cost: 0.20

sft:
  learning_rate: 0.0002
  epochs: 2
  assistant_tokens_only: true

rl:
  algorithm: episodic_grpo
  learning_rate: 0.000005
  roots_per_effective_batch: 8
  rollouts_per_root: 4
  max_updates: 120
  checkpoint_every_updates: 10
  kl_coefficient: 0.02
  kl_reference: frozen_selected_sft_same_seed
  clip_epsilon: 0.20
  advantage_epsilon: 0.000001
  temperature: 1.0
  top_p: 1.0
  stop_after_zero_signal_updates: 10

limits:
  max_input_tokens_per_phase: 4096
  max_select_tokens: 64
  max_respond_tokens: 256
  max_write_tokens: 512
  allow_silent_truncation: false

budget:
  post_sft_accelerator_seconds_cap: null
  selected_by_technical_smoke_before_final_training: true

sample:
  smoke_roots: 40
  train_roots: 600
  validation_roots: 150
  test_id_min_roots: 300
  test_id_max_roots: 1200
  test_id_locked_roots: null
  test_ood_roots: 150
  external_roots: 50
  robustness_roots: 100
  grouped_by: root_id

evaluation:
  decoding: greedy
  primary_metric: future_justified_success_macro
  bootstrap_resamples: 20000
  descriptive_ci: 0.95
  primary_contrast_ci: 0.99
  primary_contrasts: 5
  practical_future_gain: 0.05
  max_current_error_increase: 0.02
  max_unsupported_write_increase: 0.02
  include_step_zero_checkpoint: true
  include_technical_failures_in_primary_denominator: true
  test_time_weight_updates: false
```

El tamaño final de test se completa en G4, no necesariamente en G0. El validador distinguirá los campos requeridos para entrenar de los requeridos para abrir el test. La potencia no se presume por completar el campo.

---

## 20. Instrucción de entrega a un agente de programación

> Implementa esta SPEC sobre el repositorio existente, empezando por un inventario y las pruebas de G0/G1. Reutiliza componentes válidos; no supongas que la estructura sugerida ya existe. Mantén separados actor, simulador, lector y evaluador. No expongas estado oculto ni eventos futuros en los prompts. Implementa SELECT, RESPOND y WRITE como fases separadas y prueba que I0/I1/I2 impiden la persistencia indirecta de la aclaración. Entrena adaptadores del propio LLM, no una red externa presentada como entrenamiento del LLM. Compara B2-L con B1, B3 y B2-I desde checkpoints emparejados. Conserva el paso cero y todos los resultados negativos. No abras el test hasta congelar datos, evaluador, prompts, checkpoints y tamaño de muestra. Detente ante una puerta fallida y entrega evidencia del fallo y de lo realmente completado; no inventes resultados ni ajustes el benchmark para favorecer RL.

---

## 21. Riesgos de interpretación y límites de la contribución

**Circularidad del entorno.** Construir casos donde una aclaración revela el dato faltante demuestra una propiedad del mecanismo diseñado. El aporte empírico adicional está en saber si políticas no privilegiadas detectan esas oportunidades, escriben bien, evitan preguntar en controles y generalizan fuera de las plantillas.

**Lector simbólico.** Aísla el problema y abarata entrenamiento, pero no demuestra por sí solo que un LLM lector aprovechará la memoria. Por eso se exige la relectura con un modelo congelado.

**Alcance corto.** Tres sesiones permiten atribución clara, pero no validan memoria de meses ni secuencias largas. No presentar esta duración como equivalente a los bancos longitudinales de otros artículos.

**Simulación.** Un usuario sintético cooperativo no valida utilidad humana ni satisfacción. La prueba de no respuesta y las historias independientes reducen, pero no eliminan, esa limitación.

**Reglas y etiquetas.** Un dominio pequeño puede resolverse con reglas fuertes. Si ocurre, es un hallazgo, no un fallo que deba ocultarse. La recompensa contiene elecciones normativas sobre coste de preguntar, compromiso y escritura; se deben hacer explícitas.

**Potencia y semillas.** Un intervalo amplio no demuestra que RL y SFT sean equivalentes. Tres semillas tampoco representan toda la variabilidad de entrenamiento.

**Posicionamiento bibliográfico.** Esta SPEC no garantiza novedad absoluta ni publicación. Antes de la defensa o envío, actualizar el mapa de trabajos relacionados, incluyendo los antecedentes adicionales, y sostener solo la aportación que los experimentos ejecutados respalden.

### Formulación de contribución para la tesis

> «Estudiamos la aclaración como adquisición de evidencia para memoria futura, distinguiendo su uso inmediato de su incorporación persistente. En un entorno controlado de preferencias temporales y contextuales, medimos por separado observabilidad, efecto de persistencia y beneficio incremental de distintas recetas de aprendizaje, incluyendo SFT, SFT adicional y RL con horizontes de recompensa diferentes».

Esta formulación sigue siendo correcta si RL empata o pierde. Las frases “demostramos una mejora”, “generaliza” o “supera a las reglas” solo se añadirán cuando los resultados y su incertidumbre lo permitan.

---

## 22. Referencias y registro de revisión

### Material proporcionado por el proyecto

**[P0]** `Pasted markdown.md`. *Resumen general de tu tesis, experimentos y siguiente etapa*. Secciones 3, 5, 6 y 7. Fuente del estado reportado de los pilotos y de las recomendaciones atribuidas al director. No equivale a disponer de los informes completos que menciona.

**[R1]** Yan, S. et al. (2026). *Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning*. ACL 2026, pp. 12805–12825. Identificador ACL: `2026.acl-long.583`. Archivo revisado: `2026.acl-long.583.pdf`. Base de comparación: §3, figura 2, §4.1 y apéndice B.2. Referencia pública: `https://aclanthology.org/2026.acl-long.583/`.

**[R2]** Li, Y., Wu, X. y Meng, H. (2026). *When and How Should an Agent Clarify? CIGAsk: Teaching LLMs to Clarify via Counterfactual Information Gain*. `arXiv:2609.24290v1`. Archivo revisado: `2609.24290v1.pdf`. Base de comparación: §3, figura 2, §4.1 y apéndice A.3. Referencia pública: `https://arxiv.org/abs/2609.24290v1`.

**[R3]** Yang, L. et al. (2026). *When Personal Memory Has No Single Answer: Evaluating LLM Agents under Irreducible Conflict*. TANGLE. `arXiv:2608.13921v1`. Archivo revisado: `2608.13921v1.pdf`. Base de comparación: §3.1, §3.4, §4.5 y figura 1. Referencia pública: `https://arxiv.org/abs/2608.13921v1`.

### Ampliación externa focalizada

**[R4]** Li, B., Yao, J. y Zheng, Z. (2026). *Remember, Verify, or Ask? Cross-Family Evaluation of Memory Commitment in LLM Agents*. MCB. `arXiv:2608.19564v1`. Consultado el 23 de octubre de 2026. Referencia: `https://arxiv.org/abs/2608.19564v1`. Se fija esta versión para no mezclar títulos o contenidos de revisiones posteriores del mismo identificador.

**[R5]** Hossain, S. M. A., Shayoni, R. K. y Morol, M. K. (2026). *CAPTURE: Disentangling Preference Drift from Memory Poisoning in Personalized LLM Agents*. `arXiv:2609.02265v1`. Consultado el 23 de octubre de 2026. Base de comparación: §3, §6 y apéndice B. Referencia: `https://arxiv.org/abs/2609.02265v1`.

Las búsquedas focalizadas incluyeron combinaciones de `LLM clarification persistent memory longitudinal reinforcement learning user preferences` y `proactive memory clarification cross session`. Se revisaron los dos antecedentes adicionales más próximos encontrados. No se presenta este registro como revisión sistemática exhaustiva.

### Documentación técnica

**[D1]** Qwen. Tarjeta oficial de `Qwen/Qwen2.5-3B-Instruct`. Consultada el 23 de octubre de 2026: `https://huggingface.co/Qwen/Qwen2.5-3B-Instruct`.

**[D2]** Qwen. Tarjeta oficial de `Qwen/Qwen2.5-1.5B-Instruct`. Consultada el 23 de octubre de 2026: `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct`.

**[D3]** Hugging Face. Documentación de TRL, *GRPO Trainer*, apartados de entrenamiento de agentes y rollouts: `https://huggingface.co/docs/trl/grpo_trainer`. Consultada el 23 de octubre de 2026. Las llamadas concretas deben verificarse frente a la versión fijada en el lock; este documento no promete compatibilidad automática de interfaces.

---

## Criterio final de finalización

La investigación está completada cuando existe un conjunto auditable de resultados que permite responder PI1, PI2 y PI3 en el alcance efectivamente ejecutado, identificar las comparaciones inconclusas y reproducir cada conclusión desde las trazas.

**No está completada únicamente porque el entrenamiento termine, cambien los pesos o aumente la recompensa de entrenamiento. Tampoco necesita una victoria de RL para producir una contribución válida.**
