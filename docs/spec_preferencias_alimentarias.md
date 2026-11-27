# SPEC-001: contrato e implementación incremental

Fecha de diagnóstico: 2026-10-19. Implementación local determinista, sin RL ni API.
El plan de pruebas y las correcciones están autorizados por SPEC-001 del usuario.

## Diagnóstico previo a cambios de producción

Línea base: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v`:
71 pruebas, OK, 5,498 s; Python del sistema 3.14.7. El proyecto declara >=3.13.

| Funcionalidad esperada | Estado inicial | Archivos | Deficiencia | Cambio previsto |
|---|---|---|---|---|
| Cuatro comidas | Parcial | simulator/user.py:49 | Solo contraste desayuno/cena; almuerzo/merienda se aceptan fuera de ámbito | Parser compartido de cuatro comidas y regresiones |
| Días y fechas | Narrativo | simulator/user.py:66; data_generation/build.py | No se evalúa aplicabilidad por día | Contexto explícito, fechas y días concretos |
| Preferencia estructurada | Ausente | memory/store.py:13 | Texto sin polaridad ni dimensiones consultables | Tablas adicionales; mantener eventos originales |
| Resolución | Prompt/heurísticas | agents/policies.py | Sin motor determinista contextual | Resolver público separado de expectativas |
| Extracción/actualización | Texto/ADD | memory/extraction.py; evaluation/runner.py | No interpreta alcance estructurado ni vigencia | Extracción conservadora y sustitución explícita trazable |
| Restricciones | Advertencia | agents/policies.py:13 | Alergia sin representación propia | Registro de restricciones independiente |
| Usuarios/conversaciones | Sintético por plantillas | data_generation/build.py | Dos opciones/historia, plantillas compartidas | Nuevo generador compatible separado, sin reemplazar el anterior |
| Evaluación | Pilotos y métricas legacy | evaluation/metrics.py; tests/ | Sin oráculos independientes contextualizados | Expectativas declaradas, 20 categorías, tres semillas |

Reproducción inicial: `interpret_question('¿Quieres tofu para el almuerzo?',
'¿Qué compro para la merienda?', ['tofu'])` devuelve `preference`; también
merienda/desayuno y «¿Tu hijo prefiere tofu?» devuelven `preference`.

## Plan de verificación

1. Unitarias de extracción, ámbito desconocido, polaridad, fuentes y restricciones.
2. Ocho pruebas independientes de día × comida y S01–S20.
3. Integración por SQLite y reapertura, conservación y atribución de evidencia.
4. Metamórficas: orden, ruido, día, comida, fuente y excepción temporal.
5. Evaluación declarativa de al menos 200 instancias por semilla 42/43/44.
6. Métricas con denominadores, intervalos Wilson descriptivos y resultados por
   categoría, contexto y conflicto; registrar todos los fallos.

Las expectativas no importan ni ejecutan el parser o resolver. El motor recibe
solo usuario, eventos públicos, consulta y fecha; no recibe categoría ni gold.
Los resultados sintéticos verifican este contrato y no validan conducta humana.
Los conjuntos retenidos usan familias distintas, pero el vocabulario es finito;
se declarará cualquier inspección/adaptación posterior del parser a esos datos.

## Implementación disponible

- `simulator/preferences.py`: modelos Preference, Restriction, QueryContext y
  ParsedEvidence; extracción conservadora, comidas y aliases configurables.
- `memory/store.py`: tablas adicionales, conservación del original, propietario,
  colisiones/idempotencia y snapshot alimentario. Contexto y configuración de
  extracción se preservan en event_metadata. No se migra destructivamente la BD.
- `simulator/resolution.py`: recuperación exhaustiva por usuario, aplicabilidad,
  especificidad parcial, conflictos y actualización explícita. Vigencia efectiva
  se calcula sin reescribir la evidencia antigua; se puede reconstruir el pasado.
- `simulator/food_user.py`: respuestas privadas a slots solicitados. El resolver
  no recibe perfiles privados, familias, etiquetas ni expectativas.
- `data_generation/food_scenarios.py`: S01–S20 con expectativas declaradas desde
  intenciones. El comando final usa 320 casos/semilla: 16/categoría y 2 consultas
  de cada combinación de S08 por semilla (6 por combinación en tres semillas).
- `evaluation/food_preferences.py`: métricas, gates, fuentes, trazas y reportes;
  `scripts/evaluate_food_preferences.py` ofrece la misma CLI.

Se emiten ANSWER, CLARIFY, PRESERVE_CONFLICT, VERIFY y ABSTAIN. No se emite
CONDITIONAL_ANSWER en esta versión: se pregunta el contexto pendiente o se conserva
el conflicto. No hay verificación externa automática: VERIFY solicita verificar,
no ejecuta una consulta a pedidos o servicios reales.

ANY solo se extrae de generalidad explícita. Un gusto con ámbito ausente puede
reportarse como información comunicada sin afirmar universalidad; si una consulta
añade una dimensión no conocida se pide aclarar. Dos alimentos con LIKE no son por
sí solos contradictorios. La especificidad resuelve polaridades sobre el mismo
alimento con alcances comparables; dimensiones incomparables conservan conflicto.

## Versiones y evaluación retenida

La primera ejecución de desarrollo se conserva con errores. En gold v2, CPC
designa subdeterminación contextual, no mera coexistencia resuelta. S20 expresa
explícitamente «Todos los días»; su resultado esperado no se cambió para mejorar
la puntuación. Estas revisiones del instrumento ocurrieron durante desarrollo y
no se presentan como una evaluación ciega.

Tras congelar la interpretación se creó un desafío de 32 casos sin consultar ni
ejecutar el parser. No se adaptó la interpretación a sus errores. El hash original
está en challenge_review.json; este archivo añade una revisión posterior explícita
del instrumento sin sustituir las etiquetas originales. El formateo de preguntas
se hizo legible después; no cambia decisiones ni comidas seleccionadas.

## Denominadores y límites

Exactitud contextual compara únicamente dimensiones esperadas explícitamente,
incluido null. Clasificación de comida excluye consultas sin una comida concreta;
la preservación del valor desconocido se mide aparte. Decisión estructurada exige
contexto, acción aceptable, alimentos, evidencia necesaria y tipo de conflicto.
La fidelidad estructural exige referencias visibles/necesarias y ausencia de
alimentos no respaldados por el gold; no representa una valoración humana del texto.

Precisión/recall de recuperación se calculan por IDs de evento, deduplicando los
fragmentos extraídos; la búsqueda es exhaustiva y no valida BGE. La tasa de
clarificación necesaria excluye los casos donde preguntar es solo opcional;
la innecesaria usa casos donde CLARIFY no es aceptable. Restricciones se evalúan
solo en escenarios anotados explícitamente como tales. Denominador cero -> null.

La latencia informada corresponde a resolución pura y filtrado contextual,
excluyendo extracción, lecturas SQLite y scoring. Los tiempos totales incluyen
pruebas/evaluación y no son un benchmark de servicio. Los intervalos Wilson son
descriptivos del banco sintético, sin inferencia poblacional ni corrección por
plantillas compartidas. No hubo entrenamiento, API, tarifas o uso de H200.

Limitaciones: catálogo alimentario y gramática finitos; fechas absolutas ISO y
relativas referenciadas; condiciones compuestas y paráfrasis no universales.
Los pedidos conservan fuente pero la polaridad extraída de conducta es una señal
observacional, nunca autoridad para sobrescribir un gusto declarado. Restricciones
se comprueban contra un catálogo mínimo explícito, sin inferir ingredientes de
recetas, trazas de alérgenos ni seguridad clínica.

## Modelo y acciones

Los alcances desconocidos son `null`, diferentes de `ANY` explícito. Las
preferencias tienen procedencia y vigencia, y las restricciones se almacenan
separadamente. El historial no se borra al aplicar una actualización.
Acciones: ANSWER, CLARIFY, CONDITIONAL_ANSWER, PRESERVE_CONFLICT, VERIFY, ABSTAIN.
La resolución no usa recencia como desempate general. Las observaciones de
pedidos no anulan declaraciones ni garantizan que el comprador consumió algo.

## Alcance de aceptación

Gate A verifica contratos deterministas y restricciones controladas. Gate B
aplica los umbrales de ingeniería de SPEC-001. Gate C evalúa semillas, ruido y
familias reservadas. Un fallo crítico impide declarar validación completa.
Los IC describen este banco sintético, no independencia de plantillas ni una
población humana. La incertidumbre, abstención y fallos técnicos no se convierten
silenciosamente en aciertos o preferencias generales.
