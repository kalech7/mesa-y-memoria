# Experimento SPEC-LLM-LOCAL-001

Este módulo implementa el piloto local solicitado en `SPEC_LLM_LOCAL_001.md`.
No instala paquetes, descarga modelos ni utiliza inferencia remota. El preflight
lee únicamente el inventario local de Ollama; no llama a generación ni emplea
modelos cloud. Un archivo GGUF no acredita capacidad de entrenamiento.

Desde la raíz del proyecto:

```sh
python -B -m llm_local_001.run --output reports/llm_local_001_nueva_corrida
```

El código 2 significa bloqueo de recursos verificado (caso E); 0 significa que
se completó el modo solicitado; 1 indica un error. Una corrida no sobrescribe
directorios con resultados anteriores. La auditoría puede tardar por la
inspección de intérpretes locales. Solo auditoría, sin inferencia ni entrenamiento:

```sh
python -B -m llm_local_001.run --preflight-only --output reports/llm_local_001_auditoria
```

Cuando exista un checkpoint real entrenable, se especifica con `--model-path`.
Debe contener pesos de Transformers completos, tokenizer, configuración y evidencia
de licencia local. El programa reutiliza un intérprete existente que pase el
preflight. LoRA requiere PEFT instalado; solo los adaptadores integrados en el LLM
se ajustarían. `--full-finetune` opta expresamente por todos los pesos base y no
garantiza que quepan en memoria. La capacidad se comprueba con forward/backward
y guardado reales. No se inventan rutas ni se convierte GGUF automáticamente.

## Protocolo fijado antes de test

- Cinco familias y tres sesiones del generador existente; S1 ingresa evidencia y
  la política decide en S2 y S3. SQLite se cierra y reabre entre sesiones.
- 60/20/40 usuarios sintéticos en TRAIN/VALIDATION/TEST, por defecto. Los mecanismos
  abstractos son compartidos. Textos y combinaciones de contexto se separan;
  `dataset_manifest.json` cuantifica explícitamente la superposición abstracta.
- Las etiquetas del evaluador y respuestas futuras se almacenan en
  `evaluator_private/`; los prompts usan una proyección explícita de evidencias.
- B0 usa sus propios tokens para O1/O2 en VALIDATION; mayor retorno selecciona la
  observación y un empate favorece O2. No se activa O3 sin un diagnóstico residual
  específico. En este mínimo, la ampliación O3 requiere una revisión explícita.
- B1 aprende solo de demostraciones TRAIN del sistema histórico. Son etiquetas
  sintéticas de reglas; no se presentan como anotación humana.
- B2 y B3 cargan el mismo B1 seleccionado, con hashes de base/adaptadores/tokenizer.
  REINFORCE actúa con tokens realmente muestreados del modelo, incluye S3 en el
  retorno y utiliza un baseline de retornos anteriores. La implementación documenta
  el estimador de KL respecto al B1 congelado y sus coeficientes.
- B3 iguala actualizaciones y aproxima tokens de entrenamiento por actualización,
  con exceso máximo de una demostración; RL además consume generación y referencia.
  Se reportan tokens y recursos efectivos; no se afirma igualdad de tiempo.
- Semillas por defecto: 101, 202, 303. Selección por VALIDATION, empate al paso
  anterior, incluido cero; se conservan inicial, seleccionado y último checkpoint.
- Recompensa histórica congelada: +1 correcto/fundamentado, −1 incorrecto, −0.1 por
  aclaración, −0.5 actualización injustificada, −0.6 sin evidencia, −2 inseguridad,
  +0.7 preservación justificada. Se registran acciones propuestas y ejecutadas.
- Métrica primaria: B2−B1 y B2−B3 en retorno longitudinal. Complementos: errores,
  coherencia S3, aclaraciones y memoria. Intervalos emparejados por usuario, con
  semillas agrupadas dentro de usuario. B4 conserva el sistema histórico nativo.
- Formato inválido y citas inventadas son fallos observables; no hay reintento
  condicionado al resultado ni sustitución por una acción gold.

El esquema completo de la respuesta y sus validaciones está en `environment.py`.
`answer` incluye una polaridad explícita y un texto; la evaluación automática de
preferencia usa el campo estructurado, no acredita calidad semántica libre del
texto. La interpretación del texto de preguntas usa un router conservador de
dominio limitado. Estas limitaciones requieren revisión humana independiente.
El preflight técnico no sustituye esa revisión. Los resultados son de un piloto
sintético, no evidencia confirmatoria hasta cumplirla.

## Archivos y comprobaciones

`reports/llm_local_001/` contiene el informe, respuestas a Vicent, auditorías,
tablas, métricas y manifiesto de checkpoints. En caso de bloqueo, las etapas no
ejecutadas se identifican con `NOT_RUN_BLOCKED` y métricas vacías/null.
`source_snapshot/`, hashes de fuentes/configuración y el hash del test permiten
auditar qué se ejecutó. Los experimentos anteriores no se modifican.

```sh
python -B -m unittest discover -s tests
python -B -m unittest tests.test_llm_local_protocol tests.test_llm_local_environment tests.test_llm_local_training -v
```

Las pruebas de gradientes con fixtures matemáticos o modelos aleatorios temporales
son verificaciones de software. Nunca se usan para obtener métricas del experimento
ni se presentan como un LLM preentrenado disponible.
