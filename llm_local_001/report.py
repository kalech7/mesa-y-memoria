"""Reports distinguish observed measurements, unexecuted stages, and inference."""
from .artifacts import write_csv, write_json


OBS_FIELDS = ['status', 'split', 'method', 'observation', 'seed', 'users', 'decisions', 'errors',
              'error_rate', 'mean_return', 's3_accuracy', 'necessary_asks', 'unnecessary_asks',
              'incorrect_updates', 'extraction_errors', 'format_errors', 'ci_return', 'reason']
OBS_FIELDS += ['temporal_decision_errors', 'conflicts_not_detected', 'elapsed_seconds',
               'prompt_tokens', 'completion_tokens', 'generation_truncations']
BASE_FIELDS = OBS_FIELDS + ['comparison', 'delta_return', 'ci_delta_return', 'delta_error_rate']
CURVE_FIELDS = ['status', 'method', 'seed', 'step', 'loss', 'validation_return', 'tokens',
                'gradient_norm', 'seconds', 'reason']
ERROR_FIELDS = ['status', 'method', 'seed', 'user_id', 'family', 'session_index', 'causes',
                'causal_status', 'reward', 'proposed_action', 'reason']


def csv_summary(summary, method, observation, seed=None, split='test'):
    return {'status': 'EXECUTED', 'split': split, 'method': method, 'seed': seed,
            'observation': observation, 'users': summary['users'], **summary['counts'],
            'error_rate': summary['metrics']['error_rate']['mean'],
            'mean_return': summary['metrics']['return']['mean'],
            's3_accuracy': summary['metrics']['future_accuracy']['mean'],
            'ci_return': summary['metrics']['return']['ci95'],
            **{key: summary['metrics'][key]['mean'] for key in ('elapsed_seconds', 'prompt_tokens',
                                                              'completion_tokens', 'generation_truncations')}}


def blocked_report(output, audit, data_audit, metadata, b4=None):
    reasons = audit.get('reasons', [])
    reason = '; '.join(str(r) for r in reasons) or 'No trainable local model verified'
    is_blocked = audit.get('status', 'blocked') != 'ready'
    missing = {'status': 'NOT_RUN_BLOCKED' if is_blocked else 'NOT_RUN_AUDIT_ONLY', 'reason': reason}
    result_label = 'CASO E' if is_blocked else 'SOLO AUDITORÍA, SIN EXPERIMENTO'
    acquisition = ('La descarga del modelo y la instalación de dependencias fueron autorizadas '
                   'expresamente por el usuario; véase `preparation_manifest.json`. '
                   'No se utilizó inferencia remota.'
                   if (output / 'preparation_manifest.json').exists() else
                   'No se instalaron paquetes, descargaron modelos ni utilizaron servicios remotos.')
    write_csv(output / 'observability_results.csv',
              [{**missing, 'method': 'B0', 'split': 'validation', 'observation': obs}
               for obs in ('O1', 'O2')], OBS_FIELDS, metadata)
    write_csv(output / 'baseline_comparison.csv',
              [{**missing, 'method': method, 'split': 'test'} for method in ('B0','B1','B2','B3','B4')],
              BASE_FIELDS, metadata)
    write_csv(output / 'training_curves.csv',
              [{**missing, 'method': method} for method in ('B1','B2','B3')], CURVE_FIELDS, metadata)
    write_csv(output / 'error_analysis.csv', [missing], ERROR_FIELDS, metadata)
    write_json(output / 'training_metrics.json', {
        'provenance': metadata, **missing, 'optimizer_updates': None,
        'tokens_processed': None, 'trained_parameters': None, 'checkpoints': [],
        'reason_null_values': 'Unexecuted measurement, not zero training loss or zero errors',
        'algorithm_implemented': 'Autoregressive REINFORCE with past-return baseline and frozen B1 reference',
    })
    write_json(output / 'checkpoint_manifest.json', {
        'provenance': metadata, **missing, 'base_model': None, 'checkpoints': [],
        'adapters_trained': False, 'base_weights_trained': False,
    })
    detail = '\n'.join(f'- {r}' for r in reasons)
    hardware = audit.get('hardware', {})
    gpu = hardware.get('gpu_nvidia_smi', {}).get('stdout', '').strip() or 'Consultar hardware_audit.json'
    ram = hardware.get('ram', {})
    available_ram = ram.get('MemAvailable_bytes')
    ram_text = f'{available_ram / (1024**3):.2f} GiB disponibles' if available_ram else 'Consultar hardware_audit.json'
    interpreter = audit.get('selected_python')
    environment = next((e for e in hardware.get('python_environments', [])
                        if e.get('requested_executable') == interpreter), {})
    packages = environment.get('packages', {})
    ollama_models = audit.get('model_manifest', {}).get('ollama', {}).get('models', [])
    inventory = '\n'.join(f"- `{model.get('name')}`: " +
                          ('referencia cloud excluida' if model.get('excluded_cloud') else
                           f"{model.get('details', {}).get('format', 'GGUF')}, "
                           f"{model.get('details', {}).get('quantization_level', 'cuantización según manifiesto')}; "
                           'sin checkpoint diferenciable verificado') for model in ollama_models)
    abstract = data_audit['overlap']['train:test']
    b4_detail = ('Se ejecutó únicamente un diagnóstico B4 sobre VALIDATION; véase '
                 '`validation/b4_diagnostic.json`. No es una comparación final con un LLM.'
                 if b4 else 'No se ejecutaron decisiones del test.')
    (output / 'INFORME_LLM_LOCAL.md').write_text(f'''# SPEC-LLM-LOCAL-001 — informe local

Fecha UTC: {metadata['created_at']}. Procedencia: `{metadata['source']}`.
Configuración: `config.json`, SHA-256 `{metadata['config_hash']}`.

**Resultado: {result_label}. No se ha entrenado un LLM ni se ha medido una ventaja de RL.**

Bloqueos comprobados automáticamente:

{detail}

`hardware_audit.json` registra CPU, RAM, RTX, versiones, intérpretes y las pruebas
efectivas de CUDA. `model_manifest.json` distingue los pesos locales, los modelos
de embeddings y las referencias cloud; estas últimas no participan. Un GGUF
cuantizado no se trata como un checkpoint entrenable de Transformers.

GPU (nombre, VRAM total/libre en MiB, driver): `{gpu}`.
RAM: {ram_text}. Intérprete reutilizable: `{interpreter}`.
Versiones en ese entorno: PyTorch `{packages.get('torch')}`, Transformers
`{packages.get('transformers')}`, PEFT `{packages.get('peft')}`. Un valor None indica
dependencia ausente. La prueba CUDA de tensor tuvo alcance técnico y no entrenó
un LLM. El inventario de Ollama observado fue:

{inventory or 'Consultar model_manifest.json.'}

{acquisition}
Los experimentos anteriores se conservaron. Los campos de métricas no ejecutadas
están vacíos o son null; no significan cero errores. {b4_detail}

## Implementación y alcance

`llm_local_001/` reutiliza `rl_small_pilot`, `rl_observability`, el almacén SQLite
temporal y la recompensa vigente. S1 ingresa evidencia; las acciones generadas
controlan S2 y S3. Las salidas JSON inválidas se registran y puntúan como fallos,
sin excluirlas del denominador. El texto de las preguntas se comprueba antes de
permitir que el respondedor existente entregue una aclaración.

El backend carga exclusivamente pesos locales de un modelo causal y genera tokens
reales. SFT enmascara el prompt; REINFORCE optimiza los logits de esos tokens y
utiliza retorno longitudinal, baseline de retornos anteriores y referencia B1.
B2 y B3 parten de B1, con selección por validación que incluye el paso cero y
conservación del último checkpoint. La ruta de entrenamiento queda **sin verificar
con un LLM real** por el bloqueo descrito; los tests de contrato no acreditan un
entrenamiento realizado.

El ajuste preferido usa adaptadores LoRA integrados en el LLM: solo esos parámetros
se entrenarían. El ajuste completo requiere `--full-finetune` y una prueba real de
recursos. La inferencia mediante Ollama no sustituye la carga con gradientes.

## Datos y validez

Se congelaron {data_audit['splits']['train']['users']} usuarios TRAIN,
{data_audit['splits']['validation']['users']} VALIDATION y
{data_audit['splits']['test']['users']} TEST, tres sesiones y cinco familias.
El hash del test antes de ajustar hiperparámetros está en `dataset_manifest.json`.
Se reutilizan las trayectorias existentes con direcciones de preferencia
balanceadas, redacción y combinaciones de contexto separadas.

La auditoría encuentra {abstract['abstract_overlap_count']} firmas abstractas
compartidas entre TRAIN y TEST: {abstract['right_trajectories_with_seen_abstract_signature']}
de {abstract['right_trajectories']} historias TEST tienen mecanismo abstracto visto.
Esto permite estudiar variación textual y composicional, **no** demostrar
generalización a mecanismos nuevos. No se dispone de historias redactadas y
revisadas independientemente: la revisión humana queda pendiente y cualquier
ejecución posterior es un piloto técnico sintético.

O1 frente a O2 debe medirse con el mismo B0 antes de entrenar; no fue medido aquí.
O3 requiere errores residuales observados y una justificación explícita sobre
relaciones temporales. No se activa automáticamente. B0–B3 comparten observación
seleccionada en validación, tokenizer y modelo base. B4 es un método convencional
de capacidad distinta, no un LLM equivalente. La métrica primaria prefijada es
B2−B1 y B2−B3 en retorno longitudinal, con intervalos emparejados por usuario.

La extracción y el respondedor de preguntas actuales son parsers convencionales
limitados; la interpretación semántica de respuestas se restringe al esquema
documentado. No deben generalizarse los resultados a conversación libre sin
validación independiente. El test aún no se usó para evaluar modelos ni seleccionar
hiperparámetros.

## Requisitos concretos para continuar

1. Disponer de un checkpoint causal local completo compatible con Transformers,
   su tokenizer, configuración y licencia verificable. Debe caber junto con los
   adaptadores, activaciones y optimizador en los recursos medidos. No se inventa
   una ruta ni se exige una cantidad de VRAM que no se haya probado.
2. Usar un intérprete local con PyTorch CUDA compatible, Transformers y, para LoRA,
   PEFT compatible. El preflight documenta exactamente cuáles están disponibles.
   Cualquier instalación o descarga necesita autorización previa del usuario.
3. Superar la prueba real de forward/backward/guardado del candidato y ejecutar
   B0 O1/O2 en validación; revisar errores y comprobar margen frente a SFT.
4. Revisar una muestra de etiquetas, preguntas y recompensa antes de inferencias
   científicas; ampliar con historias independientes antes de escalar.

Comando reproducible desde la raíz (no instala ni descarga):

```sh
python -B -m llm_local_001.run --output reports/llm_local_001_replay
```

Para entrenar después de disponer de un modelo compatible, indicar
`--model-path /ruta/real/al/checkpoint` y ejecutar con el intérprete que apruebe
el preflight. El directorio de salida debe ser nuevo: no se sobrescriben corridas.

No hay evidencia todavía para elegir A, B, C o D. Conviene resolver el bloqueo
y medir observabilidad primero. La posibilidad técnica de RL no justificaría
convertirlo en el método principal de la tesis.
''')
    (output / 'VICENT_RESPUESTAS.md').write_text(f'''# Respuestas a Vicent

Fecha UTC: {metadata['created_at']}. Procedencia: `{metadata['source']}`.
Configuración SHA-256: `{metadata['config_hash']}`.

**P1. ¿Qué información adicional necesita el agente?**
Se implementó la comparación O1/O2 con evidencia original, fechas, vigencia,
contexto, fuente y aclaraciones. No hay mediciones con el LLM: no puede afirmarse
qué campos reducen errores. O3 queda condicionado a errores residuales.

**P2. ¿Qué causa los errores?**
No se han observado errores de un LLM en esta corrida. El diagnóstico distingue
formato, recuperación, extracción, interpretación temporal, clarificación y
operaciones de memoria. Los errores de reglas no se atribuyen al LLM.

**P3. ¿Cuál es el experimento mínimo y cómo se compara?**
Cinco familias existentes, tres sesiones y B0/B1/B2/B3/B4. B2 y B3 continúan el
mismo B1. Se congelan datos, recompensa y métricas; selección por validación y
bootstrap por usuario. Las firmas abstractas compartidas y la falta de revisión
humana limitan las conclusiones a un piloto sintético. Aún no se seleccionó un
modelo causal local entrenable.

**P4. ¿Qué falta antes de RL?**
El preflight se ejecutó y detectó: {reason}. Faltan un checkpoint entrenable
completo y dependencias compatibles verificadas, prueba de gradientes y guardado,
B0 O1/O2, diagnóstico de errores residuales y B1 validado. Antes de escalar:
historias independientes revisadas, validación del respondedor y recompensa,
tres semillas y evidencia de ventaja sobre SFT adicional con recursos comparables.

**P5. ¿Conviene continuar con RL?**
Resultado {result_label}: la falta de ejecución no demuestra que RL falle. No existen
resultados B0–B3 que justifiquen priorizar RL. Primero resolver disponibilidad
del modelo y medir observabilidad y SFT; después decidir mediante B2−B1/B2−B3
y coherencia de S3, informando el caso correspondiente sin prometer una ganancia.
''')


def completed_report(output, metadata, config, observations, selected, summaries, comparisons, training,
                     *, seed_comparisons=None):
    """Only actual measurements can reach this report; no invented superiority."""
    incremental, specific = comparisons['B2-B1']['return'], comparisons['B2-B3']['return']
    seed_zero = [name for name, result in training.items()
                 if name.startswith('B2_') and result['best_step'] == 0]
    enough_seeds = len(config['seeds']) >= 3
    consistent = bool(seed_comparisons) and all(
        scores[key]['return']['mean'] > 0 for scores in seed_comparisons.values()
        for key in ('B2-B1', 'B2-B3'))
    if enough_seeds and consistent and incremental['ci95'][0] > 0 and specific['ci95'][0] > 0 and not seed_zero:
        case = 'A — señal favorable exploratoria; requiere revisión y réplica independiente'
    elif incremental['ci95'][0] > 0 and specific['ci95'][0] <= 0:
        case = 'B — mejora incremental sin ventaja específica demostrada frente a más SFT'
    elif incremental['mean'] <= 0 or len(seed_zero) == len(config['seeds']):
        case = 'C — no se demuestra beneficio incremental en este piloto'
    else:
        case = 'F — incertidumbre estadística o replicación insuficiente'
    shortcut_risk = (incremental['mean'] > 0 and
                     summaries['B2']['metrics']['format_errors']['mean'] > summaries['B1']['metrics']['format_errors']['mean'] and
                     summaries['B2']['metrics']['error_rate']['mean'] >= summaries['B1']['metrics']['error_rate']['mean'] and
                     summaries['B2']['metrics']['future_accuracy']['mean'] <= summaries['B1']['metrics']['future_accuracy']['mean'])
    if shortcut_risk:
        case = 'F — retorno compatible con evitar penalizaciones mediante salidas inválidas, sin mejora de decisiones'
    table = ['| Método | Error | Retorno (IC95%) | Exactitud S3 |', '|---|---:|---|---:|']
    for method, summary in summaries.items():
        metrics = summary['metrics']
        table.append(f"| {method} | {metrics['error_rate']['mean']:.3f} | "
                     f"{metrics['return']['mean']:.3f} ({metrics['return']['ci95']}) | "
                     f"{metrics['future_accuracy']['mean']:.3f} |")
    deltas = '\n'.join(f"- {key}: Δ retorno {value['return']['mean']:.4f}; IC95% {value['return']['ci95']}."
                       for key, value in comparisons.items())
    obs_delta = observations['O2']['metrics']['return']['mean'] - observations['O1']['metrics']['return']['mean']
    base_format_floor = all(observations[obs]['counts']['format_errors'] >= observations[obs]['counts']['terminal_decisions']
                            for obs in ('O1', 'O2'))
    observability_limit = ('B0 falló el esquema en todas las decisiones terminales de ambas condiciones: '
                           'O1/O2 es inconcluso por efecto suelo de formato. La diferencia cero no demuestra '
                           'que la historia temporal sea inútil.' if base_format_floor else
                           'O1/O2 mezcla efectos sobre formato, interpretación y decisiones; no identifica '
                           'el efecto independiente de cada campo histórico.')
    scope = 'todos los pesos base' if config['full_finetune'] else 'solo adaptadores LoRA integrados en el LLM'
    ceiling_gap = summaries['B4']['metrics']['return']['mean'] - summaries['B1']['metrics']['return']['mean']
    ceiling = ('B1 igualó el retorno y la exactitud de B4: existe un efecto techo respecto '
               'a esta referencia observable.' if abs(ceiling_gap) < 1e-6 and
               summaries['B1']['metrics']['error_rate']['mean'] == 0 else
               f'Brecha media B4−B1 en retorno: {ceiling_gap:.4f}; el banco aún permite errores residuales.')
    seed_detail = '\n'.join(f"- Semilla {seed}: " + '; '.join(
        f"{key}={scores[key]['return']['mean']:.4f}" for key in ('B2-B1', 'B2-B3'))
        for seed, scores in (seed_comparisons or {}).items())
    acquisition = ('El modelo y las dependencias se obtuvieron con autorización explícita; '
                   'véase `preparation_manifest.json`, que incluye revisión, licencia y versiones. '
                   'Toda la inferencia y el entrenamiento fueron locales.'
                   if config.get('preparation_manifest') else
                   'No se descargaron modelos ni instalaron paquetes durante esta corrida; '
                   'se usaron recursos existentes e inferencia local.')
    text = f'''# SPEC-LLM-LOCAL-001 — piloto técnico ejecutado

Fecha UTC: {metadata['created_at']}. Procedencia: `{metadata['source']}`.
Configuración SHA-256: `{metadata['config_hash']}`.

Resultado: **{case}**. Se entrenaron {scope}. La revisión humana independiente
de historias y respuestas sigue pendiente; las conclusiones son exploratorias
y no se extienden a RL en general.

Modelo local: `{config.get('model_path')}`; revisión y licencia en
`preparation_manifest.json` y `model_manifest.json`.

{chr(10).join(table)}

Comparaciones emparejadas por usuario, con semillas promediadas dentro de usuario:

{deltas}

Diferencias por semilla (se exige signo positivo en todas para declarar A):

{seed_detail}

B0 O2−O1 en retorno VALIDATION: {obs_delta:.4f}. Observación seleccionada: {selected}.
El cambio de exposición no cambia los pesos. No se estima una contribución causal
independiente de cada campo histórico sin sus propias ablaciones.

{observability_limit}

Checkpoints RL cuyo mejor paso es cero: {seed_zero or 'ninguno'}. Para esos casos,
el entrenamiento no produjo una mejora validada y B2 seleccionado conserva B1.
Los checkpoints inicial, seleccionado y último están en `checkpoint_manifest.json`.
Cambios de pesos y gradientes se registran como evidencia técnica, no de calidad.

{ceiling} No se presenta como alcanzable una cota que conozca de antemano si el
usuario podrá resolver un conflicto. El resultado de este presupuesto reducido
no permite rechazar RL en general ni descartar el valor de la tesis.

Los denominadores incluyen salidas inválidas. Los intervalos agrupan S2 y S3 por
usuario. B4 conserva las reglas históricas y su intérprete convencional; su
capacidad es distinta de la del LLM. `final_metrics.json` contiene resultados por
familia y `baseline_comparison.csv` por semilla. `error_analysis.csv` ofrece causas
candidatas que requieren revisión; los errores del extractor fijo no se atribuyen
al modelo de lenguaje. ORACLE controla la extracción para aislar las decisiones.

SFT consume demostraciones generadas por reglas exclusivamente en TRAIN. B2 y
B3 cargan el mismo B1 por hashes y usan la misma observación. Se igualan las
actualizaciones y se aproxima el presupuesto de tokens con un exceso máximo de
una demostración por actualización; RL añade pasadas de generación y referencia.
Los recursos reales y las diferencias se conservan en `training_metrics.json`.
La recompensa se congeló antes del test. REINFORCE usa retorno con descuento 1,
baseline EMA {config['baseline_decay']} y coeficiente KL {config['kl_coefficient']}.
El estimador y su tratamiento de gradientes se documentan en `training.py`.

Las firmas abstractas compartidas en `dataset_manifest.json` impiden interpretar
usuarios nuevos como mecanismos nuevos. Se conservan los cinco mecanismos y los
tres momentos del simulador existente, con SQLite reabierto en S3. El respondedor
de aclaraciones y la evaluación de texto siguen siendo limitados al dominio y
necesitan validación humana antes de escalar.

Comando/configuración exactos: `config.json`; fuentes: `source_snapshot/`.
{acquisition}
'''
    (output / 'INFORME_LLM_LOCAL.md').write_text(text)
    (output / 'VICENT_RESPUESTAS.md').write_text(f'''# Respuestas a Vicent

Fecha UTC: {metadata['created_at']}. Procedencia: `{metadata['source']}`.
Configuración SHA-256: `{metadata['config_hash']}`.

**P1. Información adicional.** O2−O1 con el mismo B0 dio Δ retorno VALIDATION
{obs_delta:.4f}; se eligió {selected}. Véase `observability_results.csv` para
error, aclaraciones y coherencia posterior. No se atribuye el efecto a un solo
campo sin ablaciones adicionales.

{observability_limit}

**P2. Causas.** `error_analysis.csv` separa candidatos de recuperación/exposición,
formato, temporalidad, decisión y memoria. Las atribuciones semánticas requieren
revisión independiente. ORACLE permite estudiar decisiones con evidencia literal
controlada; no demuestra que el extractor convencional carezca de errores.

**P3. Experimento mínimo.** Un modelo local causal, cinco familias, tres sesiones,
B0/B1/B2/B3/B4 y semillas {config['seeds']}. B2/B3 parten exactamente de B1 con
presupuestos de actualizaciones iguales y tokens aproximados. No se construyó
otro simulador ni una cabeza clasificadora externa.

**P4. Antes de RL y de escalar.** Se completaron preflight, gradientes/guardado,
B0 O1/O2, B1 y controles técnicos. Se entrenaron {scope}. Faltan historias
redactadas y revisadas independientemente, validación humana del respondedor y
recompensa y réplica con mejor cobertura antes de una conclusión confirmatoria.

**P5. Recomendación basada en evidencia.** {case}. B2−B1: {incremental};
B2−B3: {specific}. Checkpoints con mejor paso cero: {seed_zero or 'ninguno'}.
La factibilidad técnica por sí sola no justifica priorizar RL en la tesis. Una
ventaja exploratoria requiere revisión, test independiente y reproducción antes
de escalar; un resultado nulo aquí no descarta RL en otros dominios.
''')
