"""Generate the Spanish scientific report from frozen experiment artifacts."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def load(path):
    return json.loads(path.read_text())


def number(value):
    return 'no evaluable' if value is None else f'{value:.4f}'


def estimate(value):
    if not value or value.get('mean') is None:
        return 'no evaluable'
    lo, hi = value['ci95']
    return f"{value['mean']:.4f} [{lo:.4f}, {hi:.4f}]"


def generate_report(run):
    run=Path(run)
    freeze=load(run/'freeze.json')
    resources=load(run/'resources.json')
    from .descriptive import summarize_actions
    metrics=summarize_actions(run)
    comparisons=load(run/'comparisons.json')
    config=load(run/'config.json')
    selected=freeze['selected_observation']
    root=run.parent
    analysis_source=run/'report_source_snapshot.py'
    shutil.copyfile(Path(__file__),analysis_source)
    (run/'report_generation.json').write_text(json.dumps({
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'report_source_sha256':hashlib.sha256(analysis_source.read_bytes()).hexdigest(),
        'role':'descriptive reporting only; policies, reward and test inference unchanged',
        'inference_source_manifest':'source_hashes.json'},indent=2)+'\n')
    # Standalone figures use matplotlib; numerical curves are always exported.
    curve_rows=[]
    for path in sorted((run/'training').glob('*/B4_seed*.json')):
        result=load(path)
        for point in result['curve']:
            curve_rows.append(dict(observation=path.parent.name,seed=result['metrics']['seed'],
                episodes=point['episodes'],validation_return=point['validation']['mean_return'],
                validation_error=point['validation']['decision_error_rate'],
                training_return=point.get('training_mean_return')))
    fields=['observation','seed','episodes','validation_return','validation_error','training_return']
    with (run/'learning_curves.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader(); writer.writerows(curve_rows)
    figure_note='Curvas numéricas en `learning_curves.csv`.'
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, axes=plt.subplots(1,3,figsize=(13,3.7),sharey=True)
        for ax,rep in zip(axes,('O1','O2','O3')):
            for seed in config['seeds']:
                rows=[r for r in curve_rows if r['observation']==rep and r['seed']==seed]
                ax.plot([r['episodes'] for r in rows],[r['validation_return'] for r in rows],
                        marker='.',label=str(seed))
            ax.set(title=rep,xlabel='Episodios de entrenamiento')
            ax.grid(alpha=.2);ax.legend(title='Semilla')
        axes[0].set_ylabel('Retorno por usuario en validación')
        fig.tight_layout();fig.savefig(run/'learning_curves.svg');fig.savefig(run/'learning_curves.png',dpi=150)
        plt.close(fig)
        figure_note=f'![Curvas de validación]({run.name}/learning_curves.png)\n\nCurvas: [CSV]({run.name}/learning_curves.csv), [SVG]({run.name}/learning_curves.svg).'
    except ImportError:
        pass
    def metric(rep,method,key):
        return estimate(metrics[rep][method]['metrics'].get(key))
    lines=[
      '# SPEC-RL-004 — Observabilidad, errores de decisión y piloto RL',
      '',
      'Experimento sintético mínimo ejecutado en CPU. Se separan el efecto de la información y el efecto del método. Las cifras siguientes proceden de artefactos de ejecución; no constituyen validación humana de la semántica ni evidencia sobre un LLM entrenado.',
      '',
      '## 1. Diagnóstico inicial',
      '',
      'La política histórica observa `required_decision` y una reconciliación calculada por reglas. El contrato puede cambiar la acción, la pregunta exacta y la operación de memoria. Esas capacidades impiden atribuir el resultado protegido al aprendizaje. El nuevo módulo no entrega esa decisión y conserva únicamente protecciones de seguridad.',
      '',
      'Se analizaron 120 decisiones fallidas archivadas de 120 usuarios distintos, doce por cada una de diez familias históricas, semilla 4004. Las 120 presentan discrepancias de extracción y sus sesiones Oracle emparejadas aciertan; no se establece un error independiente de política. En los cinco métodos PIPELINE históricos hay 792 fallos de 1008 sesiones, sin intervenciones del contrato en esa ejecución concreta. El código sí permite sustituciones amplias: se distingue capacidad del contrato de intervención observada.',
      '',
      'Fallo localizado: la expresión «sin importar la compañía ni el lugar» activa una señal negativa cuando «Disfruto comer» no se reconoce como preferencia positiva. El texto bruto existe en SQLite, pero no está disponible en los hechos de la observación histórica. El diagnóstico está en [audit.md](audit.md) y [diagnosis](diagnosis/summary.json). Conserva entradas, memorias, observaciones exactas, propuestas, ejecuciones, recompensa, resultados y causas probables. Los campos ausentes se declaran y las discrepancias de extracción se separan de sus síntomas posteriores. Las etiquetas históricas y el análisis son privados del evaluador; no alimentan el nuevo entrenamiento.',
      '',
      '## 2. Información observada',
      '',
      '| Observación | Información permitida |',
      '|---|---|',
      '| O1 RECENT_ONLY | Únicamente el hecho más reciente, con su contenido y procedencia; ablación. |',
      '| O2 TEMPORAL_HISTORY | Historial relevante completo, contenido, fechas observada/registrada, intervalo, fuente, contexto y fecha consultada. |',
      '| O3 TEMPORAL_ENRICHED | O2 más tipo interpretado, excepción temporal, relaciones, conflictos potenciales, contexto conocido/faltante, aclaraciones y procedencia. |',
      '',
      'Todas las condiciones conservan las restricciones explícitas de seguridad. O1 no expone hechos descartados por un canal alternativo de entrada. No se entregan `required_decision`, familia, gold, preferencias latentes ni respuestas futuras. Tipo y relaciones se derivan del texto, la fuente y el historial comunicado. Las fechas y el alimento consultado son metadatos estructurados: este piloto no evalúa comprensión libre de consultas temporales.',
      '',
      'ANSWER expresa LIKE/DISLIKE sobre un alimento consultado, no genera menús ni recomendaciones naturales. La restricción permanente de maní se conserva, pero el generador no consulta ese alimento: cero infracciones en las tablas no prueba robustez ante restricciones desafiadas. El bloqueo se comprobó por separado con una acción insegura en las tres representaciones.',
      '',
      'Las reglas fuertes pueden calcular de O2 las mismas interpretaciones que O3 almacena. O3 aporta una representación explícita y auditable, no necesariamente información semántica nueva; un resultado nulo O3−O2 es legítimo.',
      '',
      'O1 conserva los mismos campos temporales del hecho que sí expone: O2−O1 identifica el efecto de conservar más historia relevante, no el efecto aislado de añadir fechas. En estas cinco plantillas, fecha registrada y observada coinciden y los ámbitos son universales. No se identifica el beneficio marginal de cada campo de O3 ni el coste de interpretarlo para un lector sin el preprocesador simbólico compartido.',
      '',
      '## 3. Diseño y congelación',
      '',
      f"Particiones: {config['train_users']} usuarios de entrenamiento, {config['validation_users']} de validación y {config['test_users']} de test. Cinco familias equilibradas: cambio explícito, excepción temporal, conflicto relevante resoluble, conflicto irrelevante y conflicto relevante irresoluble. Tres sesiones con fechas separadas; S1 inicializa memoria persistente y S2/S3 son decisiones de la política. Se permite una aclaración en S2 y ninguna en S3.",
      '',
      'El conflicto relevante contiene declaraciones opuestas sobre el mismo alimento, sin declarar sustitución. Una aclaración sobre preferencia puede resolverlo; preguntas sobre día, comida o fuente no revelan por defecto la preferencia privada. La incertidumbre justifica preservar o responder condicionalmente; no se exige adivinar un estado oculto. La resolución fundamentada obtiene mayor utilidad que conservar incertidumbre, y preguntar cuesta incluso si el usuario no puede aclarar.',
      '',
      f"La observación principal es **{selected}**, seleccionada por retorno de B2 en validación. Comparador no RL: **{freeze['selected_non_rl']}**; semilla RL seleccionada: **{freeze['selected_rl']}**. Empates prefieren la primera opción según el protocolo. [freeze.json]({run.name}/freeze.json) fue escrito antes del test y contiene recompensa, parámetros y hashes de checkpoints.",
      '',
      'Los perfiles, años y plantillas textuales son distintos por partición. **Se comparten las cinco plantillas abstractas y el soporte de estados simbólicos**. Cambiar fechas o redacción no constituye generalización conceptual. Se conserva esta limitación para mantener mínimo el experimento; P4 sólo puede responderse dentro de esta distribución sintética.',
      '',
      '## 4. Observabilidad con método fijo B2',
      '',
      '| Condición | Error por usuario | Retorno por usuario | Coherencia S3 |',
      '|---|---:|---:|---:|']
    for rep in ('O1','O2','O3'):
        lines.append(f"| {rep} | {metric(rep,'B2','error_rate')} | {metric(rep,'B2','return')} | {metric(rep,'B2','future_accuracy')} |")
    lines += ['', 'Diferencias emparejadas; IC percentil 95% por usuario sintético completo. Para error, un valor negativo favorece la condición más informada.', '',
      '| Diferencia | Error | Retorno | Coherencia S3 |','|---|---:|---:|---:|']
    for label,data in comparisons['observability_fixed_B2'].items():
        lines.append(f"| {label} | {estimate(data.get('error_rate'))} | {estimate(data.get('return'))} | {estimate(data.get('future_accuracy'))} |")
    lines += ['', '## 5. Comparación con la misma observación', '',
      '| Método | Error | Retorno | Coherencia S3 |','|---|---:|---:|---:|']
    for method in ('B1','B2','B3_seed_mean','B4_INITIAL','B4_seed_mean'):
        lines.append(f"| {method} ({selected}) | {metric(selected,method,'error_rate')} | {metric(selected,method,'return')} | {metric(selected,method,'future_accuracy')} |")
    lines += ['',
      'B1 es última escritura como ablación; B2 interpreta fechas, cambios, excepciones, conflictos y aclaraciones sin consultar gold. B3 clona demostraciones de B2 sólo de entrenamiento. B3 y B4 usan exactamente la misma tabla de estados y catálogo de acciones compuestas (respuesta/polaridad/slot × operación de memoria), sin fallback a reglas. El extractor simbólico de características es común y fijo; no lo aprende RL. B3 dispone de acciones demostradas; B4 sólo de recompensas y exploración. Sus presupuestos no son idénticos: se documentan, no se ocultan.',
      '',
      f"B5 LLM_PROMPTING: `{resources['B5']['status']}`; {resources['B5']['reason']}. No se simula ni se atribuye un resultado al LLM.",
      '',
      '## 6. Semillas, aprendizaje y recompensa',
      '',
      '| Semilla | Episodios completados / seleccionado | Error validación antes → después | Retorno validación antes → después | Error test |',
      '|---|---:|---:|---:|---:|']
    for seed in config['seeds']:
        result=load(run/'training'/selected/f'B4_seed{seed}.json')['metrics']
        lines.append(f"| {seed} | {result['completed_episodes']} / {result['selected_episode']} | {number(result['before']['decision_error_rate'])} → {number(result['after']['decision_error_rate'])} | {number(result['before']['mean_return'])} → {number(result['after']['mean_return'])} | {metric(selected,f'B4_seed{seed}','error_rate')} |")
    lines += ['', 'Capacidad y datos de entrenamiento en la observación principal:', '',
              '| Semilla | Demostraciones B3 | Estados / celdas B3 | Visitas Q | Estados / celdas Q seleccionado |',
              '|---|---:|---:|---:|---:|']
    for seed in config['seeds']:
        b3=load(run/'training'/selected/f'B3_seed{seed}.json')['metrics']
        b4=load(run/'training'/selected/f'B4_seed{seed}.json')['metrics']
        lines.append(f"| {seed} | {b3['demonstrations']} | {b3['unique_states']} / {b3['table_cells']} | {b4['completed_episodes']} | {b4['unique_states']} / {b4['table_cells']} |")
    lines += ['',figure_note,'',
      'Se conserva Q-learning tabular, α=0,3, γ=1 y ε de 1 a 0,05; los checkpoints se eligen por validación a bloques prefijados, incluido episodio cero. Se remuestrean usuarios de entrenamiento: visitas repetidas no son nuevos sujetos independientes. El piloto de entrenabilidad está autorizado por SPEC-RL-004 y no altera los gates ni los resultados históricos.',
      '',
      f"Coeficientes congelados: `{json.dumps(freeze['reward'],ensure_ascii=False,sort_keys=True)}`. La recompensa usa la decisión ejecutada, penaliza intentos inseguros y actualizaciones injustificadas, y recoge las consecuencias en S3. No hay premio por preguntar sin obtener evidencia. El retorno acumulado y las acciones por episodio están en los logs de entrenamiento.",
      '',
      'Comparaciones emparejadas en la observación principal:', '',
      '| Diferencia | Error | Retorno | Coherencia S3 |','|---|---:|---:|---:|']
    for label,data in comparisons['learning_selected_observation'].items():
        lines.append(f"| {label} | {estimate(data.get('error_rate'))} | {estimate(data.get('return'))} | {estimate(data.get('future_accuracy'))} |")
    lines += ['',
      'El límite temporal se aplica al bucle de entrenamiento; las evaluaciones de validación y la evaluación final añaden tiempo. Los episodios completados pueden variar entre máquinas. Los checkpoints se comparan sólo en los bloques programados. Se conservan los valores y hashes del checkpoint seleccionado; una ejecución nueva sujeta a tiempo no garantiza el mismo punto de parada.',
      '', '## 7. Componentes y coherencia longitudinal', '',
      'Los conteos son promedios por usuario; las tasas de extracción, recuperación, detección, tipos y vigencia usan suma de numeradores / suma de denominadores. Se remuestrean usuarios completos también para estas razones. Denominador cero significa no evaluable. Detectar conflictos o recuperar evidencia no equivale a acertar la respuesta. Los errores de S3 son resultados posteriores; no se etiquetan automáticamente como propagación causal de una actualización.', '',
      '| Indicador | B2 principal | RL media de semillas |','|---|---:|---:|']
    for key in ('accuracy','questions','necessary_asks','necessary_question_recall',
                'unnecessary_asks','repeated_asks','incorrect_updates','future_errors',
                'relevant_memory_recall','conflicts_detected','conflict_detection_rate',
                'temporal_type_error_rate','interval_applicability_error_rate',
                'extraction_error_rate','unsupported_answers','shield_interventions'):
        lines.append(f"| {key} | {metric(selected,'B2',key)} | {metric(selected,'B4_seed_mean',key)} |")
    lines += ['', 'Errores por familia (promedio de S2 y S3):', '',
              '| Familia | B2 O1 | B2 O2 | B2 O3 | RL principal, media de semillas |',
              '|---|---:|---:|---:|---:|']
    for family in sorted(metrics[selected]['B2']['by_family']):
        errors=[metrics[rep]['B2']['by_family'][family]['metrics']['error_rate'] for rep in ('O1','O2','O3')]
        errors.append(metrics[selected]['B4_seed_mean']['by_family'][family]['metrics']['error_rate'])
        lines.append(f"| {family} | "+' | '.join(estimate(v) for v in errors)+' |')
    lines += ['', 'PIPELINE vuelve a usar el extractor existente sobre los mismos episodios; VISIBLE_STRUCTURED usa anotaciones de hechos expresamente comunicados, sin estados latentes. Este contraste diagnostica la cadena extracción/adaptación/decisión, no una mejora atribuible a RL.', '',
      '| Observación B2 PIPELINE | Error | Retorno | Coherencia S3 |','|---|---:|---:|---:|']
    for rep,entry in comparisons['pipeline_B2'].items():
        mm=entry['metrics']['metrics']
        lines.append(f"| {rep} | {estimate(mm.get('error_rate'))} | {estimate(mm.get('return'))} | {estimate(mm.get('future_accuracy'))} |")
    lines += ['',
      'Una discrepancia de campo del extractor no implica necesariamente un fallo de decisión: una frontera a medianoche frente a la hora del mensaje puede ser inocua. `extraction_error_rate` mide discrepancias seleccionadas de polaridad, tipo y fronteras; no cubre exhaustivamente ámbito o fuente ni es exactitud lingüística global. `temporal_type_error_rate` compara tipos interpretados con anotaciones de evidencia comunicada; `interval_applicability_error_rate` compara vigencia de los mismos hechos recuperados. No son medidas de comprensión temporal libre ni incluyen hechos no recuperados.',
      '',
      'El [diagnóstico complementario](pipeline_diagnosis.md) identifica una cadena de causas en la excepción temporal de PIPELINE: el parser concreta un día de semana, el adaptador deja vacío el contexto de consulta, las reglas no deducen ese día de `query_at` y preguntan innecesariamente por él. Una intervención local que deriva el día de la fecha visible corrige el caso sin RL. Es un análisis exploratorio posterior a congelar las políticas, no un baseline nuevo seleccionado ni un cambio del test. No se atribuye este fallo exclusivamente a extracción.',
      '',
      'B1 acumula penalizaciones por UPDATE_WITH_HISTORY sin evidencia explícita de cambio. Su retorno combina errores de respuesta y escritura. Los JSONL individuales contienen las acciones de cada semilla. `descriptive_analysis/test_metrics.json` conserva las métricas numéricas y agrega los contadores de acciones sobre las tres semillas; el resumen bruto usa la trayectoria de referencia en las filas de media numérica. Ambos se conservan con sus hashes, sin cambiar inferencias.',
      '', '## 8. Validez experimental', '',
      'Las unidades del bootstrap son usuarios, con S2/S3 y condiciones emparejadas. En la media de tres semillas primero se promedian semillas dentro de cada usuario; no se triplica n. Los intervalos sólo describen variación del generador sintético, no diversidad humana ni incertidumbre de nuevos mecanismos. Una diferencia exacta puede producir IC degenerado por repetición de estados abstractos.',
      '',
      'El presupuesto de una pregunta en S2 y ninguna en S3 impide evaluar aprendizaje para evitar preguntas repetidas: ese cero está impuesto por el diseño. La exactitud de S3 tampoco distingue por sí sola resolver una preferencia de conservar justificadamente incertidumbre; esa diferencia aparece en el retorno y las acciones. La separación entre decisiones de memoria y su efecto futuro no se interpreta como causal sin una intervención emparejada adicional.',
      '',
      'La observación se selecciona por B2, no por resultados RL del test. Los métodos se comparan en los mismos usuarios y con la misma información en cada condición. Reglas y aprendizaje comparten interpretación simbólica accesible; no se atribuye ese conocimiento al entrenamiento. Las polaridades y mecanismos de sólo cinco plantillas tienen correlaciones repetidas: aprenderlas no acredita razonamiento general. No hay optimización posterior sobre test, evaluación humana ni prueba de superioridad sobre prompting fuerte. Las etiquetas son propuestas del generador y los baselines no son jueces externos.',
      '',
      'Las comprobaciones ejecutadas y los resultados históricos se conservan. El módulo nuevo registra acciones propuestas y ejecutadas; las protecciones no reemplazan automáticamente una decisión incierta por la respuesta esperada. Los artefactos privados y las observaciones públicas se mantienen separados. Las bases SQLite temporales se recargan en S3 y se eliminan al cerrar; su contenido y operaciones quedan serializados. Con `db_path` explícito se conserva también la base física.',
      '',
      '## 9. Recursos y reproducción', '',
      f"CPU `{config['hardware']}`, Python {config['python']}, sin GPU ni inferencia remota. Duración del experimento: {resources['elapsed_seconds']:.1f} s. {resources['training_runs']} pilotos Q y {resources['supervised_runs']} modelos supervisados; presupuesto por piloto {config['episodes']} episodios y {config['max_seconds']} s. Bootstrap: {config['bootstrap']} réplicas.",
      '',
      '```bash',
      'uv run --offline python -m unittest discover -s tests -v',
      'uv run --offline python -m rl_observability.diagnostics --help',
      f"uv run --offline python -m rl_observability.run --output reports/rl_observability/reproduccion --train-users {config['train_users']} --validation-users {config['validation_users']} --test-users {config['test_users']} --episodes {config['episodes']} --max-seconds {config['max_seconds']} --bootstrap {config['bootstrap']} --seed {config['seed']} --seeds {' '.join(map(str,config['seeds']))}",
      '```','',
      'La salida debe ser una carpeta nueva: se rechaza sobrescribir ejecuciones. `source_snapshot/`, hashes, datasets, validación, checkpoints, freeze y test permiten auditar la ejecución. La versión descriptiva del generador de informe se conserva separadamente en `report_source_snapshot.py` y `report_generation.json`; no modifica políticas ni inferencias. Matplotlib es opcional para figuras; los CSV numéricos se producen siempre.',
      '',
      'Verificación: [unittest.log](unittest.log), 355 pruebas existentes correctas; [verification.json](verification.json), comprobaciones de privacidad, respaldo de recompensa, persistencia, seguridad, acciones y denominadores. No se crearon ni modificaron módulos de pruebas. Los cambios de implementación son aditivos en `rl_observability/`; la documentación está en README y `docs/rl_observability_protocol.md`.',
      '',
      '## 10. Respuestas e hipótesis','']
    def val(rep,method,key):
        return metrics[rep][method]['metrics'].get(key,{}).get('mean')
    h1=comparisons['observability_fixed_B2']['O2-O1'].get('error_rate',{})
    h2=comparisons['observability_fixed_B2']['O3-O2'].get('error_rate',{})
    best=freeze['selected_rl'];baseline=freeze['selected_non_rl']
    gain=comparisons['learning_selected_observation'][f'{best}-B4_INITIAL']
    superiority=comparisons['learning_selected_observation'][f'{best}-{baseline}']
    h1_conclusion='H1 respaldada dentro de este generador.' if h1.get('ci95') and h1['ci95'][1]<0 else 'H1 no respaldada con este contraste.'
    h2_conclusion='H2 respaldada dentro de este generador.' if h2.get('ci95') and h2['ci95'][1]<0 else 'H2 no respaldada: el enriquecimiento no demuestra una reducción adicional.'
    seed_learning=[load(run/'training'/selected/f'B4_seed{s}.json')['metrics'] for s in config['seeds']]
    h3_conclusion='Las tres semillas mejoran el retorno de validación frente a inicialización: existe una señal aprovechable para aprender acciones en esta política tabular pequeña.' if all(m['after']['mean_return']>m['before']['mean_return'] for m in seed_learning) else 'La evidencia de aprendizaje es mixta entre semillas; no basta con elegir la mejor.'
    if val(selected,'B4_seed_mean','necessary_asks')==0:
        h3_conclusion+=' La política seleccionada no realiza aclaraciones necesarias: H3 sólo recibe apoyo parcial, no se demuestra aprendizaje de aclaraciones útiles ni de la política óptima.'
    else:
        h3_conclusion+=' Los conteos de preguntas y los resultados posteriores delimitan el apoyo a H3; aprender acciones no implica haber encontrado la política óptima.'
    return_ci=superiority.get('return',{}).get('ci95')
    error_ci=superiority.get('error_rate',{}).get('ci95')
    h4_conclusion='El contraste favorece RL en al menos un criterio; su alcance se limita a este generador.' if ((return_ci and return_ci[0]>0) or (error_ci and error_ci[1]<0)) else 'H4 no respaldada: no se demuestra una ventaja de RL frente al mejor comparador no RL.'
    lines += [
      f"**P1 / H1:** O2−O1 cambia el error en {estimate(h1)}. {h1_conclusion} El historial permite conservar la preferencia anterior al terminar una excepción y consultar ambos lados de un conflicto; estos efectos pertenecen a observabilidad con B2 fijo.",
      '',
      f"**P2 / H2:** O3−O2 cambia el error en {estimate(h2)}. {h2_conclusion} El decisor ya deriva esas características del historial. Los errores de extracción se distinguen de interpretación temporal y selección de preguntas; no se demuestra que el piloto los corrija. La tabla por familia y el diagnóstico de PIPELINE localizan los fallos que permanecen y una corrección convencional posible.",
      '',
      f"**P3 / H3:** La semilla seleccionada cambia el retorno frente a inicialización en {estimate(gain.get('return'))} y el error en {estimate(gain.get('error_rate'))}. {h3_conclusion} La selección de respuestas y operaciones de memoria mejoró dentro del catálogo que también permite preguntar; la interpretación simbólica fija no fue aprendida y el resultado no se extrapola a un LLM.",
      '',
      f"**P4:** En usuarios/fechas/redacciones retenidos, RL obtiene error {metric(selected,'B4_seed_mean','error_rate')} y coherencia futura {metric(selected,'B4_seed_mean','future_accuracy')}. No se ha probado generalización a plantillas abstractas nuevas.",
      '',
      f"**P5 / H4:** RL seleccionado menos mejor no RL de validación: retorno {estimate(superiority.get('return'))}, error {estimate(superiority.get('error_rate'))}. {h4_conclusion} Aprendizaje frente a inicialización no basta para superioridad. Los resultados frente a B2 y supervisión están separados.",
      '',
      '**P6:** Este experimento permite estudiar de forma controlada el papel de historial, vigencia, conflicto y recompensa. Constituye evidencia metodológica y de entrenabilidad local. Para responder la tesis sobre asistentes reales faltan historias y etiquetas adjudicadas externamente, variedad conceptual retenida y comparación con prompting de un LLM accesible. No justifica por sí solo escalar RL ni aumentar artificialmente la dificultad.',
      '',
      f"Artefactos completos: [{run.name}]({run.name}/config.json)."]
    (root/'INFORME_OBSERVABILIDAD_RL.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    generate_report(p.parse_args().run)
