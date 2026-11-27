"""Spanish SPEC-RL-005 reports and auditable CSVs, derived from saved results."""
import csv
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil


def load(path):return json.loads(Path(path).read_text())

def estimate(value):
    if not value or value.get('mean') is None:return 'no evaluable'
    lo,hi=value['ci95']
    return f"{value['mean']:.4f} [{lo:.4f}, {hi:.4f}]"

def write_csv(path,rows,fields=None):
    fields=fields or list(rows[0]) if rows else (fields or ['no_cases'])
    with Path(path).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def generate_report(run):
    run=Path(run);config=load(run/'config.json');metrics=load(run/'metrics.json')
    freeze=load(run/'freeze.json');resources=load(run/'resources.json')
    obs=metrics['observations'];test=metrics['test'];comparisons=metrics['comparisons']
    diagnostics=load(run/'error_diagnostics.json');selected=metrics['selected_observation']
    llm=load(run/'llm_status.json')
    failure_path=run/'llm/technical_failures.json'
    failures=load(failure_path) if failure_path.exists() else []
    private_test_path=run/'evaluator_private/test.json'
    family_by_id={s['user_id']:s['family'] for s in load(private_test_path)} if private_test_path.exists() else {}
    failure_counts=Counter((value['condition'],value['technical_error']) for value in failures)
    failure_table=['| Condición M4 | Fallo de ejecución/formato | Episodios |','|---|---|---:|']
    for (condition,reason),count in sorted(failure_counts.items()):
        failure_table.append(f'| {condition} | {reason} | {count} |')
    failure_family_counts=Counter((value['condition'],family_by_id.get(value['user_id'],'UNAVAILABLE'),value['technical_error']) for value in failures)
    failure_family_rows=[{'condition':condition,'family':family,'reason':reason,'failed_episodes':count}
        for (condition,family,reason),count in sorted(failure_family_counts.items())]
    write_csv(run/'llm_failure_coverage.csv',failure_family_rows,['condition','family','reason','failed_episodes'])
    failure_family_table=['| Condición M4 | Familia excluida | Fallo | Episodios |','|---|---|---|---:|']
    for value in failure_family_rows:
        failure_family_table.append(f"| {value['condition']} | {value['family']} | {value['reason']} | {value['failed_episodes']} |")
    support_path=run/'symbolic_support_audit.json'
    support=load(support_path) if support_path.exists() else None
    support_note=(f"Auditoría descriptiva posterior a la congelación: {support['test_observations_with_exact_train_vector']}/{support['test_rule_observations']} observaciones de reglas en test coinciden exactamente con vectores vistos en TRAIN; {support['test_unique_vectors_seen_in_train']}/{support['test_unique_vectors']} vectores distintos. Ver symbolic_support_audit.json. No se seleccionaron ni cambiaron modelos con este análisis."
        if support else 'No se calculó cobertura exacta de los vectores de TRAIN; la separación superficial no garantiza separación simbólica.')
    # Additional descriptions use saved trajectories only; no policy is rerun.
    from .error_diagnostics import diagnose_rows,compare_paired
    from .observations import FEATURE_NAMES,FEATURE_VERSION
    (run/'feature_manifest.json').write_text(json.dumps({'version':FEATURE_VERSION,'names':FEATURE_NAMES,
        'provenance':'fixed shared transformation of public observation only; no reward, gold, future reply or family input'},indent=2)+'\n')
    def saved_rows(path):return [json.loads(line) for line in path.read_text().splitlines()]
    llm_diagnostics={}
    llm_examples=[]
    for condition in ('ORACLE','PIPELINE'):
        path=run/f'llm/{condition}.jsonl'
        if path.exists() and path.stat().st_size:
            llm_diagnostics[condition]=diagnose_rows(saved_rows(path))
            diagnostics[condition]['M4']=llm_diagnostics[condition]
            for row in saved_rows(path):
                if row['policy_errors'] and len(llm_examples)<3:
                    llm_examples.append({'condition':condition,'user_id':row['user_id'],'family':row['family'],
                        'return':row['return'],'steps':row['steps']})
    (run/'llm_error_diagnostics.json').write_text(json.dumps(llm_diagnostics,ensure_ascii=False,indent=2)+'\n')
    (run/'llm_error_examples.json').write_text(json.dumps(llm_examples,ensure_ascii=False,indent=2)+'\n')
    ablation_diagnostics={rep:diagnose_rows(saved_rows(run/f'test/observability/{rep}.jsonl')) for rep in ('O1','O2','O3')}
    (run/'observation_error_diagnostics.json').write_text(json.dumps(ablation_diagnostics,ensure_ascii=False,indent=2)+'\n')
    extraction_pairs={method:compare_paired(saved_rows(run/f'test/ORACLE/{method}.jsonl'),saved_rows(run/f'test/PIPELINE/{method}.jsonl'))
        for method in ('M3',*(f'{m}_seed{s}' for m in ('M0','M1','M2') for s in config['seeds']))}
    if all((run/f'llm/{condition}.jsonl').exists() for condition in ('ORACLE','PIPELINE')):
        extraction_pairs['M4']=compare_paired(saved_rows(run/'llm/ORACLE.jsonl'),saved_rows(run/'llm/PIPELINE.jsonl'))
        extraction_pairs['M4']['limitation']='Only common completed users; independent LLM calls and technical exclusions can affect differences. Not a unique attribution to extraction.'
    (run/'paired_extraction_diagnostics.json').write_text(json.dumps(extraction_pairs,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(Path(__file__),run/'report_source_snapshot.py')
    (run/'report_generation.json').write_text(json.dumps({
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'purpose':'descriptive reporting only; original inference snapshots and frozen policy files retained',
        'protocol_sha256':hashlib.sha256((run/'PROTOCOL.md').read_bytes()).hexdigest()},indent=2)+'\n')
    def m(method,key='return',condition='ORACLE'):
        return test[condition][method]['metrics'][key]
    observations=[]
    for rep in ('O1','O2','O3'):
        v=obs[rep];raw=saved_rows(run/f'test/observability/{rep}.jsonl')
        observations.append({'observation':rep,**v['counts'],
            **{('mean_'+key if key in v['counts'] else key):value['mean'] for key,value in v['metrics'].items()},
            'temporal_decision_errors':sum(r['temporal_decision_errors'] for r in raw),
            'temporal_field_discrepancies':sum(r['temporal_interpretation_errors'] for r in raw),
            'conflicts_detected':sum(r['conflicts_detected'] for r in raw),
            'potential_conflicts':sum(r['potential_conflicts'] for r in raw),
            'false_conflict_sessions':sum(s['conflict_detected'] and not s['potential_conflict'] for r in raw for s in r['sessions']),
            'retrieved_relevant_count':sum(r['retrieved_relevant_count'] for r in raw),
            'relevant_memory_count':sum(r['relevant_memory_count'] for r in raw)})
    write_csv(run/'observation_ablation.csv',observations)
    baseline_rows=[]
    for condition,methods in test.items():
        for method,value in methods.items():
            baseline_rows.append({'condition':condition,'method':method,**value['counts'],
                **{('mean_'+k if k in value['counts'] else k):v['mean'] for k,v in value['metrics'].items()}})
    write_csv(run/'baseline_comparison.csv',baseline_rows)
    errors=[]
    def case_details(case):
        values={key:json.dumps(case.get(key),ensure_ascii=False) for key in
            ('original_input','available_memories','retrieved_memories','observation','evidence')}
        return dict(values,query=case.get('observation',{}).get('query'),final_correct=case.get('final_correct'))
    for condition,methods in diagnostics.items():
        for method,value in methods.items():
            for case in value['cases']:
                errors.append({'condition':condition,'method':method,'user_id':case['user_id'],
                    'family':case['family'],'session_index':case['session_index'],
                    'causes':'|'.join(case['causes']),'multiplicity':case['multiplicity'],
                    'attribution':case['attribution'],'reward':case['reward'],
                    'proposed_action':json.dumps(case['proposed_action'],ensure_ascii=False),
                    'executed_action':json.dumps(case['executed_action'],ensure_ascii=False),**case_details(case)})
    for rep,value in ablation_diagnostics.items():
        for case in value['cases']:
            errors.append({'condition':'ORACLE','method':f'M3_{rep}_ablation','user_id':case['user_id'],
                'family':case['family'],'session_index':case['session_index'],
                'causes':'|'.join(case['causes']),'multiplicity':case['multiplicity'],
                'attribution':case['attribution'],'reward':case['reward'],
                'proposed_action':json.dumps(case['proposed_action'],ensure_ascii=False),
                'executed_action':json.dumps(case['executed_action'],ensure_ascii=False),**case_details(case)})
    write_csv(run/'error_analysis.csv',errors,['condition','method','user_id','family','session_index',
        'causes','multiplicity','attribution','reward','proposed_action','executed_action',
        'original_input','available_memories','retrieved_memories','observation','evidence','query','final_correct'])
    curves=[]
    for point in load(run/'training_curves.json'):
        val=point['validation'];curves.append({'method':point['method'],'seed':point['seed'],
            'lr':point['lr'],'optimizer_steps':point['optimizer_steps'],
            'episodes':point.get('episodes'),'epoch':point.get('epoch'),
            'training_mean_return':point.get('training_mean_return'),
            'training_mean_loss':point.get('training_mean_loss'),
            'validation_return':val.get('mean_return'),'validation_error_rate':val.get('decision_error_rate'),
            'necessary_asks':val.get('necessary_asks'),'unnecessary_asks':val.get('unnecessary_asks'),
            'incorrect_updates':val.get('incorrect_updates'),'future_accuracy':val.get('future_accuracy')})
    write_csv(run/'training_curves.csv',curves)
    budgets=[]
    for path in sorted((run/'training').glob('*.json')):
        value=load(path)['metrics'];last=value.get('final_checkpoint_validation',value['after'])
        budgets.append({'configuration':path.stem,'optimizer_steps':value['optimizer_steps'],
            'elapsed_seconds':value['elapsed_seconds'],'action_forward_passes':value.get('action_forward_passes'),
            'sampled_trajectories':value.get('sampled_trajectories'),
            'nonzero_gradient_updates':value.get('nonzero_gradient_updates'),
            'selected_step':value['selected_step'],'last_parameter_l2_delta':value['last_parameter_l2_delta'],
            'selected_parameter_l2_delta':value['selected_parameter_l2_delta'],
            'selected_validation_return':value['after']['mean_return'],'last_validation_return':last['mean_return'],
            'last_validation_errors':last['decision_errors']})
    write_csv(run/'training_budget.csv',budgets)
    neural_seconds=sum(row['elapsed_seconds'] for row in budgets)
    _plot(run)
    h1=comparisons['observability']['O2-O1'];h2=comparisons['observability']['O3-O2']
    improvement=comparisons['ORACLE']['M1-M0'];supervised=comparisons['ORACLE']['M1-M2'];rules=comparisons['ORACLE']['M1-M3']
    gain=improvement['return']['ci95'][0]>0
    beats_all=all(comparisons['ORACLE'][f'M1-{other}']['return']['ci95'][0]>0 and
                  comparisons['ORACLE'][f'M1-{other}']['return']['mean']>=.10 for other in ('M0','M2','M3'))
    if beats_all:classification='RESULTADO 1 — RL APRENDE Y SUPERA BASELINES'
    elif gain:classification='RESULTADO 2 — RL APRENDE, PERO NO SUPERA BASELINES'
    elif h1['error_rate']['ci95'][1]<0:classification='RESULTADO 3 — LA OBSERVABILIDAD EXPLICA PRINCIPALMENTE LA MEJORA'
    else:classification='RESULTADO 4 — NO SE OBSERVA APRENDIZAJE RL'
    verdict='EVIDENCIA INSUFICIENTE'
    training=[]
    for seed in config['seeds']:
        m0=load(run/'training'/f'M0_seed{seed}.json')['metrics']
        candidates={}
        for method in ('M1','M2'):
            lr=metrics['selected_learning_rates'][method]
            candidates[method]=load(run/'training'/f'{method}_seed{seed}_lr{lr}.json')['metrics']
        training.append((seed,m0,candidates['M1'],candidates['M2']))
    retained_m0=all(m1['selected_step']==0 for _,_,m1,_ in training)
    selection_note=('En todas las semillas, el M1 evaluado es M0 retenido por selección de validación (paso 0). Los checkpoints con pesos actualizados por RL se guardan como last.pt y su última validación se informa por separado; no se presentan como políticas desplegadas en el test principal.'
        if retained_m0 else 'Los checkpoints M1 se seleccionan por validación; la tabla distingue el último entrenado y el seleccionado.')
    weight_table=['| Semilla | Parámetros | Pasos RL | ΔL2 final RL | ΔL2 seleccionado RL | Paso seleccionado | M1/M2 mismo M0 |',
                  '|---|---:|---:|---:|---:|---:|---|']
    for seed,m0,m1,m2 in training:
        weight_table.append(f"| {seed} | {m0['parameter_count']} | {m1['optimizer_steps']} | {m1['last_parameter_l2_delta']:.6f} | {m1['selected_parameter_l2_delta']:.6f} | {m1['selected_step']} | {m1['initial_hash']==m2['initial_hash']==m0['selected_hash']} |")
    initial_table=['| Semilla | Errores red aleatoria / decisiones VALIDATION | Error tras M0 | Retorno aleatorio → M0 → M1 seleccionado | Último RL: errores / retorno |',
                   '|---|---:|---:|---:|---:|']
    for seed,m0,m1,m2 in training:
        last=m1.get('final_checkpoint_validation',m1['after'])
        initial_table.append(f"| {seed} | {m0['before']['decision_errors']}/{m0['before']['decisions']} | {m0['after']['decision_errors']} | {m0['before']['mean_return']:.3f} → {m0['after']['mean_return']:.3f} → {m1['after']['mean_return']:.3f} | {last['decision_errors']} / {last['mean_return']:.3f} |")
    pipeline_summary=extraction_pairs['M3']
    reward_audit=load(run/'reward_audit.json')
    reward_table=['| Control previo, 10 usuarios de validación | Retorno | Tasa de error | Preguntas innecesarias |',
                  '|---|---:|---:|---:|']
    for method,value in reward_audit['results'].items():
        reward_table.append(f"| {method} | {value['return']:.3f} | {value['error_rate']:.3f} | {value['unnecessary_questions']} |")
    extraction_table=['| Método PIPELINE | Eventos discrepantes / comunicados | % | Discrepancia ámbito | Fuente | Intervalo cambia aplicabilidad |',
                      '|---|---:|---:|---:|---:|---:|']
    for method in ('M3',*(f'M1_seed{s}' for s in config['seeds'])):
        rows=saved_rows(run/f'test/PIPELINE/{method}.jsonl')
        total=sum(r['extraction_total'] for r in rows);different=sum(r['extraction_errors'] for r in rows)
        extraction_table.append(f"| {method} | {different}/{total} | {100*different/max(total,1):.2f} | {sum(r['extraction_scope_errors'] for r in rows)} | {sum(r['extraction_source_errors'] for r in rows)} | {sum(r['extraction_interval_applicability_errors'] for r in rows)} |")
    obs_table=['| Observación | Aciertos | Errores | Tasa de error, IC95% | Retorno, IC95% | S3 |',
               '|---|---:|---:|---:|---:|---:|']
    for rep in ('O1','O2','O3'):
        v=obs[rep];obs_table.append(f"| {rep} | {v['counts']['correct']:.0f} | {v['counts']['incorrect']:.0f} | {estimate(v['metrics']['error_rate'])} | {estimate(v['metrics']['return'])} | {estimate(v['metrics']['future_accuracy'])} |")
    obs_details=['| Observación | Error temporal de decisión | Conflictos detectados / potenciales | Falsos conflictos (sesiones) | Memorias pertinentes recuperadas / disponibles | Preguntas pertinentes / innecesarias |',
                 '|---|---:|---:|---:|---:|---:|']
    for v in observations:
        obs_details.append(f"| {v['observation']} | {v['temporal_decision_errors']} | {v['conflicts_detected']}/{v['potential_conflicts']} | {v['false_conflict_sessions']} | {v['retrieved_relevant_count']}/{v['relevant_memory_count']} | {v['necessary_asks']}/{v['unnecessary_asks']} |")
    comparison_table=['| Método | Condición | Usuarios completos | Fallos / decisiones | Error, IC95% | Retorno por episodio, IC95% | Recompensa por sesión | S3 | Preguntas pertinentes | Innecesarias |',
                     '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    intervention_table=['| Método | Condición | Actualizaciones injustificadas | Intervenciones de seguridad/presupuesto |',
                        '|---|---|---:|---:|']
    for condition in ('ORACLE','PIPELINE'):
        for method in ('M0','M1','M2','M3',*(['M4'] if 'M4' in test[condition] else [])):
            v=test[condition][method];mtr=v['metrics']
            comparison_table.append(f"| {method} | {condition} | {v['counts']['users']} | {v['counts']['incorrect']:.0f}/{v['counts']['sessions']} | {estimate(mtr['error_rate'])} | {estimate(mtr['return'])} | {estimate(mtr['reward_per_session'])} | {estimate(mtr['future_accuracy'])} | {v['counts']['necessary_asks']:.2f} | {v['counts']['unnecessary_asks']:.2f} |")
            intervention_table.append(f"| {method} | {condition} | {v['counts']['incorrect_updates']:.2f} | {v['counts']['safety_interventions']:.2f} |")
    seed_table=['| Método / semilla | Retorno ORACLE | Error ORACLE | Retorno PIPELINE | Error PIPELINE |',
                '|---|---:|---:|---:|---:|']
    for method in ('M0','M1','M2'):
        for seed in config['seeds']:
            label=f'{method}_seed{seed}'
            seed_table.append(f"| {label} | {estimate(m(label))} | {estimate(m(label,'error_rate'))} | {estimate(m(label,'return','PIPELINE'))} | {estimate(m(label,'error_rate','PIPELINE'))} |")
    family_table=['| Familia | M0 retorno / error | M1 retorno / error | M2 retorno / error | M3 retorno / error |',
                  '|---|---:|---:|---:|---:|']
    for family in sorted(test['ORACLE']['M3']['by_family']):
        values=[test['ORACLE'][method]['by_family'][family] for method in ('M0','M1','M2','M3')]
        family_table.append('| '+family+' | '+' | '.join(f"{v['return']:.3f} / {v['error_rate']:.3f}" for v in values)+' |')
    llm_family_table=['| Condición M4 | Familia | Usuarios completados | Tasa de error | Retorno |',
                      '|---|---|---:|---:|---:|']
    for condition in ('ORACLE','PIPELINE'):
        if 'M4' in test[condition]:
            for family,value in sorted(test[condition]['M4']['by_family'].items()):
                llm_family_table.append(f"| {condition} | {family} | {value['users']} | {value['error_rate']:.3f} | {value['return']:.3f} |")
    differences=['| Diferencia (primero menos segundo) | Error | Retorno | S3 |','|---|---:|---:|---:|']
    for label,value in comparisons['ORACLE'].items():
        differences.append(f"| {label} | {estimate(value['error_rate'])} | {estimate(value['return'])} | {estimate(value['future_accuracy'])} |")
    stage_table=['| Condición / método / semilla | Fallos | Extracción probable | Tiempo probable | Contexto probable | Conflicto probable | Memoria probable | Causas múltiples |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition,methods in diagnostics.items():
        for method,value in methods.items():
            s=value['summary'];cats=s['categories']
            def cell(key):return f"{cats[key]['count']} ({cats[key]['percent']:.1f}%)"
            stage_table.append(f"| {condition}/{method} | {s['failed_decisions']} | {cell('linguistic_extraction')} | {cell('date_or_validity')} | {cell('context_interpretation')} | {cell('conflict_detection')} | {cell('memory_update')} | {s['multiplicity'].get('multiple',0)} |")
    # Trace examples of the same user with O1 and O2; labels remain evaluator-only.
    def read_rows(path):return [json.loads(line) for line in path.read_text().splitlines()]
    a={r['user_id']:r for r in read_rows(run/'test/observability/O1.jsonl')}
    b={r['user_id']:r for r in read_rows(run/'test/observability/O2.jsonl')}
    examples=[]
    for uid in a:
        if a[uid]['error_rate']>b[uid]['error_rate']:
            examples.append({'user_id':uid,'family':a[uid]['family'],
                'O1_steps':a[uid]['steps'],'O2_steps':b[uid]['steps']})
        if len(examples)==3:break
    (run/'observation_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    questions=[
      '# RESPUESTAS EXPERIMENTALES PARA VICENT','',
      '## Pregunta 1. ¿Qué información relevante y adicional necesita el agente para tomar una buena decisión?','',
      f'Respuesta directa: conservar historial temporal relevante. Con el mismo decisor M3, O2−O1 cambia la tasa de error en {estimate(h1["error_rate"])} y O3−O2 en {estimate(h2["error_rate"])}. La observación principal fue {selected}, elegida sólo en validación.',
      '',*obs_table,'',*obs_details,'',
      'O1 conserva únicamente el hecho reciente y restricciones; le faltan antecedentes que permiten reactivar una preferencia al terminar una excepción o detectar ambos lados del conflicto. O2 aporta contenido, fechas observada/registrada, vigencia, alcance, fuente, evidencia actual e historial de preguntas. O3 añade tipos, excepciones, relaciones y conflictos con procedencia. No contiene acción correcta ni estado privado.',
      '',
      'Evidencia trazable: observation_ablation.csv, test/observability/*.jsonl y observation_examples.json. O1 ya incluye los campos temporales del hecho conservado: O2−O1 no identifica por separado el beneficio de cada fecha. O3 es derivable de O2 mediante el intérprete compartido; un empate no demuestra que el enriquecimiento sea inútil para otros decisores. Recomendación: mantener historial y evaluar cada enriquecimiento sin atribuir su efecto a RL.',
      '',
      '## Pregunta 2. ¿Por qué falla el agente: información insuficiente, interpretación lingüística o política de decisión?','',
      'Respuesta directa: depende de la etapa. Los resultados ORACLE/PIPELINE y el diagnóstico separan observación insuficiente, discrepancias de extracción y selección de acciones. Las causas son probables y sus porcentajes se calculan sobre fallos terminales; pueden solaparse.',
      '',*stage_table,'',
      'error_diagnostics.json conserva las diez categorías completas, incluyendo recuperación, pregunta, respuesta, reparación e indeterminada; error_analysis.csv conserva cada caso. En ORACLE los hechos describen únicamente lo comunicado. PIPELINE usa el extractor existente y registra polaridad, tipo, intervalo, ámbito y fuente. Una frontera a medianoche frente a mediodía puede ser discrepancia sin cambiar la decisión.',
      '',
      'Antes de entrenar se ejecutó diagnostics_before_training.json y reward_audit.json. Se reutiliza además el diagnóstico previo de 120 fallos archivados, que identificó problemas de extracción; no se cuentan como nuevos errores de este piloto. Las fechas de consulta permiten derivar el día sin información oculta: el adaptador aplica esa corrección convencional uniformemente. Un error Pipeline no se declara automáticamente resoluble con RL. Recomendación: intervenir sobre cada etapa y confirmar la atribución en pares ORACLE/PIPELINE.',
      '',f"Intervención emparejada actual M3: {pipeline_summary.get('terminal_pairs',0)} decisiones; fallos sólo PIPELINE {pipeline_summary.get('pipeline_only_errors',0)}, sólo ORACLE {pipeline_summary.get('oracle_only_errors',0)}, en ambas {pipeline_summary.get('both_errors',0)}. La ablación O1 aporta {ablation_diagnostics['O1']['summary']['failed_decisions']} fallos diagnosticados en observation_error_diagnostics.json; no se mezclan con fallos de O2.",
      '',*extraction_table,'',
      'Las discrepancias anteriores se cuentan por evento comunicado, no por decisión. Si no cambian aplicabilidad o respuesta, no constituyen fallos de política. llm_error_diagnostics.json añade los fallos de M4 cuando existe cobertura completada.',
      '',*failure_table,'',*failure_family_table,'',
      *llm_family_table,'',
      '',
      '## Pregunta 3. ¿Cuál es el experimento mínimo suficiente y cómo se compara con métodos existentes?','',
      f'Respuesta directa: cinco familias cubren cambio, temporalidad, relevancia y posibilidad de aclaración; tres sesiones separan incorporación inicial, decisión actual y consecuencia futura. Se utilizaron {config["train_users"]}/{config["validation_users"]}/{config["test_users"]} usuarios TRAIN/VALIDATION/TEST, tres semillas, SQLite real y {selected}.',
      '',
      'La red tiene entrada de 51 características públicas, dos capas de 64 unidades Tanh y 32 acciones compuestas: 9.568 parámetros entrenables. Se eligió REINFORCE porque el nuevo objetivo exige actualizar pesos neuronales y la SPEC permite este gradiente de política básico; Q-learning tabular no actualiza esos pesos. El cambio no se justifica por llamar POMDP al entorno. No es entrenamiento de un LLM.',
      '',
      'M0 aprende mediante entropía cruzada de demostraciones de reglas sólo de TRAIN. M1 y M2 parten del mismo checkpoint M0; M1 recibe recompensas y M2 supervisión adicional. Ambos tienen igual número de pasos de optimizador y usuarios por lote; se reportan tiempo y pases de acciones reales porque su coste no es idéntico. M3 son reglas fuertes, sin degradarlas.',
      '',selection_note,'',
      '',*comparison_table,'',
      'No hace falta realismo completo para aislar estos contrastes. La pregunta oficial de tesis sobre aclaración en asistentes se mantiene; la red externa sólo evalúa un componente simbólico de decisión. La generalización a conversación libre queda fuera. Recomendación: interpretar este piloto como prueba metodológica y de entrenabilidad, no como resolución completa de la tesis.',
      '',
      '## Pregunta 4. ¿Qué debe completarse antes de considerar entrenamiento de un LLM?','',
      f'Dictamen: **{verdict}** para escalar a RL en un LLM. Actualizar parámetros y mejorar el rendimiento son criterios distintos.',
      '',*weight_table,'',
      *initial_table,'',
      '| Requisito | Evidencia y estado |','|---|---|',
      f'| A. Observabilidad | Historial seleccionado {selected}; O2−O1 error {estimate(h1["error_rate"])}. No se entregan gold/futuro. |',
      f'| B. Extracción | M3 PIPELINE error {estimate(m("M3","error_rate","PIPELINE"))}; esta colección controlada no valida lenguaje natural general. |',
      f'| C. Entrenabilidad | Hay gradientes y cambios reales de pesos. M1−M0 retorno {estimate(improvement["return"])}; el cambio de pesos por sí solo no demuestra aprendizaje útil. |',
      '| D. Recompensa | Coeficientes congelados; control previo de preservar siempre, preguntar siempre y responder con signo fijo. No se afirma optimalidad global. |',
      '| E. Generalización | Contextos y plantillas superficiales retenidos; se comparten cinco mecanismos y parte del soporte simbólico. Falta generalización conceptual y humana. |',
      f'| F. Comparación | M1−M2 retorno {estimate(supervised["return"])}; M1−M3 retorno {estimate(rules["return"])}. |',
      f'| G. Recursos | CPU/PyTorch disponibles; M4 {llm["status"]}. Una API de inferencia no acredita acceso a pesos/GPU de entrenamiento de LLM. |',
      '| H. Validez externa | No hay adjudicación humana independiente. Los escenarios y etiquetas proceden del generador. |',
      '',support_note,'',
      '',*reward_table,'',
      '',
      'Recomendación: completar adjudicación externa, revisar errores residuales convencionales y verificar beneficios reproducibles frente a M0/M2/M3 antes de planificar entrenamiento costoso de LLM. Si M0 ya alcanza el techo de estas reglas, no debe empeorarse ni añadir dificultad para favorecer RL; corresponde reconocer el techo del experimento.',
      '',
      '## Pregunta 5. ¿Conviene continuar con RL, reformular la hipótesis o centrar la tesis en observabilidad y gestión de memoria?','',
      f'Respuesta directa según este piloto: **{classification}**. M1−M0 retorno {estimate(improvement["return"])}; M1−M2 {estimate(supervised["return"])}; M1−M3 {estimate(rules["return"])}.',
      '',
      ('Existe una mejora incremental de RL; estudiar su magnitud, estabilidad y coste antes de escalar.' if gain else
       'No se demuestra mejora incremental de RL frente al checkpoint supervisado inicial. Los pesos finales sí se actualizaron, pero las políticas seleccionadas pueden conservar M0 cuando validación no mejora. No se presenta ese caso como RL que aprende y supera baselines.'),
      '',
      f'H1: {"sustentada dentro del generador" if h1["error_rate"]["ci95"][1]<0 else "no demostrada"} por el contraste de historial. H2: {"sustentada" if h2["error_rate"]["ci95"][1]<0 else "sin beneficio adicional demostrado del enriquecimiento para el decisor fijo"}. H3: {"hay mejora incremental observada" if gain else "no demostrada por mejora de decisiones en este piloto; el techo de M0 limita el contraste"}. H4: {"sustentada frente a los comparadores principales bajo el margen fijado" if beats_all else "no demostrada"}.',
      '',
      'La observabilidad y gestión de memoria son la prioridad si explican la reducción principal. Un resultado negativo del piloto no demuestra inutilidad general de RL. El siguiente experimento mínimo debe usar historias adjudicadas y retenidas, con errores residuales documentados, manteniendo recompensa/observación fijas y comparando la misma red inicial con RL y supervisión adicional. Sólo si las acciones tienen utilidad futura distinta y las alternativas fuertes dejan margen medido procede continuar la hipótesis de ventaja RL. No se cambia silenciosamente la pregunta oficial de tesis.'
    ]
    (run/'VICENT_RESPUESTAS.md').write_text('\n'.join(questions)+'\n')
    report=[
      '# INFORME PILOTO RL — SPEC-RL-005','',
      '## 1. Resumen ejecutivo','',
      f'**{classification}**. Se entrenó realmente una política neuronal externa con REINFORCE. No se entrenaron pesos de GLM ni de otro LLM. M1−M0 retorno {estimate(improvement["return"])}; M1−M2 {estimate(supervised["return"])}; M1−reglas {estimate(rules["return"])}.',
      '',selection_note,'',
      '',
      f'Pregunta A: {"se observa mejora incremental frente a M0" if gain else "no se observa mejora incremental del decisor determinista después de RL frente a M0"}. Pregunta B: el contraste de observación O2−O1 reduce el error en {abs(h1["error_rate"]["mean"])*100:.1f} puntos porcentuales con la política fija; O3−O2 cambia el error en {h2["error_rate"]["mean"]*100:.1f} puntos. La mejora de red aleatoria a M0 procede de supervisión.',
      '',
      '## 2. Resultados previos y limitaciones','',
      'SPEC-RL-004 entrenó Q-learning tabular y encontró ventaja del historial, sin superioridad RL. El piloto GLM posterior fue sólo inferencia y mostró errores de persistencia. Se conservan ambos resultados; el presente experimento es nuevo. historical_hashes.json verifica preservación de resultados anteriores.',
      '',
      'La auditoría existente (reports/rl_observability/audit.md y diagnosis/summary.json) examinó 120 fallos archivados de 120 usuarios, doce por cada una de diez familias históricas, semilla 4004. Todos presentaron discrepancias de extracción y sus sesiones ORACLE emparejadas acertaron. Ejemplo: «sin importar la compañía ni el lugar» activaba negación y «Disfruto comer» no se reconocía como preferencia positiva. El texto bruto estaba almacenado en SQLite, pero no llegaba a la política histórica. Ese diagnóstico no demuestra un fallo independiente de política.',
      '',
      'El sistema longitudinal histórico entregaba required_decision y reconciliaciones de reglas; su contrato podía sustituir acción, pregunta y operación de memoria. En aquella ejecución PIPELINE no se registraron intervenciones de contrato, por lo que capacidad de reparación e intervención observada se distinguen. Este piloto no reutiliza ese campo ni esa reparación general: reutiliza SQLite, extracción, restricciones explícitas y transiciones mínimas. Ver snapshots y pruebas de ausencia de gold.',
      '',
      '## 3. Información disponible','',
      'O1: una memoria reciente; O2: historia temporal pertinente; O3: interpretaciones trazables de O2. Restricciones explícitas siempre visibles. No se entregan required_decision, familia, preferencias latentes, respuestas futuras ni recompensas futuras. El significado temporal y los conflictos se calculan con un preprocesador simbólico fijo compartido: no los aprende la red desde lenguaje natural.',
      '',
      '| Información | Procedencia | Qué recibe cada decisor |','|---|---|---|',
      '| Texto, fechas, vigencia, fuente, contexto e historial | Evidencia comunicada y memoria SQLite persistente | Observación pública O1/O2/O3 según ablación. O1 descarta historia. |',
      '| Tipos, aplicabilidad, desacuerdos y aclaraciones | Intérprete fijo de evidencia pública | La red usa 51 características numéricas; nombres y procedencia en feature_manifest.json. O2/O3 permiten los mismos cálculos. |',
      '| IDs, fechas absolutas, identidades de contexto y texto bruto | Disponibles en la observación pública | La red no los recibe como valores directos; se conservan para auditoría. El LLM procesa la observación JSON pública. |',
      '| Gold, familia, estado latente y respuestas futuras | Evaluador y usuario simulado privados | Nunca son entradas de política; sólo transiciones/recompensas/diagnóstico. |',
      '',
      'M0/M1/M2 comparten exactamente vector, arquitectura y catálogo. M3 comparte la evidencia y el intérprete fijo, pero implementa decisiones manuales. M4 recibe la misma información pública en JSON, con mayor capacidad lingüística y coste de inferencia; no es igualdad de representación interna ni de capacidad computacional.',
      '',
      '## 4. Ablación O1/O2/O3','',*obs_table,'',
      *obs_details,'',
      'Los denominadores de conflictos y memorias pueden variar porque aclarar y persistir cambia las sesiones posteriores. Son medidas de trayectorias completas, no consultas aisladas idénticas después de intervenciones distintas. Las discrepancias temporales de campos se distinguen de una decisión temporal incorrecta.',
      f'O2−O1 error {estimate(h1["error_rate"])}; O3−O2 {estimate(h2["error_rate"])}. Decisor M3 fijo. Se seleccionó {selected} mediante validación antes de entrenar y observar test.',
      '',
      '## 5. Diagnóstico de errores','',*stage_table,'',
      'Las categorías son explicaciones probables, no una identificación causal completa. Se distinguen fallos terminales de discrepancias de campos. La sección de Vicent y error_diagnostics.json muestran causas múltiples y denominadores. Los diagnósticos se hacen sobre trayectorias individuales de cada semilla, no sobre una trayectoria inventada de la media.',
      '',
      '## 6. Arquitectura de política pequeña','',
      'Red 51→64(Tanh)→64(Tanh)→32, 9.568 parámetros, CPU. Acciones: ANSWER(LIKE/DISLIKE), CLARIFY(preference/meal/day/source), CONDITIONAL_ANSWER, PRESERVE_CONFLICT, combinadas con ADD_MEMORY, UPDATE_WITH_HISTORY, NOOP o PRESERVE_CONFLICT. Sólo se enmascara la pregunta cuando el presupuesto público es cero. La conversación NOOP no es una propuesta válida nueva.',
      '',
      '## 7. Algoritmo y entrenamiento real','',
      'M0: Adam y entropía cruzada sobre demostraciones de reglas de TRAIN. M1: REINFORCE con retorno Monte Carlo, γ=1 y baseline escalar calculado con lotes anteriores. M2: Adam y entropía cruzada adicional desde el mismo M0. No hay crítico, PPO, GRPO, LoRA ni entrenamiento del extractor. La elección neuronal está autorizada por la SPEC y por la instalación de PyTorch; no se cambia algoritmo sólo por observabilidad parcial.',
      '',*weight_table,'',
      *initial_table,'',
      'Se guardan checkpoint inicial, seleccionado y último entrenado. Si el seleccionado es el inicial, ΔL2 seleccionado puede ser cero aunque el último entrenado cambió: validación no justificó conservar esa actualización. Los hashes de M0 son idénticos al inicio de M1/M2 por semilla. Entrenar realmente no equivale a demostrar mejora.',
      '',
      '## 8. Recompensa y transiciones','',
      f'Coeficientes congelados: `{json.dumps(freeze["reward"],sort_keys=True)}`. Se suman costes y penalizaciones además del resultado de decisión. La recompensa por sesión agrega los pasos de S2/S3; el retorno por episodio suma ambas sesiones. S1 es incorporación determinista. Preguntar sin resolver cuesta; preservar una incertidumbre observable es correcto con +0,7. Eso no equivale a resolverla. SQLite se cierra y recarga para S3.',
      '',
      'reward_audit.json compara controles constantes antes de entrenar. Las protecciones sólo bloquean restricciones explícitas y presupuesto; no reemplazan preguntas erróneas ni fuerzan la respuesta esperada. Se conservan propuesta y ejecución.',
      '',*reward_table,'',
      '',
      '## 9. Curvas de entrenamiento y validación','',
      'training_curves.csv y training_curves.json incluyen pérdidas supervisadas, retorno de RL y validación, errores, preguntas, actualizaciones y S3. Las configuraciones exitosas y fallidas se conservan en training/. La selección usa retorno de validación y empates al checkpoint más temprano.',
      '',
      'training_budget.csv registra por configuración duración, pases de acciones, trayectorias, pasos con gradiente no nulo y rendimiento del último checkpoint frente al seleccionado. Permite detectar degradaciones que la selección por validación descarta; conservar M0 no convierte la última política RL en una mejora.',
      '',
      'El retorno de entrenamiento RL usa acciones muestreadas de la distribución; validación y test usan argmax determinista. Por eso pueden diferir mucho, especialmente si M0 acierta con baja confianza. No se interpreta su diferencia como sobreajuste por sí sola. Este piloto no midió en test la utilidad esperada de la política estocástica ni demuestra que su distribución no pueda aprender; el dictamen se refiere al decisor determinista evaluado.',
      '',
      '![Curvas](training_curves.png)' if (run/'training_curves.png').exists() else 'No se generó imagen; las curvas numéricas CSV/JSON están disponibles.',
      '',
      '## 10. Comparación M0/M1/M2/M3','',*comparison_table,'',*differences,'',*intervention_table,'',
      *seed_table,'',*family_table,'',
      'Cada semilla aparece además individualmente en baseline_comparison.csv y metrics.json. La media de semillas se calcula dentro de cada usuario; conteos y acciones se promedian correctamente. No se triplica el número de usuarios. M1/M2 tienen igual presupuesto de actualizaciones/usuarios por lote, pero RL interactúa y supervisión reutiliza demostraciones: se reportan sus diferencias de tiempo y pases.',
      '',
      '## 11. ORACLE, PIPELINE y M4','',
      *extraction_table,'',
      'ORACLE describe hechos comunicados idealmente extraídos. PIPELINE usa extracción lingüística del código existente. La adaptación deriva el día de query_at, disponible a todos los métodos; no se atribuye esa corrección a RL. Las discrepancias de campos de extracción pueden ser inocuas. No se afirma comprensión libre del lenguaje.',
      '',f'M4: `{json.dumps(llm,ensure_ascii=False)}`. Si se ejecutó sobre menos usuarios, sus resultados son complementarios sobre ese subconjunto y no sustituyen la comparación principal. Los fallos técnicos se separan de errores de decisión. Nunca se entrenan pesos del LLM por la API.',
      '',*failure_table,'',
      'La categoría técnica también incluye fallos de formato o catálogo generados por el modelo: son fallos operativos reales, aunque no exista una acción ejecutable para calcular su calidad de decisión. No deben interpretarse como ausencia de error del sistema. El listado permite distinguir límites de generación, salidas inválidas y transporte HTTP/red.',
      '',*failure_family_table,'',
      *llm_family_table,'',
      '',
      'Los resultados M4 se condicionan a episodios completados; su denominador figura en la tabla. Los fallos técnicos no se convierten en decisiones incorrectas ni se ocultan: están en llm/technical_failures.json y reducen cobertura. M4−M3 se empareja sólo con los mismos usuarios completados. Si las exclusiones dependen del escenario, existe sesgo de selección y no procede una comparación global sin reservas. ORACLE y PIPELINE pueden tener subconjuntos completados distintos.',
      '',
      'Una diferencia favorable de M1 frente a M4 no acredita un beneficio de RL si M0, M2 y M3 obtienen el mismo rendimiento que M1. Ese contraste cambia arquitectura, interpretación y entrenamiento previo; sólo M1−M0 y M1−M2 aíslan la continuación por recompensa bajo estas condiciones.',
      '',
      'llm_error_examples.json conserva hasta tres episodios fallidos completos; llm_error_diagnostics.json y error_analysis.csv incluyen las causas probables de M4. Una respuesta que preserve incertidumbre puede omitir indexar evidencia si elige NOOP: la tercera sesión permite detectar esa consecuencia. No se atribuyen estos errores a RL ni se cambian los pesos de GLM.',
      '',
      '## 12. Coherencia longitudinal','',
      'Se evalúa S3 con la memoria persistida por cada acción. Una respuesta fundamentada, una actualización incorrecta y una omisión de aclaración tienen trazas separadas. Exactitud S3 puede incluir preservación correcta de incertidumbre; se deben revisar retorno y preguntas para distinguir resolución. Con una pregunta en S2 y ninguna en S3, el cero de repeticiones está impuesto por diseño y no demuestra aprendizaje para evitarlas.',
      '',
      '## 13. Evaluación estadística','',
      f'Bootstrap percentil 95%, {config["bootstrap"]} réplicas agrupadas por usuario sintético, comparaciones emparejadas. Las sesiones no son sujetos independientes. La media de semillas se toma dentro de usuario; su IC es condicional a las semillas ejecutadas, no un intervalo de población de algoritmos. Se reporta cada semilla. Margen práctico fijado antes de test: +0,10 de retorno por episodio para superioridad, además de IC positivo.',
      '',
      'Si todos los errores o diferencias observados son cero, el bootstrap empírico produce [0,0]. Ese intervalo degenerado no demuestra que el riesgo poblacional sea exactamente cero ni garantiza equivalencia fuera del generador. Los errores de la red aleatoria y su reducción durante M0 corresponden a supervisión, no a RL.',
      '',
      'H1 se contrasta mediante O2−O1; H2 mediante O3−O2 con el intérprete fijo. H3 no queda demostrada simplemente por gradientes: si M0 ya alcanza el techo, este diseño no identifica la capacidad de aprender desde una política sin entrenar. No se ejecutó RL desde pesos aleatorios. H4 exige ventaja incremental frente a supervisión y reglas según el margen preestablecido. Un empate tras M0 no demuestra que RL sea imposible.',
      '',
      '## 14. Fallos y amenazas a la validez','',
      'Contextos, IDs, fechas y plantillas textuales están separados por partición. Se reservan combinaciones de comida/lugar/compañía, pero los cinco mecanismos y parte de las características simbólicas se comparten. La red descarta identidades concretas de contexto tras calcular aplicabilidad: el holdout no demuestra razonamiento conceptual nuevo. Los intervalos sólo expresan variación del generador, no población humana. No hay adjudicación externa de gold.',
      '',support_note,'',
      '',
      'Los ejemplos equilibran direcciones LIKE/DISLIKE para evitar la correlación previa familia→signo. Se mantiene mínimo el simulador; no se introducen casos para favorecer RL. M3 puede estar en el techo, limitando el margen para mejora. La restricción de maní está visible, pero no se consulta habitualmente ese alimento; la prueba de seguridad es técnica, no un benchmark amplio de riesgos.',
      '',
      f'Recursos: Python {resources["python"]}, PyTorch {resources["torch"]}, {resources["threads"]} hilo CPU; duración total {resources["elapsed_seconds"]:.1f} s. Se redujeron cantidades sugeridas a {config["train_users"]}/{config["validation_users"]}/{config["test_users"]} para acotar coste CPU; parámetros configurables. macOS Intel/Python3.13 no dispone de wheel PyTorch compatible: se autorizó un entorno aislado Python3.12/PyTorch2.2.2. El launcher conserva el proyecto original y reejecuta en ese entorno. NumPy no es requisito del entrenamiento.',
      '',f'Los {len(budgets)} entrenamientos sumaron {neural_seconds:.1f} s, incluidas sus evaluaciones de validación; la duración total añade pruebas, test, estadística y, si corresponde, espera de inferencia M4. No se usó GPU para entrenar esta red. No se verificó acceso a una H200 para actualizar pesos de un LLM.',
      '',
      '## 15. RESPUESTAS EXPERIMENTALES PARA VICENT','',
      *questions[2:],
      '',
      '## 16. Recomendación final para la tesis','',
      f'**{classification}**. Dictamen para escalar a un LLM: **{verdict}**. La pregunta oficial no se modifica: estudiar cuándo y cómo aclarar afirmaciones incompatibles y su efecto actual/futuro frente a prompting y heurísticas. Este piloto externo responde sólo al componente de decisión simbólica. No demuestra por sí solo ventaja de RL en LLM.',
      '',
      'Priorizar información histórica, gestión de memoria y evaluación independiente. Si no hay mejora incremental de M1, no se debe confundir con ausencia de entrenamiento: los pesos se actualizaron y sus cambios están registrados. No concluir imposibilidad general de RL, ni modificar test/recompensas/umbrales para fabricar una ganancia.',
      '',
      'Reproducción en carpeta nueva:', '', '```bash',
      f'uv run --offline python -B -m rl_small_pilot.run --output reports/rl_small_pilot/reproduccion --train-users {config["train_users"]} --validation-users {config["validation_users"]} --test-users {config["test_users"]} --m0-epochs {config["m0_epochs"]} --optimizer-steps {config["optimizer_steps"]} --batch-size {config["batch_size"]} --bootstrap {config["bootstrap"]} --seed {config["seed"]}',
      '```','',
      'M4 requiere añadir --llm-base y --llm-users, con USFQ_API_KEY configurada mediante el entorno, nunca en código o argumentos. Sin clave, el comando registra M4 pendiente. Las dependencias y versiones instaladas están documentadas; el comando offline no instala nada. Las pruebas completas se ejecutan antes del entrenamiento y su log se conserva.'
    ]
    (run/'INFORME_PILOTO_RL.md').write_text('\n'.join(report)+'\n')


def _plot(run):
    # Matplotlib already exists in the project environment. No dependency install.
    project=Path(__file__).resolve().parent.parent
    interpreter=os.environ.get('RL_SMALL_LAUNCHER_PYTHON',str(project/'.venv/bin/python'))
    script='''import csv,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
root=Path(sys.argv[1]);rows=list(csv.DictReader((root/'training_curves.csv').open()))
fig,axes=plt.subplots(2,3,figsize=(13,7))
for column,method in enumerate(('M0','M1','M2')):
 ax=axes[0,column];bottom=axes[1,column]
 for seed in sorted({r['seed'] for r in rows}):
  for lr in sorted({r['lr'] for r in rows if r['method']==method}):
   group=[r for r in rows if r['method']==method and r['seed']==seed and r['lr']==lr]
   ax.plot([int(r['optimizer_steps']) for r in group],[float(r['validation_return']) for r in group],label=seed+' lr='+lr,marker='.')
   key='training_mean_return' if method=='M1' else 'training_mean_loss'
   measured=[r for r in group if r[key] not in ('',None)]
   bottom.plot([int(r['optimizer_steps']) for r in measured],[float(r[key]) for r in measured],label=seed+' lr='+lr,marker='.')
 ax.set(title=method,xlabel='Actualizaciones del optimizador');ax.grid(alpha=.2);ax.legend(fontsize=6)
 ax.set_ylim(-2.25,2.05)
 bottom.set(xlabel='Actualizaciones del optimizador',ylabel='Retorno de entrenamiento (muestreado)' if method=='M1' else 'Pérdida supervisada de entrenamiento');bottom.grid(alpha=.2)
axes[0,0].set_ylabel('Retorno por usuario en validación');fig.tight_layout()
fig.savefig(root/'training_curves.png',dpi=150);fig.savefig(root/'training_curves.svg')
'''
    try:
        result=subprocess.run([interpreter,'-c',script,str(run)],capture_output=True,text=True,timeout=60)
        (run/'plot_status.json').write_text(json.dumps({'executed':result.returncode==0,'returncode':result.returncode,
            'error':result.stderr[-1000:] if result.returncode else None},indent=2)+'\n')
    except (OSError,subprocess.TimeoutExpired) as error:
        (run/'plot_status.json').write_text(json.dumps({'executed':False,'error_type':type(error).__name__})+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    generate_report(p.parse_args().run)
