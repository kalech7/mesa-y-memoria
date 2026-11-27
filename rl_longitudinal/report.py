"""Render the report from saved results, never from assumed outcomes."""
from collections import Counter
import csv
import html
import json
from pathlib import Path


def _load(path):
    return json.loads(Path(path).read_text())


def _pct(metric):
    if metric.get('value') is None:
        return 'no evaluable'
    return f"{100*metric['value']:.2f}% ({metric['numerator']}/{metric['denominator']})"


def _estimate(metric):
    if metric.get('mean') is None:
        return 'no evaluable'
    lo,hi=metric['ci95']
    return f"{metric['mean']:.4f} [{lo:.4f}, {hi:.4f}]"


def _table(metrics):
    lines=['| Política | Retorno/sesión, IC95% por usuario | Decisión correcta | Sesiones posteriores | Preguntas | Intervenciones del contrato |',
           '|---|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():
        lines.append(f"| {name} | {_estimate(m['return_per_session'])} | {_pct(m['decision_accuracy'])} | {_pct(m['followup_accuracy'])} | {m['clarifications']} | {m['shield_interventions']} |")
    return '\n'.join(lines)


def _bar_chart(path,metrics):
    width,height=1040,440
    left,right,top,bottom=310,960,45,390
    low,high=-1.6,1.05
    def x(value):
        return left+(value-low)/(high-low)*(right-left)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#fff"/>',
           '<g font-family="sans-serif" font-size="14" fill="#152536">',
           '<text x="20" y="24" font-size="18">SPEC-RL-003: retorno medio por sesión (IC95% por usuario)</text>']
    for tick in (-1.5,-1,-.5,0,.5,1):
        parts += [f'<line x1="{x(tick):.2f}" y1="40" x2="{x(tick):.2f}" y2="{bottom}" stroke="#dde3e9"/>',
                  f'<text x="{x(tick):.2f}" y="415" text-anchor="middle">{tick:g}</text>']
    for i,(name,m) in enumerate(metrics.items()):
        y=top+35+i*62
        value=m['return_per_session']['mean']
        lo,hi=m['return_per_session']['ci95']
        color='#246b8e' if name not in ('LAST_WRITE_WINS','HISTORICAL_RULE_BASED') else '#8a6873'
        parts += [f'<text x="20" y="{y+5}">{html.escape(name)}</text>',
                  f'<rect x="{min(x(0),x(value)):.2f}" y="{y-13}" width="{abs(x(value)-x(0)):.2f}" height="26" fill="{color}" opacity=".75"/>',
                  f'<line x1="{x(lo):.2f}" y1="{y}" x2="{x(hi):.2f}" y2="{y}" stroke="#172536" stroke-width="2"/>',
                  f'<line x1="{x(lo):.2f}" y1="{y-6}" x2="{x(lo):.2f}" y2="{y+6}" stroke="#172536"/>',
                  f'<line x1="{x(hi):.2f}" y1="{y-6}" x2="{x(hi):.2f}" y2="{y+6}" stroke="#172536"/>']
    parts += ['</g></svg>']
    Path(path).write_text('\n'.join(parts))


def generate_report(run):
    run=Path(run)
    manifest=_load(run/'manifest.json')
    allmetrics=_load(run/'longitudinal_metrics.json')
    e1,e2,shift=(allmetrics[k] for k in ('ORACLE_STRUCTURED','PIPELINE','FAMILY_HOLDOUT'))
    comparison=_load(run/'baseline_comparison.json')
    validation=_load(run/'memory_validation.json')
    sequential=_load(run/'sequentiality_analysis.json')
    headroom=_load(run/'headroom_analysis.json')
    training=_load(run/'training_metrics.json')
    audit=_load(run/'data_audit.json')
    supervised=_load(run/'supervised_training.json')
    ablation=_load(run/'ablation_metrics.json')
    preserved=_load(run/'historical_preservation.json')
    strong=e1['CLARIFICATION_RULE_BASED']
    technical=all(validation[k] for k in ('gate_a','gate_b','gate_c'))
    verdict=('NO-GO PARA ESTA FORMULACIÓN' if technical and headroom['status']=='NO_MEANINGFUL_MARGIN'
             else 'GO-RL CONDICIONAL' if technical else 'EVIDENCIA INSUFICIENTE')
    if training.get('executed'):
        verdict='GO-RL CONDICIONAL: evaluar resultados reales del piloto antes de afirmar superioridad'
    _bar_chart(run/'baseline_returns.svg',e1)
    train_text=('Se ejecutó entrenamiento; consultar checkpoints y métricas sin presumir superioridad.' if training.get('executed') else
                'No se ejecutó entrenamiento RL: Gate D no justificó el piloto. Cero pasos de RL; no hay curva de aprendizaje ni tres semillas entrenadas. La implementación Q-learning es sólo compatibilidad técnica probada, no aprendizaje demostrado.')
    (run/'learning_status.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120"><rect width="100%" height="100%" fill="#f2f5f8"/><g font-family="sans-serif" fill="#152536"><text x="24" y="42" font-size="22">'+('Entrenamiento ejecutado: ver métricas reales' if training.get('executed') else 'RL no ejecutado: Gate D sin margen suficiente')+'</text><text x="24" y="80" font-size="16">'+('No interpretar esta figura como una curva de aprendizaje.')+'</text></g></svg>')
    testlog=(run/'unittest.log').read_text() if (run/'unittest.log').exists() else 'Pruebas no ejecutadas'
    log_summary='; '.join(line for line in testlog.splitlines() if line.startswith(('Ran ','OK','FAILED')))
    sessions=strong['sessions']
    matrix=['| Política | Recall histórico | Aplicabilidad temporal | Interpretación de hechos entrantes | Aclaración necesaria | Preguntas innecesarias | Fidelidad de respuestas | Restricciones |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in e1.items():
        matrix.append('| '+name+' | '+' | '.join(_pct(m[k]) for k in (
            'historical_relevant_recall','temporal_applicability_accuracy','interpretation_accuracy',
            'necessary_clarification_recall','unnecessary_clarification_rate','answer_evidence_fidelity','restriction_violation_rate'))+' |')
    familytable=['| Familia | Usuarios | Decisiones correctas (reglas de clarificación) | Retorno/sesión |', '|---|---:|---:|---:|']
    for name,m in strong['by_family'].items():
        familytable.append(f"| {name} | {m['users']} | {_pct(m['decision_accuracy'])} | {m['return_per_session']:.4f} |")
    examples=[]
    private_test=_load(run/'evaluator_private'/'test.json')
    sample={}
    for scenario in private_test:
        sample.setdefault(scenario['family'],scenario)
    for family in ('STABLE_EXCEPTION','EXPLICIT_CHANGE','SOURCE_CHANGE','UNANSWERED'):
        scenario=sample.get(family)
        if scenario:
            examples.append(f"- `{scenario['episode_id']}` ({family}): "+'; '.join(
                f"{s['at'][:10]} / consulta {s['query_at'][:10]}: {s['query']} → expectativa sintética {s['gold']['foods'] or s['gold']['acceptable_conversations']}"
                for s in scenario['sessions']))
    paired_current=comparison['paired_test']['PLANNING_BASELINE_minus_HISTORICAL_RULE_BASED']
    limits=[]
    for split,details in audit['partitions'].items():
        limits.append(f"| {split} | {details['users']} | {details['unique_native_signatures']} | {details['unique_abstract_scripts']} | {json.dumps(details['session_counts'])} |")
    usage=e1['SUPERVISED_LONGITUDINAL'].get('policy_usage',{})
    text=f'''# Informe SPEC-RL-003: memoria temporal y decisiones longitudinales

**Dictamen: {verdict}.** Viabilidad técnica del entorno controlado y preferencia
científica por RL son conclusiones distintas. Evidencia exclusivamente sintética;
sin adjudicación humana ni validación externa completada.

## 1. Pregunta, objetivo oficial y diagnóstico

La pregunta oficial de [SPEC_CODEX.md](../../../SPEC_CODEX.md:6) pregunta cuándo y
cómo aclarar incompatibilidades de la misma preferencia/alcance/período y cuánto
mejora la respuesta actual y la coherencia posterior frente a prompting y
heurísticas. [METODOLOGIA.md](../../../METODOLOGIA.md:21) prioriza éxito posterior.
Esta revisión estudia ese componente longitudinal; no ha comparado prompting de
LLM ni entrenado un modelo lingüístico. No sustituye silenciosamente la pregunta.

El sistema anterior ya tenía Store SQLite, preferencias con vigencia y un
adaptador de una consulta. Faltaba el contrato de operaciones temporales con
recuperación en sesiones nuevas. Se añaden módulos separados en `rl_longitudinal/`;
no se reemplazan los componentes históricos. Auditoría y protocolo previo:
[protocolo](../../../docs/rl_longitudinal_protocol.md).

## 2. Implementación y corrección histórica

Archivos nuevos bajo `rl_longitudinal/`: `temporal_memory.py`,
`memory_reconciliation.py`, `clarification_policy.py`, `memory_actions.py`,
`longitudinal_environment.py`, `scenario_generator.py`, `reward.py`, `baselines.py`,
`train_rl.py`, `evaluate.py`, `audit.py`, `metrics.py`, `validation.py` y `report.py`.
Se añaden pruebas de temporalidad, reconciliación, aclaración, persistencia,
entorno, fugas, políticas y métricas. Se amplía `README.md` y se añaden protocolo y
enmienda; el código histórico de producción se verifica sin cambios.

`TemporalMemoryStore` extiende `memory.store.Store`: tablas append-only de hechos,
operaciones, aclaraciones y consultas. `observed_at`, `recorded_at`, `valid_from`,
`valid_to` y `query_at` son distintos. Fechas ISO con zona; consultas UTC,
intervalos semiabiertos [inicio, fin), fechas sin hora a medianoche UTC y rechazo
de timestamps sin zona. Una actualización enlazada cierra la aplicabilidad
proyectada del hecho antiguo; su contenido original permanece intacto.

El reconciliador evalúa todas las evidencias pertinentes recuperadas, incluyendo
antiguas y caducadas. La recencia sola no resuelve contradicciones. El cambio
explícito requiere mismo alimento/alcance y vínculo válido; una elección puntual
tiene fin de vigencia. Comportamiento observado no se convierte en declaración.
Restricciones son hechos separados y no pueden retirarse por un gusto reciente.
No existe ontología validada de ingredientes ni interpretación clínica.
Los casos de vegetarianismo se comprueban como hechos tipados en el núcleo;
las trayectorias masivas usan análogos de alimentos del catálogo, no inferencia
automática de incompatibilidad entre categorías vegetarianas y alimentos cárnicos.

Cada `step` vuelve a consultar SQLite antes de actuar; incluso una consulta vacía
queda registrada. Se cierra y reabre la conexión al terminar cada sesión. La nueva
observación procede del índice persistente, no de una conversación completa
reinyectada. El archivo bruto es append-only; omitir el índice es una ablación de
recuperación, no destrucción del archivo ni un comportamiento conforme a MEM-01.

Pruebas: **{log_summary}**. Gates A/B/C: {validation['gate_a']},
{validation['gate_b']}, {validation['gate_c']}. Detalles:
[memory_validation.json](memory_validation.json), [log](unittest.log).

## 3. Definición formal y límites del problema RL

POMDP finito \(\mathcal P=(S,A,T,O,Z,R,1,H)\). Estado privado: gustos sintéticos,
respuesta disponible a una pregunta y acontecimientos aún no comunicados. Estado
controlado: memoria persistida. Observación: consulta, contexto comunicado,
historia recuperada y evidencias entrantes. Acción factorizada:
\(a_t=(a_t^{{conv}},a_t^{{mem}})\), con conversación ANSWER/CLARIFY/CONDITIONAL_ANSWER/
PRESERVE_CONFLICT/ABSTAIN y memoria ADD/UPDATE_WITH_HISTORY/NOOP/PRESERVE_CONFLICT.
La política usa sólo historia observable; no recibe etiquetas, semilla o futuro.
Las respuestas específicas y las escrituras cambian observaciones posteriores.

La cantidad de sesiones efectiva se informa abajo. El horizonte de preguntas es
dos por sesión. El horizonte conocido se expone como presupuesto, no se expone el
calendario de mensajes futuros. La abstracción tabular no se presume estadístico
de creencia suficiente. Hay decisiones secuenciales; esto no prueba que RL sea el
método preferible.

**Restricción decisiva de alcance:** los invariantes obligatorios calculan una
`required_decision` pública y reparan propuestas inválidas. En particular,
preguntar ante incertidumbre material y conservar evidencia explícita ya están
parcialmente resueltos por reglas. No se atribuye ese razonamiento a SFT o RL.
Las operaciones correctas son gratuitas, sin límite de capacidad. No se inventó
una escasez de memoria para favorecer RL.

## 4. Secuencialidad, incertidumbre y consecuencias futuras

En {validation['controlled_users']} usuarios de validación y
{validation['controlled_sessions']} sesiones, {validation['counterfactual_users_with_future_effect']}
contrafactuales cambiaron decisiones futuras al omitir sólo la escritura inicial.
En {validation['same_current_quality_but_future_effect']} la calidad de la primera
decisión se mantuvo y cambió alguna posterior: no es sólo un coste fijo de acción.
Se conserva aleatoriedad y estado inicial idénticos. La comparación de omisión es
una ablación explícita, no una política conforme a todos los invariantes.

[Trayectorias y análisis](sequentiality_analysis.json),
[ablaciones completas](ablation_episodes.jsonl),
[conflictos trazables](temporal_conflicts.jsonl).

## 5. Observaciones, transiciones y persistencia

Reproducibilidad de reset: {validation['reset_reproducible']}. Mutar gold sin
cambiar evidencia mantiene la observación inicial:
{validation['private_gold_mutation_does_not_change_initial_observation']}.
Cada acción tiene traza real de recuperación:
{validation['all_decisions_have_actual_sqlite_retrieval']}.
Los tests incluyen futuros no disponibles, cambios de orden físico, aclaraciones
persistidas/no repetidas, desconocido distinto de ANY, consultas históricas y
seguridad ante recencia. La frontera es una API, no una caja de aislamiento frente
a código hostil con acceso al proceso del simulador.

## 6. Recompensas y métricas independientes

{json.dumps(_load(run/'freeze.json')['reward'],ensure_ascii=False)}

El retorno suma resultados entre sesiones; no hay bonus por preguntar ni por
copiar una heurística. El evaluador independiente usa expectativas del generador
y comprobación de evidencia separada del reconciliador. No representa evaluación
humana. Separa acción correcta, fundamento, seguridad y coste. Guardar una
evidencia no demuestra que su interpretación sea estable.

Cada exactitud muestra numerador/denominador. Recall histórico significa cobertura
de hechos comunicados relevantes al alimento objetivo, más restricciones, o toda
la historia alimentaria en consulta abierta, incluyendo hechos caducados; no es
un benchmark semántico de ranking. Aplicabilidad compara hechos originales con
el conjunto aplicable, con vigencia/contexto y sustitución explícita. Interpretación
compara tipo, fuente, alcance y vigencia contra estructura autorada. Fidelidad se
calcula sólo sobre ANSWER y no convierte abstención en respuesta fundamentada.
Errores propagados causales se estudian mediante intervenciones pareadas, no
contando todos los errores posteriores como propagación.
El recall de conflicto usa exposiciones iniciales anotadas con incertidumbre
material de fuente/tiempo/restricción, antes de aclarar. Numerador y denominador
usan la misma máscara. Interpretación y recall incluyen aclaraciones realmente
comunicadas aunque el extractor las pierda. La tasa de sobrescritura refleja el
contrato append-only; las propuestas erróneas se informan por separado.

{chr(10).join(matrix)}

## 7. Baselines: E1 memoria histórica y E2 clarificación

LAST_WRITE_WINS y HISTORICAL_RULE_BASED sin preguntas son ablaciones explícitas
de MEM-02/MEM-05/MEM-06. La protección de seguridad permanece habilitada.
CLARIFICATION_RULE_BASED aplica el contrato. PLANNING_BASELINE realiza búsqueda
visible de profundidad 2 sobre respuestas hipotéticas y consultas a 30/90 días;
supone respuestas uniformes, sin leer respuestas reales futuras. No es óptimo.
El contrato limita su elección de preguntas y operaciones.

SUPERVISED_LONGITUDINAL es clonación tabular del planificador con características
y acciones compartidas con Q-learning. {supervised['examples']} demostraciones,
{supervised['unique_observation_keys']} claves de observación. Uso en test:
`{json.dumps(usage,ensure_ascii=False)}`. El resultado híbrido con fallback de reglas
no equivale a generalización de una política aprendida pura.

{_table(e1)}

![Retornos observados](baseline_returns.svg)

Planificación menos reglas históricas sin preguntar: retorno/sesión
{_estimate(paired_current['return_per_session'])}; exactitud posterior, diferencia
en proporción {_estimate(paired_current['followup_accuracy'])}. Son comparaciones
pareadas de usuarios completos, no de sesiones independientes.

## 8. E3: memoria, aclaraciones y coherencia futura

Las comparaciones con/sin persistencia, con/sin conservar aclaración y sin
preguntas están en [ablaciones](ablation_episodes.jsonl). La traza incluye
propuesta, acción ejecutada, motivos del filtro, hechos anteriores/posteriores,
evidencia y operación real por hecho. Una respuesta persistida puede eliminar una
pregunta repetida en una sesión futura; una negativa a contestar conserva
incertidumbre. Contexto de una ocasión no se vuelve preferencia general.

Control y tratamiento causales comparten configuración de guardas; `retained`
muestra además la referencia protegida. Omitir persistencia rompe deliberadamente
el contrato de recuperación:

{_table(ablation['metrics'])}

[Métricas y diferencias pareadas de ablación](ablation_metrics.json).

Ejemplos reproducibles del test; expectativas sintéticas, sin adjudicación externa:

{chr(10).join(examples)}

## 9. E4: margen medido y entrenamiento

Referencia: {headroom['reference_method']}; mejor comparador no-RL de validación:
{headroom['best_non_rl_method']}. Margen = {headroom['difference']:.6f} por sesión;
IC95% {headroom['paired_ci']}; mínimo fijado antes del test = 0,10.
Gate D: **{headroom['status']}**. Se eligió el comparador usando sólo validación.
El test no seleccionó hiperparámetros. Una ganancia frente a recencia no basta.

{train_text}

[training_metrics.json](training_metrics.json),
[headroom_analysis.json](headroom_analysis.json).
Sin entrenamiento no se ha ejecutado una ablación de entrenamiento sin recompensa
diferida: una reponderación de los mismos retornos sólo sería descriptiva.
[Sensibilidad al coste de preguntas](reward_sensitivity.json): 0,05/0,10/0,20,
con trayectorias congeladas y sin seleccionar pesos según el test.

## 10. E5: generalización y extracción lingüística

PIPELINE usa extracción lingüística de conversaciones y respuestas de aclaración;
ORACLE sólo usa estructura correcta de lo ya comunicado. El parser anterior no
fue ajustado al test. Las expresiones retenidas son un desafío finito autorado,
no una muestra representativa del español. Resultados separados:

{_table(e2)}

Familias estructurales retenidas (ausentes de entrenamiento/validación):
IRRELEVANT_CONFLICT y LATE_RECORDED_HISTORY. Este conjunto comparte esas familias
con un subconjunto del test principal; no son dos réplicas independientes.

{_table(shift)}

| Partición | Usuarios | Firmas nativas distintas | Guiones abstractos distintos | Cantidad de sesiones |
|---|---:|---:|---:|---|
{chr(10).join(limits)}

Las firmas abstractas eliminan IDs, fechas absolutas, separación exacta entre fechas
y nombres de alimentos. Los solapamientos se publican en
[data_audit.json](data_audit.json). Nuevas fechas/usuarios no equivalen a historias
conceptualmente independientes. No se interpreta un bootstrap sobre variaciones
del generador como inferencia sobre usuarios humanos.

{chr(10).join(familytable)}

## 11. Fallos, validez y recursos

{manifest['errors']} decisiones incorrectas registradas en
[error_analysis.jsonl](error_analysis.jsonl), con entrada, expectativas y salida.
Las causas automáticas son propuestas, no adjudicación. Los fallos de extracción
se separan de los de decisión. No se ocultaron bajo el promedio global.
Los resultados por sesiones anteriores y categorías están en
[longitudinal_metrics.json](longitudinal_metrics.json).

Tiempo total medido: {manifest['elapsed_seconds']:.3f} s. CPU local, biblioteca
estándar, sin instalaciones nuevas, APIs pagadas ni H200. Fuentes históricas:
{preserved['checked']} archivos comprobados; sin cambios: {preserved['unchanged']}.
Resultados SPEC-001/002 conservados. Fuentes y parámetros congelados:
[freeze.json](freeze.json), [manifest.json](manifest.json).

**No se ha realizado validación humana/externa.** Se prepara
[human_review_blind.csv](human_review_blind.csv), con campos vacíos de anotador.
No se presenta revisión de subagentes como revisión humana independiente.
Faltan historias naturales, adjudicación de casos difíciles, restricciones con
ingredientes, preguntas de lenguaje libre, distribución real de disponibilidad
de respuesta/coste y comparación contra prompting de modelos. El generador usa
pocos guiones deterministas. Las protecciones limitan el espacio de política y
pueden explicar retornos altos. Las cifras no validan todo el dominio humano.

## 12. Respuestas P1–P7 y dictamen

- **P1:** Sí en los casos controlados del contrato protegido: cada decisión
  consulta SQLite, conserva historia y evita recencia como regla universal.
  La ablación LAST_WRITE_WINS permite cuantificar sus errores.
- **P2:** La estructura y reconciliación distinguen cambios explícitos,
  excepciones, contexto y comportamiento. Generalización lingüística se limita
  por los resultados PIPELINE; no se equipara tipado oracle con comprensión.
- **P3:** Gate B = {validation['gate_b']}; aclaraciones necesarias realizadas
  {validation['performed_needed_clarifications']}/{validation['needed_clarifications']},
  innecesarias {validation['unnecessary_questions']}, repetidas
  {validation['repeated_questions']} en validación controlada. Parte de esta
  garantía la impone el contrato, no una política aprendida.
- **P4:** Hay efectos entre sesiones demostrados mediante intervenciones de
  persistencia y conservación de aclaraciones; consultar las diferencias por
  usuario, no inferir causalidad sólo de correlación entre errores.
- **P5:** Gate D = {headroom['status']}; no se presume margen positivo porque
  las reglas históricas superen la recencia.
- **P6:** {'Consultar resultados reales de entrenamiento antes de concluir.' if training.get('executed') else 'No evaluable: no se ejecutó RL. No se ha demostrado aprendizaje, superioridad ni inferioridad de RL.'}
- **P7:** El protocolo puede estudiar el componente temporal/controlado de la
  pregunta oficial y medir valor de aclaración/persistencia. No responde todavía
  una superioridad causal de RL ni toda la pregunta frente a prompting en datos
  externamente validados.

**{verdict}.** Continuar la tesis sobre clarificación y coherencia temporal es
distinto de justificar RL en esta formulación compacta. Un entorno compatible
con RL y recompensas diferidas no demuestra necesidad ni ventaja de entrenamiento.

## 13. Siguientes experimentos y reproducción

Primero adjudicar historias y decisiones con revisores externos; mantener etiquetas
propuestas y desacuerdos. Evaluar modelos pequeños/prompting, recuperación real y
SFT fuerte bajo observaciones comparables. Si aparecen errores persistentes con
un margen práctico predefinido, registrar un protocolo nuevo antes de otro test;
no bajar el umbral ni degradar baselines para conseguir GO. Introducir compromisos
de memoria/coste sólo si responden al uso real, no para favorecer RL.

```bash
uv run --offline python -B -m unittest discover -s tests -v
uv run --offline python -B -m rl_longitudinal.evaluate --output reports/rl_longitudinal/mi_reproduccion
```

La carpeta de salida debe ser nueva. Se pueden configurar cantidades de usuarios,
semillas y remuestreos. El comando evalúa gates antes de cualquier entrenamiento;
fallar un gate científico no se oculta cambiando un test de software.
'''
    (run/'INFORME_RL_LONGITUDINAL.md').write_text(text)
    return {'verdict':verdict,'report':str(run/'INFORME_RL_LONGITUDINAL.md')}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    generate_report(parser.parse_args().run)
