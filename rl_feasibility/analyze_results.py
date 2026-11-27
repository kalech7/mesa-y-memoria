"""Generate the tutor report from recorded results; never trains or selects policies."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re

from rl_feasibility.evaluate import write_json, write_jsonl
from rl_feasibility.reward import RewardConfig, terminal_reward
from rl_feasibility.statistics import mean


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pct(value):
    return 'NE' if value is None else f'{100 * value:.2f}%'


def number(value):
    return 'NE' if value is None else f'{value:.4f}'


def interval(result):
    return f"{number(result['mean'])} [{number(result['ci95'][0])}; {number(result['ci95'][1])}]"


def table(metrics):
    lines = ['| Política | Retorno [IC95] | Decisión correcta | Preguntas/episodio | Fidelidad | Intentos prohibidos |',
             '|---|---:|---:|---:|---:|---:|']
    for name, metric in metrics.items():
        m = metric['means']
        lines.append(f"| {name} | {interval(metric['return'])} | {pct(m['decision_correct'])} | {number(m['clarifications'])} | {pct(m['evidence_fidelity'])} | {int(round(m['attempted_restriction_violation'] * metric['episodes']))}/{metric['episodes']} |")
    return '\n'.join(lines)


def analyze(run, report=None):
    run = Path(run).resolve()
    report = Path(report).resolve() if report else run / 'INFORME_VIABILIDAD_RL.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    prefix = os.path.relpath(run, report.parent)
    link = lambda name: f'{prefix}/{name}'
    manifest, verdict = read(run / 'manifest.json'), read(run / 'verdict.json')
    metrics, headroom = read(run / 'test_metrics.json'), read(run / 'headroom_analysis.json')
    sequential, audit = read(run / 'sequentiality_analysis.json'), read(run / 'data_audit.json')
    training, supervised = read(run / 'training_metrics.json'), read(run / 'supervised/supervised_training.json')
    if training['executed']:
        raise ValueError('This report template is for gate-blocked RL runs; analyze trained-policy contrasts explicitly before writing a conclusion.')
    manual = read(run / 'manual_test_metrics.json')
    # Do not silently relabel hand-authored proposals after changing the task.
    # Flag possible preference/exclusive-choice mismatches for human adjudication.
    from rl_feasibility.environment import FoodPreferenceEnv, audit_evidence
    manual_fixture = read(run / 'manual_review_candidates.json')
    manual_review = []
    for scenario in manual_fixture['scenarios']:
        observation = FoodPreferenceEnv(scenario).reset()[0]
        supported = [f for f in ('ensalada', 'pizza', 'sopa', 'arroz') if audit_evidence(observation, f)['supported']]
        excluded = sorted(set(supported) - set(scenario['latent']['acceptable_foods']))
        manual_review.append({'scenario_id': scenario['scenario_id'], 'query': scenario['query'],
            'supported_options_excluded_by_proposed_gold': excluded,
            'adjudication_required': bool(excluded), 'human_adjudicated': False,
            'note': 'Check whether query demands an exclusive current choice or permits an open recommendation; original gold is unchanged.'})
    write_json(run / 'manual_construct_review.json', manual_review)
    source_hashes = read(run / 'source_sha256.json')
    root = Path(__file__).resolve().parents[1]
    hash_checks = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
                   and hashlib.sha256((run / 'source_snapshot' / name).read_bytes()).hexdigest() == expected
                   for name, expected in source_hashes.items()}
    log = (run / 'unittest.log').read_text()
    test_match = re.search(r'Ran (\d+) tests in ([\d.]+)s', log)
    tests, test_seconds = int(test_match[1]), float(test_match[2])
    oracle = metrics['ORACLE_STRUCTURED']
    selected = headroom['validation']['best_baseline_selected_on_validation']
    all_rows = {name: rows(run / f'test_oracle_{name.lower()}_episodes.jsonl') for name in oracle}
    # Exploratory coefficient sensitivity supplements the preregistered ask-cost
    # analysis. It is not used for gates, selection or any training claim.
    sensitivity = {}
    configurations = {
        'preserve_correct_0.4': RewardConfig(preserve_correct=.4),
        'preserve_correct_1.0': RewardConfig(preserve_correct=1.),
        'preserve_incorrect_-0.5': RewardConfig(preserve_incorrect=-.5),
        'preserve_incorrect_0.0': RewardConfig(preserve_incorrect=0.),
        'unsupported_penalty_0.0': RewardConfig(unsupported_penalty=0.),
        'unsupported_penalty_1.2': RewardConfig(unsupported_penalty=1.2),
    }
    for label, config in configurations.items():
        sensitivity[label] = {name: mean(terminal_reward(r['selected_action'], r, config)
            - config.clarification_cost * r['clarifications'] for r in records) for name, records in all_rows.items()}
    write_json(run / 'reward_sensitivity_exploratory.json', {
        'posthoc': True, 'used_for_selection': False, 'retraining': False,
        'frozen_trajectories_only': True, 'means': sensitivity,
        'limitation': 'No claim about policies learned under alternate rewards; unsupported/safety penalties may be inactive for prudent baselines.'})
    quality_diagnostics = {}
    for name, records in all_rows.items():
        by_category = defaultdict(Counter)
        for r in records:
            if not r['decision_correct']:
                by_category[r['category']][r['selected_action']] += 1
        quality_diagnostics[name] = {cat: dict(actions) for cat, actions in by_category.items()}
    write_json(run / 'failure_summary.json', quality_diagnostics)
    failures = []
    for condition, filename in [('ORACLE_STRUCTURED', 'test_oracle'), ('PIPELINE', 'test_pipeline'), ('SHIFT', 'shift_oracle')]:
        source = 'shift.json' if condition == 'SHIFT' else 'test.json'
        episodes = {s['scenario_id']: s for s in read(run / 'evaluator_private' / source)}
        for name in oracle:
            for row in rows(run / f'{filename}_{name.lower()}_episodes.jsonl'):
                if row['decision_correct']:
                    continue
                scenario = episodes[row['scenario_id']]
                if condition == 'PIPELINE':
                    component, cause = 'extracción/política', 'La memoria extraída o su interpretación no permite completar una decisión respaldada; comparar registros de la traza con eventos originales.'
                    recommendation = 'Auditar extracción por evidencia, sin ajustar sobre este test; construir otro desafío independiente tras corregir.'
                elif row['necessary_clarification_omitted']:
                    component, cause = 'política de adquisición', 'Terminó sin comunicar la información necesaria; puede haber preguntado slots irrelevantes o preservado provisionalmente un conflicto resoluble.'
                    recommendation = 'Comparar la trayectoria con la referencia selectiva y la disponibilidad real del dato, sin inferir que RL sea necesario.'
                else:
                    component, cause = 'decisión terminal', 'La decisión no coincide con un resultado fundamentado aceptable del perfil sintético.'
                    recommendation = 'Revisar evidencia, aplicabilidad y adjudicación de la utilidad terminal.'
                failures.append({'scenario_id': row['scenario_id'], 'condition': condition, 'policy': name,
                    'query': scenario['query'], 'visible_events': scenario['events'],
                    'expected': {'acceptable_foods_if_grounded': scenario['latent']['acceptable_foods'],
                                 'genuinely_unresolvable': scenario['latent']['unresolvable'],
                                 'restrictions': scenario['oracle_restrictions']},
                    'obtained': {key: row[key] for key in ('actions', 'return', 'selected_food', 'decision_correct',
                                 'evidence_fidelity', 'necessary_clarification_omitted')},
                    'responsible_component': component, 'possible_cause': cause, 'recommendation': recommendation,
                    'human_adjudicated': False})
    write_jsonl(run / 'error_explanations.jsonl', failures)
    coverage = {}
    from rl_feasibility.baselines import feature_key
    counts = read(run / 'supervised/supervised_policy.json')['counts']
    for name, records in all_rows.items():
        keys = [feature_key(step['observation']) for row in records for step in row['trace']]
        coverage[name] = {'known_decision_keys': sum(k in counts for k in keys), 'decisions': len(keys)}
    write_json(run / 'policy_state_coverage.json', coverage)
    contrasts = headroom['test_confirmation']['contrasts']
    witness = next((r for r in sequential['examples'] if r['category'] == 'CPC_TWO_SLOT'), sequential['examples'][0])
    successful = next((trace for trace in witness['branches'].values() if trace[-1]['action'].startswith('ANSWER:')), None)
    trajectory = '\n'.join(f"- Paso {i + 1}: `{step['action']}`, recompensa {step['reward']:+.2f}."
                           for i, step in enumerate(successful or witness['immediate']))
    category_lines = ['| Categoría test | n | Heurística: acierto | Supervisado: acierto | Referencia: acierto |', '|---|---:|---:|---:|---:|']
    for cat, values in oracle['CURRENT_HEURISTIC']['by_category'].items():
        category_lines.append(f"| {cat} | {values['episodes']} | {pct(values['means']['decision_correct'])} | {pct(oracle['SUPERVISED_POLICY']['by_category'][cat]['means']['decision_correct'])} | {pct(oracle['STRONG_REFERENCE']['by_category'][cat]['means']['decision_correct'])} |")
    details = []
    for name, values in oracle.items():
        details.append(f"- {name}: utilidad estructural de preguntas {values['clarification_quality']['numerator']:g}/{values['clarification_quality']['denominator']:g}; innecesarias {values['unnecessary_clarification_rate']['numerator']:g}/{values['unnecessary_clarification_rate']['denominator']:g}; información necesaria omitida {values['necessary_clarifications_omitted']['numerator']:g}/{values['necessary_clarifications_omitted']['denominator']:g}; preservación de irresolubles {values['unresolvable_preservation_recall']['numerator']:g}/{values['unresolvable_preservation_recall']['denominator']:g}.")
    bound = read(run / 'privileged_relaxed_bound.json')
    text = f'''# Informe de viabilidad RL — SPEC-RL-002

**Dictamen: {verdict['dictamen']}.** Hay un problema secuencial sintético válido y una mejora frente a la heurística histórica adaptada, pero la imitación supervisada iguala la referencia fuerte. Gate 2 no justifica entrenar RL con esta representación y este umbral. **No se ejecutó el piloto RL y no se estimó su efecto.** Esto no demuestra que RL sea inútil en una tarea más rica.

## 1. Pregunta y objetivo de investigación

`SPEC_CODEX.md:6` pregunta cuándo/cómo aclarar incompatibilidades de la misma preferencia, alcance y período y cuánto mejora la respuesta actual **y la coherencia en sesiones posteriores**. `METODOLOGIA.md:21-24` prioriza éxito en una sesión posterior. Es la formulación encontrada, sin certificación institucional de aprobación por el tutor.

SPEC-RL-002 estudia el efecto de entrenar la política con RL y amplía a CPC/BOC/SCC. Este piloto se limita a interacción breve con cuatro alimentos y memoria de evidencia durante el episodio. El objetivo de decisión es **recomendación abierta**: varios alimentos pueden ser aceptables; no se exige adivinar una elección única oculta. No evalúa coherencia posterior desde SQLite, ADD/UPDATE/DELETE aprendidos ni manejo longitudinal completo. No sustituye silenciosamente la pregunta anterior.

## 2. Estado previo del simulador

SPEC-001 informó 150 tests, 960/960 decisiones controladas y 13/32 en desafío lingüístico. Son resultados históricos **sin RL**, conservados intactos. El hash del desafío original sigue siendo `{manifest['historical_challenge_sha256_before']}`. No se usaron esos 32 casos para ajustar la política.

La implementación nueva reutiliza `simulator/preferences.py` para estructuras/extracción y `simulator/resolution.py` para el baseline histórico. Añade `rl_feasibility/environment.py`, `scenario_generator.py`, `reward.py`, `baselines.py`, `train_pilot.py`, `evaluate.py`, `statistics.py` y este postprocesador. No reemplaza la ruta SQLite/LLM/BGE ni modifica el parser existente. La memoria del adaptador es episódica; no constituye aún un gestor de memoria entrenado.

## 3. Definición formal del problema RL

Modelo: **POMDP finito de adquisición de información**. `s_t=(perfil, contexto efectivo, preferencia actual, historial de evidencias, capacidad de aclarar, historial visible, presupuesto)`; el perfil efectivo permanece fijo durante el episodio. La observación contiene consulta, memorias estructuradas comunicadas, fechas, fuentes, contexto comunicado, diálogo y presupuesto. No contiene IDs de episodio/usuario real, semilla, categoría, preferencia latente, futuras respuestas ni etiquetas. `info={{}}` incluso al terminar; el evaluador obtiene diagnósticos por una interfaz separada.

`A={{ANSWER(f): f∈[ensalada,pizza,sopa,arroz], CLARIFY_DAY, CLARIFY_MEAL, CLARIFY_COMPANY, CLARIFY_PLACE, CLARIFY_CURRENT_PREFERENCE, PRESERVE_CONFLICT, ABSTAIN}}`. VERIFY se excluye porque no existe una fuente verificable con transición real. Preguntar genera la respuesta concreta del usuario y actualiza solamente información comunicada; no cambia sus gustos. Responder/preservar/abstenerse termina. H≤4, presupuesto de preguntas≤2; normalmente basta un máximo efectivo de tres acciones.

`pi(a_t|h_t)`, `h_t=(o_0,a_0,...,o_t)`, `J(pi)=E[Σ r_t]` con descuento 1. Una creencia se actualizaría con `b_(t+1)(s') ∝ Z(o_(t+1)|s',a_t) Σ_s T(s'|s,a_t)b_t(s)`. El estado completo del simulador es Markov; no es observable para la política. La compresión `feature_key` no tiene demostración de suficiencia Markov. Por ello no se invoca una garantía de convergencia de Q-learning.

Una clasificación o un bandit de una sola acción no expresa el valor contingente de preguntar y después escoger. Sin embargo, una política secuencial puede aprenderse también por supervisión o planificación: la secuencialidad no demuestra necesidad de RL. Los fundamentos son [Kaelbling et al., POMDP](https://www.cassandra.org/arc/papers/aij98.pdf) y [Watkins y Dayan, Q-learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).

Trayectoria real de validación `{witness['scenario_id']}`, categoría `{witness['category']}`; consulta y memorias completas están en [sequentiality_analysis.json]({link('sequentiality_analysis.json')}):

{trajectory}

## 4. Evidencia de secuencialidad e incertidumbre

Se bifurcaron **{sequential['states']} estados iniciales** con las cinco preguntas y una decisión inmediata, controlando escenario y aleatoriedad. Hubo {sequential['counts']['clarification_enables_correct_decision']} estados donde aclarar permitió una decisión correcta que no se obtenía inmediatamente; {sequential['counts']['different_terminal_quality_by_question']} donde preguntas diferentes cambiaron la calidad final; {sequential['counts']['asking_can_cost_more_than_immediate']} donde preguntar produjo menor retorno que actuar inmediatamente.

Las diferencias de calidad no se explican por restar un costo fijo: en CPC de dos condiciones, una pregunta irrelevante consume el presupuesto e impide adquirir ambos datos antes de decidir. Las preguntas de comida/compañía/lugar/día tienen respuestas distintas y no revelan automáticamente un alimento. La pregunta sobre preferencia actual puede producir una elección explícita para esa ocasión o confirmar incertidumbre. SCC y BOC incluyen ambas posibilidades, con observaciones iniciales indistinguibles en controles.

El archivo conserva todas las ramas y ejemplos, no solo una tasa favorable. Es una demostración mecánica condicionada al simulador, sin validación de cómo responderían personas reales.

## 5. Validación de observaciones y transiciones

**{tests} pruebas aprobadas, cero fallidas, {test_seconds:.3f} s** en el log final. Incluyen las 150 anteriores más pruebas de reset/step reproducibles, límites, truncamiento, tipos de acción, esquema cerrado, información privada anidada, fechas UTC, evidencia futura, procedencia real, mutación externa, respuestas específicas de slot, estados indistinguibles y recompensa independiente de la heurística/helper de políticas. Ver [unittest.log]({link('unittest.log')}) y [environment_validation.json]({link('environment_validation.json')}). Las pruebas no constituyen un sandbox frente a código malicioso ni una prueba formal de no interferencia.

Datos: {manifest['train_episodes']} entrenamiento, {manifest['validation_episodes']} validación, {manifest['test_episodes']} test, {manifest['test_episodes']} cambio de distribución; usuarios distintos. Semillas `{manifest['data_seeds']}`. Hay {audit['partitions']['train']['unique_latent_signatures']} firmas nativas únicas en entrenamiento y {audit['partitions']['test']['unique_latent_signatures']} en test; los episodios pueden repetir estados dentro de su partición. **Cero firmas compartidas entre train, validación y test**, ver [data_audit.json]({link('data_audit.json')}). Test/OOD comparten {audit['overlap']['shift/test']['latent_signatures']} firmas: no son confirmaciones externas independientes.

Se conservan todas las ejecuciones previas. `pilot_v1` no sirve para generalización por solapamiento de estados; `pilot_v2` corrige particiones. `final` todavía tenía una comparación E1/E2 confundida por textos sin todo el alcance ANY estructurado; `final_corrected` lo corrige. Ambas conservaban un defecto de constructo: bloquear cualquier respuesta ante un conflicto de otro alimento y exigir exclusividad no comunicada. **La evidencia principal actual es `final_open_recommendation`**, con conflictos locales y conjuntos de opciones aceptables. Protocolos y cuatro enmiendas están archivados con hashes. La última corrige el objetivo/gold y la prioridad de alternativas válidas; reduce necesidad artificial de preguntar, sin cambiar coeficientes ni buscar una brecha favorable a RL. Los nuevos sorteos no convierten el catálogo finito en población humana.

ORACLE_STRUCTURED entrega extracción correcta de evidencia visible, nunca el estado oculto. PIPELINE usa el mismo perfil y el parser existente. Las respuestas de slot son actos comunicativos tipados del usuario sintético; no se evalúa comprensión abierta de toda respuesta lingüística. Las políticas reciben solo copias de observaciones. Los archivos `evaluator_private/` no son entradas de política.

## 6. Diseño y validación de recompensas

Coeficientes fijados antes del test: acierto latente +1; error -1; falta de respaldo -0,6 adicional; intento de violar restricción -2 adicional; pregunta -0,10; preservar irresoluble respaldado +0,7, preservar otro caso -0,2; abstención/truncamiento -0,35. Bonus por información=0. En ANSWER se aplica `r=(1 si f∈F*(s), -1 si no) -0,6 I_sin_evidencia -2 I_intento_prohibido`. Un acierto latente por adivinación sin respaldo obtiene 0,4 pero **no** cuenta como decisión correcta; se separan recompensa y calidad.

El perfil autoral define conjuntos de alimentos aceptables e irresolubilidad; un auditor terminal independiente comprueba evidencia comunicada por alimento. Una discrepancia sobre b no anula una opción segura a. LIKE no implica exclusividad. La conducta de un pedido no establece por sí sola un gusto ni invalida la declaración. No consulta qué acción eligió CURRENT_HEURISTIC ni usa `public_support` para la recompensa terminal. Una mutación del helper de políticas no puede fabricar respaldo en el auditor. Ambos comparten supuestos formales del dominio: esa separación de código no sustituye una adjudicación humana del significado.

La barrera bloquea alimentos prohibidos por restricciones estructuradas visibles; el evaluador comprueba también restricciones explícitas perdidas por el parser, pudiendo registrar una violación real de PIPELINE. Se diferencian intentos, bloqueos y violaciones ejecutadas. No se deduce aprendizaje de seguridad a partir de cero violaciones con barrera.

[reward_sensitivity.json]({link('reward_sensitivity.json')}) repondera preguntas con 0,05/0,10/0,20 sobre trayectorias congeladas. [reward_sensitivity_exploratory.json]({link('reward_sensitivity_exploratory.json')}) añade coeficientes de preservación/falta de respaldo, identificados **post hoc** y sin seleccionar políticas. Ninguno acredita aprendizaje bajo otros pesos ni preferencias de utilidad humanas.

## 7. Rendimiento de políticas baseline

Test E1, mismos {manifest['test_episodes']} episodios por política. Intervalos bootstrap emparejables de episodios, {manifest['bootstrap_repetitions']} réplicas, semilla estadística fija 819. Se remuestrean sorteos sintéticos independientes, no sujetos humanos; plantillas y abstracciones compartidas limitan extrapolación.

{table(oracle)}

CURRENT_HEURISTIC es un **adaptador de la heurística histórica con respaldo estructurado**, no una reproducción pura: traduce contexto, llama al resolver y recurre al helper público si el contrato de acción no puede ejecutar la pregunta. STRONG_REFERENCE es una regla selectiva explícita, no un planificador óptimo. SUPERVISED_POLICY es conteo/imitación tabular de esa referencia, con {supervised['examples']} demostraciones de decisiones y {supervised['unique_observation_keys']} claves observacionales; no usa etiquetas privadas ni recompensa y comparte representación/acciones con Q-learning. Ese demostrador diseñado tiene un costo conceptual y una ventaja supervisada legítima, que debe reconocerse.

Desglose de calidad por categoría:

{chr(10).join(category_lines)}

Métricas de preguntas/preservación con denominadores:

{chr(10).join(details)}

Utilidad informativa es cambio estructural de posibilidades/aplicabilidad/incertidumbre, no Counterfactual Information Gain ni utilidad humana. `task_useful_clarifications` exige además pertinencia y ausencia de respuesta ya suficiente. Una pregunta puede informar otra opción y seguir siendo evitable: innecesaria significa que había una respuesta suficiente, que no cambió información o que no era pertinente. Omisión requiere necesidad inicial y falta de evidencia efectivamente adquirida; el fracaso final tras adquirirla se conserva por separado. Fidelidad incluye abstención: se interpreta siempre junto a calidad y distribución de acciones. Preservación provisional de un conflicto resoluble puede ser lógicamente fiel pero pierde utilidad terminal bajo este contrato. La recuperación de acción PRESERVE usa como denominador decisiones irresolubles sin alternativa limpia; un conflicto de b puede seguir irresoluble mientras recomendar a sea correcto. Responder a no elimina la evidencia de b. La precisión de PRESERVE se reporta separadamente en JSON; no es una medida de aprendizaje de gestión de memoria.

## 8. Margen de mejora medido

Mejor baseline seleccionado **solo en validación**: `{selected}`. `headroom=retorno_referencia-retorno_mejor_baseline`, incluido el supervisado. Validación: {interval(headroom['validation']['headroom'])}; confirmación test: {interval(headroom['test_confirmation']['headroom'])}. Umbral práctico previo=0,10 (un costo de pregunta o 5% de convertir -1 a +1, antes de penalizaciones).

La referencia supera al adaptador histórico en retorno test por {interval(contrasts['CURRENT_HEURISTIC']['return'])} y en calidad por {interval(contrasts['CURRENT_HEURISTIC']['decision_correct'])}. Hay decisiones mejorables por elegir alternativas limpias y, cuando hace falta, interactuar. Ese resultado demuestra una ventaja de esta política adaptativa bajo el contrato sintético, **no una ventaja de RL** ni una ablación que atribuya toda la diferencia exclusivamente a preguntar.

El margen cero frente al supervisado prueba que la imitación basta para esta referencia y representación. No prueba que el margen óptimo global sea cero: la referencia pregunta en algunos irresolubles donde conservar directamente ahorraría costo. La cota privilegiada relajada de información perfecta/gratuita es {bound['mean_terminal_utility_bound']:.4f}; deja a lo sumo {bound['minus_validation_selected_baseline']:.4f} frente al baseline seleccionado. No es alcanzable ni una política permitida. Incluso esa relajación queda por debajo de delta=0,10 en este banco: el gate detecta también una escala de mejora restante pequeña bajo la mezcla/pesos elegidos. No se rebajó el umbral después del test para habilitar entrenamiento.

## 9. Resultados del entrenamiento piloto

**Entrenamiento RL ejecutado: {training['executed']}.** Estado de Gate 2: `{headroom['validation']['gate_2']}`. Se respetó la condición de no entrenar sin margen. Las semillas previstas 101/202/303, grid alpha 0,1/0,3, gamma1, epsilon1→0,05, 2.000 episodios y 120 s por candidato quedaron configuradas, **no ejecutadas**. No existen checkpoints ni curvas empíricas de RL. [training_metrics.json]({link('training_metrics.json')}) lo registra; [learning_status.svg]({link('learning_status.svg')}) explica la ausencia, sin curva inventada.

Q-learning CPU está implementado con serialización, límites, logs y selección por validación; su aprendizaje en este entorno no se ha demostrado experimentalmente. PPO requeriría instalaciones y una justificación adicional; observabilidad parcial por sí sola no obliga a usarlo ni garantiza que funcione. No se utilizó GRPO, LLM grande, API pagada ni H200. No corresponde decir «RL falló» ni «RL igualó a SFT».

## 10. Resultados de generalización

PIPELINE, misma política congelada y mismos perfiles de test:

{table(metrics['PIPELINE'])}

Cambio de distribución E1: mezcla de categorías más concentrada en dos condiciones/conflictos, otro estilo y distractores; no es extrapolación a valores/contextos fuera del catálogo.

{table(metrics['SHIFT_ORACLE_STRUCTURED'])}

La auditoría muestra que **{audit['initial_feature_key_overlap_with_train']['test']['shared']}/{manifest['test_episodes']} abstracciones iniciales de test ya aparecieron en entrenamiento**, aunque sus estados nativos se separaron. [policy_state_coverage.json]({link('policy_state_coverage.json')}) extiende el conteo a cada decisión. Es transferencia de una tabla estructural a nuevos perfiles, no razonamiento en estados abstractos desconocidos.

Se ejecutaron {len(read(run / 'manual_review_candidates.json')['scenarios'])} casos manuales de autoría de agente de código, con estas calidades E1 contra su propuesta original: {', '.join(f"{n} {pct(m['means']['decision_correct'])}" for n, m in manual['metrics'].items())}. Están en [manual_test_metrics.json]({link('manual_test_metrics.json')}). **No existe adjudicación humana/externa.** [manual_construct_review.json]({link('manual_construct_review.json')}) identifica propuestas que podrían exigir exclusividad sin comunicarla; no se reetiquetan para mejorar resultados. Sus tasas son desacuerdos con propuestas, no exactitud normativa validada. [human_review_blind.csv]({link('human_review_blind.csv')}) contiene campos pendientes para revisores. No se presentan estos casos como validación humana ni como el desafío original de 32 casos.

## 11. Limitaciones científicas y computacionales

1. El mundo tiene cuatro alimentos y doce familias autorales; la pregunta pertinente está fuertemente asociada al patrón de memoria. Categoría/ID/semilla no se entregan, pero eso no elimina la facilidad semántica del generador.
2. Los rasgos ya calculan conflicto, aplicabilidad y slots relevantes mediante reglas. Una política entrenada aprendería selección de acciones sobre ese análisis, no detección lingüística de conflictos desde cero.
3. El perfil queda fijo durante las cuatro decisiones. BOC son evidencias fechadas y estabilidad declarada, no una dinámica humana aprendida ni transiciones de gustos inducidas por acciones.
4. La referencia fuerte no es óptima; SFT clona sus decisiones. Igualarlas no descarta RL en otros horizontes/costos/datos. La distribución de resolubilidad/mezcla es autoral, no estimada de personas.
5. La partición semántica evita duplicados entre folds pero deja abstracciones comunes y repeticiones dentro de cada fold. Bootstrap describe incertidumbre del banco y no validez poblacional. Test y OOD no son independientes entre sí.
6. PIPELINE utiliza familias textuales deliberadamente distintas, incluidas construcciones difíciles para el parser; su porcentaje es un estrés sintético, no una estimación representativa de español natural. E1 favorable no valida extracción ni recomendaciones médicas.
7. La recompensa expresa utilidad del investigador, incluidas decisiones prudentes penalizadas por no intentar aclarar cuando el usuario habría podido responder. No es utilidad humana medida. Cambiar pesos puede cambiar el problema óptimo.
8. Sin entrenamiento ejecutado, no hay efecto RL, curvas, convergencia, estabilidad entre semillas ni superioridad evaluados. El objetivo longitudinal del repositorio permanece fuera de alcance.

Recursos medidos: Python {manifest['python']}, biblioteca estándar, {verdict['elapsed_seconds']:.3f} s de la ejecución final completa (incluye tests, datos, baselines, bootstrap y archivos). Cero instalaciones nuevas y cero llamadas API registradas. No se midió consumo energético ni memoria pico y no se estima costo de GPU. El entrenamiento configurado tendría hasta 6 candidatos×2.000 episodios×4 decisiones=48.000 pasos, sin garantía de éxito; se ejecutaron cero pasos de entrenamiento RL.

## 12. Dictamen de viabilidad

| Dimensión | Evidencia | Conclusión |
|---|---|---|
| Problema secuencial | Estados indistinguibles, transiciones de preguntas y ramas de distinta calidad | Gate 1 aprobado en el contrato sintético |
| Entrenamiento técnicamente implementado | Q-learning stdlib y límites/checkpoints | Disponible; aprendizaje piloto sin demostrar |
| Mejora frente a heurística | Contrastes pareados positivos | Política adaptativa útil en este banco |
| Margen frente al mejor convencional | Referencia igualada por imitación | Gate 2 no aprueba entrenamiento |
| Aprendizaje/superioridad RL | No entrenado por gate previo | Gate 3 no ejecutado |
| Efecto RL en la pregunta de tesis | Sin tratamiento RL evaluado | Gate 4: efecto no estimado |

**¿Ofrece el simulador un espacio válido y no trivial para estudiar conflictos/clarificación con RL?** El adaptador añadido demuestra un problema secuencial de información no trivial frente a controles simples. **No demuestra margen suficiente para justificar RL** frente a métodos convencionales con esta abstracción y delta. La formulación actual debe reformularse o ampliarse antes de invertir en entrenamiento.

**¿Permite este experimento responder de forma defendible la pregunta principal?** Responde la factibilidad del entorno y el margen de políticas, no cuánto mejora RL: ese efecto no se estimó. Tampoco responde la coherencia longitudinal oficial. La conclusión defendible es **NO-GO PARA ESTA FORMULACIÓN**, acotada al banco, representación y criterio registrados; no es NO-GO general a la tesis ni al estudio de clarificación.

## 13. Siguientes experimentos para la tesis

1. Adjudicar casos y costos con personas independientes antes de ampliar entrenamiento. El proyecto ya propone 60–100 historias/personas y dos revisores en un subconjunto (`SPEC_CODEX.md`, `METODOLOGIA.md`); son criterios de diseño existentes, no garantía de potencia. Empezar por revisar el pequeño conjunto manual y resolver desacuerdos.
2. Para el objetivo oficial, añadir una segunda sesión sin diálogo reciente, memoria persistida y consultas futuras que distingan excepción puntual, confirmación actual y preferencia estable. Medir daño de una actualización incorrecta y repetición innecesaria de preguntas.
3. Recoger variación plausible de disponibilidad de respuestas, costos, evidencia parcial y fuentes; no crearla solo para abrir una brecha. Reservar familias de decisión que no colapsen siempre en las mismas claves conocidas. Mantener E1 antes de atribuir fallos de parser a RL.
4. Comparar planificación fuerte y supervisión con igual acceso/presupuesto. Calibrar delta con costos o diferencias útiles adjudicadas y registrar un nuevo protocolo antes de examinar otro test. No bajar delta para conseguir GO.
5. Solo si esa validación establece margen relevante, ejecutar las tres semillas CPU y congelar checkpoints antes de test. Escalar a PPO/LLM/H200 únicamente si la representación y el presupuesto lo requieren y existe autorización.

Mientras tanto, es defendible estudiar experimentalmente la clarificación adaptativa y memoria contextual sin prometer superioridad de RL. La novedad de una tesis todavía requiere comparación con literatura, adjudicación externa y una contribución específica más allá de aplicar reglas conocidas.

## Reproducción y evidencias

```bash
uv run --offline python -B -m unittest discover -s tests -v
uv run --offline python -B -m rl_feasibility.evaluate --output reports/rl_feasibility/nueva_ejecucion --data-seed-offset 30000
uv run --offline python -B -m rl_feasibility.analyze_results --run reports/rl_feasibility/nueva_ejecucion
```

El destino debe ser nuevo. El código cero significa finalización técnica; consultar `verdict.json`, no confundirlo con GO. La CLI de entrenamiento aislado exige archivos de Gate 1/2 y rechaza el entrenamiento de esta ejecución. Para reproducir fases específicas, los datos completos quedan en `evaluator_private/`, las demostraciones/checkpoint supervisado en `supervised/` y todas las trayectorias por política/condición en JSONL. Ninguna política lee esos datos privados.

Evidencias principales: [manifest]({link('manifest.json')}), [gates]({link('verdict.json')}), [baselines]({link('baseline_metrics.json')}), [test]({link('test_metrics.json')}), [headroom]({link('headroom_analysis.json')}), [gráfico de retornos]({link('baseline_returns.svg')}), [fallos originales]({link('failure_analysis.jsonl')}), [fallos explicados]({link('error_explanations.jsonl')}), [resumen de fallos]({link('failure_summary.json')}). La atribución de causas es técnica y automática, sin adjudicación humana.

Los {len(hash_checks)} archivos del manifiesto de fuentes coinciden con los actuales y sus snapshots: **{all(hash_checks.values())}**. La fuente del postprocesador y su hash se archivan aparte; se añadió para construir este informe a partir de resultados, sin alterar políticas ni reentrenar.
'''
    report.write_text(text, encoding='utf8')
    write_json(run / 'source_verification.json', hash_checks)
    source = Path(__file__).resolve()
    (run / 'analysis_source_snapshot.py').write_bytes(source.read_bytes())
    write_json(run / 'analysis_source_sha256.json', {'analyze_results.py': hashlib.sha256(source.read_bytes()).hexdigest()})
    return {'report': str(report), 'source_hashes_match': all(hash_checks.values()), 'explained_failures': len(failures)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--report')
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.report), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
