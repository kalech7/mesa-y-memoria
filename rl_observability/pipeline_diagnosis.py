"""Exploratory post-freeze analysis; never imported by a selected policy.

This evaluator helper changes no frozen model or inference source. A transparent
wrapper derives calendar day from the already exposed query_at solely to test a
candidate explanation of the legacy PIPELINE failure. It is not another selected
baseline and is not used for tuning or superiority claims.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

from .evaluation import evaluate
from .policies import HistoricalClarificationRules, public_analysis
from .scenarios import generate_scenarios


class DeriveKnownDay:
    name = 'EXPLORATORY_PUBLIC_QUERY_DATE_DAY_WRAPPER'

    def select_action(self, observation):
        value = deepcopy(observation)
        weekday = datetime.fromisoformat(value['query_at'].replace('Z', '+00:00')).weekday()
        value['context'].setdefault('day', 'WEEKEND' if weekday >= 5 else 'WEEKDAY')
        return HistoricalClarificationRules().select_action(value)


def _brief(row):
    return {key: row[key] for key in ('user_id', 'family', 'error_rate', 'return',
            'future_accuracy', 'questions', 'necessary_asks', 'unnecessary_asks')}


def _trace(row):
    return [dict(session_index=s['session_index'], query_at=s['query_at'],
                 observation=s['observation'], proposed=s['proposed_action'],
                 executed=s['executed_action'], reward=s['reward'],
                 correct=s['correct'], response=s.get('response'),
                 missing_context=public_analysis(s['observation'])['missing'])
            for s in row['steps']]


def diagnose(prefix='reports/rl_observability/pipeline_diagnosis'):
    scenarios = generate_scenarios(5, seed=61004, split='test')
    original = evaluate(HistoricalClarificationRules(), scenarios, 'O2', 'PIPELINE')
    temporary = [s for s in scenarios if s['family'] == 'TEMPORARY_EXCEPTION']
    intervention = evaluate(DeriveKnownDay(), temporary, 'O2', 'PIPELINE')[0]
    target = next(r for r in original if r['family'] == 'TEMPORARY_EXCEPTION')
    before = target['steps'][0]['observation']
    weekday = datetime.fromisoformat(before['query_at'].replace('Z', '+00:00')).weekday()
    inferred = 'WEEKEND' if weekday >= 5 else 'WEEKDAY'
    summary = {
        'analysis_role': 'EXPLORATORY_POST_FREEZE_DIAGNOSTIC_NOT_SELECTED_BASELINE',
        'private_evaluator_artifact': True,
        'inference_sources_or_frozen_checkpoints_modified': False,
        'used_for_training_selection_or_test_tuning': False,
        'cohort': {'users': 5, 'seed': 61004, 'split': 'test', 'representation': 'O2',
                   'extraction': 'PIPELINE',
                   'same_first_five_users_as_main_test': True,
                   'independent_confirmatory_sample': False},
        'original_five_family_results': [_brief(r) for r in original],
        'extraction_records_all_five_users': [dict(user_id=r['user_id'], family=r['family'],
                                                  records=r['extraction_records']) for r in original],
        'temporary_exception': {
            'original': _brief(target), 'original_steps': _trace(target),
            'wrapper': _brief(intervention), 'wrapper_steps': _trace(intervention),
            'visible_context_derivation': {'query_at': before['query_at'],
                'original_context': before['context'], 'calendar_weekday_index': weekday,
                'derived_day': inferred, 'source': 'already exposed observation.query_at',
                'wrapper_changes_only_policy_local_copy_of_context': True},
            'probable_stages': [
                {'category': 'extraction_or_adapter',
                 'evidence': 'Hoy is represented both by a bounded interval and a concrete WEEKDAY/WEEKEND scope. Scope mismatch is not counted by the selected-field extraction audit.'},
                {'category': 'temporal_context_interpretation',
                 'evidence': 'The rule policy treats calendar day as missing despite an exposed absolute query_at date.'},
                {'category': 'clarification_selection',
                 'evidence': 'B2 asks day, receives a generic unresolved reply, then gives a wrong conditional answer although the date-derived day suffices.'}],
            'origin_multiplicity': 'multiple_probable_stages_not_independently_identified_causes',
        },
        'inert_field_discrepancies': [
            'Explicit changes and actual preference clarifications can have parsed valid_from at midnight rather than the authored noon; this does not change applicability for a noon or later query in these cases.',
            'Habitualmente declarations have parsed scope values None rather than authored ANY. The minimal public policy treats both as unrestricted; those differences do not explain these decisions.',
            'The extraction_error_rate checks polarity, selected kind differences and validity bounds. It does not compare scope or source and is not comprehensive linguistic extraction accuracy.'],
        'interpretation': 'One Pipeline failure has a mixed representation/temporal-policy explanation. No claim that RL would correct it; no modified wrapper result enters the primary comparisons.',
        'limitations': [
            'Five deterministic authored families, one user each; no estimate of population prevalence.',
            'The wrapper is an exploratory matched diagnostic on test cases after freeze, not a new confirmatory comparator.',
            'The intervention supplies a deterministic transformation of public information but does not isolate every stage independently.',
            'Sources and observations retain the original Pipeline data; frozen inference, reward and selected checkpoints are unchanged.'],
        'reproduction': 'python3 -B -m rl_observability.pipeline_diagnosis --output-prefix reports/rl_observability/pipeline_diagnosis_reproduction',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = prefix.with_suffix('.json'), prefix.with_suffix('.md')
    if json_path.exists() or md_path.exists():
        raise FileExistsError('Supplementary artifacts are never overwritten')
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    md_path.write_text('''# Diagnóstico exploratorio PIPELINE posterior a la congelación

Este análisis complementario no cambia fuentes de inferencia, recompensas,
checkpoints ni selección. No entra en las comparaciones primarias. Usa cinco
usuarios, uno por familia, generados con semilla **61004**, partición **test**,
observación **O2** y extracción **PIPELINE**. Son los primeros cinco usuarios del
test principal; no es una muestra confirmatoria independiente.

El único fallo de B2 en estos cinco usuarios aparece en S2 de excepción temporal.
La frase «Solamente hoy no me gusta lentejas.» se interpreta como DISLIKE con
vigencia hasta el día siguiente y ámbito day=WEEKDAY. La consulta es
`2026-02-02T12:00:00Z` y el contexto expuesto es `{}`. B2 considera que falta el día,
pregunta `CLARIFY(day)`, recibe una respuesta genérica no resolutiva y propone una
respuesta condicional incorrecta. En S3 responde LIKE correctamente. El error por
usuario es 0,5; retorno -0,1; una pregunta innecesaria.

Una envoltura exploratoria obtiene el día de la semana únicamente de `query_at`,
ya visible. El 2 de febrero de 2026 es lunes: WEEKDAY. Añade ese valor a una copia
local del contexto antes de llamar al mismo B2. No cambia la extracción, la
memoria, el simulador ni el evaluador. Con esa transformación pública responde
DISLIKE en S2 y LIKE en S3: error 0; retorno 2; cero preguntas.

La causa probable comprende varias etapas: representación redundante del día en
extracción/adaptador, interpretación temporal que no deriva un dato disponible y
selección de aclaración innecesaria. No se identifica causalmente por separado
cada etapa. No corresponde atribuir todo el fallo a extracción ni afirmar que RL
lo resolvería. La envoltura no se incorpora al baseline congelado después del test.

El audit de extracción cuenta polaridad, algunas clases e intervalos; **no compara
scope ni source**. El cambio de intervalo a medianoche frente al inicio anotado a
mediodía se cuenta como discrepancia, pero es inocuo para estas consultas. También
hay diferencias None/ANY de ámbito tratadas como equivalentes por el intérprete
mínimo. Por tanto, discrepancia de campos no equivale a error de decisión y la
tasa registrada no mide toda la fidelidad lingüística.

El JSON adjunto conserva las anotaciones originales, campos extraídos, fechas,
ámbitos, contextos, observaciones, acciones propuestas/ejecutadas y resultados de
los cinco usuarios, junto con la intervención emparejada y el hash del código.
Contiene datos privados del evaluador; no debe alimentar una política.

Reproducción en salida nueva:

```sh
python3 -B -m rl_observability.pipeline_diagnosis \\
  --output-prefix reports/rl_observability/pipeline_diagnosis_reproduction
```
''')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', default='reports/rl_observability/pipeline_diagnosis')
    result = diagnose(parser.parse_args().output_prefix)
    print(json.dumps({'original': result['temporary_exception']['original'],
                      'wrapper': result['temporary_exception']['wrapper']}, ensure_ascii=False))
