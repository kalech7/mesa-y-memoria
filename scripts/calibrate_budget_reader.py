"""TRAIN-only frozen-reader interface development; no SGD, adapters or held-out data.

Reads only one explicit FULL_HISTORY_READER.json. Observations are recovered from
its logged protocol JSON, never by extracting facts or resolving preferences.
All alternatives remain in each attribute's original full public menu. Evaluation
expectations are an external file read only after inference has completed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = Path('models/Qwen2.5-0.5B-Instruct')
VARIANTS = ('baseline_json', 'explicit_json', 'readable_json', 'readable_two_examples', 'readable_letters')
TOPICS = {'breakfast_preference': 'desayuno', 'dinner_preference': 'cena', 'drink_preference': 'bebida'}
JSON_RULE = (
    'Responde exclusivamente con un objeto JSON de la forma {"answer":"valor"}. '
    'El valor debe ser una opción exacta del menú completo, o DEFER. '
    'Contesta únicamente sobre el atributo consultado, usando afirmaciones del usuario. '
    'Aplica su última corrección, su contexto y su vigencia. '
    'Una sugerencia del asistente no confirma un hecho. '
    'Si no se puede saber la preferencia en esa fecha y lugar, responde {"answer":"DEFER"}. '
    'No elijas una opción solo porque aparezca en el menú.'
)
EXAMPLES = (
    '\nDos ejemplos independientes, ajenos a la conversación real:\n'
    'Ejemplo 1: usuario: "Habitualmente hago natación." Luego: '
    '"Corrijo: habitualmente prefiero caminar." Consulta posterior: actividad habitual. '
    'Respuesta: {"answer":"caminar"}.\n'
    'Ejemplo 2: usuario: "Habitualmente leo novelas; ahora prefiero poesía." '
    'No se aclaró cuánto dura el cambio. Consulta posterior: lectura preferida otra semana. '
    'Respuesta: {"answer":"DEFER"}.\n'
)
LETTER_RULE = (
    'Elige una letra del menú público completo. Responde exclusivamente JSON '
    '{"answer":"A"}, sustituyendo A por la letra elegida. '
    'Busca la preferencia expresada por el usuario para el tema, la fecha y el lugar consultados. '
    'Respeta correcciones y excepciones temporales o de lugar. '
    'Las preguntas y sugerencias del asistente no son preferencias del usuario. '
    'Si la evidencia no basta, elige la letra de DEFER. No adivines.'
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_from_logged_case(row):
    logged = row['answer']['metadata']['prompt']
    raw = logged.split('\nOBSERVACION\n', 1)[1].split('\nCATALOGO\n', 1)[0]
    obs = json.loads(raw)
    if set(obs) != {'slots', 'query', 'answer_menu'}:
        raise ValueError('Calibration input must have only the logged reader fields')
    if obs['query'] != row['query']:
        raise ValueError('Logged query mismatch')
    return obs


def readable_observation(obs, *, letters=False):
    lines = ['CONVERSACIONES CONSERVADAS (sin interpretar ni resumir):']
    for slot in obs['slots']:
        lines.append(f"Memoria {slot['slot_id']}; fecha {slot['stored_at']}; sesión de origen {slot['source_session']}:")
        for turn in slot['exchange']:
            role = 'USUARIO' if turn['role'] == 'user' else 'ASISTENTE'
            lines.append(f"{role} [{turn['turn_id']}]: {turn['text']}")
    query = obs['query']
    lines.extend(['CONSULTA ACTUAL:', f"Tema: {TOPICS[query['attribute']]}. Fecha: {query['date']}. "
                  f"Lugar: {query['context']}. Sesión: {query['session_id']}.",
                  f"¿Qué prefiere el usuario para {TOPICS[query['attribute']]} en esa fecha y lugar?",
                  'MENÚ PÚBLICO COMPLETO:'])
    if letters:
        lines.extend(f'{chr(65+i)}: {value}' for i, value in enumerate(obs['answer_menu']))
    else:
        lines.append(canonical(obs['answer_menu']))
    return '\n'.join(lines)


def build_prompt(obs, variant, baseline_prompt=None):
    if variant == 'baseline_json':
        if baseline_prompt is None:
            raise ValueError('The unchanged baseline requires its original logged prompt')
        return baseline_prompt
    if variant == 'explicit_json':
        return JSON_RULE + '\nOBSERVACION\n' + canonical(obs) + '\nMENÚ COMPLETO\n' + canonical(obs['answer_menu'])
    if variant == 'readable_json':
        return JSON_RULE + '\n' + readable_observation(obs)
    if variant == 'readable_two_examples':
        return JSON_RULE + EXAMPLES + '\nCASO REAL:\n' + readable_observation(obs)
    if variant == 'readable_letters':
        return LETTER_RULE + '\n' + readable_observation(obs, letters=True)
    raise ValueError(variant)


def score_case(backend, obs, prompt, variant):
    from llm_local_001.model import Generation
    public_values = obs['answer_menu']
    encoded_values = [chr(65+i) for i in range(len(public_values))] if variant == 'readable_letters' else public_values
    ids = backend.encode_prompt(prompt)
    if len(ids) > 1024:
        raise ValueError(f'Calibration prompt exceeds 1024 tokens: {len(ids)}')
    results = []
    with backend.torch.no_grad(), backend.likelihood_mode():
        for value, encoded in zip(public_values, encoded_values, strict=True):
            text = canonical({'answer': encoded})
            tokens = backend.tokenizer.encode(text, add_special_tokens=False)
            tokens.append(backend.tokenizer.eos_token_id)
            if backend.tokenizer.decode(tokens, skip_special_tokens=True) != text:
                raise ValueError('Candidate does not roundtrip through tokenizer')
            lp = backend.sequence_logprobs(Generation(ids, tokens, text), sampling_distribution=False)
            results.append({'value': value, 'encoded_answer': encoded, 'text': text, 'token_ids': tokens,
                            'token_count_including_eos': len(tokens), 'token_logprobs': lp.detach().cpu().tolist(),
                            'mean_including_eos': float(lp.mean().cpu()), 'sum_including_eos': float(lp.sum().cpu())})
    mean_index = max(range(len(results)), key=lambda i: results[i]['mean_including_eos'])
    sum_index = max(range(len(results)), key=lambda i: results[i]['sum_including_eos'])
    return {'prompt': prompt, 'prompt_ids': ids, 'prompt_tokens': len(ids), 'candidates': results,
            'mean_answer': public_values[mean_index], 'sum_answer_diagnostic': public_values[sum_index],
            'mean_index': mean_index, 'sum_index_diagnostic': sum_index}


def run(source, expectations, output, device):
    from research.src.policy.llm_controller import SpecLocalPolicy
    source, expectations, output = Path(source).resolve(), Path(expectations).resolve(), Path(output).resolve()
    rows = json.loads(source.read_text())
    if len(rows) != 10 or source.name != 'FULL_HISTORY_READER.json':
        raise ValueError('This development calibration is restricted to the ten original logged TRAIN cases')
    output.mkdir(parents=True, exist_ok=False)
    declaration = {'created_at': datetime.now(timezone.utc).isoformat(), 'stage': 'TRAIN reader-interface development only',
                   'source': str(source), 'source_sha256': sha(source), 'script_sha256': sha(__file__),
                   'evaluation_expectations_sha256': sha(expectations), 'variants': list(VARIANTS),
                   'model': str(BASE), 'device': device, 'precision': 'bfloat16', 'adapters': False,
                   'scoring_primary': 'mean completion token logprob including EOS',
                   'scoring_diagnostic': 'sum completion token logprob including EOS',
                   'sampling': False, 'model_weight_updates': 0, 'SFT_updates': 0, 'optimizer_created': False,
                   'heldout_or_validation_inputs': 0,
                   'letters': 'A through G map in original complete public menu order, including DEFER'}
    (output/'PREDECLARATION.json').write_text(json.dumps(declaration,ensure_ascii=False,indent=2)+'\n')
    backend = SpecLocalPolicy(BASE, device=device, precision='bfloat16', lora=False, seed=101, gradient_checkpointing=False)
    if any(p.requires_grad for p in backend.model.parameters()) or hasattr(backend.model,'peft_config'):
        raise ValueError('Calibration must use the frozen provider Instruct without adapters')
    started = time.monotonic()
    results = []
    for index, row in enumerate(rows):
        obs = observation_from_logged_case(row)
        for variant in VARIANTS:
            prompt = build_prompt(obs, variant, row['answer']['metadata']['prompt'])
            result = score_case(backend, obs, prompt, variant)
            result.update(case_index=index, episode_id=row['episode_id'], query=obs['query'], variant=variant)
            if variant == 'baseline_json':
                result['max_replay_mean_score_delta'] = max(abs(a['mean_including_eos']-b)
                    for a,b in zip(result['candidates'],row['answer']['metadata']['scores'],strict=True))
                result['logged_answer'] = row['answer']['answer']
            results.append(result)
            with (output/'INFERENCE.jsonl').open('a') as stream:
                stream.write(canonical(result)+'\n');stream.flush()
        print('calibrated TRAIN case',index+1,'/ 10',flush=True)
    # Evaluation labels never enter prompt construction or any LM call.
    expected = json.loads(expectations.read_text())
    if set(expected) != {str(i) for i in range(10)}:
        raise ValueError('Evaluation expectations must cover the ten original TRAIN cases only')
    summaries = {}
    for variant in VARIANTS:
        subset = [r for r in results if r['variant']==variant]
        stats = {}
        for key,label in [('mean_answer','mean_primary'),('sum_answer_diagnostic','sum_diagnostic')]:
            matched = [r[key]==expected[str(r['case_index'])]['answer'] for r in subset]
            successes = [ok and r[key]!='DEFER' for ok,r in zip(matched,subset,strict=True)]
            stats[label] = {'query_success_count':sum(successes),'queries':10,'query_success':sum(successes)/10,
                            'matches_including_justified_defer':sum(matched),
                            'answers':[r[key] for r in subset], 'defer_count':sum(r[key]=='DEFER' for r in subset)}
        stats['max_prompt_tokens'] = max(r['prompt_tokens'] for r in subset)
        stats['candidate_lengths'] = sorted({c['token_count_including_eos'] for r in subset for c in r['candidates']})
        summaries[variant] = stats
    report = {'declaration':declaration,'elapsed_seconds':time.monotonic()-started,'summaries':summaries,
              'baseline_replay_max_score_delta':max(r.get('max_replay_mean_score_delta',0) for r in results),
              'baseline_answers_reproduced':all(r['mean_answer']==r['logged_answer'] for r in results if r['variant']=='baseline_json'),
              'development_on_same_ten_TRAIN_cases':True,'proof_of_RL_improvement':False,
              'all_choices_LM_categorical_global_menu':True,'inference_calls':len(results)*7,
              'source_files_modified':False}
    (output/'RESULTS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'summaries':summaries,'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--expectations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    run(args.source,args.expectations,args.output,args.device)


if __name__=='__main__':
    main()
