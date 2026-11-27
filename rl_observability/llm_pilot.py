"""Complementary B5 pilot through a documented OpenAI-compatible service.

API keys are read exclusively from an environment variable, never serialized.
Technical failures are recorded separately and never converted into decisions.
This module does not modify the frozen CPU experiment.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import monotonic
import urllib.error
import urllib.request

from .environment import MinimalEnv
from .policies import HistoricalClarificationRules, action_catalog
from .evaluation import evaluate, paired, summarize


PROMPT = """Eres un asistente de preferencias alimentarias con memoria temporal.
Recibirás únicamente la observación disponible ahora. Decide usando esa evidencia.
No adivines preferencias ocultas ni respuestas futuras. Comprueba fecha consultada,
vigencia, excepciones temporales, alcance, cambios explícitos y restricciones.
Una última declaración sin sustitución explícita no resuelve por sí sola un conflicto.
Si un conflicto relevante impide una respuesta definitiva, puedes pedir una aclaración
por el slot pertinente si queda presupuesto. Preguntas distintas pueden obtener
respuestas distintas; no preguntes por información ya disponible ni repitas preguntas.
Si no puede resolverse, conserva justificadamente el conflicto o responde condicionalmente.
No preguntes por conflictos irrelevantes para el alimento consultado.
ANSWER significa declarar LIKE o DISLIKE sobre target_food, no proponer un menú.
Conserva la evidencia entrante e historial; UPDATE_WITH_HISTORY sólo para sustituciones
explícitas fundamentadas; una excepción temporal no es una actualización estable.
Si no hay evidencia entrante, normalmente basta NOOP como operación de memoria.
Selecciona exactamente una acción del catálogo proporcionado. Devuelve únicamente
un objeto JSON con conversation, memory_operation, polarity y slot, sin explicación.
"""


class TechnicalFailure(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class LLMPrompting:
    name = 'LLM_PROMPTING'

    def __init__(self, base, model, *, token_env='USFQ_API_KEY', timeout=60):
        self.base=base.rstrip('/')
        self.model=model
        self.token=os.environ.get(token_env,'')
        if not self.token:
            raise ValueError(f'Missing environment variable: {token_env}')
        self.timeout=timeout
        self.opener=urllib.request.build_opener(NoRedirect)
        self.calls=[]

    def select_action(self, observation):
        actions=[json.loads(a) for a in action_catalog(observation)]
        user={'observation':deepcopy(observation),'allowed_actions':actions}
        # Both payloads below contain only the public observation and fixed action
        # catalog. The environment object, gold and future replies are never sent.
        messages=[{'role':'system','content':PROMPT},
                  {'role':'user','content':json.dumps(user,ensure_ascii=False)}]
        for limit in (2048,4096):
            payload={'model':self.model,'messages':messages,'temperature':0,'seed':41004,
                     'max_completion_tokens':limit,'stream':False,
                     'response_format':{'type':'json_object'}}
            request=urllib.request.Request(self.base+'/chat/completions',
                data=json.dumps(payload,ensure_ascii=False).encode(),
                headers={'Content-Type':'application/json','Authorization':'Bearer '+self.token})
            start=monotonic()
            record={'model_requested':self.model,'max_completion_tokens':limit,
                    'messages_sha256':hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()}
            try:
                with self.opener.open(request,timeout=self.timeout) as response:
                    result=json.load(response)
                choice=result['choices'][0]
                content=choice['message'].get('content')
                record.update(model_returned=result.get('model'),content=content,
                    finish_reason=choice.get('finish_reason'),usage=result.get('usage'),
                    elapsed_seconds=monotonic()-start)
                if choice.get('finish_reason')=='length' or not isinstance(content,str) or not content.strip():
                    record['technical_error']='token_limit_or_empty_content'
                    self.calls.append(record)
                    if limit==2048:
                        continue
                    raise TechnicalFailure(record['technical_error'])
                try:
                    action=json.loads(content.strip().removeprefix('```json').removesuffix('```').strip())
                except (ValueError,TypeError):
                    record['technical_error']='invalid_json'
                    self.calls.append(record)
                    raise TechnicalFailure('invalid_json') from None
                if action not in actions:
                    record['technical_error']='action_outside_declared_catalog'
                    self.calls.append(record)
                    raise TechnicalFailure('action_outside_declared_catalog')
                self.calls.append(record)
                return action
            except urllib.error.HTTPError as error:
                record.update(technical_error='http_error',http_status=error.code,elapsed_seconds=monotonic()-start)
                self.calls.append(record)
                raise TechnicalFailure(f'http_status_{error.code}') from None
            except (urllib.error.URLError,TimeoutError,OSError) as error:
                record.update(technical_error=type(error).__name__,elapsed_seconds=monotonic()-start)
                self.calls.append(record)
                raise TechnicalFailure(record['technical_error']) from None


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')


def run(args):
    if not 0 < args.users <= 100:
        raise ValueError('Pilot bounded to 1–100 users per representation')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    source=Path(args.reference_run)
    scenarios=json.loads((source/'evaluator_private/test.json').read_text())[:args.users]
    policy=LLMPrompting(args.base,args.model,token_env=args.token_env,timeout=args.timeout)
    frozen={'created_at':datetime.now(timezone.utc).isoformat(),
        'purpose':'complementary technical pilot; previously inspected synthetic test, no independent confirmatory claim',
        'users_per_representation':len(scenarios),'representations':['O1','O2','O3'],
        'reference_run':str(source),'model':args.model,'temperature':0,'seed':41004,
        'max_completion_tokens':[2048,4096],'retry':'one retry only for empty/token-limited content',
        'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'private_fields_sent':False,'credentials_serialized':False,
        'reward_unchanged_from_reference':True,'hardware_gpu_model_verified':False}
    dump(out/'protocol.json',frozen)
    (out/'prompt.txt').write_text(PROMPT)
    (out/'source_snapshot.py').write_text(Path(__file__).read_text())
    summaries,comparisons,failures={},{},[]
    start=monotonic()
    for representation in ('O1','O2','O3'):
        rows=[]
        with (out/f'{representation}.jsonl').open('w') as file:
            for index,scenario in enumerate(scenarios):
                env=MinimalEnv(scenario,representation=representation)
                first_call=len(policy.calls)
                try:
                    obs=env.reset()
                    for _ in range(8):
                        obs,_,done,truncated,_=env.step(policy.select_action(obs))
                        if done or truncated:
                            break
                    else:
                        raise TechnicalFailure('episode_step_limit')
                    row=env.diagnostic_metrics()
                    row['method']=policy.name
                    row['api_calls']=deepcopy(policy.calls[first_call:])
                    rows.append(row)
                    file.write(json.dumps(row,ensure_ascii=False)+'\n');file.flush()
                    print(f'{representation} {index+1}/{len(scenarios)} error={row["error_rate"]} return={row["return"]}',flush=True)
                except TechnicalFailure as error:
                    failures.append({'representation':representation,'user_id':scenario['user_id'],
                        'family':scenario['family'],'technical_error':str(error),
                        'api_calls':deepcopy(policy.calls[first_call:])})
                    print(f'{representation} {index+1}: technical failure {error}',flush=True)
                finally:
                    env.close()
        summaries[representation]=summarize(rows,1000) if rows else None
        baseline=evaluate(HistoricalClarificationRules(),scenarios,representation)
        matched=[r for r in baseline if r['user_id'] in {v['user_id'] for v in rows}]
        comparisons[representation]={'rules':summarize(baseline,1000),
            'llm_minus_rules_complete_matched_users_only':paired(rows,matched,1000) if rows else None,
            'completed':len(rows),'attempted':len(scenarios)}
        dump(out/'metrics.json',summaries)
        dump(out/'comparisons.json',comparisons)
        dump(out/'technical_failures.json',failures)
    calls=policy.calls
    tokens={field:sum((c.get('usage') or {}).get(field,0) or 0 for c in calls)
            for field in ('prompt_tokens','completion_tokens','total_tokens')}
    tokens['reasoning_tokens']=sum(((c.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens',0) or 0 for c in calls)
    resources={'elapsed_seconds':monotonic()-start,'api_calls':len(calls),
        'technical_failure_episodes':len(failures),'token_usage':tokens,
        'hardware_gpu_model_verified':False,'training_performed':False}
    dump(out/'resources.json',resources)
    lines=['# Prueba del servicio USFQ y piloto B5','',
        'Autenticación Bearer y consulta mínima verificadas. Modelo reportado: `'+args.model+'`. La API permite inferencia; no verifica el modelo de GPU ni concede acceso de entrenamiento.',
        '',f'Piloto complementario: {len(scenarios)} usuarios por condición, cinco familias cuando n≥5. Se usa la evidencia estructurada visible del experimento previo. El prompt se fija antes de los llamados. El test ya había sido inspeccionado: no es evidencia confirmatoria independiente ni generalización conceptual.',
        '', '| Observación | Usuarios completos | Error LLM | Retorno LLM | Error reglas | Retorno reglas |',
        '|---|---:|---:|---:|---:|---:|']
    for rep in ('O1','O2','O3'):
        if summaries[rep]:
            m=summaries[rep]['metrics'];b=comparisons[rep]['rules']['metrics']
            lines.append(f'| {rep} | {comparisons[rep]["completed"]} | {m["error_rate"]["mean"]:.3f} | {m["return"]["mean"]:.3f} | {b["error_rate"]["mean"]:.3f} | {b["return"]["mean"]:.3f} |')
    lines += ['',f'Llamados de inferencia: {len(calls)}. Episodios con error técnico: {len(failures)}. Tokens: `{json.dumps(tokens)}`. Tiempo: {resources["elapsed_seconds"]:.1f} s.',
        '', 'Los errores técnicos quedan separados de errores de decisión. No se sustituyen acciones por reglas y no se envían gold ni respuestas futuras. Las reglas usan el mismo historial permitido; el LLM recibe el texto completo y dispone de mayor capacidad computacional que las políticas tabulares.',
        '', 'La conexión sirve para ejecutar B5. Este piloto pequeño permite detectar comportamientos y problemas de integración; no demuestra ventaja estadística frente a reglas, supervisión o RL. Los resultados históricos permanecen intactos.',
        '', 'Reproducción: configurar `USFQ_API_KEY` en el entorno y ejecutar `python -m rl_observability.llm_pilot --base <base-API-terminada-en-/v1> --output <carpeta-nueva> --users 5`. La clave no se escribe en los artefactos.']
    (out/'INFORME_PILOTO_LLM.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(resources,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True)
    p.add_argument('--model',default='zai-org/GLM-5.3-Flash')
    p.add_argument('--token-env',default='USFQ_API_KEY')
    p.add_argument('--reference-run',default='reports/rl_observability/run_001')
    p.add_argument('--output',required=True)
    p.add_argument('--users',type=int,default=5)
    p.add_argument('--timeout',type=float,default=60)
    run(p.parse_args())


if __name__=='__main__':
    main()
