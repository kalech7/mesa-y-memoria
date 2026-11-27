"""Optional authenticated M4 inference; credentials never enter artifacts."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
import json
import os
from time import monotonic
import urllib.error
import urllib.request

from rl_observability.llm_pilot import PROMPT,NoRedirect,TechnicalFailure
from .observations import ACTIONS,action_mask
from .environment_adapter import PilotEnv
from .evaluate import summarize


class PromptPolicy:
    name='M4_LLM_PROMPTING'
    def __init__(self,base,key,model='zai-org/GLM-5.3-Flash'):
        self.base,self.key,self.model=base.rstrip('/'),key,model
        self.calls=[]
        self.opener=urllib.request.build_opener(NoRedirect)

    def select_action(self,obs):
        allowed=[a for a,valid in zip(ACTIONS,action_mask(obs)) if valid]
        messages=[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps(
            {'observation':obs,'allowed_actions':allowed},ensure_ascii=False)}]
        for limit in (2048,4096):
            start=monotonic()
            payload={'model':self.model,'messages':messages,'temperature':0,'seed':51005,
                'max_completion_tokens':limit,'stream':False,'response_format':{'type':'json_object'}}
            req=urllib.request.Request(self.base+'/chat/completions',data=json.dumps(payload).encode(),
                headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
            record={'max_completion_tokens':limit}
            try:
                with self.opener.open(req,timeout=60) as response:data=json.load(response)
                choice=data['choices'][0];content=choice['message'].get('content')
                record.update(finish_reason=choice.get('finish_reason'),content=content,
                    model=data.get('model'),usage=data.get('usage'),elapsed_seconds=monotonic()-start)
                if choice.get('finish_reason')=='length' or not content:
                    record['technical_error']='token_limit_or_empty'
                    self.calls.append(record)
                    if limit==2048:continue
                    raise TechnicalFailure('token_limit_or_empty')
                action=json.loads(content.strip().removeprefix('```json').removesuffix('```').strip())
                if action not in allowed:raise TechnicalFailure('action_outside_catalog')
                self.calls.append(record)
                return deepcopy(action)
            except urllib.error.HTTPError as error:
                record.update(technical_error='http_error',http_status=error.code)
                self.calls.append(record)
                raise TechnicalFailure(f'http_{error.code}') from None
            except (OSError,ValueError,KeyError,TypeError) as error:
                record.update(technical_error=type(error).__name__)
                self.calls.append(record)
                raise TechnicalFailure(type(error).__name__) from None


def run_comparator(scenarios,representation,out,base,token_env,workers=2,repetitions=1000):
    root=out/'llm';root.mkdir()
    key=os.environ[token_env]
    opener=urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(urllib.request.Request(base.rstrip('/')+'/models',
            headers={'Authorization':'Bearer '+key}),timeout=8) as response:models=json.load(response)
    except Exception as error:
        return {'status':'PENDING_TECHNICAL_FAILURE','stage':'models_preflight','error_type':type(error).__name__}
    (root/'prompt.txt').write_text(PROMPT)
    summaries={};all_calls=[];failures=[];start=monotonic()
    def episode(scenario,condition):
        policy=PromptPolicy(base,key)
        env=PilotEnv(scenario,representation=representation,condition=condition)
        try:
            obs=env.reset()
            for _ in range(8):
                obs,_,done,truncated,_=env.step(policy.select_action(obs))
                if done or truncated:break
            else:raise TechnicalFailure('episode_budget')
            row=env.diagnostic_metrics();row['api_calls']=policy.calls;row['method']=policy.name
            return row,None
        except TechnicalFailure as error:
            return None,{'user_id':scenario['user_id'],'condition':condition,'technical_error':str(error),'api_calls':policy.calls}
        finally:env.close()
    for condition in ('ORACLE','PIPELINE'):
        rows=[]
        with (root/(condition+'.jsonl')).open('w') as stream:
            with ThreadPoolExecutor(max_workers=max(1,min(workers,4))) as executor:
                futures=[executor.submit(episode,s,condition) for s in scenarios]
                for future in as_completed(futures):
                    row,failure=future.result()
                    if failure:
                        failures.append(failure);all_calls.extend(failure['api_calls'])
                    else:
                        rows.append(row);all_calls.extend(row['api_calls'])
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    print(f'M4 {condition}: completed {len(rows)}, technical failures {sum(f["condition"]==condition for f in failures)} / {len(scenarios)}',flush=True)
        summaries[condition]=summarize(rows,repetitions) if rows else None
        (root/'metrics.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
        (root/'technical_failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
    usage={k:sum((c.get('usage') or {}).get(k,0) or 0 for c in all_calls)
           for k in ('prompt_tokens','completion_tokens','total_tokens')}
    return {'status':'EXECUTED','users_per_condition':len(scenarios),'completed':{
        c:summaries[c]['counts']['users'] if summaries[c] else 0 for c in summaries},
        'models_reported':[v.get('id') for v in models.get('data',[])],
        'api_calls':len(all_calls),'technical_failure_episodes':len(failures),'usage':usage,
        'elapsed_seconds':monotonic()-start,'weights_trained':False,
        'hardware_gpu_verified':False,'same_observations':True}
