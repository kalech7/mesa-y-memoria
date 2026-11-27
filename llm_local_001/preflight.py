"""Offline, evidence-based audit; never installs, downloads, or calls inference APIs.

Only loopback Ollama inventory endpoints are read. GGUF and cloud pointers are
listed for transparency, never converted into a surrogate training experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = ('torch', 'transformers', 'peft', 'trl', 'accelerate', 'datasets',
            'bitsandbytes', 'safetensors', 'tokenizers', 'numpy')
PROBE = r'''
import json,sys,importlib.metadata as metadata
r={'executable':sys.executable,'prefix':sys.prefix,'python':sys.version,'packages':{}}
for name in PACKAGES:
 try:r['packages'][name]=metadata.version(name)
 except metadata.PackageNotFoundError:r['packages'][name]=None
try:
 import torch
 r['torch']={'version':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
 if torch.cuda.is_available():
  r['torch']['devices']=[{'name':torch.cuda.get_device_name(i),'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory,'free_memory_bytes':torch.cuda.mem_get_info(i)[0],'capability':list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]
  r['torch']['bf16_supported']=torch.cuda.is_bf16_supported()
except Exception as e:r['torch_error']=repr(e)
if r['packages'].get('transformers'):
 try:
  from transformers import AutoTokenizer,AutoModelForCausalLM
  r['causal_loader_importable']=True
 except Exception as e:r['causal_loader_error']=repr(e)
print(json.dumps(r))
'''
TENSOR_SMOKE = r'''
import torch,tempfile,pathlib,json
r={'scope':'tiny CUDA tensor only; this does NOT prove LLM trainability'}
x=torch.tensor([1.,2.,3.],device='cuda',requires_grad=True)
loss=x.square().sum();loss.backward()
r.update(device=str(x.device),loss=float(loss.detach().cpu()),gradient=x.grad.detach().cpu().tolist(),finite=bool(torch.isfinite(x.grad).all().item()))
with tempfile.TemporaryDirectory(prefix='llm_local_001_tensor_') as d:
 p=pathlib.Path(d)/'tensor.pt';torch.save({'x':x.detach().cpu(),'gradient':x.grad.cpu()},p)
 y=torch.load(p,weights_only=True)
 r['checkpoint_save_load']=torch.equal(y['x'],x.detach().cpu());r['checkpoint_bytes']=p.stat().st_size
print(json.dumps(r))
'''
MODEL_SMOKE = r'''
import os,json,sys,tempfile,pathlib,time
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
p=sys.argv[1];plan=json.loads(sys.argv[2]);started=time.monotonic()
r={'execution_plan':plan,'trained_parameter_scope':plan['parameter_scope'],'max_smoke_tokens':32}
try:
 tok=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False)
 dtype={'float32':torch.float32,'bfloat16':torch.bfloat16}[plan['precision']]
 model=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,trust_remote_code=False,torch_dtype=dtype)
 if getattr(model.config,'is_encoder_decoder',False) or getattr(model,'is_quantized',False):
  raise ValueError('Only unquantized causal decoder checkpoints are audited by this smoke')
 if plan['parameter_scope']=='lora_adapters_integrated_in_llm':
  from peft import LoraConfig,get_peft_model
  model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0.,bias='none'))
  base_parameters=[(name,q) for name,q in model.named_parameters() if 'lora_' not in name]
  if any(q.requires_grad for _,q in base_parameters):raise RuntimeError('LoRA must freeze every base parameter')
 else:
  for q in model.parameters():q.requires_grad_(True)
  base_parameters=[]
 device=plan['device'];model.to(device);model.train();model.config.use_cache=False
 base_versions={name:(q.data_ptr(),q._version) for name,q in base_parameters}
 batch=tok('Preferencias para la cena: antes evitaba queso, ahora lo prefiero.',return_tensors='pt',truncation=True,max_length=32)
 batch={k:v.to(device) for k,v in batch.items()};out=model(**batch,labels=batch['input_ids']);out.loss.backward()
 grads=[q.grad for q in model.parameters() if q.grad is not None]
 finite=bool(grads) and all(bool(torch.isfinite(g).all().item()) for g in grads)
 nonzero=any(bool(torch.count_nonzero(g).item()) for g in grads)
 unchanged=all((q.data_ptr(),q._version)==base_versions[name] and q.grad is None for name,q in base_parameters)
 r.update(device=device,precision=plan['precision'],loss=float(out.loss.detach().cpu()),parameters=sum(q.numel() for q in model.parameters()),trainable_parameters=sum(q.numel() for q in model.parameters() if q.requires_grad),parameters_with_grad=sum(q.numel() for q in model.parameters() if q.grad is not None),finite_gradients=finite,nonzero_gradients=nonzero,base_weights_unchanged=unchanged if base_parameters else None,base_verification='frozen parameters, absent gradients, unchanged tensor storage and version counters' if base_parameters else None,tokenizer_class=type(tok).__name__)
 if not finite or not nonzero:raise RuntimeError('No finite nonzero causal-LLM gradients')
 if not unchanged:raise RuntimeError('Frozen base weights changed or received gradients')
 with tempfile.TemporaryDirectory(prefix='llm_local_001_model_') as d:
  model.save_pretrained(d,safe_serialization=True);tok.save_pretrained(d)
  files=[f for f in pathlib.Path(d).iterdir() if f.is_file()]
  r['checkpoint_bytes']=sum(f.stat().st_size for f in files);r['checkpoint_files']=[f.name for f in files]
  if base_parameters and not any(f.name.startswith('adapter_model') for f in files):raise RuntimeError('Adapter checkpoint missing')
  r['checkpoint_saved']=True
 r['status']='passed'
except Exception as e:r.update(status='failed',error=repr(e))
r['elapsed_seconds']=time.monotonic()-started
print(json.dumps(r))
'''


def _run(args: list[str], timeout: int = 30) -> dict:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                                'HF_HUB_DISABLE_TELEMETRY': '1'})
        return {'returncode': p.returncode, 'stdout': p.stdout[-200000:], 'stderr': p.stderr[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def _json(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else {'value': obj}
    except (OSError, ValueError) as exc:
        return {'read_error': str(exc)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _discover() -> tuple[list[Path], list[Path], dict]:
    """Bounded scan with explicit exclusions; never claims exhaustive filesystem coverage."""
    home = Path.home()
    interpreters = [Path(sys.executable)]
    for name in ('python', 'python3', 'python3.11', 'python3.12', 'python3.13', 'python3.14'):
        if found := shutil.which(name):
            interpreters.append(Path(found))
    for root in (home/'miniconda3', home/'anaconda3', home/'.local/share/uv/python'):
        interpreters.extend(root.glob('bin/python*'))
        interpreters.extend(root.glob('*/bin/python*'))
        interpreters.extend(root.glob('envs/*/bin/python*'))
    roots = [home, Path('/opt/models'), Path('/models'), Path('/mnt/models')]
    model_dirs: list[Path] = []
    skip = {'node_modules', '.git', '.npm', '.rustup', '.cargo', '.mozilla', '.cache',
            '.local', '.config', '.steam', '.docker', '.var', 'site-packages', 'pkgs',
            'target', '__pycache__', '.gradle', 'venv-lib', 'lib64'}
    unreadable: list[str] = []
    for base in roots:
        if not base.exists():
            continue
        for directory, dirs, files in os.walk(base, onerror=lambda e: unreadable.append(str(e))):
            depth = len(Path(directory).relative_to(base).parts)
            dirs[:] = [d for d in dirs if d not in skip and depth < 7]
            p = Path(directory)
            if 'pyvenv.cfg' in files:
                interpreters.append(p/'bin/python')
                dirs[:] = [d for d in dirs if d not in {'lib', 'bin', 'include', 'share'}]
            if 'config.json' in files:
                cfg = _json(p/'config.json')
                if 'model_type' in cfg or 'architectures' in cfg:
                    model_dirs.append(p)
    cache_roots = [home/'.cache/huggingface/hub']
    for name in ('HF_HOME', 'HUGGINGFACE_HUB_CACHE', 'HF_HUB_CACHE'):
        if os.environ.get(name):
            p = Path(os.environ[name]).expanduser()
            cache_roots.extend([p, p/'hub'])
    for root in cache_roots:
        model_dirs.extend(p.parent for p in root.glob('models--*/snapshots/*/config.json'))
    unique_interpreters = []
    seen = set()
    for p in interpreters:
        # Keep virtual environments distinct, even when their executable symlinks match.
        if p.is_file() and 'config' not in p.name and str(p) not in seen:
            seen.add(str(p)); unique_interpreters.append(p)
    return unique_interpreters, sorted(set(model_dirs)), {
        'roots': [str(x) for x in roots], 'hf_cache_roots': [str(x) for x in cache_roots],
        'max_depth': 7, 'excluded_directory_names': sorted(skip),
        'unreadable_paths': unreadable[:100],
        'scope': 'Known caches and bounded home/model-root scan; inaccessible locations explicitly recorded.'}


def _model_info(path: Path) -> dict:
    cfg = _json(path/'config.json')
    weights = sorted({*path.glob('*.safetensors'), *path.glob('pytorch_model*.bin')})
    tokenizer_files = [p for p in path.iterdir() if p.name in {'tokenizer.json', 'tokenizer_config.json',
                       'vocab.json', 'vocab.txt', 'merges.txt', 'tokenizer.model', 'spiece.model', 'special_tokens_map.json', 'sentencepiece.bpe.model'}]
    arch = cfg.get('architectures', [])
    causal = any('CausalLM' in str(a) or 'LMHeadModel' in str(a) for a in arch)
    readme = path/'README.md'
    license_files = [p for p in path.iterdir() if p.is_file() and p.name.lower().startswith(('license', 'copying'))]
    license_evidence = []
    for p in license_files + ([readme] if readme.is_file() else []):
        data = p.read_text(errors='replace')
        if p == readme:
            relevant = '\n'.join(line for line in data.splitlines()[:100] if 'license' in line.lower())
        else:
            relevant = data[:2000]
        if relevant.strip():
            license_evidence.append({'path': str(p), 'sha256': _sha(p), 'excerpt': relevant})
    return {'source': 'local_huggingface_directory', 'path': str(path), 'config': cfg,
            'config_sha256': _sha(path/'config.json'), 'architectures': arch,
            'is_causal_candidate': causal, 'weight_format': sorted({p.suffix for p in weights}),
            'weight_files': [{'path': str(p), 'bytes': p.stat().st_size} for p in weights if p.exists()],
            'weight_bytes': sum(p.stat().st_size for p in weights if p.exists()),
            'tokenizer_files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': _sha(p)} for p in tokenizer_files],
            'tokenizer_config': _json(path/'tokenizer_config.json') if (path/'tokenizer_config.json').exists() else None,
            'license_evidence': license_evidence, 'license_review': 'local evidence only; not legal advice',
            'trainability': 'candidate_requires_load_gradient_checkpoint_smoke' if causal and weights and tokenizer_files else 'not_a_complete_local_causal_llm',
            'reason': None if causal else 'Embedding, reranking, encoder, or speech architecture cannot substitute for the causal LLM.'}


def _ollama_inventory() -> dict:
    url = 'http://127.0.0.1:11434'
    result: dict = {'inventory_endpoint': url+'/api/tags', 'inference_calls': 0, 'models': []}
    try:
        # No environment proxy may redirect even inventory traffic outside loopback.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url+'/api/tags', timeout=5) as response:
            models = json.load(response).get('models', [])
        for model in models:
            entry = dict(model)
            cloud = bool(model.get('remote_host') or model.get('remote_model') or model.get('name', '').endswith(':cloud'))
            entry['excluded_cloud'] = cloud
            entry['trainability'] = 'excluded_remote_service' if cloud else 'not_verified_trainable_GGUF'
            if not cloud:
                req = urllib.request.Request(url+'/api/show', data=json.dumps({'model': model['name'], 'verbose': False}).encode(), headers={'Content-Type': 'application/json'})
                try:
                    with opener.open(req, timeout=15) as response:
                        details = json.load(response)
                    info = details.get('model_info', {})
                    entry['model_configuration'] = {k: v for k, v in info.items() if not isinstance(v, list)}
                    entry['tokenizer_arrays'] = {k: {'length': len(v), 'sha256': hashlib.sha256(json.dumps(v).encode()).hexdigest()} for k, v in info.items() if isinstance(v, list) and k.startswith('tokenizer.')}
                    entry['license_text'] = details.get('license')
                    entry['modelfile'] = details.get('modelfile')
                    entry['reason'] = 'Ollama GGUF inference artifact; no compatible local differentiable training implementation or original trainable checkpoint verified.'
                except Exception as exc:
                    entry['show_error'] = repr(exc)
            result['models'].append(entry)
    except Exception as exc:
        result['inventory_error'] = repr(exc)
    return result


def _parameter_estimate(candidate: dict) -> tuple[int, str]:
    """Read safetensors shapes without loading weights; otherwise use dtype/bytes."""
    count = 0
    files = candidate.get('weight_files', [])
    if files and all(Path(f['path']).suffix == '.safetensors' for f in files):
        try:
            for entry in files:
                with Path(entry['path']).open('rb') as stream:
                    size = struct.unpack('<Q', stream.read(8))[0]
                    if size > 32 * 1024 * 1024:
                        raise ValueError('Unusually large safetensors header')
                    header = json.loads(stream.read(size))
                count += sum(math.prod(v['shape']) for k, v in header.items() if k != '__metadata__')
            if count > 0:
                return count, 'safetensors tensor shapes (shared tensors may be counted separately)'
        except (OSError, ValueError, KeyError, TypeError, struct.error):
            pass
    config = candidate.get('config', {})
    dtype = str(config.get('torch_dtype', config.get('dtype', 'float16')))
    bytes_per_parameter = 4 if dtype in {'float32', 'torch.float32'} else 2
    return math.ceil(candidate['weight_bytes'] / bytes_per_parameter), 'file bytes / declared dtype width; 2-byte assumption if unspecified'


def _smoke_plan(candidate: dict, environment: dict, ram: dict,
                training_mode: str = 'auto', device: str | None = None) -> dict:
    """Conservative allocation estimate, not an empirical impossibility claim.

    Includes an optimizer allowance even though this audit performs only backward.
    A 20% free-memory reserve avoids claiming every free byte can be allocated.
    Actual trainability still requires the selected-mode smoke; its 32-token
    context is deliberately not proof that the entire experiment will fit.
    """
    if training_mode not in {'auto', 'lora', 'full'} or device not in {None, 'cuda', 'cpu'}:
        raise ValueError('training_mode must be auto/lora/full and device cuda/cpu/None')
    has_peft = bool(environment.get('packages', {}).get('peft'))
    mode = 'lora' if training_mode == 'auto' and has_peft else 'full' if training_mode == 'auto' else training_mode
    if mode == 'lora' and not has_peft:
        return {'status': 'blocked', 'reason': 'Selected LoRA mode needs a compatible existing PEFT installation.'}
    torch_info = environment.get('torch', {})
    parameters, source = _parameter_estimate(candidate)
    devices = [device] if device else (['cuda', 'cpu'] if torch_info.get('cuda_available') else ['cpu'])
    attempts = []
    for execution_device in devices:
        if execution_device == 'cuda' and not torch_info.get('cuda_available'):
            attempts.append({'device': 'cuda', 'reason': 'PyTorch CUDA unavailable'})
            continue
        gpu = (torch_info.get('devices') or [{}])[0]
        available = gpu.get('free_memory_bytes', 0) if execution_device == 'cuda' else ram.get('MemAvailable_bytes', 0)
        precision = 'bfloat16' if execution_device == 'cuda' and torch_info.get('bf16_supported') else 'float32'
        element_size = 2 if precision == 'bfloat16' else 4
        # 2% is a documented conservative planning allowance, not the actual
        # LoRA parameter count. The smoke records the true count separately.
        trainable = math.ceil(parameters * .02) if mode == 'lora' else parameters
        weights = parameters * element_size
        gradient_optimizer = trainable * 16
        activation_reserve = max(64 * 1024 * 1024, weights // 10)
        estimate = weights + gradient_optimizer + activation_reserve
        # from_pretrained first materializes CPU weights before .to(cuda).
        host_estimate = weights * 2
        host_available = ram.get('MemAvailable_bytes', 0)
        attempt = {'device': execution_device, 'precision': precision,
                   'parameter_scope': 'lora_adapters_integrated_in_llm' if mode == 'lora' else 'base_weights',
                   'parameter_estimate': parameters, 'parameter_estimate_source': source,
                   'trainable_parameter_estimate': trainable, 'lora_fraction_assumption': .02 if mode == 'lora' else None,
                   'weight_allocation_bytes': weights, 'gradient_and_optimizer_allowance_bytes': gradient_optimizer,
                   'activation_reserve_bytes': activation_reserve, 'estimated_peak_bytes': estimate,
                   'available_memory_bytes': available, 'memory_utilization_limit': .8,
                   'host_loading_estimate_bytes': host_estimate, 'host_available_bytes': host_available,
                   'scope': '32-token batch-one backward/checkpoint smoke; full training context and RL copies need additional runtime checks'}
        if available and estimate <= available * .8 and host_available and host_estimate <= host_available * .8:
            return {**attempt, 'status': 'planned', 'attempts': attempts}
        attempts.append({**attempt, 'reason': 'Estimated allocation exceeds reserve or measured free memory is unavailable; feasibility remains unverified.'})
    return {'status': 'blocked', 'reason': 'No requested execution mode fits the documented conservative estimate; this is not a measured OOM.', 'attempts': attempts}


def run_preflight(output: Path, model_path: str | None = None, *,
                  training_mode: str = 'auto', device: str | None = None) -> dict:
    """Write two provenance-rich audits and return a fail-closed experiment gate."""
    started = time.monotonic()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    provenance = {'spec': 'SPEC-LLM-LOCAL-001', 'created_at_utc': stamp,
                  'producer': 'llm_local_001.preflight.run_preflight', 'code_sha256': _sha(Path(__file__)),
                  'configuration': {'model_path': model_path, 'training_mode': training_mode, 'device': device,
                                    'network_policy': 'No external requests; only loopback Ollama inventory; no inference, installation or downloads.'}}
    interpreters, paths, scan = _discover()
    if model_path:
        supplied = Path(model_path).expanduser().absolute()
        paths = [supplied] + [p for p in paths if p != supplied]
    environments = []
    seen = set()
    for interpreter in interpreters:
        probe = _run([str(interpreter), '-c', 'PACKAGES='+repr(PACKAGES)+'\n'+PROBE], timeout=35)
        try:
            info = json.loads(probe.get('stdout', '').splitlines()[-1])
            key = (info['prefix'], info['python'])
            if key in seen:
                continue
            seen.add(key); info['requested_executable'] = str(interpreter)
            if probe.get('stderr'):
                info['probe_stderr'] = probe['stderr']
        except (ValueError, IndexError, KeyError):
            info = {'requested_executable': str(interpreter), 'probe_failure': probe}
        environments.append(info)
    compatible = [e for e in environments if e.get('causal_loader_importable') and e.get('torch')]
    compatible.sort(key=lambda e: (not e['torch'].get('cuda_available', False), not e['packages'].get('peft')))
    selected_env = compatible[0] if compatible else None
    cuda_envs = [e for e in environments if e.get('torch', {}).get('cuda_available')]
    cuda_smoke = None
    if cuda_envs:
        env = selected_env if selected_env and selected_env['torch'].get('cuda_available') else cuda_envs[0]
        raw = _run([env['requested_executable'], '-c', TENSOR_SMOKE], timeout=35)
        try:
            cuda_smoke = {**json.loads(raw['stdout'].splitlines()[-1]), 'python': env['requested_executable']}
        except (KeyError, IndexError, ValueError):
            cuda_smoke = {'failure': raw}
    mem = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'}:
                mem[key+'_bytes'] = int(value.strip().split()[0])*1024
    except OSError:
        pass
    smi = _run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits'])
    hardware = {**provenance, 'os': {'platform': platform.platform(), 'release': platform.release(), 'os_release': platform.freedesktop_os_release()},
                'cpu': {'logical_count': os.cpu_count(), 'architecture': platform.machine(), 'lscpu': _run(['lscpu'])},
                'ram': mem, 'gpu_nvidia_smi': smi, 'python_environments': environments,
                'cuda_tensor_gradient_checkpoint_smoke': cuda_smoke, 'discovery': scan}
    models = []
    for path in paths:
        try:
            models.append(_model_info(path))
        except (OSError, ValueError) as exc:
            models.append({'path': str(path), 'inspection_error': str(exc)})
    ollama = _ollama_inventory()
    candidates = [m for m in models if m.get('is_causal_candidate') and m.get('weight_bytes') and m.get('tokenizer_files')]
    if model_path:
        candidates = [m for m in candidates if m['path'] == str(Path(model_path).expanduser().absolute())]
    candidates.sort(key=lambda m: m['weight_bytes'])
    reasons = []
    selected = None
    if not candidates:
        reasons.append('No complete local trainable causal-language-model checkpoint with tokenizer was found. Cached encoder/reranker/speech models and Ollama GGUF/cloud references are not substitutes.')
    if not selected_env:
        reasons.append('No existing Python environment successfully imported both PyTorch and the Transformers causal-model loader.')
    for candidate in candidates if selected_env else []:
        if not candidate.get('license_evidence'):
            candidate['trainability'] = 'blocked_missing_local_license_evidence'
            continue
        plan = _smoke_plan(candidate, selected_env, mem, training_mode, device)
        candidate['smoke_execution_plan'] = plan
        if plan['status'] != 'planned':
            candidate['trainability'] = 'not_verified_execution_plan_blocked'
            continue
        raw = _run([selected_env['requested_executable'], '-c', MODEL_SMOKE,
                    candidate['path'], json.dumps(plan)], timeout=175)
        try:
            candidate['gradient_checkpoint_smoke'] = json.loads(raw['stdout'].splitlines()[-1])
        except (KeyError, ValueError, IndexError):
            candidate['gradient_checkpoint_smoke'] = {'status': 'failed', 'process': raw}
        if candidate['gradient_checkpoint_smoke'].get('status') == 'passed':
            selected = candidate
            candidate['trainability'] = 'verified_causal_llm_gradients_and_checkpoint'
            break
    if candidates and selected_env and not selected:
        reasons.append('Local causal candidates did not pass memory, local-license-evidence, gradient and checkpoint gates; inspect per-model details.')
    status = 'ready' if selected else 'blocked'
    manifest = {**provenance, 'status': status, 'reasons': reasons, 'selected_model': selected,
                'selected_python': selected_env['requested_executable'] if selected_env else None,
                'models': models, 'ollama': ollama,
                'trainable_parameter_statement': 'No LLM parameters were trained during this resource audit. A model smoke, if present, performs backward only; tensor smoke is not LLM evidence.',
                'concrete_requirements': [
                    'Provide an existing local Hugging Face causal LM directory with config, compatible tokenizer, complete safetensors/PyTorch weights and license evidence, or explicitly authorize a model download.',
                    'Reuse an environment whose causal-model loader imports successfully. LoRA additionally requires a compatible existing peft installation or explicit authorization to install it.',
                    'Verify actual forward/backward and checkpoint saving within measured CPU/GPU memory before any B0/B1/B2/B3 results; 4 GiB GPU inference capacity does not establish training feasibility.'
                ] if status == 'blocked' else [],
                'elapsed_seconds': time.monotonic()-started}
    hardware['preflight_status'] = status
    for filename, data in [('hardware_audit.json', hardware), ('model_manifest.json', manifest)]:
        target = output/filename
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(target)
    return {'status': status, 'reasons': reasons, 'hardware': hardware, 'models': models,
            'selected_model': selected, 'selected_python': manifest['selected_python'],
            'model_manifest': manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/llm_local_001'))
    parser.add_argument('--model-path')
    parser.add_argument('--training-mode', choices=('auto', 'lora', 'full'), default='auto')
    parser.add_argument('--device', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    result = run_preflight(args.output, args.model_path, training_mode=args.training_mode, device=args.device)
    print(json.dumps({k: result[k] for k in ('status', 'reasons', 'selected_python')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
