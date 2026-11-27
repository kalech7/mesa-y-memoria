#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_home="${XDG_DATA_HOME:-$HOME/.local/share}/tesis-clarificacion"
runtime="${THESIS_GPU_VENV:-$runtime_home/a100}"
python="$runtime/bin/python"
cd "$project"
export PYTHONDONTWRITEBYTECODE=1

if [[ "${1:-}" == setup ]]; then
    uv="${THESIS_UV:-$runtime_home/bin/uv}"
    if [[ ! -x "$uv" ]]; then
        uv="$(command -v uv)"
    fi
    "$uv" venv --allow-existing --python python3 "$runtime"
    "$uv" pip install --python "$python" --no-progress \
        --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1'
    "$uv" pip install --python "$python" --no-progress \
        'transformers==4.49.0' 'peft==0.14.0' 'accelerate==1.6.0' 'safetensors==0.5.3'
    "$uv" pip freeze --python "$python" > "$runtime/dependency-versions.txt"
    exit
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.__version__, flush=True)'
if [[ "${1:-}" == matched ]]; then
    shift
    exec "$python" -u -m research.tesis_shared_budget_qwen_compare "$@"
fi
if [[ "${1:-}" == matched-context ]]; then
    shift
    exec "$python" -u -m research.tesis_shared_budget_qwen_context_run "$@"
fi
exec "$python" -u -m research.tesis_shared_budget_qwen \
    --model models/Qwen2.5-0.5B-Instruct "$@"
