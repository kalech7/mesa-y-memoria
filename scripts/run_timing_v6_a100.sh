#!/usr/bin/env bash
# Execute inside a remote tmux session. A second invocation resumes from completed rounds.
set -euo pipefail
cd "$(dirname "$0")/.."
timing_run=${1:?Indica un directorio de corrida}
timing_python=${TESIS_PYTHON:-"$HOME/.local/share/tesis-clarificacion/a100/bin/python"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
mkdir -p "$timing_run"
exec 9>"$timing_run/.pipeline.lock"
flock -n 9 || { echo 'La corrida ya está activa.' >&2; exit 1; }
"$timing_python" -u -m research.tesis_timing_v6_experiment prepare --run "$timing_run"
if [[ ! -f "$timing_run/TRAIN_RESULT.json" ]]; then
    "$timing_python" -u -m research.tesis_timing_v6_train --run "$timing_run" --seeds 101 202 303 --rounds 2 --batch-size 8 --resume
fi
"$timing_python" -u -m research.tesis_timing_v6_experiment select --run "$timing_run"
"$timing_python" -u -m research.tesis_timing_v6_experiment score --run "$timing_run"
if [[ ! -f "$timing_run/DEMO_PAR.json" ]]; then
    "$timing_python" -u -m research.tesis_timing_v6_demo --run "$timing_run"
fi
