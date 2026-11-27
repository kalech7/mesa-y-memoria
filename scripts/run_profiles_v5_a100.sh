#!/usr/bin/env bash
# Start inside a server-side tmux session to continue across SSH disconnects.
set -eu
cd "$(dirname "$0")/.."
run=${1:?Supply a run directory as the first argument}
shift
tesis_python=${TESIS_PYTHON:-"$HOME/.local/share/tesis-clarificacion/a100/bin/python"}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$tesis_python" -u -m research.tesis_profiles_v5 train --run "$run" --workers 16 --epochs 2 --batch-size 8 "$@"
if [[ ! -e "$run/RESERVED_RESULTS.json" ]]; then
    "$tesis_python" -u -m research.tesis_profiles_v5 score --run "$run" --workers 12
fi
