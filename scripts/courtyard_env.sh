# Source this in every new Linux shell; no conda activation is required.
set -euo pipefail
export ROOT=/autodl-fs/data/tags_architecture_ablation_20261112
export CODE="$ROOT/code"
export FOLLOW="$ROOT/audit/followup_20261116"
export OLD=/root/autodl-tmp/TAGS/experiments/playground_projection_v2
export DATA=/autodl-fs/data/eth3d_courtyard
export NEW_SCENE="$DATA/tags_courtyard"
export NEW_DEPTH="$DATA/courtyard_mono_prior"
export NEW_OUT="$ROOT/audit/courtyard_projection_v2_confirm_20261124"
export TOOLS="$DATA/tools/courtyard_commands_20261124"
export PIPE="$TOOLS/courtyard_pipeline.py"
export PY=/root/miniconda3/envs/TAGS/bin/python
export PRIOR_PY=/root/miniconda3/envs/dsine/bin/python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$NEW_OUT/audit" "$DATA/tools/weights"
printf 'PY=%s\nPRIOR_PY=%s\nNEW_SCENE=%s\nNEW_OUT=%s\nPIPE=%s\n' "$PY" "$PRIOR_PY" "$NEW_SCENE" "$NEW_OUT" "$PIPE"
