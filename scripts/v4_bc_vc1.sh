#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export VC1_CHECKPOINT="${VC1_CHECKPOINT:-/workspace/.cache/vc1-large/pytorch_model.bin}"
POETRY="${POETRY:-poetry}"
"$POETRY" run python -m v4.train_bc \
  --policy transformer --rgb-backbone vc1_vitl --camera-fusion concat --rgb-pool-size 1 \
  --no-previous-action --history-length 3 --image-size 128 \
  --exp-name v4-vc1-vitl-cls-no-prev --split-manifest v4/splits/diverse100.json \
  --epochs 30 --batch-size 64 --feature-batch-size 16 --device cuda \
  --train-rollout-episodes 5 --rollout-eval-episodes 5 "$@"
