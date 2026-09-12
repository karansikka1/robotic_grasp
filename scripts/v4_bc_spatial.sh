#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
POETRY="${POETRY:-poetry}"

"$POETRY" run python -m v4.train_bc \
  --policy transformer --rgb-backbone resnet18 --camera-fusion concat --rgb-pool-size 4 \
  --exp-name v4-diverse100-resnet18-concat-grid4 --split-manifest v4/splits/diverse100.json \
  --epochs 30 --batch-size 64 --history-length 3 --image-size 128 --device cuda \
  --train-rollout-episodes 5 --rollout-eval-episodes 5 "$@"
