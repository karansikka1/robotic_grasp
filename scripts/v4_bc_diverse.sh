#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
POETRY="${POETRY:-poetry}"
SPLIT=v4/splits/diverse100.json

"$POETRY" run python -m v4.collect_trajectories --target-total 100 --seed 20260909 --workers 4
if [[ ! -f "$SPLIT" ]]; then
  "$POETRY" run python -m v4.data --output "$SPLIT" \
    --validation-fraction 0.1 --test-fraction 0.1 --seed 20260909
fi
"$POETRY" run python -m v4.report_diversity "$SPLIT" --reference-manifest v4/splits/default.json
for backbone in mobilenet_v3_small resnet18; do
  "$POETRY" run python -m v4.train_bc --policy transformer --rgb-backbone "$backbone" \
    --exp-name "v4-diverse100-$backbone" --split-manifest "$SPLIT" \
    --epochs 30 --batch-size 64 --history-length 3 --device cuda \
    --train-rollout-episodes 5 --rollout-eval-episodes 5
done
