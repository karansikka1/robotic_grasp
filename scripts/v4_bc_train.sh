#!/usr/bin/env bash
set -euo pipefail

poetry run python -m v4.train_bc \
  --exp-name v4-history-bc \
  --policy history \
  --history-length 3 \
  --epochs 30 \
  --batch-size 64 \
  --validation-fraction 0.2 \
  --test-fraction 0.2 \
  --rollout-eval-episodes 5 \
  "$@"
