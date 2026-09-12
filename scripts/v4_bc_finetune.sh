#!/usr/bin/env bash
set -euo pipefail

bash scripts/v4_bc_train.sh \
  --exp-name v4-history-bc-finetune \
  --finetune-backbone \
  --backbone-learning-rate 0.00001 \
  --split-manifest v4/splits/default.json \
  "$@"
