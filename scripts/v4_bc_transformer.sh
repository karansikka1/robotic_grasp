#!/usr/bin/env bash
set -euo pipefail

bash scripts/v4_bc_train.sh \
  --exp-name v4-transformer-bc \
  --policy transformer \
  --history-length 3 \
  --transformer-layers 2 \
  --transformer-heads 4 \
  --transformer-feedforward-dim 512 \
  --transformer-dropout 0 \
  --split-manifest v4/splits/default.json \
  "$@"
