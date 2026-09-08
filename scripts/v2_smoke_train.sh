poetry run python -m v2.train \
  --exp-name v2-history \
  --total-timesteps 200000 \
  --rollout-steps 1024 \
  --update-epochs 8 \
  --minibatch-size 256 \
  --reach-reward-scale 1.0 \
  --policy history \
  --history-length 3 \
  --mini-eval-episodes 10 \
  "$@"
