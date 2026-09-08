poetry run python -m v3.train \
  --exp-name v3-state \
  --total-timesteps 400000 \
  --max-episode-steps 1000 \
  --num-envs 1 \
  --rollout-steps 2048 \
  --update-epochs 8 \
  --minibatch-size 256 \
  --reach-reward-scale 1.0 \
  --mini-eval-episodes 5 \
  "$@"
