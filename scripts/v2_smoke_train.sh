poetry run python -m v2.train \
  --exp-name v2-smoke \
  --total-timesteps 100000 \
  --rollout-steps 1024 \
  --update-epochs 4 \
  --minibatch-size 256