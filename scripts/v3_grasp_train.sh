# New reward experiment, initialized from the preceding state-policy run.
# Later CLI arguments can override the checkpoint or any training setting.
poetry run python -m v3.train \
  --exp-name v3-maintained-grasp \
  --init-checkpoint v3/runs/v3-state-20d9d6e6-320d-4241-a047-f0b7b965ca26/checkpoints/step_000204800.pt \
  --grasp-hold-steps 5 \
  --total-timesteps 400000 \
  --max-episode-steps 1000 \
  --num-envs 4 \
  --rollout-steps 2048 \
  --update-epochs 8 \
  --minibatch-size 256 \
  --reach-reward-scale 1.0 \
  --mini-eval-episodes 5 \
  --eval-max-steps 500 \
  --evaluate-untrained \
  "$@"
