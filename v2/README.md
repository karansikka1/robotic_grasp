# v2: grasp and lift green

This experiment reuses v1's frozen MobileNet RGB/depth encoders and PPO optimizer.
It trains on the same scene but targets only the green object, cubeB.

## Reward and success

- +1 for the first grasp: both finger contact groups touch green.
- +5 for lifting green at least 5 cm above its post-reset resting height while
  maintaining grasp for 5 consecutive control steps (0.25 seconds).
- Each bonus is paid once per episode. Dropping the object or going below the
  target height resets the consecutive hold counter. Success ends the episode.
- No distance reward or step penalty. Default episode returns are 0, 1, or 6.
  Distance shaping is reserved for a possible v3.

The zero-action bootstrap after reset earns no reward, contributes no hold steps,
and establishes the resting-height reference. The task adapter reads green-object
poses and finger contacts to compute rewards. These fields are not policy inputs.
The policy still uses privileged depth, like v1.

## Run

From the repository root:

    bash scripts/v2_smoke_train.sh

The smoke script uses 2,048 total steps, 256 rollout steps, four update epochs,
and minibatches of 256. It accepts extra training arguments. For a longer run:

    bash scripts/v2_smoke_train.sh --exp-name v2-green-lift --total-timesteps 100000 --rollout-steps 1024 --evaluate-untrained

The optional untrained evaluation provides a before-training comparison on the
same seeds as final evaluation. With no successes yet, zero return is expected.

Task parameters are configurable:

    poetry run python -m v2.train --grasp-reward 1 --lift-reward 5 --lift-height-m 0.05 --hold-steps 5

The KL penalty defaults to zero; PPO runs all configured optimization epochs.

## Artifacts and evaluation

- Model, config, terminal log, TensorBoard: v2/runs/&lt;experiment&gt;-&lt;uuid&gt;/
- Mini evaluations: evaluation/v2/mini/&lt;experiment&gt;-step-&lt;step&gt;-&lt;uuid&gt;/
- Final evaluation: evaluation/v2/&lt;experiment&gt;-&lt;uuid&gt;/
- Optional initial evaluation: evaluation/v2/untrained/&lt;experiment&gt;-&lt;uuid&gt;/

Automatic evaluations share the training UUID. By default mini evaluation runs
on seeds 0–4 every 10 updates, and final evaluation runs on seeds 0–24. Each
evaluation records videos. Use --mini-eval-interval-updates 0 to disable mini
evaluation, or --skip-evaluation to disable all evaluations.

Evaluation success means a held green lift, not the full stack. Metrics include
grasp rate, lift success rate, steps to grasp/lift, maximum lift height, and reward
components. Training logs per-episode task metrics and evaluation summaries to
stdout and TensorBoard.

    poetry run tensorboard --logdir v2/runs
    poetry run python -m v2.evaluate v2/runs/<run>/checkpoint_final.pt --name v2-held-out --seed 100 --episodes 25

Standalone evaluation restores task thresholds from the checkpoint and creates
a fresh evaluation UUID so repeated evaluations do not overwrite each other.
Checkpoints retain the v1 model serialization format; task metadata identifies v2.

## Validation

    poetry run python -m unittest v2.test_task

For a short real rendering/training check, including resuming after mini evaluation:

    poetry run python -m v2.train --exp-name v2-check --total-timesteps 4 --rollout-steps 2 --minibatch-size 2 --update-epochs 2 --max-episode-steps 2 --mini-eval-interval-updates 1 --mini-eval-episodes 1 --final-eval-episodes 1 --no-pretrained

This checks execution only: two-step episodes cannot achieve the default
five-step held lift.
