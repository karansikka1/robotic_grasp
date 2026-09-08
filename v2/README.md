# v2: grasp and lift green

This experiment reuses v1's frozen MobileNet RGB/depth encoders and PPO optimizer.
It trains on the same scene but targets only the green object, cubeB.

## Observation history

New runs default to --policy history --history-length 3: the current observation,
the previous two observations, and the previous 7-D action. The frozen encoders
process each new observation once; cached RGB/depth/proprioceptive features are
assembled oldest to newest. Per-frame embeddings are concatenated with the previous
action and passed through a learned fusion layer before the actor and value heads.

At episode reset, missing history repeats the first observation and the previous
action is zero (matching the zero-action bootstrap). History survives PPO update
boundaries but never crosses episode boundaries. Evaluation has separate history
and resets it for every episode. Three frames at 20 Hz cover 0.1 seconds.

Use --policy single to reproduce the earlier single-observation model, without
previous-action input. The reward and PPO settings are the same for both variants.
New temporal runs train a new model; the old single-observation checkpoints remain
loadable for evaluation with their original architecture.


## Reward and success

Every policy step adds a distance-progress term:

    reach_reward = reach_reward_scale * (previous_distance_m - current_distance_m)

Distance is measured from the gripper's grip site (between the fingers) to green's
body center. The default scale is 1.0 reward per meter: approaching by 1 cm earns
+0.01, retreating by 1 cm earns -0.01, and hovering earns zero. The undiscounted
reaching return equals scale times the initial-to-final distance reduction;
moving away and back adds no net reaching reward. This is a progress heuristic,
not a claim of policy invariance under PPO's discounted objective.


- +1 for the first grasp: both finger contact groups touch green.
- +5 for lifting green at least 5 cm above its post-reset resting height while
  maintaining grasp for 5 consecutive control steps (0.25 seconds).
- Each bonus is paid once per episode. Dropping the object or going below the
  target height resets the consecutive hold counter. Success ends the episode.
- No constant step penalty. Total episode return is reaching return plus the
  one-time milestone bonuses (0, 1, or 6). Reaching alone never counts as success.

The zero-action bootstrap after reset earns no reward, contributes no hold steps,
and establishes the resting-height and initial-distance references. The task adapter reads green-object
poses and finger contacts to compute rewards. These fields are not policy inputs.
The policy still uses privileged depth, like v1.

## Run

From the repository root:

    bash scripts/v2_smoke_train.sh

The script currently uses 200,000 total steps, 1,024 rollout steps, eight update
epochs, and minibatches of 256. It names new runs v2-history and accepts extra training arguments. To include
an untrained-policy comparison:

    bash scripts/v2_smoke_train.sh --exp-name v2-green-lift --total-timesteps 100000 --rollout-steps 1024 --evaluate-untrained

The optional untrained evaluation provides a before-training comparison on the
same seeds as final evaluation. Reaching rewards can be nonzero before any grasp.

Task parameters are configurable:

    poetry run python -m v2.train --reach-reward-scale 1.0 --grasp-reward 1 --lift-reward 5 --lift-height-m 0.05 --hold-steps 5

Use --reach-reward-scale 0 to reproduce the original sparse v2 reward. Restart
training to use the changed reward; an already running process keeps its old code.

The KL penalty defaults to zero; PPO runs all configured optimization epochs.

## Artifacts and evaluation

- Model, config, terminal log, TensorBoard: v2/runs/&lt;experiment&gt;-&lt;uuid&gt;/
- Mini evaluations: evaluation/v2/mini/&lt;experiment&gt;-step-&lt;step&gt;-&lt;uuid&gt;/
- Final evaluation: evaluation/v2/&lt;experiment&gt;-&lt;uuid&gt;/
- Optional initial evaluation: evaluation/v2/untrained/&lt;experiment&gt;-&lt;uuid&gt;/

Automatic evaluations share the training UUID. By default mini evaluation runs
on seeds 0–9 (10 episodes) every 10 updates, and final evaluation runs on seeds 0–24. Each
evaluation records videos. Use --mini-eval-interval-updates 0 to disable mini
evaluation, or --skip-evaluation to disable all evaluations.

Evaluation success means a held green lift, not the full stack. Metrics include
grasp rate, lift success rate, steps to grasp/lift, maximum lift height, and reward
components, plus initial/final/minimum gripper-to-green distances and reaching
return. Training logs per-episode task metrics and evaluation summaries to
stdout and TensorBoard.

    poetry run tensorboard --logdir v2/runs
    poetry run python -m v2.evaluate v2/runs/<run>/checkpoint_final.pt --name v2-held-out --seed 100 --episodes 25

Standalone evaluation restores task thresholds from the checkpoint and creates
a fresh evaluation UUID so repeated evaluations do not overwrite each other.
History length is stored in model config; the evaluator automatically loads the
correct architecture and resets its action/observation history. The reaching scale
is saved in task metadata. Older checkpoints without this
field are evaluated with scale 0, preserving their sparse rewards. Checkpoints
retain the v1 model serialization format; task metadata identifies v2.

## Experiment results

The temporal policy has produced gripping actions and one successful evaluation
lift, but learning remains slow and inconsistent near 200,000 training steps.
See the [experiment report](REPORT.md) for the progress snapshot and
[preserved successful episode at step 122,880](assets/green_lift_step_000122880_episode_006.mp4).

## Validation

    poetry run python -m unittest v2.test_task v2.test_history

For a short real rendering/training check, including resuming after mini evaluation:

    poetry run python -m v2.train --exp-name v2-check --total-timesteps 4 --rollout-steps 2 --minibatch-size 2 --update-epochs 2 --max-episode-steps 2 --mini-eval-interval-updates 1 --mini-eval-episodes 1 --final-eval-episodes 1 --no-pretrained

This checks execution only: two-step episodes cannot achieve the default
five-step held lift.
