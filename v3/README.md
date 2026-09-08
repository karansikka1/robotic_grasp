# v3: state-based green grasp-and-lift

This diagnostic replaces the visual policy with a small MLP that receives exact
simulator state. Its purpose is to test whether learning grasp-and-lift becomes
easier when object localization does not need to be learned from images. If it
becomes reliable, it can later provide actions for training a visual student.
Student imitation is a separate experiment and is not implemented here.

## Run

From the repository root, using the existing Poetry environment:

```bash
bash scripts/v3_smoke_train.sh
```

Defaults are 200,000 environment steps, 1,024 total steps per
rollout, eight PPO optimization epochs, minibatches of 256, learning rate 0.0003,
and a maximum of 1,000 policy steps per training episode (50 simulated seconds). The KL penalty remains zero.
The launch script and training CLI accept overrides; for example:

```bash
poetry run python -m v3.train --total-timesteps 20000 --exp-name v3-state-short
```

The actor and critic each have two 128-unit tanh hidden layers. Both are trained
from scratch. The seven-dimensional tanh-squashed Gaussian action distribution,
PPO optimizer, randomized resets, 20 Hz control frequency, and success criterion
are shared with v2. Use `--hidden-dim` to change MLP width or `--device cpu` to
run without CUDA. There are no pretrained weights to download.

## Next experiment: maintained grasp, initialized from v3

```bash
bash scripts/v3_grasp_train.sh
```

This starts `v3-maintained-grasp` from checkpoint
`v3-state-20d9d6e6-320d-4241-a047-f0b7b965ca26/checkpoints/step_000204800.pt`
under `v3/runs`. The script pins that source for reproducibility; it is the latest
checkpoint available when this experiment was prepared, not a selected best model.
It uses four simulators, 2,048 total rollout steps, eight PPO epochs, minibatches
of 256, 1,000-action training episodes, and 400,000 **additional** training steps.
Mini evaluation uses five seeds every ten updates; initial/final evaluation uses
25 seeds, with a 500-action evaluation limit. Extra CLI arguments override defaults.

The +1 grasp bonus now requires `--grasp-hold-steps 5`: both fingerpad groups must
contact green for five consecutive control steps (0.25 seconds). A contact loss
resets the streak; the bonus is awarded only once per episode. Reaching reward and
the +5 held-lift condition remain unchanged. Maintained contact is still a proxy
for a secure grip; successful lifting is the stronger check.

The basic training defaults and checkpoints lacking `grasp_hold_steps` retain
one-step contact semantics. Standalone evaluation uses each checkpoint's saved
threshold. Compare new `grasp_rate` values only against an evaluation using the
same threshold, rather than interpreting old one-contact rates as maintained grips.

New metrics in stdout, TensorBoard, and evaluation JSON include:

- `contact_seen` / summary `contact_rate`: bilateral contact happened at least once.
- `grasped` / `grasp_rate`: the configured consecutive-contact threshold was reached.
- `bilateral_contact` / `final_contact_rate`: contact remains at episode end.
- `max_contact_streak_steps`: longest uninterrupted contact, in control steps;
  divide by 20 for seconds. Summary: `mean_max_contact_streak_steps`.
- `contact_loss_count`: transitions from bilateral contact to its loss, including
  short contacts; summary: `mean_contact_loss_count`.
- `contact_steps` and `contact_streak_steps`: total and current consecutive contact
  observations. The reset bootstrap does not count toward these or the reward.

`--init-checkpoint PATH` loads actor, critic, and exploration `log_std` weights and
the saved model architecture. If supplied, `--hidden-dim` must match the source.
The new run creates fresh Adam state, rollout buffers, step counters, and UUID.
Source path, SHA-256, source step/update, and original task settings are saved in
its config and checkpoints. The new task settings come from this run's CLI.

The script enables `--evaluate-untrained`, whose existing name means evaluation
of the **initialized** policy before any new updates. For this warm start it uses
the loaded weights with the stricter reward, and stores the baseline in
`evaluation/v3/untrained/<new-run>/`. This gives a before/after comparison under
the same grasp definition. Initial evaluation and all later automatic evaluations
share the new run UUID.

## Parallel rollouts

`--num-envs` defaults to **1**, which uses the original single-simulator rollout
path and seed sequence. To run four simulators:

```bash
bash scripts/v3_smoke_train.sh --num-envs 4
```

For `num_envs > 1`, each simulator runs in its own spawned CPU process with
cameras disabled. The main process runs one batched policy forward pass for all
active simulators, steps them concurrently, then waits for their results. Workers
remain alive during PPO optimization and evaluation. Evaluation remains sequential
and records the same videos and metrics.

`--rollout-steps` is the **total transitions per update across all simulators**.
For example, 1,024 rollout steps with four simulators gives 256 steps per simulator
per update, while the episode limit remains 1,000 actions per simulator. PPO batch
size, total training-step budget, and checkpoint/evaluation frequency therefore
keep their existing meanings. Uneven batches and a partial final update are
supported without exceeding the requested budget. `num_envs` cannot exceed
`rollout_steps`.

Each worker resets independently on success or its episode limit. With N workers,
worker i uses episode seeds `training_seed + i`, then increments by N on each
reset. Advantages and value targets are computed independently along each
worker's timeline, with the same terminal/episode-limit handling as the single
path, before concatenating transitions for PPO minibatches. Increasing N shortens
each worker's trajectory segment per update, so multi-simulator learning need not
match a single-simulator run even with the same overall step budget.

Episode logs include worker IDs and seeds. TensorBoard additionally records
`charts/num_envs` and `charts/rollout_steps_per_second` for parallel runs. Worker
errors are reported to the main process; workers are closed when training finishes
or fails. Workers use one BLAS/OpenMP thread each; choose the worker count to fit
your available CPU cores and memory. More workers add process and communication
overhead, so throughput gains must be measured on your machine.

## State and reward

Both actor and critic receive the same 47 state values:

| Input | Values |
| --- | --- |
| Robot joint positions and velocities | 7 + 7 |
| End-effector position and quaternion | 3 + 4 |
| Gripper finger positions and velocities | 2 + 2 |
| Green object position and quaternion | 3 + 4 |
| Vector from gripper grip site to green's center | 3 |
| End-effector linear and angular velocities | 3 + 3 |
| Green object linear and angular velocities | 3 + 3 |

Positions and velocities use world coordinates; the relative vector is green
minus grip-site position. Quaternions use xyzw order. Inputs are float32 in
physical units (meters, radians, seconds), without running normalization.
The ordered input schema is saved in checkpoints. There is no image/depth input,
observation history, previous action, reward, or success flag in the policy input.
Velocities provide direct motion information for this diagnostic.

Reward is inherited from `v2/task.py`, including the bootstrap and reset rules:

- Signed distance progress: `previous_distance_m - current_distance_m` (scale 1).
- +1 once for a bilateral finger-contact grasp of green.
- +5 for a grasped lift at least 5 cm above the post-reset reference, maintained
  for five consecutive steps (0.25 seconds). Success ends the episode.
- No constant step penalty.

The state adapter lives in `v3/task.py`; the supplied simulator/environment files
are unchanged. Cameras are disabled during training, so no image rendering or
visual encoding is performed. Evaluation enables both RGB cameras for videos,
with depth disabled. This policy requires privileged simulator state at inference
and is a diagnostic/potential teacher, not the final allowed RGB policy.

## Logs, checkpoints, and evaluation

- Training config, terminal log, TensorBoard, and final checkpoint:
  `v3/runs/<experiment>-<uuid>/`.
- Periodic checkpoints: `checkpoints/step_<step>.pt` inside that run directory,
  saved every ten PPO updates.
- Mini evaluations with videos: `evaluation/v3/mini/<experiment>-step-<step>-<uuid>/`.
- Final evaluation with videos: `evaluation/v3/<experiment>-<uuid>/`.

Automatic evaluations share the training UUID. Mini evaluation uses deterministic
actions on seeds 0–9 every ten updates (10,240 training steps); final evaluation
uses seeds 0–24. Evaluation stays at 500 policy steps per episode for comparison;
use training option `--eval-max-steps 1000` to extend automatic evaluations too.
The simulator horizon includes one extra step for the initial zero-action
bootstrap, allowing the full configured number of policy actions. Success still
ends episodes early. Longer training episodes mean fewer resets within a fixed
training-step budget. Restart training to use these changes.

Training starts at seed 10,000 and advances seeds each episode.
Grasp/lift rates, distances, individual reward returns, and PPO metrics are logged
to stdout and TensorBoard, with episode metrics also in evaluation `metrics.json`.

```bash
poetry run tensorboard --logdir v3/runs
poetry run python -m v3.evaluate v3/runs/<run>/checkpoint_final.pt --seed 100 --episodes 25
```

Standalone evaluation restores the saved state schema and reward settings.
Use `--no-video` to disable its rendering. Training supports `--evaluate-untrained`
for an initial comparison, or `--skip-evaluation` to disable automatic evaluations.

Compare success over multiple checkpoints and held-out placements before deciding
whether this policy is useful as a teacher. Better performance would support a
perception/representation bottleneck; failure would motivate investigating reward,
exploration, and optimization. This comparison also changes network architecture,
so it does not isolate perception as the sole cause.

## Validation

```bash
poetry run python -m unittest v2.test_task v2.test_history v3.test_state v3.test_parallel v3.test_grasp_run
```

All 33 tests passed. An eight-step CUDA integration run crossed rollout and episode
boundaries, ran two video mini evaluations, and saved a final checkpoint and final
evaluation video. Three-step evaluation episodes cannot satisfy the five-step
held-lift criterion: this checks execution, not learning.

A two-simulator CUDA check (`v3-parallel-check-f9795afe-a1e0-45c7-930f-09a5e3a90b7e`)
completed exactly 13 transitions in rollouts of 5, 5, and 3 steps, with independent
episode resets, three mini video evaluations, final checkpoint saving, and final
video evaluation. Tests cover independent advantage calculations and episode
boundaries, worker seed streams, worker error propagation/cleanup, and default
single-simulator routing. These short checks do not measure learning or throughput.

The default single-simulator regression run produced checkpoint weights exactly
matching the earlier eight-step v3 baseline before parallel rollout support.

Warm-start check `v3-grasp-check-d5ca474f-0eb9-40dc-bcbd-9e43f57646a7` loaded the
actual 204,800-step checkpoint and completed 16 new CUDA transitions with two
simulators, initial evaluation, two mini evaluations, and final evaluation. Tests
cover brief contacts, streak resets, one-time bonuses, unchanged lift criteria,
old-checkpoint reward compatibility, and identical initialized policy weights.
These short episodes verify execution only; no learning improvement is claimed.
