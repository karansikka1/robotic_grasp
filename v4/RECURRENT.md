# Persistent LSTM BC comparison — 2026-09-09

This experiment replaces the short observation-window temporal model with an
LSTM whose hidden and cell state persist for the whole episode. It tests two
input sets from [the earlier diagnostic report](DIAGNOSTICS.md): spatial visual
BC without previous actions, and exact object positions plus robot state and
teacher stage.

The earlier transformer is a causal transformer, but it recomputes only three
observation frames and retains no state beyond that window. These LSTMs process
one new frame/state per control step. Neither receives a previous action.

## Final comparison

| Input set / temporal model | Training stacks | Fresh stacks | Successful speed |
| --- | ---: | ---: | --- |
| Spatial visual, three-frame transformer, no previous action | 2/5 | 0/5 | Train mean 310 actions / 15.50 s |
| Spatial visual, persistent LSTM, no previous action | **0/5** | **0/5** | No completion |
| Exact state + teacher stage, MLP | 4/5 | 3/5 | Train 297.75 / 14.89 s; fresh 299.33 / 14.97 s |
| Exact state + teacher stage, persistent LSTM | **1/5** | **0/5** | One training success: 314 actions / 15.70 s |

Neither LSTM improved full-stack success at these tested settings. Keep the
previous transformer and state MLP checkpoints as the stronger baselines.
The experiments establish working persistent recurrence, not that memory alone
solves imitation error or recovery. The small seed sample and differences in
architecture/sequence optimization limit broader conclusions about LSTMs.

The visual LSTM's five fresh rollouts all reached 900 actions (45 simulated
seconds), without a complete stack. Their video directory is linked below.

## Architecture and training

| Setting | Spatial visual LSTM | State + teacher-stage LSTM |
| --- | --- | --- |
| Inputs | Front/wrist RGB, front depth, 16 robot proprioception values | Same 54 exact-state/relative-position/stage values as the state MLP |
| Visual encoding | Frozen ImageNet ResNet-18 4×4 grids, camera concatenation; frozen MobileNet depth | None |
| Input fusion | Existing learned 256-dimensional visual/depth/proprio fusion | Train-normalized inputs, 128-dimensional ReLU projection |
| Recurrent model | One unidirectional LSTM layer, hidden/cell size 256 | One unidirectional LSTM layer, hidden/cell size 128 |
| Action head | Existing 256-dimensional MLP; seven tanh outputs | Linear 128→7; tanh outputs |
| Training epochs | 30 | 200 |
| Learning rate | 3e-4 | 1e-3 |
| Trajectories per training batch | 2 | 8 |
| Backpropagation chunk | 32 consecutive steps | 32 consecutive steps |
| Maximum valid transitions per update | 64 | 256 |

Both runs use Smooth L1 over the seven normalized action outputs, AdamW with
weight decay 1e-5, gradient clipping at norm 1, initialization/shuffle seed 0,
and the existing 80/10/10 trajectory split. Frozen pretrained image weights are
retained; learned components are initialized anew, without a BC warm start.
The best checkpoint is selected by validation loss; test metrics are measured
only after selection.

Trajectory order is shuffled each epoch, but steps within each trajectory remain
ordered. Hidden/cell states start at zero at trajectory boundaries and carry
between chunks. They are detached after each optimizer update, so gradients span
32 steps, while forward memory can span the episode. Shorter trajectories are
padded only at their ends; padding is excluded from loss and metrics. Each batch
starts new trajectories and cannot carry hidden state from an earlier batch.

At inference, hidden/cell state reset only at episode boundaries. They do not
reset at 32 steps or teacher-stage transitions. Runtime hidden state is not stored
in checkpoints; loading a policy starts with empty episode memory.

This comparison changes the temporal architecture, parameter count, and training
order. Epochs, datasets, learning rates, action objective, and evaluation seeds
are retained, but sequential chunking changes the exact optimizer update count.
It is not a parameter-count- or compute-matched ablation of memory alone.

## Evaluation

Each selected checkpoint runs on the same five sampled training seeds and five
fresh seeds 1000000–1000004, with videos, exact initial-observation checks on
training seeds, green lift/placement telemetry, full-stack success, and speed.
Success is ten consecutive official full-stack successes. The one post-reset
zero-action observation step is excluded from control time and included in
simulator-step counts. Visual failures allow 900 policy actions.

The state LSTM retains the live privileged teacher stage provider. The teacher
observes student-visited states, supplies stage transitions, and discards its
own action. Its guards may terminate a rollout early with a recorded reason.
This remains a diagnostic with privileged sequencing. The visual policy still
uses depth. Neither model meets the final RGB/proprioception-only input contract.

## Reproduction

```bash
export TORCH_HOME=/workspace/.cache/torch
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

/root/.local/bin/poetry run python -m v4.train_recurrent visual
/root/.local/bin/poetry run python -m v4.train_recurrent state
```

Defaults are shown above. `--chunk-length`, `--batch-trajectories`, `--epochs`,
`--learning-rate`, `--device`, and `--split-manifest` are explicit overrides.
The visual checkpoint can also be evaluated through the existing
`python -m v4.evaluate <checkpoint>` command. State checkpoints load through
`v4.recurrent.load_state_lstm` and use `v4.control_diagnostics.rollout_state`,
which resets their memory before each episode.

## Runs

- [Visual LSTM](runs/v4-lstm-visual-75940047-a07e-4190-8109-8dfb9f1f5b19).
- [State-plus-stage LSTM](runs/v4-lstm-state-e50ef4fe-e95e-4ef6-8f61-b81afde66f2a).

Both directories contain configuration, split, epoch metrics, best/final
checkpoints, and offline action metrics. Visual video locations are recorded in
`train_rollout_metrics.json` and `rollout_metrics.json`; state videos are named
`seed_<seed>.mp4` next to `rollouts.json`.

## Validation

Tests cover full-sequence/streaming equivalence, dependence on observations older
than three frames, no future-state influence, episode reset, end-padding masks,
exact trajectory coverage, chunk-state carry, gradient updates, and checkpoint
reloads with empty runtime memory. Existing transformer and diagnostic tests are
included in regression checks.


## State-plus-stage result

The state LSTM selected epoch **198/200**. Training translation MAE is **0.00939**
and test translation MAE is **0.01846**, compared with **0.00958 / 0.02349** for
the earlier state MLP. Offline gripper-sign accuracy is 100% in each split.

Closed-loop success worsened from **4/5 training and 3/5 fresh** for the MLP to
**1/5 training and 0/5 fresh** for the LSTM. Training seed 190447585 completed
in **314 actions / 15.70 seconds**. Failures included approach/transfer timeouts,
blue descent timeouts, and a failed green grasp. These are observed outcomes;
the experiment does not isolate whether memory, capacity, or sequential training
caused the degradation. Lower offline action error again failed to predict
better execution. The simpler state MLP remains the stronger tested controller
when privileged stage is available.

[Successful state-LSTM video](runs/v4-lstm-state-e50ef4fe-e95e-4ef6-8f61-b81afde66f2a/seed_190447585.mp4).


## Visual fitting and training-seed result

The visual LSTM selected epoch **30/30** by validation loss. Train/test translation
MAE is **0.06428 / 0.08642**; test gripper-switch accuracy is **19/40**. The previous
three-frame transformer without previous actions had test translation MAE
**0.08031** and switch accuracy **26/40**. Both frozen image backbones exactly
match that baseline checkpoint.

Training-seed full stacks fell from **2/5 to 0/5**. Three training episodes
achieved held green lifts, but none registered green-on-red placement. Seed
1390308241 lifted both green and blue and registered blue-on-green placement,
but did not achieve the required complete stack. All initial observations
matched the demonstrations. These partial milestones do not count as full
success.

The recurrent tests and all relevant regression tests passed: **32 tests**.
No simulator/environment source was changed and no dependencies were added.


## Final artifacts

- [Fresh visual-LSTM videos](../evaluation/v4/v4-lstm-visual-75940047-a07e-4190-8109-8dfb9f1f5b19-best-06448caa-86de-4802-a046-2fe55de60d3f), files `episode_000.mp4` through `episode_004.mp4`.
- [Successful state-LSTM video](runs/v4-lstm-state-e50ef4fe-e95e-4ef6-8f61-b81afde66f2a/seed_190447585.mp4).
- [Machine-readable comparison](runs/recurrent20260909-comparison.json), including full-stack success, speed, and green manipulation milestones.
- Each run has a `video_validation.json`; **20/20 new videos** decoded with expected frame counts.

Both training/evaluation jobs completed. The implementation passed 32 regression
tests and source syntax/whitespace checks. Further optimization, different
backpropagation horizons, or recovery-data experiments were not launched.
