# v4: scripted privileged teacher

The first v4 objective is deliberately narrow: produce and preserve one complete
red-green-blue stacking trajectory without learning. The supplied environment and
simulator are not modified.

The initial real MuJoCo run succeeded with seed 0 on its first attempt in 320
actions (16.0 simulated seconds).

The teacher reads exact object poses, sizes, body velocities, gripper position,
grasp state, object contacts, and the official task condition from the underlying
MuJoCo environment. It executes a closed-loop state machine:

```text
green: approach -> descend -> grasp -> lift -> transfer -> place -> release
blue:  approach -> descend -> grasp -> lift -> transfer -> place -> release
settle -> verify stable task_complete
```

Run it from the repository root:

```bash
poetry run python -m v4.record_trajectory \
  --name first-full-stack \
  --seed 0 \
  --max-attempts 20
```

Or use:

```bash
make teacher-v4 ARGS="--name first-full-stack --seed 0 --max-attempts 20"
```

The collector tries consecutive deterministic seeds until it finds one success.
Failed attempts are summarized in `metrics.json` but their large temporary files
are discarded. A successful run directory contains:

```text
v4/trajectories/first-full-stack-<uuid>/
├── metrics.json
├── trajectory.h5
└── trajectory.mp4
```

`trajectory.h5` stores each pre-action observation, its teacher action and stage,
the next-step success flag, and aligned oracle state. It also stores the terminal
observation. The legal student observations can later be selected without using
depth or oracle state.

The initial controller intentionally keeps the Panda's reset orientation and uses
only XYZ OSC deltas plus open/close gripper commands. Its tolerances and waypoint
clearances are collected in `TeacherConfig` so the first cloud rollout can be
tuned from observed failure stages without touching the task environment. The
most useful settings are also CLI flags, for example
`--grasp-z-offset-m 0.01` and `--max-translation-action 0.4`. Stage transitions
are printed as the trajectory runs and the final stage for each attempt is saved
in `metrics.json`.

## Collect a dataset

For the resumed 100-demonstration comparison:

```bash
POETRY=/root/.local/bin/poetry bash scripts/v4_bc_diverse.sh
```

The script fills the dataset to 100 successful distinct simulator seeds using four
spawned workers, creates `v4/splits/diverse100.json` (80/10/10), and trains two
30-epoch three-frame transformers. The first uses the original frozen MobileNet
RGB encoder; the second uses frozen ImageNet ResNet-18 for both RGB cameras.
The depth encoder remains MobileNet in both. Neither model is RGB-only.
ResNet uses torchvision's [pretrained model](https://docs.pytorch.org/vision/main/models/resnet.html)
and the existing full-image resize/normalization convention, without cropping.

Both selected checkpoints receive five training-seed and five fresh-seed rollouts.
Training diagnostics verify the initial observations against their demonstrations.
Rollouts record bilateral contacts, held lifts (5 cm for five steps), and partial
stack observations, separately from the authoritative stable full-stack success.
Offline metrics separate translation MAE, gripper sign accuracy, and accuracy at
teacher gripper switches. A switch is a timestep whose nonzero previous gripper
action has a different sign from the target; reset padding is excluded.

The collection command is independently resumable:

```bash
poetry run python -m v4.collect_trajectories --target-total 100 --seed 20260909 --workers 4
```

Completed demonstrations and recorded failed seeds are excluded on restart.
The combined script reuses an existing split but starts new training runs.
Collection manifests, checkpoints, and per-run logs are stored in the workspace.
`diverse100.coverage.json` and `.coverage.png` beside the split summarize placement
coverage relative to the original dataset. These generated artifacts are ignored
by Git and must be retained with the workspace volume.

For a standalone training-seed diagnostic:

```bash
poetry run python -m v4.evaluate v4/runs/<run>/best.pt --training-seeds --episodes 5
```

`--zero-rotation` is a separate evaluation ablation that sets the three rotation
outputs to the teacher's exact zero commands. The executed command is also used
in previous-action history. It does not overwrite the checkpoint, and its metrics
and videos are named separately from normal evaluation. This is not enabled by
the training script. Offline diagnostics can be recomputed with
`python -m v4.evaluate_actions <checkpoint> --partition validation`.

### Spatial camera concatenation

```bash
POETRY=/root/.local/bin/poetry bash scripts/v4_bc_spatial.sh
```

This adds a third comparison using the same data split and 30-epoch training
settings: ResNet-18 RGB features retain a 4x4 grid instead of global pooling.
Each camera yields 512x4x4 features. Flattening in fixed spatial order and
concatenating front then wrist gives 16,384 features, projected to 256 dimensions
before combining with depth/proprioception and entering the temporal transformer.
The projection has distinct weights for each camera, channel, and spatial cell.
It does not average the two views together. These are spatial grids within each
frame, not extra transformer time tokens.

Use `--rgb-pool-size 2` for a 2x2 grid, or `--rgb-pool-size 1 --camera-fusion concat`
to test camera concatenation without spatial retention. `--camera-fusion sum`
retains the original camera combination. Spatial grids require ResNet-18, and
input resolution must produce a feature map at least as large as the grid.
All settings are saved in checkpoints; older global-pooling checkpoints still load.

To compare binary gripper supervision on exactly this spatial architecture:

```bash
POETRY=/root/.local/bin/poetry bash scripts/v4_bc_spatial.sh \
  --exp-name v4-diverse100-resnet18-concat-grid4-bce --gripper-loss bce
```

Motion keeps Smooth L1 on the six tanh-normalized outputs. Gripper BCE operates
on the raw seventh logit with targets mapped from -1/+1 to 0/1. The six motion
losses and one gripper loss are averaged together. Validation selects checkpoints
with the same objective used for training; raw loss values should not be compared
between BCE and Smooth L1 runs. Action MAE, gripper-switch accuracy, and rollouts
remain comparable. Inference still uses tanh, and previous-action history records
the executed normalized command. BCE rejects nonbinary teacher gripper labels.

```bash
poetry run python -m v4.collect_trajectories --count 12 --seed 20260908
```

This draws distinct random simulator seeds, excluding successful seeds already
stored locally. It records every attempt in a collection manifest; each attempt
uses one seed, and failed seeds are not silently retried. By default, at most
`2 * count` attempts are allowed to collect `count` successful demonstrations.

The local check collected 12/12 successes (282–357 actions). Together with the
initial random-seed check, there are 13 demonstrations / 4,042 transitions. These
are a small pipeline dataset; collection success measures the scripted teacher,
not the learned student.

## Visual history behavior cloning

The default is the existing **v2 visual history policy**: current + previous two
RGB/depth/proprio observations and the previous action. The same frozen pretrained
MobileNet encoders process every frame once. Cached features form causal windows
within each trajectory; learned per-frame projections, temporal fusion, and actor
are optimized with Smooth L1 action regression. The critic and exploration
parameters are not trained by BC.

At the first step, history repeats the first observation and previous action is
zero. At training step t, previous action is the teacher's action at t-1; target
action t and future observations/actions never enter the input. At deployment,
the previous action is the student's own action, and history resets each episode.
Use `--policy single` to retain the original memoryless baseline.

The policy includes privileged depth, like the earlier v2 diagnostic. Oracle
poses, teacher stages, reward, and completion flags are not student inputs. A final
RGB-only policy would need a separate input configuration and training experiment.

## Train, validation, and test separation

```bash
make split-v4 ARGS="--validation-fraction 0.2 --test-fraction 0.2 --seed 0"
bash scripts/v4_bc_train.sh --split-manifest v4/splits/default.json
```

Splits are by **whole trajectory and simulator seed**, before history windows are
built. Repeated demonstrations with the same seed stay in one partition. The
current fixed split has 7 training trajectories (2,162 transitions), 3 validation
trajectories (968), and 3 test trajectories (912).

Validation loss selects `best.pt` over 30 epochs. The test partition is evaluated
only once, after loading that checkpoint. Per-epoch logs call the held-out metric
validation loss; `test_metrics.json` stores the final test loss, MAE, RMSE, and
per-action MAE. Offline temporal evaluation uses the preceding teacher action,
so simulator evaluation is needed to assess accumulated student errors.

Reusing `--split-manifest` keeps the partitions fixed when more trajectories are
collected. The loader rejects missing partitions, changed metadata, duplicate
paths, or overlapping seeds. Old two-way manifests must be regenerated before
using this three-way training command.

Runs under `v4/runs/<experiment>-<uuid>/` contain `split.json`, `config.json`,
`metrics.jsonl`, `best.pt`, `final.pt`, `test_metrics.json`, `rollout_metrics.json`,
and `train.log`. `--no-pretrained` is for architecture tests only; the default
uses cached ImageNet weights. No weights are loaded from a v3 state-policy run.

## Backbone fine-tuning comparison

```bash
bash scripts/v4_bc_finetune.sh --split-manifest v4/runs/<baseline-run>/split.json
```

This keeps the MobileNetV3-Small architecture, three-frame history, previous
action, and BC loss unchanged. It starts a new model from ImageNet backbone
weights with the same initialization seed as the frozen baseline. Both RGB and
depth backbones receive gradients at learning rate `1e-5`; heads use `3e-4`.
BatchNorm running statistics remain fixed, while affine weights can train.
The critic and exploration parameters remain outside the BC optimizer.

The default BC command still freezes the backbones. `--finetune-backbone` enables
this experiment; `--backbone-learning-rate` controls its separate optimizer group.
Fine-tuning caches normalized input images on CPU instead of backbone features,
then recomputes visual features for every minibatch. History uses the same causal
windows and reset padding; all three timesteps contribute gradients to the
backbone weights.
Checkpoint loading and simulator inference use the existing v2 policy.

## Closed-loop evaluation

Training automatically evaluates the selected model on five fresh seeds starting
at 1,000,000, with videos and a 900-action episode limit. Evaluation checks that
these seeds appear in none of the demonstration partitions. Success requires the
full red–green–blue stack to satisfy the official task condition for ten
consecutive control steps, matching the teacher's stable-success check.

```bash
poetry run python -m v4.evaluate v4/runs/<run>/best.pt --seed 1000000 --episodes 5
```

Videos and detailed episode metrics are stored under `evaluation/v4/`. To change
the automatic check, use `--rollout-eval-episodes`, `--rollout-eval-max-steps`, or
`--rollout-eval-seed` on the training command. Setting episodes to zero skips it.
Offline action error and fresh-seed rollout success measure different things;
passing the split checks alone does not establish generalization.

## Validation

```bash
poetry run python -m unittest v4.test_teacher v4.test_bc v4.test_history_bc v4.test_finetune v4.test_transformer v4.test_collection v2.test_history
```

Tests cover collection/data loading, seed grouping, causal history and reset
padding, agreement with online history, temporal-fusion optimization, checkpoint
loading, fresh-seed checks, the consecutive-success criterion, backbone weight
updates, fixed BatchNorm statistics, and image-cache/inference agreement.

## Frozen-backbone temporal transformer

```bash
bash scripts/v4_bc_transformer.sh --split-manifest v4/runs/<baseline-run>/split.json
```

This replaces concatenation-based temporal fusion with a transformer over the
same frame embeddings. Both MobileNet backbones remain frozen. Each frame still
combines RGB, depth, and robot state through the existing projections and fusion
normalization. Two pre-norm attention layers use four heads, 256-dimensional
embeddings, a 512-dimensional feedforward block, and learned temporal positions.
The newest encoded token is combined with the previous action before the existing
actor. Dropout defaults to zero, as in the original temporal MLP experiment.

Attention is causally masked, and windows never cross trajectory boundaries.
Initial frames are repeated and the previous action is zero at reset. During
rollout, previous action comes from the student. Only the final action in each
window is supervised. The transformer recomputes the window at every decision;
it does not retain hidden state outside that window.

The first comparison retains **three observations** to isolate the temporal
encoder change. This still spans 0.1 seconds at 20 Hz. A separate longer-context
run can use `--history-length 16` (0.75 seconds); this is not an LSTM or an
unbounded memory. Other options are `--transformer-heads`, `--transformer-layers`,
`--transformer-feedforward-dim`, and `--transformer-dropout`.

Transformer checkpoints use `v4.model.load_policy` and the existing
`python -m v4.evaluate` command. The v4 loader also accepts older single-frame and
history BC checkpoints. The v2/v3 training code is unchanged.

## Tested baseline: 2026-09-08

Run `v4-history-bc-5da21dfd-668f-4b76-9b81-592a18d1a4ab` completed 30 CUDA
epochs on the 7/3/3 split above. The best validation checkpoint was epoch 30:
training loss 0.00245, validation loss 0.01779, held-out test loss 0.01425 and
normalized action MAE 0.07655. Test translation MAE was 0.126/0.153/0.173 for
X/Y/Z; the almost constant rotation commands make the overall average lower.

Fresh-seed simulator evaluation achieved **0/5 full-stack successes**, each
hitting 900 actions. This checks execution and data isolation; it does not show
reliable closed-loop generalization. The five videos each contain 901 frames.

Local artifacts (generated runs and datasets are ignored by Git):

- [Best checkpoint](runs/v4-history-bc-5da21dfd-668f-4b76-9b81-592a18d1a4ab/best.pt)
- [Test metrics](runs/v4-history-bc-5da21dfd-668f-4b76-9b81-592a18d1a4ab/test_metrics.json)
- [Rollout metrics](../evaluation/v4/v4-history-bc-5da21dfd-668f-4b76-9b81-592a18d1a4ab-best-510035f6-70fd-4d8c-98a6-d0a94681d390/metrics.json)
- [First rollout video](../evaluation/v4/v4-history-bc-5da21dfd-668f-4b76-9b81-592a18d1a4ab-best-510035f6-70fd-4d8c-98a6-d0a94681d390/episode_000.mp4)

## Fine-tuning result: 2026-09-08

The 30-epoch comparison used the same initialization seed and exact split as the
frozen baseline. Both selected epoch 30 on validation loss.

| Metric | Frozen backbone | Fine-tuned backbone |
| --- | ---: | ---: |
| Validation loss | 0.01779 | 0.02028 |
| Test action MAE | 0.07655 | 0.08059 |
| Training-seed full-stack successes | 0/7 | 0/7 |
| Fresh-seed full-stack successes | 0/5 | 0/5 |

Fine-tuning both backbones at 1e-5 with fixed BatchNorm statistics did not improve
this run. This is one setting and initialization; it does not settle whether a
stronger representation or longer temporal context would help. Those variants
remain untested. All initial training-seed observations matched the demonstrations
exactly; all 12 new rollout videos decoded successfully, with 901 frames each.

[Comparison metrics and evaluation paths](runs/v4-history-bc-finetune-c8db0cbf-ada6-40b7-921c-ffe10107a500/comparison.json).

## Transformer result: 2026-09-08

The frozen-backbone transformer completed 30 epochs on the exact baseline split,
selecting epoch 15 by validation loss. Test action MAE improved by about 26%, but
closed-loop full-stack success did not improve.

| Metric | Frozen history MLP | Frozen transformer |
| --- | ---: | ---: |
| Validation loss | 0.01779 | 0.01021 |
| Test action MAE | 0.07655 | 0.05643 |
| Training-seed full-stack successes | 0/7 | 0/7 |
| Fresh-seed full-stack successes | 0/5 | 0/5 |

Both saved visual backbones exactly match the frozen baseline. All training-seed
initial observations match their demonstrations, and all 12 videos decoded
successfully (901 frames each). This comparison retains a three-frame window;
longer transformer context and a binary gripper loss remain separate experiments.
Twenty regression/transformer tests passed.

[Checkpoint, metrics, and video evaluation paths](runs/v4-transformer-bc-f403fca1-928f-4070-a8d8-1d6c678bc452/comparison.json).


## Action-persistence diagnostic

```bash
poetry run python -m v4.report_action_persistence v4/splits/diverse100.json --output v4/splits/diverse100.action_persistence.json
```

This offline baseline repeats the previous demonstration action, with zero at
trajectory start. It is not a closed-loop controller. On diverse100 its test MAE
is 0.01455 and gripper accuracy is 98.75%, despite missing all 40 switches.
Compare switch accuracy and simulator milestones as well as aggregate error.


## Perception and control diagnostics

See [the September 9 diagnostic report](DIAGNOSTICS.md) for the no-previous-action comparison, privileged-state/stage ablation, same-seed visual overfit, RGB-position replacement, commands, and evaluation videos.


## Persistent recurrent policies

[Persistent LSTM comparison](RECURRENT.md) implements episode-long hidden/cell state for spatial visual BC without previous actions and exact state plus teacher stage. Use `python -m v4.train_recurrent visual` or `python -m v4.train_recurrent state`; both include video rollout evaluation.


## VC-1 backbone

[VC-1 comparison](VC1.md) uses the official ViT-L CLS embedding with the non-LSTM, no-previous-action temporal policy. Run `POETRY=/root/.local/bin/poetry bash scripts/v4_bc_vc1.sh` after downloading the pinned weights as documented.


## Categorized demonstrations and review page

The 300-trajectory clean/noise/drop/delay/bias dataset, fixed 270/30 train/validation split, and ten interactive video examples are documented in [ROBUSTNESS.md](ROBUSTNESS.md).
