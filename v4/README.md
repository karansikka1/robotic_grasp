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

## Privileged behavior cloning

Create a deterministic split by whole trajectory (never by adjacent frames):

```bash
make split-v4 ARGS="--test-fraction 0.2 --seed 0"
```

With the current six demonstrations this produces five training trajectories
(1,566 transitions) and one held-out trajectory (315 transitions). Seed 347024
is held out by split seed 0.

Train the v1 privileged-depth network from the repository root:

```bash
make train-v4-bc ARGS="--exp-name v4-privileged-bc --epochs 30"
```

The two RGB images and metric depth image are passed through the frozen
ImageNet-pretrained MobileNetV3-Small encoders once. Their cached features and
robot proprioception train the v1 projection, fusion, and actor layers using
Smooth L1 behavior-cloning loss. Every epoch reports held-out loss, MAE, RMSE,
and MAE for each of the seven action dimensions. Runs contain `split.json`,
`config.json`, `metrics.jsonl`, `best.pt`, `final.pt`, and `train.log`.

`--no-pretrained` exists only for offline architecture tests. A useful training
run should retain the default pretrained encoders. Offline test loss does not
establish closed-loop success; evaluate the selected checkpoint in the simulator
after training.
