# v3: scripted privileged teacher

The first v3 objective is deliberately narrow: produce and preserve one complete
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
poetry run python -m v3.record_trajectory \
  --name first-full-stack \
  --seed 0 \
  --max-attempts 20
```

Or use:

```bash
make teacher-v3 ARGS="--name first-full-stack --seed 0 --max-attempts 20"
```

The collector tries consecutive deterministic seeds until it finds one success.
Failed attempts are summarized in `metrics.json` but their large temporary files
are discarded. A successful run directory contains:

```text
v3/trajectories/first-full-stack-<uuid>/
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
