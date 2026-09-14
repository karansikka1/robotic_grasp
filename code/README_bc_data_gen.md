# BC trajectory generation

We began by generating GT trajectories for behavior cloning. For this we relied on the simulator's access to the world state and mechanically collecting trajectories. For each trajectory we collected the images and other relevant features needed for training the final policy.

## Collection approach

We first defined a collection taxonomy covering clean stacking and four types of
execution error: 
- Gaussian command bursts
- block drops
- sideways errors
- gripper interruptions. 

Using this taxonomy, we collected **300** successful expert trajectories
and split them by whole trajectory into 270 training and 30 validation examples,
with each category represented in both partitions and no shared seeds between them.

After training the initial state-input BC policy, rollout inspection still showed
green-block drops and failed grasps caused by green/blue misalignment. We therefore
collected additional trajectories by running that policy until a detected failure
and handing control to the recovery teacher. We added 12 successful corrections
(four per failure type) to training, producing 282 training trajectories while
keeping the original 30 validation trajectories unchanged. Only the expert portion
of each correction supplies supervised targets; the learner prefix supplies context.

## Code

- `bc_data/collect_demonstrations.py`: collect clean and disturbed expert trajectories.
- `bc_data/collect_corrections.py`: run the source BC policy until a detected error, then hand control to the recovery teacher.
- `bc_data/teacher.py`: waypoint stacking expert.
- `bc_data/recovery.py`: disturbances, failure detection and retreat/regrasp recovery.
- `bc_data/recording.py`: HDF5 recording, optional videos and collection metadata.
- `bc_data/state_policy.py`: inference for the state-input BC policy used to collect corrections.
- `plans/`: fixed seeds, disturbance settings and train/validation assignments.

## Run

Run from the assignment repository root, with `code/` alongside the supplied
`motion_planning/` directory:

```bash
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

poetry run python -m bc_data.collect_demonstrations \
  --output code/outputs/base --workers 1

poetry run python -m bc_data.collect_corrections \
  --checkpoint path/to/source_state_bc.pt \
  --output code/outputs/corrections --workers 1
```

Add `--videos` to record every selected demonstration.

Supply the original state-input BC checkpoint matching `plans/corrections.json`
via `--checkpoint`. The collector saves correction trajectories and videos, and
lists the 12 selected recoveries in `collection.json`.

## Sample collection videos

Each video shows the front and wrist cameras side by side:

- [Clean expert demonstration](videos/clean.mp4)
- [Injected block drop and expert recovery](videos/drop_block.mp4)
- [Learner green-drop failure followed by expert correction](videos/green_drop.mp4)

## Assignment contract

Note: Privileged state is used for training-data generation. The final submitted policy
must consume only the two RGB images and four permitted proprioceptive fields;
it must not read depth, object poses. The supplied simulator
files remain unchanged.
