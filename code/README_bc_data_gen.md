# BC trajectory generation

We generated expert demonstrations for behavior cloning (BC), which trains a policy to imitate expert actions. A scripted teacher uses the simulator's exact block positions to complete the stack. Each trajectory records one attempt, including camera images, robot measurements, and the teacher's actions.

## Collection approach

We collected clean stacking demonstrations and demonstrations with four types of
execution error:

| Execution error | How it is introduced |
| --- | --- |
| Gaussian command bursts | Temporarily add random sideways movement to the teacher's commands. |
| Block drops | Open the gripper after lifting a block. |
| Sideways errors | Temporarily push the commanded movement to one side. |
| Gripper interruptions | Delay closing or briefly reopen the gripper during a grasp. |

Across these categories, we collected **300** successful expert trajectories
and split them by whole trajectory into 270 training and 30 validation examples,
with each category represented in both partitions and no shared seeds between them.

After training the initial state-input BC policy, rollout inspection still showed
green-block drops and failed grasps caused by green/blue misalignment. We therefore
collected additional trajectories by running that policy until a detected failure
and handing control to the recovery teacher. We added 12 successful corrections
(four per failure type) to training, producing 282 training trajectories while
keeping the original 30 validation trajectories unchanged. Only the expert portion
of each correction supplies the actions the model is trained to imitate. The earlier
learner-controlled part provides memory context. See [correction training](experiment_details.md#correction-training).

## Code

- `bc_data/collect_demonstrations.py`: collect clean and disturbed expert trajectories.
- `bc_data/collect_corrections.py`: run the source BC policy until a detected error, then hand control to the recovery teacher.
- `bc_data/teacher.py`: waypoint stacking expert.
- `bc_data/recovery.py`: disturbances, failure detection and retreat/regrasp recovery.
- `bc_data/recording.py`: HDF5 recording, optional videos and collection metadata.
- `bc_data/state_policy.py`: inference for the state-input BC policy used to collect corrections.
- `plans/`: fixed seeds, disturbance settings and train/validation assignments.

## Run

Run from inside the delivered `code/` folder after [environment setup](README.md#setup). The simulator is included in `motion_planning/`:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m bc_data.collect_demonstrations \
  --output outputs/base --workers 1

python -m bc_data.collect_corrections \
  --checkpoint path/to/source_state_bc.pt \
  --output outputs/corrections --workers 1
```

Add `--videos` to record every selected demonstration.

Supply the original state-input BC checkpoint matching `plans/corrections.json`
via `--checkpoint`. The collector saves correction trajectories and videos, and
lists the 12 selected recoveries in `collection.json`.

## Sample collection videos

Each video shows the front and wrist cameras side by side:

| Example | Video |
| --- | --- |
| Clean expert demonstration | <video src="media/clean.mp4" controls preload="metadata" width="400"></video><br>[Open video](media/clean.mp4) |
| Injected blue-block drop and expert recovery | <video src="media/drop_block.mp4" controls preload="metadata" width="400"></video><br>[Open video](media/drop_block.mp4) · Blue drops at 10.15–10.40 s; regrasp at 13.55 s. |
| Learner green-drop failure followed by expert correction | <video src="media/green_drop.mp4" controls preload="metadata" width="400"></video><br>[Open video](media/green_drop.mp4) |

Inline playback requires a Markdown viewer that permits video elements. The links also open the clips directly.

## Assignment contract

Exact simulator state is used to generate training data. The final submitted policy
must use only the two RGB images and four permitted robot measurements: joint positions,
end-effector position, end-effector orientation, and gripper positions.
It must not receive depth or exact object poses as inputs.
