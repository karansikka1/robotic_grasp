# Categorized BC demonstrations — 2026-09-12

> **Retired after storage cleanup (2026-09-12):** 290 HDF5 trajectories were deleted; the ten video-review trajectories remain. The split and counts below are historical and cannot be used for training. Resume the stronger batch using [DATA_CREATION.md](DATA_CREATION.md).

The initial robustness dataset contains **300 newly generated successful teacher
trajectories**: 100 clean and 50 in each of four movement-disturbance categories.
Ten percent of each category is reserved for validation. No policy was trained
or evaluated as part of collection.

| Category | Training trajectories | Validation trajectories | Total |
| --- | ---: | ---: | ---: |
| Clean | 90 | 10 | 100 |
| Gaussian noise | 45 | 5 | 50 |
| Dropped commands | 45 | 5 | 50 |
| Delayed commands | 45 | 5 | 50 |
| Sustained bias | 45 | 5 | 50 |
| **Total** | **270** | **30** | **300** |

The new trajectories contain **97,998 transitions**: **88,223 training** and
**9,775 validation**, occupying approximately **23.58 GB** of HDF5 data. Collection
required **312 attempts**; 12 failed attempts are retained as metrics.

The new split is [robustness300.json](splits/robustness300.json). Its `test` list
references the original ten clean `diverse100` test demonstrations unchanged;
these are additional legacy references, not part of the 300 new demonstrations.
The previous 100 demonstration seeds and the five previously inspected fresh
seeds are excluded from every new collection attempt. The original dataset and
split are retained. No disturbed test set was added in this initial batch;
the 30 validation trajectories include all five categories.

## Review videos

[Open the review page](trajectories/robustness300/index.html). Ten selected
validation videos are saved, two per category, with front and wrist views.
The page includes category/example selection, playback-speed control, frame
stepping, and synchronized XYZ plots of intended and executed commands. Clicking
the plot seeks the video; the event button seeks to the first command of the next dropped-command
or sustained-bias burst. Gaussian noise and delay affect commands
throughout the episode and can be inspected with frame stepping.

The marker refers to the command applied after the displayed frame. Frame zero
is the post-reset observation; frame T is the terminal observation after T
commands. Video length is T+1 frames at 20 Hz. Reported control time is T/20 and
excludes the one reset observation action.

Serve the page locally if the IDE does not play videos in a file preview:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory v4/trajectories/robustness300
```

Then open `http://localhost:8765/`. On a remote pod, forward port 8765 through
the VS Code Ports panel. The current session started this server; its PID is
saved in `v4/trajectories/robustness300/review_server.pid`.

## Disturbance definitions

Only the first three normalized action values (XYZ movement) are modified.
Rotation and gripper commands always retain the teacher's current values.
Every simulator control step and its observations remain recorded. Disturbance
schedules use a dedicated RNG and no teacher stage, contacts, or exact state;
the same action wrapper can be used for later learned-policy evaluation.

| Category | Fixed initial settings |
| --- | --- |
| Clean | Intended and executed actions are identical. |
| Gaussian noise | Independent XYZ Gaussian samples every step; sigma 0.015; clipped componentwise to ±0.045. |
| Dropped commands | On each idle step, probability 0.03 of starting a 1–3-step burst of zero XYZ movement. Gripper and rotation remain current. |
| Delayed commands | Execute the XYZ command from one step earlier, a 50 ms delay. Missing initial history is zero movement. Gripper and rotation remain current. |
| Sustained bias | On each idle step, probability 0.02 of starting a 5–10-step burst. A uniformly random direction has constant normalized magnitude 0.02 throughout that burst. |

The controller scales normalized translation by 0.05 m. Thus Gaussian sigma
corresponds to 0.75 mm of commanded translation, the Gaussian component bound to
2.25 mm, and the bias norm to 1 mm. These are command perturbations, not claims
about measured end-effector displacement. Executed translations are clipped to
[-1, 1]. All phases, including close/release/settle, use the same step-based
schedule; no privileged phase gating is used.

A pilot completed one trajectory per category on the first attempt using these
settings. Those five trajectories occupy their preassigned training slots in
the final dataset. The settings were not changed after the pilot.

## Labels and file format

The existing teacher and supplied simulator/environment are unchanged. New
HDF5 files retain legal RGB/proprioceptive observations, privileged depth and
oracle telemetry, teacher stage, and terminal observations. Added format-v2
fields are:

- `actions[T,7]`: the intended teacher action at the actual visited state;
  this remains the BC target consumed by existing dataset loaders.
- `executed_actions[T,7]`: the perturbed command passed to the simulator.
- Attributes `category`, `perturbation_seed`, `perturbation_config`, and
  `action_semantics` record trajectory-level provenance.
- `perturbation/{active,changed,delta,event_id,event_step,event_duration}` records
  per-transition intervention metadata. `delta` is executed minus intended;
  `active` records the schedule, whereas `changed` records an actual command
  difference. Clean/inactive event IDs are -1; continuous delay uses event ID 0
  with duration -1. Gaussian events last one step.

A correction label is recomputed by the closed-loop teacher at every new state.
Noise is applied during simulation, not afterward to saved training labels.
Category tags are metadata, not additional actor inputs. Visual training that
uses depth is still a privileged diagnostic; the final RGB/proprioception-only
contract has not changed.

## Collection, restart, and interpretation

[The frozen plan](trajectories/robustness300/plan.json) assigns category,
train/validation membership, video selection, and up to eight distinct candidate
layout/noise seeds to each slot before simulation. Failed candidates remain
reserved and recorded. A failed attempt is replaced only within its original
slot and partition. Seeds are unique across all candidates and partitions.

The collector resumes completed slots and recovers saved attempt metrics after
an interrupted job-ledger write. HDF5 data is streamed to temporary local storage
and copied to persistent storage on success. Failed HDF5/video files are removed;
failure metrics remain in the per-slot records. Success requires the unchanged
teacher's ten consecutive official full-stack success checks. The teacher can
abort on lost grasps or stage timeouts and does not implement arbitrary regrasp
recovery. These mild disturbances demonstrate the corrections it can complete.

Consequently, the dataset is conditioned on teacher success. Collection
completion rates measure teacher attempts, not student-policy robustness, and
the categories use different layouts rather than paired counterfactual trials.
There is no claim that these perturbations improve learning before training and
closed-loop evaluation are run. Five disturbed validation examples per category
are a small diagnostic set.

```bash
# First pilot; its successful trajectories are retained for the full collection.
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/.local/bin/poetry run python -m v4.collect_robustness --pilot --workers 5

# Fill/resume the frozen 300-trajectory plan. Completed datasets collect nothing.
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/.local/bin/poetry run python -m v4.collect_robustness --workers 8

# Check artifacts and replay one complete recorded trajectory per category.
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/.local/bin/poetry run python -m v4.audit_robustness --replay

/root/.local/bin/poetry run python -m v4.build_robustness_review
```

Use a new output directory and split filename for different counts or settings;
the collector rejects changes to an existing frozen plan. A larger dataset can
be constructed later while explicitly preserving existing partition assignments.

Artifacts live under [trajectories/robustness300](trajectories/robustness300):
`summary.json`, `jobs/*.json`, `episodes/*/*/metrics.json`, `videos.json`,
`validation.json`, `replay_validation.json`, source hashes, collection logs, and
`index.html`. The source additions are `perturbations.py`, `collect_robustness.py`,
`audit_robustness.py`, and `build_robustness_review.py`; `record_trajectory.py`
adds optional metadata, video selection, and temporary-storage support while
preserving the original recorder defaults.

## Completed validation

- **21 relevant tests passed**, including disturbance bounds and timing, intended/executed label alignment, split counts and seed exclusions, restart recovery, old teacher recording, and existing BC/history loaders.
- **300/300 new trajectories passed** artifact checks: every intended/executed action and perturbation field, all proprioceptive/oracle values, initial/middle/final camera frames, terminal observations, and ten consecutive official success flags. RGB/depth payloads were sampled at three frames per trajectory rather than exhaustively decoded.
- **10/10 videos fully decoded**, each with exactly T+1 frames.
- **5/5 complete simulator replays passed**, one per category. Every pre-action object/grip-site position and teacher action/stage matched exactly, as did initial/terminal observations. Every replay reached stable official success.
- The final page and ten video URLs passed local HTTP checks; JavaScript syntax passed. Collection-source hashes still match the saved provenance.
