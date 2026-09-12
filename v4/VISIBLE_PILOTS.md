# Three visible disturbance previews — 2026-09-12

The user found the initial robustness300 disturbances too subtle and clarified
that a dropped block should physically leave the gripper. The earlier
`dropped_commands` category only issued zero movement commands and never opened
the gripper. Collection is now paused after **three review-only examples**.

[Open the new review page](trajectories/robustness300/visible_pilots/index.html),
served at `http://localhost:8765/visible_pilots/`. The original review page also
links to it. Each example has an annotated full video, half-speed playback,
frame stepping, “Play disturbance”, and “Watch recovery” controls. The player
initially seeks just before the event.

| Example | Disturbance window | Full-stack completion actions / seconds |
| --- | --- | --- |
| gaussian_burst | 0.35–1.35 s | 309 / 15.45 s |
| drop_block | 2.70–3.30 s | 373 / 18.65 s |
| sideways_error | 2.70–3.20 s | 296 / 14.80 s |

- **Physical block drop:** open the gripper for 12 control steps while holding
  green above the table. Green falls **10.50 cm**, loses grasp contact, and is
  regrasped at **6.25 s**. An external recovery controller clears the gripper,
  waits for green to settle, and restarts the green pick at its actual position.
- **Gaussian burst:** one 20-step / 1-second burst near the first approach.
  XY Gaussian sigma is 0.7 normalized units, clipped componentwise to ±0.9;
  each draw persists for five steps. This is temporally correlated, bounded
  Gaussian noise. Maximum measured gripper displacement from burst start is
  **3.73 cm**. Z and gripper commands are unchanged by this disturbance.
- **Sideways error:** one 10-step / 0.5-second Y-command bias of magnitude 0.95
  while holding green, directed toward the center of the table. Maximum measured
  block displacement from event start is **5.32 cm**. Grasp is retained during
  the event; the teacher then resumes stacking.

All three use the same starting layout (seed **220769969**) and disturbance
seed **34**. Each completed on its first generation attempt. These are stronger,
stage-triggered illustrative interventions with explicit hold/recovery logic,
not the earlier oracle-independent random step schedules. They are not evidence
of recovery over arbitrary dropped-object states or of improved learned policies.
Movement figures are measured displacements during the event, not differences
against a paired clean rollout.

The examples live outside all BC train/validation/test manifests. Each HDF5 file
is marked `review_only`; preview/recovery stage labels extend the ordinary
teacher stage set and need explicit treatment before any future BC ingestion.
Intended supervisor commands and actually executed commands remain separate.
The original teacher, simulator, environment, and 300-trajectory dataset are
unchanged. No additional large collection or training was launched.

Validation: **3/3 simulator replays** reproduce all recorded pre-action object
and grip-site positions, intended/executed commands, stage labels, and
perturbation metadata exactly. Initial and terminal observations match exactly,
and all three replays achieve stable official stacking success. All three raw
and three annotated MP4s decode to T+1 frames. The page and three video URLs
pass local HTTP checks, and page JavaScript passes syntax checking. Frames
before/after the drop were visually inspected.

Source: `visible_perturbation_pilots.py`, `build_visible_pilot_review.py`, and
`audit_visible_pilots.py`. Artifacts: `examples.json`, `validation.json`, and
per-example `metrics.json`, `trajectory.h5`, `raw.mp4`, `review.mp4` under
`v4/trajectories/robustness300/visible_pilots`.

Rebuild the page with `poetry run python -m v4.build_visible_pilot_review`.
Generation uses `python -m v4.visible_perturbation_pilots` with the existing
Poetry environment and `MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`;
rerunning it generates new preview artifacts. **Wait for user feedback on these
three examples before generating more.**
