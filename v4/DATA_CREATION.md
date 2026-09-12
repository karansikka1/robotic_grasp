# Data creation handoff — 2026-09-12

The user approved the three visible disturbance previews and a stronger 300-trajectory batch. Collection is **paused for a pod rebuild**, not cancelled. Resume data creation next; no new BC training has been requested or launched.

## Preserve these before replacing the pod

Git preserves code and these notes. `.gitignore` excludes trajectory data, splits, checkpoints, and evaluation videos. A fresh clone alone cannot resume the current collection. Keep the existing persistent volume, or copy the directories below to the new volume **before deleting the old one**. Use the same `/workspace/robotic_grasp` path because saved manifests contain absolute paths.

- `/workspace/robotic_grasp/` including `.git`, `v4/trajectories/`, `v4/splits/`, `v4/runs/`, and `evaluation/`.
- `/workspace/.cache/vc1-large/` if continuing VC-1 training: official source weights and provenance. Saved BC checkpoints are self-contained for evaluation.
- The Poetry environment lives under `/root/.cache/pypoetry/virtualenvs/` and can be rebuilt from `poetry.lock`; it is not required to preserve the demonstrations.

The finalized recovery recordings stranded in `/tmp/robotic-grasp-recovery300` were moved into the persistent dataset with SHA-256 verification and full HDF5 dataset reads. Do not rely on `/tmp` surviving a rebuild. The recovery ledger and original scratch inventory are saved with the dataset.

50 GB refers to disk storage, not RAM or GPU memory. Removing 290 superseded mild trajectories freed **22,795,731,313 bytes (22.8 GB)**. All repository MP4s occupied only about **72 MB** at cleanup, so downsampling review videos would not materially help. The full stronger batch is projected to bring workspace usage to approximately **42–44 GB**, before additional feature caches/checkpoints. **100 GB of persistent disk** gives room for the planned expansion; 50 GB can accommodate this batch with limited headroom. `df` reports the shared backing filesystem and did not expose the 50 GB pod-volume quota.

## Approved batch and collection status

Dataset root: `v4/trajectories/robustness300/recovery300/`.

| Category | Saved | Remaining | Target |
| --- | ---: | ---: | ---: |
| Clean | 100 | 0 | 100 |
| Gaussian bursts | 46 | 4 | 50 |
| Actual block drops | 41 | 9 | 50 |
| Sideways errors | 46 | 4 | 50 |
| Gripper interruptions | 19 | 31 | 50 |
| **Total** | **252** | **48** | **300** |

Currently saved: **225 train / 27 validation episodes**, **86,243 transitions**, **20.73 GB** of HDF5 data. Ten review examples are available.

The final split will contain **270 train / 30 validation** trajectories: clean 90/10 and each disturbance 45/5. These are complete episodes, not individual action samples. The original ten clean test episodes remain separate references; no disturbed test set was collected.

Within each disturbance category, the planned 50 episodes target **25 green and 25 blue** manipulations. Each category has five validation slots (indices 0, 11, 22, 33, 44), covering both objects; clean validation is every tenth slot. Layout and perturbation seeds are distinct and frozen. Exclusions cover the original 100 layouts, prior diagnostic fresh seeds, and all reserved candidates from the superseded mild collection.

| Category | Intervention |
| --- | --- |
| Clean | Original teacher with no injected disturbance. |
| Gaussian burst | One XY burst during approach, descent, lift, or transfer; sigma 0.5–0.8, component clipping 0.85–1.0, held draws 3–5 steps, event 14–24 steps. |
| Actual block drop | Open the gripper during lift, 8–14 steps, triggered after an 8–12 cm lift. Accepted demonstrations must lose grasp and show at least 3.5 cm downward travel. |
| Sideways error | XY directional bias during lift/transfer, magnitude 0.8–1.0 for 8–13 steps. Direction varies and is oriented toward table center at onset. |
| Gripper interruption | Delay closure or reopen immediately after contact around picking, 4–10 steps. Reopen examples must actually lose grasp. |

Commands are normalized; one control step is 0.05 s. Stage-triggered events and recovery use privileged state. Their schedules are not yet an actor-independent robustness evaluation protocol.

The external `RecoveryTeacher` recomputes intended actions at the actual visited state. After a lost grasp or recoverable approach/pick failure it opens, retreats, waits for landing, and retries the same object. Blue recovery preserves the green-red base; a broken base or exhausted recovery budget fails the attempt. The original teacher, environment, and simulator are unchanged. Unlike the three preview demonstrations, the ordinary teacher continues during each burst instead of holding an artificial waypoint.

HDF5 `actions` are intended teacher/recovery commands (BC labels); `executed_actions` include injected faults. `perturbation/*` records event masks, command differences, and recovery activity; `supervisor/mode_id` distinguishes ordinary teaching from recovery retreat. Canonical 17-stage labels are preserved. Stage/mode information is privileged diagnostic context, not an approved visual policy input. Only successful trajectories enter the dataset; ordinary failed attempts retain metrics.

The disk quota interrupted publication after 141 successful job records. Another **111 finalized successful HDF5s (9.51 GB)** were recovered without rerunning simulations. Their event measurements and recovery intervals were reconstructed from stored arrays. Original wall-clock timing and textual recovery reasons were unavailable and are explicitly `null`, with provenance in each result. These were all first-candidate episodes. Six incomplete scratch HDF5s are excluded and their slots remain available for normal retries; technical interruptions should not be interpreted as teacher failures.

## Resume on the rebuilt pod

Use Python 3.12 and the existing lockfile. If the new image lacks EGL or Poetry, run the repository setup script as root, then install the locked environment:

```bash
cd /workspace/robotic_grasp
bash scripts/setup_runpod.sh
export PATH="/root/.local/bin:$PATH"
poetry install
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

If the environment already exists, only the `cd` and environment exports are needed. Run the data checks and resume the approved batch:

```bash
poetry run python -m unittest v4.test_teacher v4.test_perturbations v4.test_recovery_teacher
poetry run python -m v4.collect_recovery --workers 8
```

The collector reads `jobs/*.json`, skips successful slots, and writes each remaining slot atomically. It reconstructs the frozen plan from `v4/splits/diverse100.json` and the earlier mild `plan.json`, so preserve those files even though most mild HDF5s were deleted. It checks source hashes for `recovery_teacher.py`, `collect_recovery.py`, `record_trajectory.py`, `teacher.py`, and the two simulator files. **Do not change those files or overwrite the plan when resuming this batch**; a deliberate implementation change needs a new dataset version.

The final `v4/splits/recovery300.json` and dataset `videos.json` are published only when all 300 slots succeed. They are not yet present. `summary.json` and `jobs/` are the current partial collection record. Inspect `collection.log` for the quota incident, and `storage_recovery.json` / `storage_recovery.log` for recovery provenance. Use a new log file when restarting if redirecting output.

## Video review

Ten validation examples are selected in the plan (two per category; both target objects for disturbances). Raw videos have both camera views; annotated versions add event/recovery banners. The page also displays intended/executed XYZ and gripper commands, slow playback, frame stepping, and a jump-to-event button. Recovered videos were rebuilt directly from stored RGB frames.

```bash
poetry run python -m v4.build_recovery_review
python -m http.server 8765 --bind 127.0.0.1 --directory v4/trajectories/robustness300
```

Forward port 8765 in VS Code and open **http://localhost:8765/recovery300/**. The three approved previews remain at **http://localhost:8765/visible_pilots/**. The server must be restarted after a pod rebuild; an old PID file is not evidence that it is running.

## What remains after collection

1. Confirm 300 successful trajectories, category counts, 270/30 split, layout exclusions, and 25/25 target-object balance per disturbance.
2. Audit intended/executed actions, perturbation metadata, canonical stages, finite state and sampled image/depth data, and stable terminal success. Recompute perturbations from the frozen configs and seeds.
3. Replay representative trajectories through the simulator (the 12 pilot slots cover important phase/fault combinations); compare states, actions, and labels. Fully decode the ten raw and ten annotated review videos and check T+1 frames. Some of these checks were completed for the handoff; see the status below.
4. Rebuild the review page and report final transition counts and dataset bytes. Then discuss BC training using the new data; do not treat data collection as learned-policy evaluation.

The earlier mild dataset and three visible pilots passed their separate full audits. Those reports do **not** validate the stronger dataset. A dedicated `audit_recovery.py` was not completed before the quota interruption; its empty placeholder was removed. Use `audit_robustness.py` / `audit_visible_pilots.py` as references, adapting for the recovery teacher and gripper faults. Do not run an empty audit command and infer success.

Handoff verification: **12 focused teacher/perturbation/recovery tests passed**; all 252 successful job files and stable-success flags were checked; all original 100 HDF5s remain; frozen plan/source hashes match. All 111 recovered HDF5s passed copy SHA-256 verification and full dataset reads. **20/20 raw and annotated stronger-dataset videos decoded with T+1 frames**. The ten-example page was rebuilt. These checks do not replace a complete action/state audit or simulator replay of the stronger batch.

## Superseded data and code map

- Original `diverse100`: **80 train / 10 validation / 10 test**, 25,648 / 3,139 / 3,191 transitions. All original trajectories and checkpoints are retained.
- Earlier `robustness300`: generated 300 mild trajectories, but **290 HDF5s were deleted** with user authorization. Ten HDF5s supporting review videos remain. Its split and audit reports are historical, not a usable training dataset. `retirement.json` and `storage_cleanup.json` record this; the old collector now refuses to resume that retired root.
- `visible_pilots`: three review-only trajectories, excluded from training manifests. User approved these and requested varied phases/objects plus gripper faults.
- Collection implementation: `recovery_teacher.py`, `collect_recovery.py`, and optional recorder extensions in `record_trajectory.py`.
- Review: `build_recovery_review.py`.
- Recovery utility: `recover_interrupted_collection.py`. This is specific to this first-candidate scratch inventory; no need to rerun it after successful preservation.
- Versioned handoff metadata: `handoffs/recovery300-20260912/`. These are small provenance snapshots, not a backup of the HDF5 data or checkpoints. The active dataset files remain authoritative.
