# BC perception and control diagnostics — 2026-09-09

Removing previous-action input produced the first multi-demonstration visual BC
training-seed stacks, but did not produce fresh-seed full stacks. A sufficiently
precise one-demonstration fit executed successfully. A small state BC policy
worked substantially better when supplied the teacher stage, showing that
perception and task sequencing must be tested separately.

These are diagnostic policies. The visual action policies still consume depth;
the state policies consume exact object/grip-site positions, and some receive
teacher stage. None is a final actor satisfying the RGB/proprioception-only contract.

## Results

| Experiment | Training-seed stacks | Fresh-seed stacks | Mean successful control steps / seconds |
| --- | ---: | ---: | --- |
| Previous spatial Smooth L1 baseline | 0/5 | 0/5 | No successes |
| Same spatial policy, previous action disabled | 2/5 | 0/5 | Training: 310 / 15.50 |
| Exact object positions + robot state + teacher stage, 128×128 MLP | 4/5 | 3/5 | Training: 297.75 / 14.89; fresh: 299.33 / 14.97 |
| Same state MLP, teacher stage removed | 0/5 | 0/5 | No successes |
| Fixed stage-assisted state controller, RGB-predicted object positions | 0/5 | 0/5 | No successes |
| One visual demonstration, 300 epochs | 0/1 | Not evaluated | No full stack; green placed on red |
| Same one-demo checkpoint, 500 further epochs at lower LR | 1/1 | Not evaluated | 320 / 16.00 |

All training-seed initial observations matched their recordings exactly. The
multi-demo models use the existing 80/10/10 split and the same five sampled
training seeds and five fresh seeds (1000000–1000004). Five episodes are a small
diagnostic sample, not a precise estimate of success probability.

Success requires ten consecutive official full-stack successes. Timing excludes
the one post-reset zero-action observation step; saved simulator-step counts
include it. Visual and stage-free state failures run to 900 policy actions.
Stage-assisted state rollouts stop early when a teacher guard fails and record
the reason; this is a privileged sequencing diagnostic, not autonomous recovery.

### No-previous-action comparison

Run: `v4-diverse100-resnet18-concat-grid4-no-prev-88b5c91b-77eb-48c6-8f61-358fcd880e52`.
Selected epoch 23 of 30 by validation loss. Kept the spatial ResNet-18 4×4 camera
concatenation, depth backbone, three-observation history, initialization seed,
Smooth L1 objective, optimizer settings, and split. Both frozen backbones exactly
match the previous spatial checkpoint. The seven previous-action channels are
zeroed at the shared model boundary in training and inference, retaining layer
shape and random initialization. The original action cache remains available
for correctly identifying gripper switches in offline diagnostics.

Test translation MAE worsened from **0.03097 to 0.08031**, while training-seed
stacking improved from **0/5 to 2/5**. Test overall MAE is 0.04042, gripper-switch
accuracy 26/40. Thus lower aggregate teacher-forced action error did not rank
closed-loop behavior correctly. This controlled result supports investigating
previous-action dependence; it does not prove a single cause of all failures.

Fresh seeds 1000000 and 1000003 achieved held green lifts; seed 1000003 placed
green on red. Seed 1000001 lifted incorrect objects. None completed the full stack.

### Alignment and scaling audit

Replayed all 336 recorded actions for seed 1390308241. Initial observations,
pre-action exact object positions, and grip-site positions matched exactly.
A live teacher produced identical actions and stage labels at every step:
maximum action error 0, position error 0, and no stage mismatches. Replay finished
with stable official success. Translation actions span approximately [-0.6, 0.6],
rotation labels are exactly zero, and gripper labels are -1/+1. Teacher translation
scaling is 0.05 m per normalized unit. This checks one entire trajectory; it is
not a proof about every possible data path.

### Privileged state and stage

A 128×128 ReLU MLP predicts seven tanh-bounded normalized actions with Smooth L1.
Inputs are object positions (9), exact grip-site position (3), legal robot
proprioception (16), derived object-to-grip-site offsets (9), and a 17-way stage
one-hot encoding containing object/phase identity. Continuous inputs are
standardized on training data only. Training: 200 epochs, batches of 256,
AdamW LR 1e-3, seed 0; epoch 193 selected by validation loss.

Training/test translation MAE: **0.00958 / 0.02349**. Offline gripper accuracy:
100%. During rollout, the unchanged teacher state machine observes student
states and supplies the stage; its action is discarded. It still provides
privileged transition tests, timing, grasp memory, and guards. Consequently the
3/5 fresh success result validates a learned low-level controller with privileged
sequencing, not a learned full-task planner or a perception-only conclusion.

Removing stage uses the same model shape with zero stage channels in training
and inference. This controller never calls the teacher during rollout. Its
training/test translation MAE is **0.05646 / 0.09974**, and all ten rollouts fail.
Despite no full stacks, the stage-free model achieved three held green lifts
and two green-on-red placements in each five-seed partition. This implicates
missing phase/history or fitting difficulty when stage is absent;
a feedforward state policy cannot establish that a recurrent state policy would
also fail. It is not evidence that more vision data alone will solve sequencing.

### Single-demonstration fitting

Used training seed 1390308241, the first seed in the existing fixed sample.
Same frozen spatial visual/depth policy, three observations, no previous action.
Checkpoint selection uses training resubstitution error only; there is no
validation or held-out test claim. The generic trainer's `validation_loss` log
means training resubstitution error for this diagnostic and is labeled as such
in `train_action_metrics.json` and `split.json`.

| Fit | Training Smooth L1 | Translation MAE | Switches | Same-seed result |
| --- | ---: | ---: | ---: | --- |
| 300 epochs, LR 3e-4 | 0.00004131 | 0.007441 | 4/4 | Green placed; blue not stacked |
| Best checkpoint + 500 epochs, LR 1e-5, fresh optimizer | 0.000001877 | 0.002010 | 4/4 | Full stack in 320 actions |

The successful refinement shows this visual architecture can execute one learned
trajectory. Small remaining errors mattered in this case. It does not show
recovery robustness or unseen-layout generalization.

### Progressive replacement with RGB-predicted object positions

Kept the successful state-controller checkpoint fixed and retained the live
privileged teacher stage and exact grip-site position. Replaced only the three
object positions with a learned regressor using the two RGB cameras and legal
robot proprioception. Teacher stage transitions still use exact simulator state;
this deliberately isolates the low-level controller's object-position input.

The regressor uses frozen ImageNet ResNet-18 4×4 spatial features, separate camera
slots concatenated with proprioception, and a 128-unit MLP predicting nine position
coordinates. Feature and target normalization use training statistics only.
Training: 30 epochs, batches of 64, AdamW LR 3e-4, seed 0; epoch 25 selected on
validation coordinate MAE. No depth or true object positions enter the RGB
regressor at inference.

Training/validation/test coordinate MAE is **4.26 / 20.67 / 23.55 mm**. Test mean
3D position errors for red/green/blue are **57.90 / 46.75 / 51.56 mm**. Coordinate
MAE is not Euclidean position error. These errors are large relative to the
teacher's 8 mm waypoint tolerance. The bridge achieved **0/5 training and 0/5
fresh stacks**; all fresh episodes timed out in the first approach stage.

The tested frozen-feature localization head generalizes poorly, and the fixed
controller does not tolerate its error. This is evidence about this estimator
and interface, not proof that RGB cannot provide sufficient precision. The
end-to-end visual action policies also use depth, so this RGB-only bridge is
not a matched architecture comparison against them.

[RGB-position run, pose metrics, rollout metrics, and seed videos](runs/v4-diagnostic-rgb-positions-39a6a311-8cca-447e-92bc-eb4a1661a007).

## Reproduction

Use the existing Poetry environment and cached ImageNet weights:

```bash
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_HOME=/workspace/.cache/torch
export POETRY=/root/.local/bin/poetry

# Controlled visual comparison: automatic five train / five fresh video rollouts.
"$POETRY" run python -m v4.train_bc --policy transformer --rgb-backbone resnet18 \
  --camera-fusion concat --rgb-pool-size 4 --no-previous-action \
  --exp-name v4-diverse100-resnet18-concat-grid4-no-prev \
  --split-manifest v4/splits/diverse100.json --epochs 30 --batch-size 64 \
  --history-length 3 --image-size 128 --device cuda \
  --train-rollout-episodes 5 --rollout-eval-episodes 5

"$POETRY" run python -m v4.control_diagnostics replay
"$POETRY" run python -m v4.control_diagnostics privileged --epochs 200 --device cpu
"$POETRY" run python -m v4.control_diagnostics privileged --without-stage --epochs 200 --device cpu
"$POETRY" run python -m v4.control_diagnostics one-demo --epochs 300 --device cuda
"$POETRY" run python -m v4.control_diagnostics one-demo --epochs 500 --device cuda \
  --learning-rate 1e-5 --warm-start v4/runs/<one-demo-run>/best.pt

# Progressive replacement: same successful state controller, RGB-predicted object positions.
"$POETRY" run python -m v4.pose_bridge v4/runs/<privileged-with-stage-run>/best.pt
```

## Videos and artifacts

- [Visual no-previous-action run](runs/v4-diverse100-resnet18-concat-grid4-no-prev-88b5c91b-77eb-48c6-8f61-358fcd880e52): `best.pt`, `test_metrics.json`, `train_rollout_metrics.json`, `rollout_metrics.json`.
- [Successful multi-demo visual rollout](../evaluation/v4/v4-diverse100-resnet18-concat-grid4-no-prev-88b5c91b-77eb-48c6-8f61-358fcd880e52-train-1390308241-f451c44a-8c12-45fc-9b99-723980dcfc36/episode_000.mp4).
- [Fresh visual evaluations](../evaluation/v4/v4-diverse100-resnet18-concat-grid4-no-prev-88b5c91b-77eb-48c6-8f61-358fcd880e52-best-5de554bf-d6ed-41f4-8e6a-faa65f4c0523).
- [Privileged controller](runs/v4-diagnostic-privileged-866abd17-544b-4eef-8b8e-ef93027b6ed1): `best.pt`, `action_metrics.json`, `rollouts.json`, `seed_<seed>.mp4`.
- [Privileged successful fresh rollout](runs/v4-diagnostic-privileged-866abd17-544b-4eef-8b8e-ef93027b6ed1/seed_1000000.mp4).
- [Stage-free state controller](runs/v4-diagnostic-privileged-no-stage-1b5b03e3-7e05-420a-a12b-f36e7cd558b5).
- [Single-demo initial fit](runs/v4-diagnostic-one-demo-7ba2fd63-66a7-4bc7-9b25-862ded81a13b).
- [Single-demo refined fit](runs/v4-diagnostic-one-demo-4087c423-acf5-4927-a59b-3c7e2546136b).
- [Successful refined single-demo video](../evaluation/v4/v4-diagnostic-one-demo-4087c423-acf5-4927-a59b-3c7e2546136b-train-1390308241-56d1452f-de5b-47ac-b79a-ef2756ad3df3/episode_000.mp4).
- [Replay audit](runs/v4-diagnostic-replay-c6444073-65e1-4a23-acbc-9cbea8868c6e/replay_audit.json).

## Interpretation and next work

Keep previous-action input disabled for the next visual experiment. Avoid
selecting approaches by aggregate action MAE alone. The single-demo result
supports checking fit precision and transition-specific errors before adding
another backbone. The stage ablation motivates a learned task-phase representation
or longer memory; supplying oracle stage is only a diagnostic upper bound.

The 100 demonstrations cover initial placements but contain successful,
mechanical teacher trajectories. More random layouts are different from recovery
coverage. A useful diversity experiment would add valid teacher corrections after
small state/action perturbations or student-induced deviations, especially around
grasp, lift, transfer, and release. The existing teacher can throw on failed
contacts or unreachable stages, so successful recovery labeling must be checked
before applying a DAgger-style collection scheme. No recovery collection was run
in this diagnostic batch.

Validation: 28 regression tests passed, covering model/checkpoint compatibility,
action-history masking, causal windows, privileged feature packing, stage removal,
and batch/online RGB-position encoding. Supplied environment and simulator files
were not modified.

All **42 evaluation videos** decoded successfully with exactly the expected
number of frames. [Machine-readable comparison](runs/diagnostics20260909-comparison.json)
and [video validation](runs/diagnostics20260909-video-validation.json) preserve
per-run success, timing, milestones, and video paths.


## Persistent recurrent follow-up

The requested episode-persistent LSTM comparison for spatial visual BC without previous actions and exact state plus teacher stage is recorded in [RECURRENT.md](RECURRENT.md), including architecture, training semantics, measured results, and video paths.
