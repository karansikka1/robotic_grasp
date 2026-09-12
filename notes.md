# Ultra Policy Training - Handoff Notes

**Current data-creation handoff (2026-09-12):** see [v4/DATA_CREATION.md](v4/DATA_CREATION.md) before resuming collection or rebuilding the pod. It supersedes the older “wait for preview review” status below and records storage cleanup, preserved trajectories, and restart commands.

## Objective and hard constraints

- Train and evaluate a policy that stacks **red (bottom) -> green -> blue (top)**.
- Do **not** modify `motion_planning/environment.py` or `motion_planning/simulator.py`. New training, evaluation, policy, and utility files are fine.
- Do not add dependencies beyond the existing lockfile. PyTorch, torchvision, NumPy, OpenCV, TensorBoard, etc. are already available; do not assume Stable-Baselines3.
- The final checkpoint may consume only:
  - `robot0_joint_pos`
  - `robot0_eef_pos`
  - `robot0_eef_quat`
  - `robot0_gripper_qpos`
  - `frontview_image`
  - `robot0_eye_in_hand_image`
- Training may use privileged information such as `frontview_depth` and `task_complete`, but neither may be an inference-time policy input.
- Evaluation must report both success rate and speed (for example, control steps or seconds to first success).
- Final deliverables: code ZIP, three rollout videos, and a short write-up including approaches that were tried but not retained.

## Simulator contract

- `Simulator.step(action)` returns the six allowed policy observations plus:
  - `frontview_depth`: metric front-camera depth, shape `(256, 256, 1)`
  - `task_complete`: boolean success signal
- `Simulator.reset()` returns `None`; the first public observation can be obtained with a zero-action step after reset. Account for that dummy step consistently in evaluation timing.
- Action space is a normalized 7-D vector in `[-1, 1]`:
  - first 3: end-effector XYZ delta
  - next 3: orientation delta (described by the task as roll/pitch/yaw)
  - final value: gripper; negative opens and positive closes
- The default controller is Panda `OSC_POSE`. Maximum scaled deltas are approximately `0.05 m` translation and `0.5 rad` rotation per command.
- Control frequency is 20 Hz, so one action is `0.05 s`; MuJoCo runs 25 internal `0.002 s` physics steps per action.
- Default horizon is 1,000 actions (50 simulated seconds). Success does not automatically set robosuite's `done`; the external evaluation loop should stop when `task_complete` becomes true.
- Cube placement is randomized on reset. Seed NumPy before constructing/resetting the simulator when reproducible episodes are needed.
- Raw camera images are vertically inverted by MuJoCo. `run_sim.py` flips them only for video output. Use one consistent preprocessing convention during training and inference.

## Success and reward caveat

- `_check_success()` is the authoritative task condition: red below green below blue, with red-green and green-blue contact.
- The underlying robosuite `env.step()` produces a reward, but `Simulator.step()` intentionally discards it.
- The existing `staged_rewards()` / `reward()` logic is not aligned with the complete objective: it considers only red and green and rewards a red-on-green-style interaction while ignoring blue.
- Do not repair this inside the supplied environment. Use `task_complete` as the official success metric and construct any training-only reward externally.

## Current experiment plan: PPO first

Start with a deliberately basic PPO implementation as a baseline:

1. Filter observations so the actor receives only the six permitted fields.
2. Downsample and normalize both RGB images, encode them with small CNNs (or a shared CNN), encode proprioception with an MLP, concatenate the embeddings, and produce policy/value outputs.
3. Initially use a Gaussian policy for all seven normalized actions and clip to `[-1, 1]`. The gripper effectively uses the sign of the final output; a separate binary gripper head can be added later if needed.
4. Minimal sparse training signal:

   ```python
   reward = 10.0 * float(obs["task_complete"]) - 0.001
   episode_finished = bool(obs["task_complete"])
   ```

   The step penalty also encourages faster solutions.
5. Track episodic return, success rate, time-to-success, entropy, value loss, policy loss, KL divergence, and whether the rollout buffer ever contains a positive reward.

Treat sparse PPO as a short diagnostic. Random exploration is extremely unlikely to discover the full two-pick/two-place sequence. If no successful rollout is collected, tuning PPO hyperparameters alone will not provide a learning signal.

## Escalation path if basic PPO stalls

Add complexity one item at a time and preserve the same final actor input contract:

1. Training-only, success-aligned dense rewards for reach -> grasp -> lift -> align -> place green on red, followed by the same stages for blue on green.
2. Curriculum training over those stages.
3. Asymmetric actor-critic: actor uses only allowed RGB/proprioception; critic may use privileged training state.
4. Generate successful trajectories with a privileged scripted teacher, behavior-clone the RGB policy, then use PPO for fine-tuning and recovery behavior.
5. If behavior cloning suffers from compounding errors, collect corrective teacher labels on states visited by the learned policy (DAgger-style collection).

Exact MuJoCo cube poses are likely covered by the permission to use privileged training information, but depth and `task_complete` are the examples explicitly named in the prompt. If using exact internal poses, keep access entirely in external training/data-generation code and explain clearly that the final checkpoint does not require them.

## Suggested next-session starting point

1. Run `make run-sim` on the GPU machine and confirm rendering works.
2. Build a thin external rollout adapter that filters policy inputs, handles the post-reset zero action, stops on `task_complete`, and records episode statistics.
3. Implement and overfit PPO on a small fixed set of seeds as a pipeline sanity check.
4. Evaluate on held-out randomized seeds and apply the go/no-go rule above.
5. Save checkpoints containing only the actor and its RGB/proprioceptive preprocessing state; verify inference never reads depth or completion.

## Experiment log

| exp_name | network | reward | results | note |
| --- | --- | --- | --- | --- |
| `v1-vanilla-ppo` | Frozen ImageNet MobileNetV3-Small shared by front/wrist RGB + a second frozen MobileNetV3-Small for depth; learned projections and proprio MLP are summed, then separate MLP actor/value heads. | `10 * task_complete - 0.001` per control step | Pending | Privileged-depth pipeline baseline. Mini-eval on fixed seeds 0–4 every 10 PPO updates; final eval on fixed seeds 0–24. Not a valid final actor because depth is unavailable at inference. |
| `v2-green-lift` | Reuse v1's privileged-depth PPO policy initially, to isolate the task/reward change. | +1 first green grasp; +5 held green lift (5 cm, 5 consecutive steps). No step penalty or distance shaping. | Reward tests and short CUDA integration passed; learning results pending | Simpler learning diagnostic before returning to stacking. |
| `v2-history` | Three consecutive observations + previous action; frozen RGB/depth encoders and learned temporal fusion. | Signed reaching progress (scale 1), +1 first grasp, +5 held lift. | Slow, inconsistent learning: 8 grasp episodes and 1 held lift in 368 training episodes at 184,320 steps; mini evaluation at 122,880 achieved 1/10 lifts, subsequent six checkpoints 0/10. | [Report and preserved episode 006](v2/REPORT.md); next experiment undecided. |
| `v3-state` | MLP actor/critic on 47 privileged state values. | Reaching progress +1 contact +5 held lift. | At 204,800 steps: 1/5 evaluation contact episodes, 0/5 held lifts; contact did not establish secure gripping. | Next: five-step contact bonus, initialized from this checkpoint. [Details](v3/README.md). |
| `v4-scripted-teacher` | No learned network: privileged closed-loop waypoint controller using exact object poses, gripper pose, bilateral grasps, object contacts, and official success. | None; actions come from the scripted state machine. | 6/6 single-attempt successes: seed 0 plus five randomly selected seeds. The random-seed runs took 295–327 actions (mean 312.2). | Streams legal observations, privileged oracle state, stage labels, and actions to HDF5; retains HDF5/MP4 only after stable official success. |
| `v4-privileged-bc` | Reuses v1's frozen pretrained RGB/depth MobileNetV3-Small encoders, summed learned projections, proprio MLP, and continuous actor. Frozen CNN features are cached once. | Smooth L1 regression from privileged observation to the teacher's normalized 7D action. | Pipeline tests pass; pretrained cloud training pending. Split seed 0 gives 1,566 train samples from 5 trajectories and 315 test samples from held-out seed 347024. | Split is strictly by whole trajectory. This is a privileged-depth behavior-cloning baseline, not a valid final inference policy. |

## Original v2: grasp and lift the green bar (baseline commit 7762977)

- Motivation: v1 rollouts reported no successes and repeated -0.500 episode returns. Test whether PPO learns a shorter manipulation task before attempting the full stack.
- Target the green object (`cubeB` / `cubeB_body_id`), not cubeA, which the existing staged environment reward targets.
- Implemented starting reward: +1 once per episode for a confirmed green-object grasp, then +5 for lifting it 5 cm above its post-reset resting height while still grasped for 5 consecutive control steps (0.25 s). End the episode on the successful held lift. These are initial design choices, not tuned settings.
- Detect grasp using contacts from both gripper finger groups with the green object; a closed gripper alone is insufficient. Give each milestone bonus only once, so repeated release/regrasp cannot accumulate grasp bonuses.
- Two binary bonuses are still sparse before the first grasp. If v2 does not learn, try a gripper-to-green distance reward as v3. No distance reward or initialization curriculum is included in v2.
- Read object poses and contacts only in an external training/reward adapter. Keep the simulator task definition intact and preserve v1 for comparison. Reuse v1's observation/model setup for this diagnostic, including its explicitly privileged depth input.
- v2 evaluation must use the same grasp-and-held-lift success criterion, rather than the existing full-stack `task_complete` flag.
- Log grasp rate, lift success rate, maximum lift height, steps to grasp/lift, and individual reward components. Compare with an untrained policy on fixed seeds; inspect rollout videos. Begin on a small fixed seed set, then evaluate on held-out seeds.
- Implementation: external task adapter in `v2/task.py`, shared v1 model/PPO optimizer, v2 training and checkpoint evaluation commands. Model and automatic evaluations share the run UUID. Task settings are saved in checkpoints.
- Run: `bash scripts/v2_smoke_train.sh`. For a longer comparison run: `bash scripts/v2_smoke_train.sh --total-timesteps 100000 --rollout-steps 1024 --evaluate-untrained`.
- Validation: five reward/adapter/evaluation tests passed. A four-step CUDA training run completed two PPO updates, two video mini evaluations, and final video evaluation. This checks the pipeline, not learning performance.
- v2 returns are 0 without grasp, 1 for grasp without lift, and 6 for successful grasp-and-held-lift with default settings.

### v2 smoke result — 2026-09-07

- Run: `v2-smoke-70fbf0c2-528c-4603-b78a-05a302e08ab9`, launched with `bash scripts/v2_smoke_train.sh`.
- CUDA, pretrained encoders, 2,048 training steps, 256-step rollouts, four optimization epochs, minibatches of 256; eight updates completed.
- Four completed training episodes: no grasps or lifts, return 0 each. Training took approximately 61 seconds.
- Final deterministic evaluation on seeds 0–24: 0/25 grasps and 0/25 successful held lifts; every episode reached 500 steps. Mean maximum green height increase: approximately 1.33 mm, below the 50 mm target.
- Checkpoint: `v2/runs/v2-smoke-70fbf0c2-528c-4603-b78a-05a302e08ab9/checkpoint_final.pt`.
- Evaluation metrics and 25 videos: `evaluation/v2/v2-smoke-70fbf0c2-528c-4603-b78a-05a302e08ab9/`.
- Execution completed without errors. This short smoke run produced no positive reward; it does not establish whether a longer v2 run can learn. Distance shaping remains a possible v3 experiment.

### v2 distance-progress reward — 2026-09-07

- Decision: add reaching progress to v2, instead of creating v3. Baseline commit: `7762977`.
- Motivation: run `7366757e-aed0-4d49-9e09-46899c4901db` had 70 completed episodes without grasps/lifts at the inspected snapshot (~34,816 completed update steps). Mini evaluations at 10,240, 20,480, and 30,720 also had zero grasp/lift success.
- New per-step term: `reach_reward_scale * (previous_distance_m - current_distance_m)`, using the distance from the gripper grip site to green's body center. Default scale 1.0 gives +0.01 for 1 cm closer and -0.01 for 1 cm farther; hovering earns zero.
- The undiscounted reaching return is scale times the initial-to-final distance reduction. A round trip adds no net reaching reward. This is a progress heuristic, not an invariant shaping claim for the discounted PPO objective.
- Keep the original +1 first-grasp and +5 held-lift bonuses and success definition. No constant step penalty or extra actor inputs. Set `--reach-reward-scale 0` for the original sparse reward.
- Record initial, final, and minimum gripper distance and reaching return per episode in stdout/TensorBoard, plus their evaluation averages in metrics.json/TensorBoard.
- Saved checkpoints contain the reaching scale; standalone evaluation interprets missing scale in old checkpoints as 0. Existing runs need a restart to load these changes.
- Validation: 10 reward/adapter/evaluation tests passed; short CUDA integration collected nonzero reaching rewards before any grasp. Long-run learning results are pending.

### v2 temporal policy variant — 2026-09-07

- Added current + previous two observations and the previous 7-D normalized action. Keep the reaching, grasp, and held-lift reward settings unchanged for comparison.
- Cache frozen visual/proprioceptive features; fuse each frame, concatenate the three frame embeddings and previous action, then apply a learned fusion layer before the existing actor/value heads.
- Start each episode with repeated first-frame features and zero previous action. Preserve training history across PPO updates; evaluation uses independent history and resets every episode. Bootstrap value reads do not append duplicate frames.
- New training defaults: `--policy history --history-length 3`. Use `--policy single` for the previous architecture. History covers 0.1 seconds at 20 Hz.
- `scripts/v2_smoke_train.sh` starts a new `v2-history` experiment and uses 10 mini-evaluation episodes (seeds 0–9), still every 10 updates. Final evaluation remains 25 episodes.
- Checkpoints save history length, and the standalone evaluator loads old single-observation and new temporal checkpoints with matching architectures.
- Validation: 16 reward/history tests passed; a short CUDA run crossed rollout and episode boundaries with mini evaluations between updates. Long-run progress is recorded below and in the [v2 experiment report](v2/REPORT.md).

### Reaching run progress before shutdown — 2026-09-07

- Run: `v2-smoke-6b47c187-e82d-4e22-9d88-71bd698fb63d`. This job uses the single-observation policy with reaching reward, not the later history variant.
- Snapshot: 71,680 completed training steps (update 70), 143 completed training episodes. Five episodes registered a contact-based grasp; none achieved a held lift.
- All completed mini evaluations through step 71,680 had zero grasps and lifts. Evaluation uses deterministic actions on seeds 0–4, unlike stochastic training on different seeds.
- At step 51,200, mean closest distance was 11.4 cm, final distance 24.0 cm, and reaching return +0.0312. At step 61,440 these were 15.7 cm, 25.6 cm, and +0.0148. Reaching improved over the 10,240-step evaluation (23.0 cm closest, 69.5 cm final, -0.4239 return), but progress is uneven.
- Latest saved checkpoint: `v2/runs/v2-smoke-6b47c187-e82d-4e22-9d88-71bd698fb63d/checkpoints/step_000071680.pt`. The 71,680-step evaluation was the best completed one by mean closest and final distance: 11.1 cm closest, 23.1 cm final, reaching return +0.0396.
- The 71,680-step mini evaluation finished during inspection. The assistant did not stop the job. No final 100,000-step result is implied by this snapshot.

### Temporal run results approaching 200,000 steps — 2026-09-08

- Run: `v2-history-ba0c88b3-bcda-4c82-b4a7-2892e9a0fe4f`. Adding temporal context has not produced fast or reliable learning, although this policy has discovered gripping actions and a held lift.
- Fixed training snapshot at 184,320 steps (180 PPO updates): 368 completed episodes, eight with grasps and one with a successful held lift. These are environment steps, not 184,320 optimization iterations; the final 200,000-step result is not included.
- At checkpoint 122,880, deterministic mini evaluation achieved 1/10 grasps and 1/10 held lifts. Episode index 6 (seed 6) grasped at step 273 and completed the held lift at step 283 (14.15 simulated seconds), with a maximum height increase of 11.53 cm.
- All six subsequent completed mini evaluations, at steps 133,120 through 184,320, had 0/10 grasps and lifts. The preserved success is an example of discovered behavior, not evidence of a dependable policy.
- Saved the original [episode 006 video](v2/assets/green_lift_step_000122880_episode_006.mp4) in `v2/assets`, with [source metrics and provenance](v2/assets/green_lift_step_000122880_episode_006.json), for inclusion in the repository. The [v2 experiment report](v2/REPORT.md) references it and records the result in context.
- History is not isolated as the cause of improvement: this run also uses eight PPO epochs and ten mini-evaluation seeds versus four and five in the earlier single-observation run. Decide the next plan of action separately.

### v3 state policy: setup and results — 2026-09-08

- Separate 128-by-128 MLP actor/critic on 47 privileged state values (robot/object poses, velocities, and relative position). No vision or history; same reaching reward, +1 bilateral-contact bonus, and +5 held green lift. This is a diagnostic/potential teacher; visual student training is deferred.
- Parallel CPU simulators with batched policy inference; `--num-envs` defaults to 1. Rollout steps count transitions across all workers, with advantages computed per worker. Training episodes allow 1,000 actions; evaluation allows 500. Commands and artifact paths: [v3 README](v3/README.md).
- Run `20d9d6e6-320d-4241-a047-f0b7b965ca26`: four simulators, 2,048-step rollouts, eight PPO epochs, minibatches of 256, 400K-step budget. At 143,360 steps, evaluations registered contact in 2/5 episodes; at 204,800, 1/5, with no evaluation held lifts. **The original “grasp” counted even one instant of bilateral contact, so these rates do not establish secure gripping.**

### v3 next run: maintained grasp from checkpoint — 2026-09-08

- Launch: `bash scripts/v3_grasp_train.sh`. Award +1 only after five consecutive bilateral-contact steps (0.25 s); contact loss resets the streak. Keep reaching and the +5 lift criterion (5 cm for five contact steps). Log contact occurrence, duration, losses, and final contact state. Maintained contact remains a proxy; lifting is the stronger check. Old checkpoints retain their original grasp definition.
- Initialize actor, critic, and exploration weights from `v3/runs/v3-state-20d9d6e6-320d-4241-a047-f0b7b965ca26/checkpoints/step_000204800.pt`; start fresh optimizer state, counters, and UUID, with source provenance saved. Keep the above training settings for 400K additional steps. Evaluate loaded weights under the new definition first; five mini-eval episodes every ten updates, 25 initial/final episodes.
- Validation: 33 tests and a 16-step CUDA warm-start check passed, including parallel rollouts, checkpoint saving, and initial/mini/final evaluations. Full experiment prepared for launch; details in the [v3 README](v3/README.md).

### v4 visual history BC baseline — 2026-09-08

- Reused the v2 three-observation RGB/depth/proprio history policy plus previous action. Teacher demonstrations target the full red-green-blue stack. History resets at trajectory boundaries; teacher stages and oracle state are excluded from policy inputs. Frozen pretrained visual encoders, learned projections/temporal fusion/actor, Smooth L1 regression; depth remains privileged, so this is a diagnostic policy.
- Collected 12 successes from 12 new random seeds, giving 13 demonstrations / 4,042 actions locally. Fixed seed-disjoint split: 7 train / 3 validation / 3 test trajectories. Validation selects the checkpoint; test action error is measured once afterward, then simulator evaluation uses five fresh seeds.
- Run `v4-history-bc-5da21dfd-668f-4b76-9b81-592a18d1a4ab`: 30 CUDA epochs, best at epoch 30. Train loss 0.00245, validation loss 0.01779, test loss 0.01425 and action MAE 0.07655. Translation MAE is higher (X/Y/Z: 0.126/0.153/0.173); nearly constant rotation commands lower the overall average.
- Fresh seeds 1,000,000–1,000,004: **0/5 full-stack successes**, all reaching 900 actions. Success requires the official stack condition for ten consecutive steps, matching the teacher. Five videos decoded successfully (901 frames each). Low offline action error does not establish closed-loop generalization; this small dataset validates the pipeline, not a reliable controller.
- Added collection, temporal BC, fixed splits, and automatic video evaluation. Fifteen relevant tests passed, including causal history, reset boundaries, temporal weight updates, checkpoint reload, and seed isolation. Commands and artifact links are in the [v4 README](v4/README.md).

- Training-seed diagnostic of the same BC checkpoint: **0/7 full-stack successes**, all capped at 900 actions. Every initial RGB/depth/proprio observation exactly matched its demonstration; all seven videos decoded (901 frames each). Failure also occurs on training initial conditions, so unseen-seed generalization alone cannot explain it. This test does not isolate model capacity from action errors or recovery behavior. [Diagnostic metrics](evaluation/v4/v4-history-bc-5da21dfd-668f-4b76-9b81-592a18d1a4ab-train-seeds-29f23549-3cea-4e6b-bb1c-0e2f930db15d/metrics.json).

### v4 backbone fine-tuning comparison — 2026-09-08

- Added optional RGB/depth backbone fine-tuning (`bash scripts/v4_bc_finetune.sh`). Same MobileNetV3-Small, three-frame history, initial seed, 7/3/3 split, 30 epochs, batch 64, and head learning rate 3e-4 as the frozen baseline; backbone learning rate 1e-5. Start from ImageNet again; no BC checkpoint warm start. BatchNorm statistics stay fixed. Cache normalized pixels, recompute features with gradients each minibatch; the original frozen path remains the default.
- Run `v4-history-bc-finetune-c8db0cbf-ada6-40b7-921c-ffe10107a500`, selected epoch 30. Validation loss **0.02028** versus frozen **0.01779**; test action MAE **0.08059** versus **0.07655**. Full-stack success remains **0/7 training seeds and 0/5 fresh seeds**, all at 900 actions. Initial observations match the demonstrations exactly; all 12 evaluation videos decoded correctly. Unfreezing at these settings did not help; this does not rule out capacity/representation limits or establish their cause. Stronger backbones and LSTM/transformer variants are still untested.
- Sixteen regression/fine-tuning tests passed, including actual updates to both backbones, fixed BatchNorm statistics, causal history, and checkpoint reload. [Comparison metrics](v4/runs/v4-history-bc-finetune-c8db0cbf-ada6-40b7-921c-ffe10107a500/comparison.json) link to the evaluated checkpoints and video metrics.

### v4 frozen-backbone transformer comparison — 2026-09-08

- Added `--policy transformer` and `scripts/v4_bc_transformer.sh`: two causal attention layers, four heads, learned temporal positions, dimension 256 / feedforward 512, zero dropout. Keep the same frozen RGB/depth backbones, three-observation window, previous action, 7/3/3 split, and 30-epoch BC settings. This isolates the temporal encoder; it does not add longer memory. Checkpoints and evaluation use the architecture-aware `v4.model.load_policy`.
- Run `v4-transformer-bc-f403fca1-928f-4070-a8d8-1d6c678bc452`, selected epoch 15. Validation loss **0.01021** versus frozen-MLP **0.01779**; test MAE **0.05643** versus **0.07655** (26% lower). Full-stack success still **0/7 training seeds and 0/5 fresh seeds**, all capped at 900 actions. Both saved backbones exactly match the frozen baseline; demonstration resets match exactly; all 12 videos decoded successfully. Twenty tests passed, including causal attention, history/reset equivalence, frozen weights, and checkpoint reload. [Comparison metrics](v4/runs/v4-transformer-bc-f403fca1-928f-4070-a8d8-1d6c678bc452/comparison.json).
- Next discussion: increase demonstration diversity (only seven training trajectories), and separately compare Smooth L1 motion loss plus BCE on the binary gripper command. Actual teacher gripper labels are exactly -1/+1. The completed transformer run still uses Smooth L1 for all seven outputs.

### Resumed diversity experiment - 2026-09-09

- User requested a larger diverse teacher dataset, transformer retraining, a stronger image-backbone comparison, and actual closed-loop success checks. Original demonstrations and runs survived the VM restart. Poetry is installed at `/root/.local/bin/poetry`; this path is absent from the assistant's default shell PATH. Its environment has torch 2.13.0+cu130, torchvision 0.28.0+cu130, robosuite 1.5.1, MuJoCo 3.3.2, and working CUDA on a 24 GB RTX PRO 4000.
- Plan: fill `v4/trajectories` to 100 distinct successful seeds using the unchanged teacher and environment randomization. Four spawned simulator workers; collection seed 20260909. Preserve successes and failed-attempt records, and exclude their seeds on restart. The first resumed demonstration succeeded in 342 actions.
- Compare frozen MobileNetV3-Small RGB against frozen ImageNet ResNet-18 RGB, retaining the MobileNet depth encoder, 128-pixel inputs, three-frame transformer, previous action, Smooth L1 loss, seed 0, and 30 epochs. Use one new 80/10/10 trajectory split (`v4/splits/diverse100.json`, split seed 20260909). This remains a privileged-depth diagnostic, not the final RGB-only policy.
- Evaluate each selected checkpoint on five fixed sampled training seeds (including exact initial-observation comparison) and five fresh seeds, with videos and stable full-stack success. Do not infer rollout success from offline action error.
- Reproduction command: `POETRY=/root/.local/bin/poetry bash scripts/v4_bc_diverse.sh`. Collection resumes to the target, but training starts new runs. Dataset coverage is summarized and plotted alongside the new split. Results pending at this entry; 24 regression tests passed after adding collection and backbone support.
- Collection completed: 100 distinct successful demonstrations / 31,978 transitions. The main resumed batch produced 86 successes in 98 attempts, plus the one successful restart check. Each object's initial XY positions cover all 16 bins of a 4-by-4 grid over [-0.2, 0.2] m; yaw spans almost the full circle. Teacher lengths: 263-391 actions (mean 319.78). This demonstrates placement coverage, not recovery-state coverage. New split: 80 train / 10 validation / 10 test trajectories, with 25,648 / 3,139 / 3,191 transitions. Coverage artifacts: `v4/splits/diverse100.coverage.json` and `.coverage.png`.
- User then requested concatenating the two camera features and retaining a 2x2 or 4x4 spatial grid. Added `--camera-fusion concat --rgb-pool-size 4` with ResNet-18. Each RGB camera contributes 512x4x4 features, flattened in a fixed spatial order; concatenation yields 16,384 values before a learned 256-dimensional projection. Depth and proprioception retain their existing paths. The transformer still has three-frame history. Pool sizes 1, 2, and 4 are supported; pooling cannot upsample a smaller feature map. The new run uses 128-pixel images, whose ResNet output is naturally 4x4.
- Spatial experiment command: `POETRY=/root/.local/bin/poetry bash scripts/v4_bc_spatial.sh`. Same 80/10/10 split, 30 epochs, seed, training/fresh evaluation samples, and loss as the two baseline runs. It changes both camera fusion and retained spatial resolution, so its comparison does not isolate their individual effects. Twenty-one relevant tests passed, including camera identity, spatial location preservation, batched/online feature agreement, optimization, and checkpoint round trips.
- Larger-data MobileNet result (`v4-diverse100-mobilenet_v3_small-6234f010-1bbd-4908-8bfe-6b36db00b4dc`): test action MAE 0.01855, translation MAE 0.03493, 0/5 training-seed stacks and 0/5 fresh-seed stacks. One training seed achieved a held green lift; all five training initial observations matched their demonstrations. Validation gripper-switch accuracy was 35/40 (87.5%). An evaluation-only zero-rotation ablation on two of the same training seeds also had 0/2 stacks and no held lifts; suppressing rotation did not fix those episodes. The checkpoint itself was not changed.
- At 02:42 UTC, the earlier `evaluation/v4` contents were absent despite completed summaries in `v4/runs`. The assistant did not delete them and asked whether this was intentional cleanup. Training and test summaries, training-seed detailed metrics, and the zero-rotation summaries remain in their run directories. New ResNet/spatial evaluations continue to use `evaluation/v4`.
- The spatial model's validation translation MAE improved to 0.03072 but its gripper-switch accuracy was 22/40 (55%), versus 33/40 (82.5%) for the global ResNet model. Started the previously discussed separate loss comparison on the same 4x4 concatenated architecture: `POETRY=/root/.local/bin/poetry bash scripts/v4_bc_spatial.sh --exp-name v4-diverse100-resnet18-concat-grid4-bce --gripper-loss bce`. Six Smooth L1 motion losses plus one BCE-with-logits gripper loss are averaged; labels must be exactly -1/+1. All other settings remain fixed. This is a new ImageNet-initialized run, not a checkpoint warm start. Loss values are not directly comparable across objectives; use action errors and rollouts. Tests verify useful BCE gradients even for confidently wrong gripper predictions.
- Global ResNet-18 run `v4-diverse100-resnet18-a8a41916-2752-415c-86e0-f4da1272531d` selected epoch 25: test MAE 0.01721, translation MAE 0.03386, gripper switch accuracy 33/40. Full stacks: 0/5 training seeds, 0/5 fresh seeds; no held green lifts in the five training episodes. All training initial observations matched. All ten videos decoded to 901 frames.
- Spatial concatenation run `v4-diverse100-resnet18-concat-grid4-63159c1c-70f2-493e-93cc-aac2e5e0b9e7` selected epoch 29: test MAE 0.01643, translation MAE 0.03097, gripper switch accuracy 21/40. Full stacks: 0/5 training seeds, 0/5 fresh seeds. Training seed 190447585 achieved a held green lift (maximum height increase 0.2495 m, 162 bilateral-contact steps), but no green-on-red placement. All training initial observations matched. At this point BCE results remain pending. Thirty-one regression tests pass after all implementation changes.


### BC restart audit and action-persistence diagnostic — 2026-09-09

- Recovered completed BCE run `v4-diverse100-resnet18-concat-grid4-bce-7431ce74-b6d4-4d1a-aa92-d70d1110b720`: test MAE **0.01899**, translation MAE **0.03702**, gripper sign accuracy **99.06%**, switch accuracy **23/40 (57.5%)**. Saved rollout results show **0/5 training stacks and 0/5 fresh stacks**, all at 900 policy actions (45 simulated seconds, plus the reset observation step). No completion time exists. All training initial observations matched demonstrations. Seed 190447585 lifted **red**, not green; none of the five achieved a held green lift or green-on-red placement. BCE did not resolve the failure at these settings.
- Added `v4/report_action_persistence.py` and ran it on all 100 demonstrations with the preserved 80/10/10 split. It predicts each teacher action using the previous teacher action, with zero padding at each trajectory start, matching the BC history convention. It does not use images or run a controller.
- Test persistence baseline: overall MAE **0.01455**, translation MAE **0.02454**, gripper accuracy **98.75%**, switch accuracy **0/40**. Only **1.25%** of test steps change gripper sign. The baseline's overall and translation MAE are lower than all three diverse-data Smooth L1 models and the BCE model. This shows aggregate offline scores can be low without learning transitions; it does **not** establish that the trained policies actually copy their previous action or isolate the cause of rollout failure.
- Reproduce: `/root/.local/bin/poetry run python -m v4.report_action_persistence v4/splits/diverse100.json --output v4/splits/diverse100.action_persistence.json`. Report includes train, validation, and test results. Existing checkpoints and simulator files were unchanged.
- Proposed next controlled experiment (not launched): retrain the spatial Smooth L1 baseline with previous-action input disabled consistently during training and inference, preserving observation history, initialization seed, split, and other settings. Do not merely zero this input on a checkpoint trained to use it. Evaluate the same five training and five fresh seeds, reporting green contact/lift/placement, full stacks, and time, alongside switch metrics. If this still cannot reproduce training demonstrations, run a one-trajectory overfit/closed-loop diagnostic before another backbone or more clean demonstrations. Recovery-state collection remains a later experiment; the current dataset covers successful teacher states only.


### Executed BC diagnostics — 2026-09-09

User authorized the no-previous-action comparison, privileged perception/control separation, and one-demonstration visual overfit. Completed these plus controlled removal of teacher stage, a lower-LR single-demo refinement, and RGB replacement of exact object positions. Full settings, interpretation, commands, checkpoints, and video links: [v4/DIAGNOSTICS.md](v4/DIAGNOSTICS.md).

| Experiment | Training-seed stacks | Fresh-seed stacks | Successful speed |
| --- | ---: | ---: | --- |
| Spatial Smooth L1 without previous action | **2/5** | **0/5** | Mean 310 actions / 15.50 s on train |
| Small BC MLP, exact positions + robot state + teacher stage | **4/5** | **3/5** | Mean 297.75 / 14.89 s train; 299.33 / 14.97 s fresh |
| Same state MLP, no supplied stage | **0/5** | **0/5** | No completion |
| Visual single demo, 300 epochs | **0/1** | Not tested | Green placed on red, blue not stacked |
| Same demo, 500 further epochs at LR 1e-5 | **1/1** | Not tested | **320 actions / 16.00 s** |
| Fixed stage-assisted controller with RGB-predicted object positions | **0/5** | **0/5** | No completion |

- No-previous-action run: `v4-diverse100-resnet18-concat-grid4-no-prev-88b5c91b-77eb-48c6-8f61-358fcd880e52`, selected epoch 23. Same frozen backbones (verified exact equality), spatial camera concatenation, split, initialization seed, three-observation window, 30 epochs, and Smooth L1 objective as the earlier spatial model. Model-level masking disables previous action in both cached training and online inference; original action labels remain available for switch metrics. Old checkpoints retain prior behavior. Test MAE **0.04042**, translation MAE **0.08031**, switch accuracy **26/40**. Despite worse offline errors, training stacks improved from 0/5 to 2/5. Fresh rollouts included **2/5 held green lifts and 1/5 green-on-red placement**; all five failed the full stack at 900 actions.
- Replayed all 336 teacher actions from training seed 1390308241: exact initial observation, pre-action object/grip-site positions, action values, and stage labels; stable official success. This sampled trajectory has no alignment/scaling discrepancy. `v4-diagnostic-replay-c6444073-65e1-4a23-acbc-9cbea8868c6e/replay_audit.json` records the audit.
- Privileged state run: `v4-diagnostic-privileged-866abd17-544b-4eef-8b8e-ef93027b6ed1`. 128×128 MLP, normalized exact positions, legal robot proprioception, derived relative positions, and 17-way stage encoding. 200 epochs; train/test translation MAE **0.00958 / 0.02349**. Stage comes from the unchanged teacher state machine acting on student-visited states; teacher actions are discarded. It still supplies privileged sequencing, timing, and guards. Success therefore does not isolate perception alone. Guard failures terminate these diagnostic rollouts with a recorded reason.
- Stage-free state run: `v4-diagnostic-privileged-no-stage-1b5b03e3-7e05-420a-a12b-f36e7cd558b5`. Same architecture/training settings, stage channels zeroed, no teacher called at inference. Train/test translation MAE **0.05646 / 0.09974**. All ten rollouts fail. This supports investigating phase/history and fitting; it does not rule out a recurrent state policy.
- Single-demo runs: initial `v4-diagnostic-one-demo-7ba2fd63-66a7-4bc7-9b25-862ded81a13b`, refinement `v4-diagnostic-one-demo-4087c423-acf5-4927-a59b-3c7e2546136b`. Seed 1390308241, no previous action, training-only checkpoint selection. Refinement starts from the best initial checkpoint with a fresh optimizer. Translation MAE fell **0.007441 -> 0.002010**, Smooth L1 **0.00004131 -> 0.000001877**; both get 4/4 switches correct offline. The first model places green but fails full stacking; the refined model stacks successfully. This is a same-seed fitting diagnostic, not generalization evidence.
- Progressive perception replacement: `v4-diagnostic-rgb-positions-39a6a311-8cca-447e-92bc-eb4a1661a007`. Fixed the successful state controller and teacher-stage provider; replaced only object positions with a frozen spatial ResNet-18 + 128-unit RGB/proprio regressor. Exact grip-site position and teacher transitions remain privileged. Train/validation/test coordinate MAE **4.26 / 20.67 / 23.55 mm**; test mean 3D object errors **57.90 / 46.75 / 51.56 mm** for red/green/blue. All fresh episodes time out in the initial approach. This estimator generalizes poorly; its failure does not rule out better visual localization. It uses RGB only, whereas the direct visual BC models still use depth.
- All training-seed initial observations match demonstrations. Timing excludes one reset observation step and success requires ten consecutive official successes. All newly launched experiments completed. 28 regression tests passed; source syntax/whitespace checks passed. Environment and simulator source were unchanged; no dependencies were added.
- Next recommendation (not launched): retain the no-previous-action variant, improve phase/history representation and visual localization, and evaluate recovery-rich teacher corrections around manipulation transitions. More clean, mechanically repeated trajectories alone do not target the demonstrated fitting/phase/localization failures. The existing teacher may reject student-deviated states, so validate recovery labels before scaling DAgger-style collection. See the report for limits on each inference.

- Final artifact validation: **42/42 evaluation videos** decoded with expected frame counts. Aggregate success, speed, and manipulation milestones are saved in `v4/runs/diagnostics20260909-comparison.json`; video checks in `v4/runs/diagnostics20260909-video-validation.json`. The stage-free state policy still achieved 3/5 green lifts and 2/5 green-on-red placements in each partition, despite 0/5 full stacks.


### Persistent LSTM follow-up — 2026-09-09

- User requested an actual recurrent transformer/LSTM comparison for spatial visual BC without previous actions and exact state + robot state + teacher stage. Implemented one-layer causal LSTMs: 256 hidden/cell units for visual, 128 for state. Each consumes one new observation at a time and carries h/c until episode reset; neither takes previous action. Frozen spatial RGB/depth encoders and the existing state feature contract are retained. See [v4/RECURRENT.md](v4/RECURRENT.md).
- Training preserves ordered trajectories and carries h/c between 32-step chunks, detaching only the gradient graph. Two trajectories per visual batch (up to 64 valid transitions), eight per state batch (up to 256). End padding is masked; no history crosses trajectory boundaries. Same seed 0, 80/10/10 split, Smooth L1, baseline learning rates, 30 visual / 200 state epochs, five sampled train and five fresh rollout seeds. Parameter count, sequence order, and exact update count differ from the nonrecurrent baselines; this is not a compute-matched memory-only ablation.
- Code: `v4/recurrent.py`, `v4/train_recurrent.py`, `v4/test_recurrent.py`; visual checkpoints load through `v4.model.load_policy`, state checkpoints through `load_state_lstm`. Runtime hidden state is not checkpointed. The shared state rollout resets recurrent memory before each episode.
- State LSTM run `v4-lstm-state-e50ef4fe-e95e-4ef6-8f61-b81afde66f2a`: selected epoch 198. Train/test translation MAE **0.00939 / 0.01846**, 100% offline gripper accuracy. Stacks **1/5 train, 0/5 fresh**, worse than the state MLP's 4/5 and 3/5. Training seed 190447585 succeeds in **314 actions / 15.70 s**. Ten state videos validated. Stage remains supplied by the live teacher; its actions are discarded and guards can end episodes early.
- Visual LSTM run `v4-lstm-visual-75940047-a07e-4190-8109-8dfb9f1f5b19`: completed all training and evaluation; selected epoch 30. Train/test translation MAE **0.06428 / 0.08642**, test switch accuracy **19/40**. Full stacks **0/5 train and 0/5 fresh**, versus 2/5 and 0/5 for the previous no-action-history transformer. Three training episodes achieve held green lifts but no green-on-red placement; all visual failures reach 900 actions. Frozen ResNet-18 RGB 4x4 spatial concatenation, frozen MobileNet depth, legal proprioception, no previous action, 256-unit persistent LSTM replacing the three-frame transformer.
- 32 regression tests passed, including full-sequence/online equivalence, memory beyond three frames, causal behavior, episode reset, padding masks, exact sequence coverage, training weight updates, and checkpoint round trips. Source syntax/whitespace checks pass. No environment/simulator changes or new dependencies.

- Final recurrent comparison: neither LSTM improves full-stack success at these settings. Visual LSTM **0/5 train, 0/5 fresh**; state+stage LSTM **1/5 train, 0/5 fresh**. Preserve the prior transformer and state MLP as stronger baselines. Both visual backbones match the preceding no-previous-action checkpoint exactly. These changes establish episode-persistent recurrence, but do not settle whether other recurrent optimization settings or recovery data would help.
- Both recurrent jobs completed; **20/20 new videos** validated. Summary: `v4/runs/recurrent20260909-comparison.json`. Fresh visual videos: `evaluation/v4/v4-lstm-visual-75940047-a07e-4190-8109-8dfb9f1f5b19-best-06448caa-86de-4802-a046-2fe55de60d3f/`. State videos: `v4/runs/v4-lstm-state-e50ef4fe-e95e-4ef6-8f61-b81afde66f2a/seed_<seed>.mp4`. Successful state seed: 190447585. Full report: [v4/RECURRENT.md](v4/RECURRENT.md).


### VC-1 backbone experiment — 2026-09-09

- User requested VC-1 for the non-LSTM, no-previous-action visual baseline, then explicitly allowed changing the CNN-style 4x4 readout. Chose the official VC-1 ViT-L native CLS embedding per camera (1024 each), concatenated to 2048 before the existing projection/three-frame transformer. RGB uses the official 224 crop; depth remains 128. This changes backbone, resolution/cropping, readout, and feature dimension. See [v4/VC1.md](v4/VC1.md).
- Official `facebook/vc1-large` weights pinned to revision `8a47f311ef3a8e0b3f58e4249700c9c5b36012c9`, downloaded with the existing HF CLI. Implemented strict conversion to torchvision ViT, avoiding new dependencies. All 294 encoder tensors load; full-model reference parity max difference **4.29e-6**. 34 relevant regression tests passed.
- Launch: `POETRY=/root/.local/bin/poetry bash scripts/v4_bc_vc1.sh`. Run `v4-vc1-vitl-cls-no-prev-177ce458-18f1-48e1-80cb-08f0121de867`. Same split, seed, 30 epochs, Smooth L1, batch 64, head LR 3e-4, fixed train/fresh evaluation seeds. Feature extraction batch 16; frozen float32 VC-1. Training completed (30 epochs, selected epoch 26); training-seed evaluation completed at 0/5 stacks. Fresh-seed evaluation was still incomplete at the shutdown handoff below.
- Real-data verification found inherited depth-encoder TF32 batch/single-frame rounding: depth pixels match exactly, but feature max difference **0.03565** (mean **0.00595**) with cuDNN TF32 versus **1.91e-6** when disabled. VC-1 RGB batch/online features pass. Preserve the baseline arithmetic for this backbone comparison and record full-precision depth as a separate follow-up; this has not been shown to cause the observed control failures.


### Pod shutdown handoff — 2026-09-09

Latest snapshot: VC-1 training and all five training-seed evaluations are complete; **fresh-seed evaluation is incomplete at this handoff**. The evaluation process was still running when these notes were saved; no need to repeat training after restarting. Any fresh videos from an interrupted episode may be incomplete. The snapshot is timestamped in `v4/runs/v4-vc1-vitl-cls-no-prev-177ce458-18f1-48e1-80cb-08f0121de867/shutdown_snapshot.json`; console output was copied from ephemeral `/tmp` into `shutdown_console.log` in the same run directory.

| Diagnostic | Train stacks | Fresh stacks | Finding |
| --- | ---: | ---: | --- |
| Spatial visual transformer, previous action included | 0/5 | 0/5 | Low offline error did not yield execution success. |
| Spatial visual transformer, previous action removed | **2/5** | 0/5 | Strongest multi-demo visual baseline so far; successful train episodes averaged 15.50 s. |
| Exact positions + robot state + teacher stage, MLP | **4/5** | **3/5** | Strongest diagnostic; fresh successes averaged 14.97 s. |
| Exact positions + robot state, stage removed, MLP | 0/5 | 0/5 | Supplied sequencing matters at these settings. |
| One-demo visual transformer, refined fit | **1/1** | Not tested | Same-seed stack in 16.00 s after tighter fitting; no generalization claim. |
| RGB-estimated positions replacing exact positions | 0/5 | 0/5 | Localization errors remain substantial (test coordinate MAE 23.55 mm). |
| Persistent visual LSTM, no previous action | 0/5 | 0/5 | Episode memory alone did not improve this baseline. |
| Persistent exact-state + stage LSTM | 1/5 | 0/5 | Worse closed-loop results than the state MLP despite lower test translation error. |
| **VC-1 ViT-L CLS, three-frame transformer, no previous action** | **0/5** | **Incomplete** | Better offline fit than spatial no-action baseline, but all train rollouts failed. |

**VC-1 completed findings.** Run `v4-vc1-vitl-cls-no-prev-177ce458-18f1-48e1-80cb-08f0121de867`, selected `best.pt` at epoch 26; `final.pt` also saved. Test action MAE **0.02618** versus **0.04042** for the spatial no-previous-action baseline; translation MAE **0.05395** versus **0.08031** (~33% lower); gripper switches **28/40** versus **26/40**. Nevertheless all five training seeds reached 900 actions / 45 simulated seconds without a stack, green bilateral contact, held green lift, or green-on-red placement. Initial observations exactly matched demonstrations. Therefore the native CLS backbone configuration has not improved training-seed execution; the fresh result remains unknown. This comparison changes RGB backbone, crop/resolution, and spatial readout together. All these visual policies still use privileged depth, so they are not the final RGB-only actor.

**Implementation checks and precision finding.** 34 relevant tests passed. Official VC-1 conversion passed full-model reference agreement (max difference 4.29e-6), saved policy reload was bit-exact, and frozen RGB/depth weights were verified. VC-1 videos have not yet undergone full decode validation. An inherited MobileNet depth path discrepancy was found: identical depth pixels produce batch/single-frame feature differences up to 0.03565 with cuDNN TF32, versus 1.91e-6 without it. On 16 demonstration frames, the selected VC-1 policy's normalized batch/online action max difference was 0.01955 with TF32 versus 0.000088 without. This is a plausible sensitivity to investigate, **not an established cause of failed stacks**. Baseline arithmetic was preserved for the VC-1 comparison; no full-precision retraining experiment has been launched.

**Saved artifacts and videos.** Preserve `/workspace/robotic_grasp` on the pod's persistent storage, including uncommitted source changes, `v4/trajectories`, `v4/splits`, `v4/runs`, and `evaluation/v4`. The selected checkpoint is `v4/runs/v4-vc1-vitl-cls-no-prev-177ce458-18f1-48e1-80cb-08f0121de867/best.pt` and is self-contained for evaluation. This run contains `test_metrics.json`, `train_rollout_metrics.json`, `metrics.jsonl`, `config.json`, `split.json`, `backbone_provenance.json`, `frozen_weight_validation.json`, `action_precision_audit.json`, and the shutdown snapshot/log. Training videos are `evaluation/v4/v4-vc1-vitl-cls-no-prev-177ce458-18f1-48e1-80cb-08f0121de867-train-<seed>-<uuid>/episode_000.mp4`; exact directories are in the training metrics and snapshot. Check for a later `rollout_metrics.json` or `evaluation/v4/v4-vc1-vitl-cls-no-prev-177ce458-18f1-48e1-80cb-08f0121de867-best-*/metrics.json` on restart before treating fresh evaluation as unfinished. Official source weights and verification records are in `/workspace/.cache/vc1-large`; preserve this directory separately if retraining is planned. Cached source weights are unnecessary to evaluate the saved BC checkpoint.

**Resume evaluation only** (from `/workspace/robotic_grasp`, using the existing Poetry environment; restore the environment if the pod image changes):

```bash
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/.local/bin/poetry run python -m v4.evaluate \
  v4/runs/v4-vc1-vitl-cls-no-prev-177ce458-18f1-48e1-80cb-08f0121de867/best.pt \
  --episodes 5 --seed 1000000 --device cuda
```

This reruns the full fixed fresh-seed set (1000000–1000004), saves metrics/videos in a new evaluation directory, and prints its path. Training seeds are already complete; add `--training-seeds --episodes 5` only if intentionally repeating them. `scripts/v4_bc_vc1.sh` starts a new training run; it is not a resume-evaluation command.

Next work: finish/inspect the fresh VC-1 rollouts and validate videos; then test consistent full-precision feature caching and online inference as a separate controlled experiment (fresh caches and retraining, not just toggling inference on this checkpoint). Preserve the spatial no-previous-action transformer and exact-state + stage MLP as reference baselines. Native patch-token readouts, better phase/localization inputs, and validated recovery-state demonstrations remain untested alternatives; more successful repetitive trajectories alone have not been shown to solve the problem. Teacher action replay matched exactly, so no alignment/scaling error was detected in that diagnostic.

Detailed reports: [initial BC diagnostics and videos](v4/DIAGNOSTICS.md), [persistent LSTM comparison](v4/RECURRENT.md), [VC-1 configuration and audits](v4/VC1.md). Prior completed aggregate/video reports: `v4/runs/diagnostics20260909-comparison.json`, `v4/runs/diagnostics20260909-video-validation.json` (42 videos), and `v4/runs/recurrent20260909-comparison.json` (20 videos).


### Categorized BC collection and video review — 2026-09-12

- User requested a first batch of 100 clean trajectories and 50 each with Gaussian noise, dropped commands, delayed commands, and sustained bias; keep 10% in validation. Generated **300 new successful demonstrations**, with **270 train / 30 validation**, stratified as clean 90/10 and each disturbance 45/5. There are **88,223 training transitions / 9,775 validation transitions** (97,998 total), approximately 23.58 GB of HDF5 data. All 300 are distinct new layout seeds.
- Split: `v4/splits/robustness300.json`. The existing ten clean `diverse100` test trajectories remain its test references, outside the 300 new trajectories. All original 100 seeds and diagnostic fresh seeds 1000000–1000004 were excluded from new collection. The old dataset/split are unchanged. No new disturbed test set or learned-policy evaluation was run.
- Perturb only XYZ commands: Gaussian sigma 0.015 clipped to ±0.045; dropped zero-XYZ bursts 1–3 steps with 0.03 start probability per idle step; one-step XYZ delay; sustained random-direction bias norm 0.02 for 5–10 steps with 0.02 start probability. Rotation/gripper retain current teacher commands. Schedules depend only on step count and a separate seed, without oracle phase gating. These are mild command disturbances, not guaranteed measured displacements or arbitrary failed-grasp recovery.
- HDF5 `actions` remain the intended teacher command at the actual visited state (BC targets); `executed_actions` store simulator commands. Category/config/noise seed attributes and per-step active/changed/delta/event metadata preserve interventions. The teacher recomputes corrective labels online. Clean defaults remain compatible with old recordings.
- One pilot per category completed on the first attempt and was retained in its preassigned training slot. Full collection required **312 attempts, 12 failures**. Successful-only dataset filtering remains explicit; failed attempts retain seed, stage, and reason. The teacher/environment/simulator definitions and dependencies were not changed.
- User also requested videos and a webpage. Saved **10 validation videos, two per category**. Review page: `v4/trajectories/robustness300/index.html`, served locally on port **8765**; forward that port in VS Code for the remote pod. Category/example selection, slow playback, frame stepping, next-disturbance seeking, and synchronized intended/executed XYZ plots support inspection. Server PID/log live in the same dataset directory.
- Reproduction, settings, labels, constraints, and artifact links: [v4/ROBUSTNESS.md](v4/ROBUSTNESS.md). Collection/resume: `MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/.local/bin/poetry run python -m v4.collect_robustness --workers 8`. Audit: same environment, `python -m v4.audit_robustness --replay`. Rebuild webpage: `python -m v4.build_robustness_review`. Source: `v4/perturbations.py`, `v4/collect_robustness.py`, `v4/audit_robustness.py`, `v4/build_robustness_review.py`, optional recorder extensions, and `v4/test_perturbations.py`. No training launched.
- Final validation: **21 relevant tests passed**. All **300 new HDF5 trajectories** passed split/exclusion, action/metadata, finite-state, sampled-image, and stable-success checks. All **10 videos** decoded with exactly T+1 frames. Replaying one full episode per category reproduced every teacher action/stage, every pre-action object/grip-site position, and initial/terminal observations exactly; all five reached stable official success. Reports: `validation.json` and `replay_validation.json` in the dataset directory. Final review HTML and all ten video URLs passed HTTP checks; page JavaScript syntax checked.


### Visible disturbance correction: three previews only — 2026-09-12

- User inspected the 300-trajectory batch and found disturbances too subtle. Clarified that “dropping” should include an actual dropped object, whereas the prior implementation only dropped motion commands. User requested **three examples for visual review before collecting more**.
- Generated exactly three new preview trajectories on the same layout seed 220769969. All three completed on their first attempt: Gaussian burst **309 actions / 15.45 s**; physical block release/regrasp **373 / 18.65 s**; sustained sideways movement while holding green **296 / 14.80 s**. No larger follow-up batch or training was started.
- Drop event: 2.70–3.30 s; green falls **10.50 cm**, loses contact, and is regrasped at **6.25 s**. Added an external preview hold/recovery controller that clears, waits for landing, and restarts the pick. Original teacher and simulator files are unchanged.
- Gaussian: one 1-second XY burst, sigma 0.7, clipped ±0.9 normalized units, each draw held for five steps; **3.73 cm** measured gripper movement during the event. Sideways: 0.5-second Y bias magnitude 0.95, **5.32 cm** measured block movement, grasp retained during the event. These are deliberately stronger, stage-triggered previews with explicit hold/recovery logic, not the earlier step-only disturbance process.
- New page: **http://localhost:8765/visible_pilots/**, also linked from the old review page. Three annotated videos, direct event/recovery buttons, slow playback, and frame stepping. Full files: `v4/trajectories/robustness300/visible_pilots/`. Report: [v4/VISIBLE_PILOTS.md](v4/VISIBLE_PILOTS.md). The three HDF5 files are marked review-only and excluded from all dataset manifests; new stage labels need handling before future training.
- **3/3 exact simulator replays passed** for positions, intended/executed commands, stage/event metadata, and initial/terminal observations. All reach stable success; all raw/annotated videos fully decode. Page/video HTTP and JavaScript syntax checks pass. Before/after drop frames were visually inspected. **Next: wait for user review of these three before generating more.**


### Stronger recovery dataset and pod rebuild handoff — 2026-09-12

- User approved the three visible previews, variation across green/blue and task stages, and the five-category batch: **100 clean + 50 Gaussian bursts + 50 physical block drops + 50 sideways errors + 50 gripper interruptions**, with 10% validation. No new policy training was requested.
- **Collection paused for pod rebuild: 252/300 saved, 48 remaining.** Current counts: clean 100, Gaussian 46, drop 41, sideways 46, gripper 19; **225 train / 27 validation**, **86,243 transitions**, **20.73 GB HDF5**. Resume with `python -m v4.collect_recovery --workers 8` in the Poetry/EGL environment. The final 270/30 split is published only when collection completes.
- The 50 GB disk quota interrupted publication. Recovered **111 finalized trajectories (9.51 GB)** from temporary storage into the persistent dataset, verified SHA-256 copies and read every HDF5 dataset. Metadata reconstructed from arrays explicitly leaves unavailable wall-clock timing and textual recovery reasons null. Six incomplete recordings were excluded. Collection processes are stopped.
- User authorized deleting superseded trajectory data: removed **290 mild robustness300 HDF5s**, freeing **22.8 GB**, while keeping its ten review HDF5s/videos, all original diverse100 data, three previews, and checkpoints. The old mild split is now a historical record and cannot be used for training; its collector refuses to resume the retired root.
- All **10 stronger review examples** are available at `http://localhost:8765/recovery300/` after restarting the review server. **20/20 raw/annotated videos** passed full T+1 decode checks. Twelve focused tests passed; source/plan hashes match and all 252 stored success flags passed. Full stronger-batch action/state audit and simulator replay remain outstanding.
- Read **[v4/DATA_CREATION.md](v4/DATA_CREATION.md)** for exact restart commands, frozen-source constraints, remaining audits, and storage migration requirements. Git excludes data/splits/checkpoints: preserve the persistent `/workspace/robotic_grasp` tree, plus `/workspace/.cache/vc1-large` for VC-1 training. A fresh clone alone does not restore the dataset. Versioned metadata snapshots live in `v4/handoffs/recovery300-20260912/`.
