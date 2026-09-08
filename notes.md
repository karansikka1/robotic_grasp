# Ultra Policy Training - Handoff Notes

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
| `v4-scripted-teacher` | No learned network: privileged closed-loop waypoint controller using exact object poses, gripper pose, bilateral grasps, object contacts, and official success. | None; actions come from the scripted state machine. | 6/6 single-attempt successes: seed 0 plus five randomly selected seeds. The random-seed runs took 295–327 actions (mean 312.2). | Streams legal observations, privileged oracle state, stage labels, and actions to HDF5; retains HDF5/MP4 only after stable official success. |

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
