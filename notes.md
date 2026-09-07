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
