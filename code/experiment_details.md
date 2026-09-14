# Experiment details

This document explains the training and evaluation choices behind [the solution](solution.md). The main results and example videos are in that report.

## Reading the results

| Term | Meaning |
| --- | --- |
| Policy | The model that chooses the robot's next action. |
| Trajectory or episode | A sequence of observations and actions from one attempt. |
| Layout | The starting arrangement of the blocks. |
| Seed | A number used to reproduce a randomized starting arrangement or training run. |
| Epoch | One pass through the training data. |
| Checkpoint | A saved version of the model. |
| Action loss | How closely predicted actions match the teacher's actions. Lower loss does not necessarily mean more completed stacks. |

The results in the solution describe the reported September 14 snapshot. The initial visual baseline and later visual models differ in architecture and training budget. The four later visual variants are compared at the same 50-epoch budget. These comparisons guide model selection; they do not prove that one architecture is always better.

## State policy and stage training

The state-based models receive exact block positions and robot measurements. They help test control without requiring the model to locate blocks in images. They do not satisfy the final policy's input restrictions.

The teacher divides each pick-and-place into eight stages: approach from above, descend, close the gripper, lift, move to the support block, lower, release, and retreat. Repeating these for green and blue, followed by settling, gives 17 stages.

During training, the LSTM predicts both the action and the stage. The action objective encourages matching the teacher's movement and gripper commands. A separate stage-classification objective encourages the LSTM's memory to represent task progress; its loss has weight 0.01. The stage labels and predictions are never supplied as action inputs. At evaluation, the model acts without a teacher and resets its memory between episodes.

## Correction training

The correction approach is inspired by DAgger, or dataset aggregation: add expert guidance for situations the learner encounters itself. Here, the teacher takes over after a detected error and completes the episode.

| Training approach | Starting weights | Training trajectories |
| --- | --- | ---: |
| Original model | Random initialization | 270 |
| Fine-tuning with corrections | Original model's saved weights | 282 |
| Training from scratch with corrections | Random initialization | 282 |

Each trajectory was visited once per epoch; corrections were not sampled more often than other trajectories.

Only the teacher-controlled part of a correction provides action and stage targets. The earlier learner-controlled part updates the LSTM's memory so the teacher's correction is seen in context.

On the separate 20-layout test:

| Outcome after fine-tuning | Layouts |
| --- | ---: |
| Previously failed, now succeeds | 4 |
| Previously succeeded, now fails | 3 |
| Net increase in successes | 1 (12/20 to 13/20) |

Completion on the 12 correction layouts was measured from normal resets, not by forcing identical drops or failed grasps. Those layouts were part of training.

## Visual model

ResNet18 is the image-processing network. Its intermediate **features** are learned descriptions of image regions. The initial visual baseline used a 4×4 grid per camera. The later model preserves a finer 16×16 grid and learns separate heatmaps for red, green, and blue.

Localization training teaches the model to estimate block centers in 3D. The simulator supplies those targets and the projected image positions used to supervise heatmaps. The reported millimeter error is the average distance between predicted and true block centers on validation data.

The strongest reported visual variant reuses the localization-trained backbone, keeps its weights fixed during BC, and trains the heatmap layers, adapter, LSTM, and action and stage outputs. The LSTM receives image features and permitted robot measurements. It does not receive true or predicted block coordinates. Supplying predicted coordinates was a separate comparison.

## Evaluation sets and timing

| Set | Purpose |
| --- | --- |
| 270 original training trajectories, later 282 with corrections | Teach the model through demonstrations. |
| 30 validation layouts | Check behavior on layouts excluded from training; used in model selection. |
| 20 development layouts | Inspect behavior during experimentation. These became familiar through repeated evaluations. |
| 12 correction-training layouts | Check performance on known training cases. |
| Separate 20-layout state-policy test | Compare the frozen original and corrected state policies on previously unused layouts. |

The visual model's reported evaluation covers the 30 validation, 20 development, and 12 correction-training layouts: 62 episodes in total. Its untouched final test is still pending in this snapshot.

Full success requires the official stack-completion check to remain true for ten consecutive actions. Intermediate events, such as briefly placing green, do not count as full success. Episodes are limited to 900 policy actions.

At 20 actions per second, completion time is the number of policy actions divided by 20. This is simulated time, not the computer's runtime. Timing excludes the initial step used to obtain an observation and includes the ten-action success confirmation. Mean completion time is calculated only over successful episodes.
