# v2 temporal-policy experiment report

The temporal policy has produced gripping actions and a successful held lift,
but learning remains slow and inconsistent as the run approaches 200,000 training
steps. It has not yet learned a reliable grasp-and-lift policy.

## Setup and progress snapshot — 2026-09-08

Run: `v2-history-ba0c88b3-bcda-4c82-b4a7-2892e9a0fe4f`.
The policy receives three consecutive observations and the previous action
(0.1 seconds of observation history at 20 Hz), with frozen RGB/depth encoders.
Reward combines signed reaching progress (scale 1), +1 for the first contact-based
grasp, and +5 for a held lift. PPO uses 1,024 rollout steps, eight optimization
epochs, and minibatches of 256, with a 200,000-step training budget.

These counts describe a fixed snapshot, not the final run result:

| Measurement | Result |
| --- | --- |
| Training steps / completed PPO updates | 184,320 / 180 |
| Completed training episodes | 368 |
| Training episodes with a grasp | 8 / 368 |
| Training episodes with a successful held lift | 1 / 368 |
| Mini evaluation at step 122,880 | 1 / 10 grasps; 1 / 10 held lifts |
| Subsequent mini evaluations, steps 133,120–184,320 | 0 / 10 grasps and lifts at each of six checkpoints |

Mini evaluations use deterministic actions on fixed seeds 0–9. Training uses
stochastic actions and different seeds, so its counts are reported separately.
All earlier mini evaluations through step 112,640 also had zero grasps and lifts.
The later zero-success evaluations show that the successful behavior has not
become consistent. This is not a controlled comparison of observation history:
the earlier single-observation run used four optimization epochs and five
mini-evaluation seeds, whereas this run uses eight and ten respectively.

## Preserved successful episode

[Watch episode 006 at training step 122,880](assets/green_lift_step_000122880_episode_006.mp4).
This is episode index 6 (seed 6; the seventh episode in the evaluation batch).
It is a selected successful example, not representative of the overall success rate.

- First contact-confirmed grasp: policy step 273.
- Successful held lift: policy step 283, or 14.15 simulated seconds.
- Maximum recorded lift above the post-reset reference: 11.53 cm.
- Reward components: +0.2853 reaching, +1 grasp, +5 lift (total +6.2853).

Success requires lifting green at least 5 cm while maintaining grasp for five
consecutive control steps (0.25 seconds). The initial zero-action bootstrap is
excluded from the 283 policy steps; the recorded video includes its frame.

The video is copied unchanged from
`evaluation/v2/mini/v2-history-step-000122880-ba0c88b3-bcda-4c82-b4a7-2892e9a0fe4f/episode_006.mp4`.
The [episode metadata](assets/green_lift_step_000122880_episode_006.json) preserves
the source path, run UUID, evaluation settings, batch summary, episode metrics,
and video SHA-256. The selected video is allowed by `.gitignore` so it can be
committed with this report; bulk evaluation outputs remain local artifacts.

The next experiment remains undecided. These results document that the policy
can discover gripping and lifting, while reliable execution is still unresolved.
