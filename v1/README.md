# v1: vanilla PPO with privileged depth

This is the first pipeline baseline. It uses the two RGB observations, metric
front-camera depth, and robot proprioception. Depth makes this a privileged
baseline rather than a final deployable policy.

## Network

- One frozen ImageNet MobileNetV3-Small backbone is shared by front and wrist RGB.
- A second frozen ImageNet MobileNetV3-Small backbone encodes depth replicated to
  three channels.
- Learned projections map all visual features to 256 dimensions.
- A proprioceptive MLP maps the 16 robot-state values to 256 dimensions.
- The four embeddings are summed and layer-normalized.
- Separate two-layer MLP heads predict the squashed-Gaussian action distribution
  and scalar value.

The raw simulator images are flipped vertically, resized to 128 x 128, and
ImageNet-normalized consistently in training and evaluation. Metric depth is
clipped to 2 m before encoding.

## Train

Install the locked environment once and train from the repository root:

```bash
poetry install
poetry run python -m v1.train --exp-name v1-vanilla-ppo
```

The equivalent Make shortcut is:

```bash
make train-v1 ARGS="--exp-name v1-vanilla-ppo"
```

Training outputs are written to a UUID-suffixed directory under `v1/runs/`.
After training, the command evaluates the deterministic policy over the 25 fixed
evaluation initializations and writes videos plus metrics under `evaluation/`.

During training, a deterministic five-episode mini evaluation runs every 10 PPO
updates (about 10,240 policy steps with the default rollout size). It uses fixed
seeds 0 through 4, records no video, writes metrics under `evaluation/mini/`, and
logs evaluation success and completion speed to TensorBoard. Configure it with:

```bash
poetry run python -m v1.train \
  --mini-eval-interval-updates 5 \
  --mini-eval-episodes 5
```

Set `--mini-eval-interval-updates 0` to disable only periodic evaluation, or
`--skip-evaluation` to disable both periodic and final evaluation.

For a short architecture and pipeline check without downloading ImageNet weights:

```bash
poetry run python -m v1.train \
  --exp-name v1-smoke \
  --total-timesteps 128 \
  --rollout-steps 128 \
  --update-epochs 1 \
  --no-pretrained \
  --skip-evaluation
```

Monitor a run with:

```bash
poetry run tensorboard --logdir v1/runs
```

Evaluate a saved checkpoint again with:

```bash
poetry run python -m v1.evaluate v1/runs/<run>/checkpoint_final.pt \
  --name v1-vanilla-ppo
```

The sparse reward is `10 * task_complete - 0.001` per policy action. Training
logs episode return, length, success, rolling success rate, PPO losses, entropy,
KL divergence, clipping fraction, gradient norm, and whether any positive reward
was observed in the rollout.
