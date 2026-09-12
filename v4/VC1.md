# VC-1 visual backbone comparison — 2026-09-09

This experiment replaces the frozen ResNet-18 RGB encoder in the non-LSTM,
no-previous-action visual BC baseline with **VC-1 ViT-L**. Following the user's
request to adapt the CNN spatial readout, the first run uses VC-1's native
**CLS embedding**, not a 4×4 CNN-style grid.

## Model and input changes

The [official VC-1 model card](https://github.com/facebookresearch/eai-vc/blob/main/MODEL_CARD.md)
describes a ViT-L/16 encoder with 24 layers, 16 attention heads, 1024-dimensional
embeddings, and 224-pixel RGB inputs. The official implementation's default
readout is its normalized CLS token. Architecture and preprocessing were checked
against [Meta's source](https://github.com/facebookresearch/eai-vc/tree/main/vc_models/src/vc_models).

Each camera produces a 1024-value CLS embedding. The two cameras are concatenated
(2048 values) and projected to 256 dimensions, retaining camera identity. The
existing depth/proprioception paths, three-frame causal temporal transformer,
and action head remain. There is no previous-action input and no persistent
LSTM state.

RGB preprocessing vertically flips MuJoCo images, uses the official bicubic
short-edge resize to 256, center-crops to 224, and applies ImageNet normalization.
The depth path retains the existing 128-pixel preprocessing and frozen MobileNet
encoder. Thus `image_size: 128` in the generic model configuration describes the
retained depth preprocessing; VC-1 RGB preprocessing is fixed at 224 in its adapter.

This is a backbone/readout/preprocessing comparison, not an isolated change of
pretraining weights. Compared with the spatial ResNet baseline it changes RGB
resolution/cropping, encoder architecture, feature dimension, and spatial readout.
The visual policy still consumes privileged depth, so it is not the final
RGB/proprioception-only actor.

## Weights and implementation verification

- Official checkpoint: [facebook/vc1-large](https://huggingface.co/facebook/vc1-large).
- Pinned revision: `8a47f311ef3a8e0b3f58e4249700c9c5b36012c9`.
- Checkpoint SHA-256: `89d59b6e0fe627ee13dd49011d9e3178fc5749237b3963608213ba210257d79a`.
- Encoder conversion: all **294 encoder tensors** load strictly into existing
  torchvision ViT modules; only MAE decoder/mask tensors are discarded.
- Full-model parity: maximum token difference **4.29e-6** versus an independent
  functional computation following the official timm-style architecture.
- No timm, Hydra, OmegaConf, or other new dependency was installed. Saved policy
  checkpoints include the VC-1 encoder weights and load without the download file.
- Model use remains subject to the official model card/license.

The adapter supports native CLS at `rgb_pool_size=1`; optional patch-token spatial
pooling at 2/4 is available but is not used in this run. The RGB encoder remains
frozen and runs in float32. Batch/online preprocessing is shared.

## Training and evaluation

Same 80/10/10 trajectory split (`v4/splits/diverse100.json`), seed 0, 30 epochs,
Smooth L1 action objective, batch size 64, head learning rate 3e-4, three-frame
history, and disabled previous action as the baseline. Frozen features are cached
once, with extraction batches of 16 for the larger encoder. Validation loss selects
the checkpoint; test action error is measured afterward.

Automatic video evaluation uses the same five sampled training seeds and five
fresh seeds 1000000–1000004. Training initial observations are checked against the
recordings. Official success must hold for ten consecutive steps. Failures allow
900 policy actions, and reported control time excludes the one reset observation
step. Green grasp/lift/placement telemetry complements full-stack success.

## Numerical precision finding

A real-demonstration check found an inherited CUDA precision difference in the
unchanged depth encoder. Batch and single-frame depth **pixels match exactly**,
but MobileNet depth features differ with cuDNN TF32 enabled: maximum absolute
error 0.03565 on the checked frame (mean 0.00595). Disabling cuDNN TF32 reduces
the maximum feature difference to 1.91e-6. Native VC-1 RGB features pass the
batch/online comparison.

This experiment preserves the baseline's default TF32 convolution setting,
rather than changing depth arithmetic alongside the RGB backbone. The finding
is recorded as a separate precision-control follow-up; the existing CPU
regression tests did not expose it. It could matter for a sensitive controller,
but its effect on stacking success has not been isolated. Do not infer that this
alone explains earlier failures.

## Reproduction

```bash
# Download the pinned official checkpoint with the existing HF CLI.
HF_HOME=/workspace/.cache/huggingface /root/.local/bin/poetry run hf download \
  facebook/vc1-large pytorch_model.bin config.yaml \
  --revision 8a47f311ef3a8e0b3f58e4249700c9c5b36012c9 \
  --local-dir /workspace/.cache/vc1-large

POETRY=/root/.local/bin/poetry bash scripts/v4_bc_vc1.sh
```

`VC1_CHECKPOINT` overrides the initial pretrained-weight path. The saved BC
checkpoint is self-contained. `TORCH_HOME` controls the existing depth-backbone
cache. The supplied simulator/environment and dependency lockfile are unchanged.

## Run and artifacts

[Run directory](runs/v4-vc1-vitl-cls-no-prev-177ce458-18f1-48e1-80cb-08f0121de867)
contains configuration, source/checkpoint provenance, training log, best/final
checkpoints, offline metrics, and rollout summaries as they complete. Video paths
are recorded in `train_rollout_metrics.json` and `rollout_metrics.json`.


## Selected checkpoint and offline comparison

Epoch **26/30** was selected by validation loss. All official VC-1 encoder weights
remain exactly unchanged, and the depth encoder weights exactly match the
preceding ResNet baseline checkpoint.

| Held-out action metric | ResNet-18 spatial, no previous action | VC-1 native CLS, no previous action |
| --- | ---: | ---: |
| Overall MAE | 0.04042 | **0.02618** |
| Translation MAE | 0.08031 | **0.05395** |
| Gripper-switch accuracy | 26/40 | **28/40** |

The translation-error reduction is approximately 33%. It is not a success-rate
claim; simulator evaluation is required.

The selected policy was also checked on the first 16 demonstration frames of
training seed 25035255. Batch-vs-online normalized action differences reached
**0.01955 max / 0.000750 mean** under the preserved cuDNN TF32 setting, compared
with **0.0000880 max / 0.00000858 mean** with cuDNN TF32 disabled. These comparisons
recomputed both paths under each setting; they are not a full-precision retraining
or rollout ablation. Details are saved in `action_precision_audit.json`.

The full regression suite passed **34 tests**. Actual-dataset VC-1 feature checks
and a full policy save/reload test also passed; reloaded predictions matched
exactly. The separate inherited depth rounding discrepancy is documented above.
