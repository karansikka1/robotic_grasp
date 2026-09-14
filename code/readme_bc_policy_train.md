# Train policies with behavior cloning

This package trains the state and RGB policy families compared in [the solution](solution.md). It reads the HDF5 demonstrations produced by [the data collectors](README_bc_data_gen.md), including optional teacher corrections.

## Policy choices

| `--policy` | Model and inputs | Stage prediction during training |
| --- | --- | --- |
| `state_mlp` | Two 128-unit layers; exact block positions and robot measurements | No |
| `state_lstm` | 128-unit LSTM with episode memory; same state inputs | No |
| `state_lstm_aux` | Same LSTM, with a separate stage-prediction output | Yes |
| `rgb_lstm` | Frozen ResNet18, 4×4 features per camera, and an LSTM; two RGB views and robot measurements | Yes |
| `rgb_spatial` | 16×16 features and color heatmaps, an adapter, and an LSTM; starts from saved localization weights | Yes |

State policies are control experiments: exact block positions are not allowed inputs to the final submitted policy. Both RGB families use only the permitted camera and robot observations. Teacher stages are targets, never inputs. All models learn XYZ movement and gripper opening/closing; rotation commands stay zero.

The earlier short-history transformer and teacher-assisted diagnostic are not included in this compact package.

## Data and environment

Run all commands from inside the delivered `code/` folder after [environment setup](README.md#setup). No parent repository or `PYTHONPATH` setting is needed.

| Input | Where it comes from |
| --- | --- |
| `outputs/base/split.json` | Completed demonstration collection; contains training and validation trajectories |
| `outputs/corrections/collection.json` | Optional completed correction collection; only its selected episodes are added to training |
| Localization weights | Required only for spatial RGB initialization; described below |

Trajectory paths are resolved relative to their manifest; absolute paths also work. The trainer rejects incomplete collections, duplicated files, and seeds shared between training and validation/test. A correction may reuse a training seed. Validation is never expanded with corrections, and test episodes are not used for training or checkpoint selection.

## State policies

```bash
python -m bc_policy.train \
  --policy state_lstm_aux \
  --split outputs/base/split.json \
  --epochs 200 --device cpu \
  --output outputs/state_lstm_aux
```

Change `--policy` to `state_mlp` or `state_lstm` and choose a new output directory to train the other baselines.

To fine-tune the auxiliary LSTM on all demonstrations plus corrections:

```bash
python -m bc_policy.train \
  --policy state_lstm_aux \
  --split outputs/base/split.json \
  --corrections outputs/corrections/collection.json \
  --init-checkpoint outputs/state_lstm_aux/best.pt \
  --epochs 100 --learning-rate 0.0001 --device cpu \
  --output outputs/state_lstm_aux_corrected
```

Omit `--init-checkpoint` and use 200 epochs with learning rate `0.001` to train from scratch on the combined data. With an initial checkpoint, normalization and model weights are retained but the optimizer starts fresh. This is fine-tuning, not an exact interrupted-run resume.

## Initial RGB baseline

```bash
python -m bc_policy.train \
  --policy rgb_lstm \
  --split outputs/base/split.json \
  --corrections outputs/corrections/collection.json \
  --epochs 200 --device cuda \
  --output outputs/rgb_lstm
```

The first fresh baseline run loads torchvision's ImageNet ResNet18 weights, downloading them if they are not already cached. A saved RGB checkpoint can instead be supplied with `--init-checkpoint`. Images are read in small chunks, flipped upright, resized to 128×128, and normalized before encoding. No full image dataset or feature cache is loaded into memory.

## Spatial RGB policies

Localization pre-training is a separate step. This package consumes its saved weights; it does not rerun localization training. `prepare_visual` accepts the existing ResNet18 localization checkpoint formats:

| `--kind` | `--localizer` | `--encoder` |
| --- | --- | --- |
| `frozen` | Spatial-softmax head checkpoint | Full frozen encoder checkpoint |
| `finetuned` | Checkpoint containing the trained layer2 suffix and localization heads | Frozen encoder prefix checkpoint |

Supply those files explicitly. They are not bundled in this code increment. For the localization-fine-tuned model:

```bash
python -m bc_policy.prepare_visual \
  --kind finetuned \
  --localizer path/to/localization_best.pt \
  --encoder path/to/frozen_prefix.pt \
  --output outputs/spatial_initial.pt

python -m bc_policy.train \
  --policy rgb_spatial \
  --init-checkpoint outputs/spatial_initial.pt \
  --split outputs/base/split.json \
  --corrections outputs/corrections/collection.json \
  --epochs 50 --device cuda \
  --output outputs/rgb_spatial
```

The initializer retains the trained image features and heatmap head, and creates a new random controller. It saves one self-contained initialization checkpoint; later BC runs need only that checkpoint and the demonstrations.

| Comparison | Change to the commands above |
| --- | --- |
| Original frozen image backbone with spatial features | Prepare with `--kind frozen` and its matching localizer/encoder files |
| Localization-trained backbone, frozen during BC | Use the commands as shown |
| Also fine-tune the backbone during BC | Add `--finetune-cnn`; layer1 and layer2 update at `0.00001`, with fixed BatchNorm statistics |
| Also supply predicted positions and gripper offsets | Add `--predicted-geometry --split outputs/base/split.json --corrections outputs/corrections/collection.json` to **prepare_visual**, choose a new initialization filename, then train it with `rgb_spatial` |

The predicted-position branch uses a frozen localization model and training-only normalization. It receives no true block coordinates during policy inference. Its backbone stays frozen. The other spatial variants do not feed predicted positions to the controller. No extra position-prediction loss is used during BC.

## Training and saved results

| Setting | Default |
| --- | --- |
| Action objective | Smooth L1 between predicted and expert actions |
| Stage objective, where enabled | Classification loss with weight `0.01` |
| State learning rate | `0.001` from scratch; `0.0001` for fine-tuning |
| Initial RGB baseline learning rate | `0.0003` |
| Spatial RGB learning rate | `0.0001` |
| MLP batch | 256 supervised frames |
| LSTM batch | 8 state trajectories or 2 RGB trajectories |
| LSTM gradient chunk | 32 actions; memory carries across chunks and resets between episodes |
| Checkpoint selection | Lowest validation action loss |

The learner-controlled prefix of a correction updates LSTM memory but contributes no action or stage loss. Padding is also excluded. Only expert labels are used as targets; disturbed executed commands are not substituted for the teacher's intended actions.

| Output | Contents |
| --- | --- |
| `best.pt` | Model with the lowest validation action loss, including input normalization |
| `final.pt` | Model after the last epoch |
| `metrics.jsonl` | Training and validation action loss, stage loss, translation error, and gripper accuracy per epoch |
| `config.json` | Training settings |
| `split.json` | Exact resolved training/validation/test file lists |

Use a new output directory for each run. This compact trainer preserves the model designs and supervision rules, but does not reproduce the historical scheduling, batching changes, or optimizer resumes exactly. Its losses measure imitation; full-stack success and completion time require the simulator rollouts below.

Short checks passed for all three state models, state fine-tuning, and the four RGB training variants using recorded data. Checks also covered correction masking, checkpoint reloads, matching existing state/spatial-policy predictions, frozen backbone weights, and running without the experiment folders. These checks used the existing working Python environment; no full training experiment or new success-rate evaluation was run.

## Evaluate a checkpoint

The [evaluation runner](evaluate.py) uses the packaged policy loaders and [30 validation seeds](plans/evaluation_validation.json). It uses the bundled [simulator](motion_planning/simulator.py) directly and counts consecutive successes in the evaluation loop. It does not import files from outside this bundle.

The bundle includes [best_visual_bc.pt](checkpoints/best_visual_bc.pt): the localization-trained ResNet18 fine-tuned during BC that scored 24/30 validation stacks in the report. It contains the trained visual backbone and controller; no separate localization weights are needed for evaluation.

From inside the delivered `code/` folder, run all 30 validation seeds:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python -m evaluate \
  --policy rgb_spatial \
  --checkpoint checkpoints/best_visual_bc.pt \
  --device cuda --video \
  --seeds \
    155284722 225192514 246848522 316164238 446701367 \
    527142122 535212849 570746810 576497590 589563505 \
    652863466 701863950 731014221 758699772 797951687 \
    798569765 873629338 944123758 1311387844 1369671429 \
    1429315851 1570055189 1628365314 1641469845 1798284844 \
    1811276011 1888632319 1953236062 2014303324 2035399579 \
  --output outputs/evaluation
```

The full seed list matches the bundled manifest; omitting `--seeds` also runs these same 30 layouts. Use `--device cpu` if needed. Choose the matching `--policy` from the policy table when evaluating other checkpoints; all spatial variants, including predicted geometry, use `rgb_spatial`. Omit `--video` to save only metrics, or replace the seed list with `--seeds 155284722 873629338` to reproduce the report's two example layouts. The output directory must be new.

`results.json` records the checkpoint hash, evaluated seeds, per-episode outcomes, success rate, and mean completion actions and seconds among successes. Videos show both cameras at 20 frames per second. Episodes stop after ten consecutive official successes or 900 policy actions; the initial observation step is excluded from timing. RGB policies receive only the six permitted observations. State-policy evaluation uses exact block positions and is a control diagnostic. See [evaluation details](experiment_details.md#evaluation-sets-and-timing).

The packaged runner reproduced the best model's two report examples exactly: 267 actions for the successful stack and 900 for the failure. Video frame counts, timing, memory resets, input filtering, and loading all packaged policy families without research-folder imports were checked.

## Code map

| File | Purpose |
| --- | --- |
| [train.py](bc_policy/train.py) | Shared training loop, masking, validation, and checkpoints |
| [data.py](bc_policy/data.py) | Manifest reading, target loading, and ordered trajectory chunks |
| [models.py](bc_policy/models.py) | State models; reuses the collection package's LSTM definitions |
| [visual.py](bc_policy/visual.py) | RGB models, chunked image reading, and checkpoint loading |
| [prepare_visual.py](bc_policy/prepare_visual.py) | Transfer localization weights into a new spatial BC controller |
