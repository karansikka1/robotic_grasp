# BC stacking submission

This folder contains the [report](solution.md), trained BC checkpoint, evaluation runner, fixed validation seeds, simulator, and training/data-collection code. All commands below run from inside this folder; the original repository is not needed.

## HTML report

Open [solution.html](solution.html) in a browser. The single file includes videos, heatmaps, the architecture diagram, supporting notes, and a downloadable code bundle with the trained checkpoint. No server or internet connection is needed to read it.

After editing the Markdown or bundled files, regenerate the HTML from this folder:

```bash
python -m pip install -r requirements_report.txt
python build_report.py
```

The builder excludes generated outputs, virtual environments, and the HTML itself from the embedded code download.

## Setup

Use Python 3.12. If you already have the working assignment environment active, you can use it directly. Otherwise:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) records the main library versions used for checkpoint verification. On Linux, offscreen rendering also requires an EGL library; on Ubuntu/Debian, install it with `apt-get install -y libegl1` if missing. CUDA evaluation requires a compatible NVIDIA GPU and driver.

## Evaluate the delivered checkpoint

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python evaluate.py \
  --policy rgb_spatial \
  --checkpoint checkpoints/best_visual_bc.pt \
  --device cuda --video \
  --output outputs/validation_all
```

This runs all [30 validation seeds](plans/evaluation_validation.json). For a quick check, add `--seeds 155284722 873629338`: the verified outcomes are a successful stack in 267 actions and a failure at 900 actions. Use `--device cpu` if needed. Omit `--video` to record metrics only. Choose a new output directory for each run.

`results.json` contains per-episode outcomes, checkpoint hash, success rate, and mean completion length and time among successes. MP4 files show the front and wrist cameras. The reported checkpoint scored 24/30 validation successes.

## Contents

| Path | Purpose |
| --- | --- |
| [solution.md](solution.md) | Report and example videos |
| [evaluate.py](evaluate.py) | Evaluation loop using the simulator directly |
| [checkpoints/best_visual_bc.pt](checkpoints/best_visual_bc.pt) | Trained visual BC policy, including its backbone |
| [motion_planning/simulator.py](motion_planning/simulator.py) | Copy of the supplied simulator, with unchanged code |
| [readme_bc_policy_train.md](readme_bc_policy_train.md) | Training and evaluation commands |
| [README_bc_data_gen.md](README_bc_data_gen.md) | Demonstration and correction collection |

Evaluation needs no demonstration files or separate localization weights. Training instructions describe any additional data or initialization weights they require.
