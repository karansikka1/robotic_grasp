"""Train the v1 privileged-depth policy by behavior cloning v4 trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from v1.model import PROPRIO_KEYS, PrivilegedPPOPolicy
from v4.data import (
    H5TransitionDataset,
    discover_trajectories,
    split_trajectories,
    write_manifest,
)


logger = logging.getLogger("v4.train_bc")
FEATURE_NAMES = ("front", "wrist", "depth", "proprio")


@dataclass(frozen=True)
class BCConfig:
    epochs: int = 30
    batch_size: int = 64
    feature_batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    training_seed: int = 0

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "feature_batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_run_dir(root: Path, experiment_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", experiment_name).strip("-._")
    if not safe_name:
        raise ValueError("experiment name must contain a letter or number")
    run_dir = root.expanduser().resolve() / f"{safe_name}-{uuid.uuid4()}"
    run_dir.mkdir(parents=True)
    return run_dir


def _normalize_rgb(policy: PrivilegedPPOPolicy, images: Tensor) -> Tensor:
    images = images.to(policy.device).permute(0, 3, 1, 2).float().div_(255.0)
    images = images.flip(-2)
    images = F.interpolate(
        images,
        size=(policy.image_size, policy.image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (images - policy.image_mean) / policy.image_std


@torch.no_grad()
def extract_frozen_feature_batch(
    policy: PrivilegedPPOPolicy,
    observations: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Vectorized equivalent of v1's single-observation feature extractor."""
    front = observations["frontview_image"]
    wrist = observations["robot0_eye_in_hand_image"]
    if front.ndim != 4 or front.shape[-1] != 3:
        raise ValueError(f"Expected BHWC front RGB, got {tuple(front.shape)}")
    if wrist.ndim != 4 or wrist.shape[-1] != 3:
        raise ValueError(f"Expected BHWC wrist RGB, got {tuple(wrist.shape)}")

    depth = observations["frontview_depth"].to(policy.device).float()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected BHW or BHW1 depth, got {tuple(depth.shape)}")
    depth = torch.nan_to_num(
        depth,
        nan=policy.max_depth_m,
        posinf=policy.max_depth_m,
        neginf=0.0,
    )
    depth = depth.clamp_(0.0, policy.max_depth_m).div_(policy.max_depth_m)
    depth = depth.unsqueeze(1).repeat(1, 3, 1, 1).flip(-2)
    depth = F.interpolate(
        depth,
        size=(policy.image_size, policy.image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    depth = (depth - policy.image_mean) / policy.image_std

    proprio = torch.cat(
        [observations[key].to(policy.device).flatten(1) for key in PROPRIO_KEYS],
        dim=1,
    ).float()
    if proprio.shape[1] != 16:
        raise ValueError(f"Expected 16 proprio values, got {proprio.shape[1]}")

    policy.rgb_backbone.eval()
    policy.depth_backbone.eval()
    return {
        "front": policy.rgb_backbone(_normalize_rgb(policy, front)),
        "wrist": policy.rgb_backbone(_normalize_rgb(policy, wrist)),
        "depth": policy.depth_backbone(depth),
        "proprio": proprio,
    }


def cache_frozen_features(
    policy: PrivilegedPPOPolicy,
    dataset: H5TransitionDataset,
    *,
    batch_size: int,
) -> tuple[dict[str, Tensor], Tensor]:
    """Run frozen CNNs once; subsequent epochs train only compact features."""
    feature_chunks: dict[str, list[Tensor]] = {name: [] for name in FEATURE_NAMES}
    action_chunks: list[Tensor] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch_index, batch in enumerate(loader, start=1):
        features = extract_frozen_feature_batch(policy, batch["observations"])
        for name, value in features.items():
            feature_chunks[name].append(value.cpu())
        action_chunks.append(batch["action"].float().cpu())
        if batch_index % 10 == 0 or batch_index == len(loader):
            logger.info("Cached features for %d/%d batches", batch_index, len(loader))
    return (
        {name: torch.cat(chunks) for name, chunks in feature_chunks.items()},
        torch.cat(action_chunks),
    )


def _predict_actions(
    policy: PrivilegedPPOPolicy,
    features: Mapping[str, Tensor],
) -> Tensor:
    return torch.tanh(policy.actor(policy.fused_embedding(features)))


@torch.no_grad()
def evaluate_cached(
    policy: PrivilegedPPOPolicy,
    features: Mapping[str, Tensor],
    actions: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    policy.eval()
    losses: list[Tensor] = []
    absolute_errors: list[Tensor] = []
    squared_errors: list[Tensor] = []
    for start in range(0, len(actions), batch_size):
        stop = min(start + batch_size, len(actions))
        batch_features = {
            name: value[start:stop].to(device) for name, value in features.items()
        }
        targets = actions[start:stop].to(device)
        predictions = _predict_actions(policy, batch_features)
        losses.append(F.smooth_l1_loss(predictions, targets, reduction="none").cpu())
        absolute_errors.append((predictions - targets).abs().cpu())
        squared_errors.append((predictions - targets).square().cpu())
    loss_values = torch.cat(losses)
    absolute = torch.cat(absolute_errors)
    squared = torch.cat(squared_errors)
    return {
        "loss": float(loss_values.mean()),
        "mae": float(absolute.mean()),
        "rmse": float(squared.mean().sqrt()),
        "action_mae": [float(value) for value in absolute.mean(dim=0)],
    }


def _save_checkpoint(
    path: Path,
    policy: PrivilegedPPOPolicy,
    optimizer: AdamW,
    *,
    epoch: int,
    metrics: Mapping[str, Any],
    config: BCConfig,
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "version": 1,
            "algorithm": "behavior_cloning",
            "epoch": epoch,
            "metrics": dict(metrics),
            "config": asdict(config),
            "model_config": policy.get_model_config(),
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "uses_privileged_depth": True,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def train_behavior_cloning(
    policy: PrivilegedPPOPolicy,
    train_features: Mapping[str, Tensor],
    train_actions: Tensor,
    test_features: Mapping[str, Tensor],
    test_actions: Tensor,
    config: BCConfig,
    run_dir: Path,
    *,
    device: torch.device,
) -> Path:
    config.validate()
    trainable_modules = (
        policy.rgb_projection,
        policy.depth_projection,
        policy.proprio_encoder,
        policy.fusion_norm,
        policy.actor,
    )
    parameters = [
        parameter
        for module in trainable_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.training_seed)
    policy.to(device)
    best_loss = math.inf
    best_path = run_dir / "best.pt"
    history_path = run_dir / "metrics.jsonl"

    for epoch in range(1, config.epochs + 1):
        policy.train()
        permutation = torch.randperm(len(train_actions), generator=generator)
        total_loss = 0.0
        total_samples = 0
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            features = {
                name: value[indices].to(device) for name, value in train_features.items()
            }
            targets = train_actions[indices].to(device)
            predictions = _predict_actions(policy, features)
            loss = F.smooth_l1_loss(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            total_samples += len(indices)

        metrics = evaluate_cached(
            policy,
            test_features,
            test_actions,
            batch_size=config.batch_size,
            device=device,
        )
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_samples,
            }
        )
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(metrics, sort_keys=True) + "\n")
        logger.info(
            "epoch=%d train_loss=%.6f test_loss=%.6f test_mae=%.6f",
            epoch,
            metrics["train_loss"],
            metrics["loss"],
            metrics["mae"],
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            _save_checkpoint(
                best_path,
                policy,
                optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )

    _save_checkpoint(
        run_dir / "final.pt",
        policy,
        optimizer,
        epoch=config.epochs,
        metrics=metrics,
        config=config,
    )
    return best_path


def parse_args() -> argparse.Namespace:
    defaults = BCConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", type=Path, default=Path("v4/trajectories"))
    parser.add_argument("--runs-dir", type=Path, default=Path("v4/runs"))
    parser.add_argument("--exp-name", default="v4-privileged-bc")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--training-seed", type=int, default=defaults.training_seed)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--feature-batch-size", type=int, default=defaults.feature_batch_size
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BCConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        feature_batch_size=args.feature_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        training_seed=args.training_seed,
    )
    config.validate()
    device = select_device(args.device)
    run_dir = create_run_dir(args.runs_dir, args.exp_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
        ],
        force=True,
    )
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.training_seed)

    manifest = split_trajectories(
        discover_trajectories(args.trajectories_dir),
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    manifest_path = write_manifest(manifest, run_dir / "split.json")
    run_config = {
        "algorithm": "behavior_cloning",
        "device": str(device),
        "pretrained": not args.no_pretrained,
        "uses_privileged_depth": True,
        "split_manifest": str(manifest_path),
        "bc": asdict(config),
        "model": {
            "image_size": args.image_size,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "max_depth_m": args.max_depth_m,
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Run directory: %s", run_dir)
    logger.info("Split: %s", manifest["summary"])
    logger.info("Initializing v1 privileged policy (pretrained=%s)", not args.no_pretrained)
    policy = PrivilegedPPOPolicy(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        image_size=args.image_size,
        max_depth_m=args.max_depth_m,
        pretrained=not args.no_pretrained,
    ).to(device)
    train_dataset = H5TransitionDataset(manifest["train"])
    test_dataset = H5TransitionDataset(manifest["test"])
    try:
        logger.info("Caching frozen train features")
        train_features, train_actions = cache_frozen_features(
            policy,
            train_dataset,
            batch_size=config.feature_batch_size,
        )
        logger.info("Caching frozen test features")
        test_features, test_actions = cache_frozen_features(
            policy,
            test_dataset,
            batch_size=config.feature_batch_size,
        )
    finally:
        train_dataset.close()
        test_dataset.close()

    checkpoint = train_behavior_cloning(
        policy,
        train_features,
        train_actions,
        test_features,
        test_actions,
        config,
        run_dir,
        device=device,
    )
    logger.info("Best checkpoint: %s", checkpoint)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Behavior-cloning run failed")
        raise
