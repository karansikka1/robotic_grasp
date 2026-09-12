"""Train visual single-frame or history policies by cloning v4 demonstrations."""

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

from v1.model import PrivilegedPPOPolicy
from v2.model import TemporalPPOPolicy
from v4.model import POLICY_CLASSES, TransformerTemporalPolicy, load_policy, policy_kind
from v4.history import add_history
from v4.visual_inputs import (
    VisualInputCache, cache_visual_inputs, preprocess_observation_batch, select_feature_batch,
)
from v4.data import (
    H5TransitionDataset,
    discover_trajectories,
    split_trajectories,
    write_manifest,
    validate_manifest,
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
    finetune_backbone: bool = False
    backbone_learning_rate: float = 1e-5
    gripper_loss: str = 'smooth_l1'

    def validate(self) -> None:
        if self.gripper_loss not in ('smooth_l1', 'bce'):
            raise ValueError('gripper_loss must be smooth_l1 or bce')
        for name in ("epochs", "batch_size", "feature_batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "backbone_learning_rate", "max_grad_norm"):
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


@torch.no_grad()
def extract_frozen_feature_batch(policy, observations):
    """Vectorized equivalent of the policy's inference feature extractor."""
    prepared = preprocess_observation_batch(policy, observations)
    policy.rgb_backbone.eval()
    policy.depth_backbone.eval()
    return {
        "front": policy.rgb_backbone(prepared["front"]),
        "wrist": policy.rgb_backbone(prepared["wrist"]),
        "depth": policy.depth_backbone(prepared["depth"]),
        "proprio": prepared["proprio"],
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
    features = {name: torch.cat(chunks) for name, chunks in feature_chunks.items()}
    actions = torch.cat(action_chunks)
    if isinstance(policy, TemporalPPOPolicy):
        features = add_history(features, actions, dataset.entries, policy.history_length)
    return features, actions


def _predict_actions(
    policy: PrivilegedPPOPolicy,
    features: Mapping[str, Tensor],
) -> Tensor:
    return torch.tanh(policy.actor(policy.fused_embedding(features)))


def action_losses(logits, targets, gripper_loss='smooth_l1'):
    motion = F.smooth_l1_loss(logits[..., :6].tanh(), targets[..., :6], reduction='none')
    if gripper_loss == 'bce':
        if not torch.all((targets[..., 6] == -1) | (targets[..., 6] == 1)):
            raise ValueError('Binary gripper loss requires exactly -1/+1 teacher labels')
        gripper = F.binary_cross_entropy_with_logits(logits[..., 6], (targets[..., 6] + 1) / 2,
                                                      reduction='none')
    elif gripper_loss == 'smooth_l1':
        gripper = F.smooth_l1_loss(logits[..., 6].tanh(), targets[..., 6], reduction='none')
    else:
        raise ValueError('Unknown gripper loss')
    return torch.cat((motion, gripper.unsqueeze(-1)), dim=-1)


@torch.no_grad()
def evaluate_cached(
    policy: PrivilegedPPOPolicy,
    features: Mapping[str, Tensor] | VisualInputCache,
    actions: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    gripper_loss: str = 'smooth_l1',
) -> dict[str, Any]:
    policy.eval()
    losses: list[Tensor] = []
    absolute_errors: list[Tensor] = []
    squared_errors: list[Tensor] = []
    gripper_correct: list[Tensor] = []
    gripper_switches: list[Tensor] = []
    for start in range(0, len(actions), batch_size):
        stop = min(start + batch_size, len(actions))
        batch_features = select_feature_batch(policy, features, slice(start, stop), device)
        targets = actions[start:stop].to(device)
        logits = policy.actor(policy.fused_embedding(batch_features))
        predictions = logits.tanh()
        losses.append(action_losses(logits, targets, gripper_loss).cpu())
        absolute_errors.append((predictions - targets).abs().cpu())
        squared_errors.append((predictions - targets).square().cpu())
        gripper_correct.append(((predictions[:, -1] > 0) == (targets[:, -1] > 0)).cpu())
        if 'previous_action' in batch_features:
            previous = batch_features['previous_action'][:, -1]
            gripper_switches.append(((previous != 0) & ((previous > 0) != (targets[:, -1] > 0))).cpu())
    loss_values = torch.cat(losses)
    absolute = torch.cat(absolute_errors)
    squared = torch.cat(squared_errors)
    correct = torch.cat(gripper_correct)
    switches = torch.cat(gripper_switches) if gripper_switches else torch.zeros_like(correct)
    return {
        "loss": float(loss_values.mean()),
        "gripper_loss": gripper_loss,
        "mae": float(absolute.mean()),
        "rmse": float(squared.mean().sqrt()),
        "action_mae": [float(value) for value in absolute.mean(dim=0)],
        "translation_mae": float(absolute[:, :3].mean()),
        "gripper_sign_accuracy": float(correct.float().mean()),
        "gripper_switch_samples": int(switches.sum()),
        "gripper_switch_accuracy": float(correct[switches].float().mean()) if switches.any() else None,
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
            "task_name": "full-stack",
            "policy_type": policy_kind(policy),
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
    train_features: Mapping[str, Tensor] | VisualInputCache,
    train_actions: Tensor,
    validation_features: Mapping[str, Tensor] | VisualInputCache,
    validation_actions: Tensor,
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
    if isinstance(policy, TemporalPPOPolicy):
        trainable_modules += (policy.temporal_fusion,)
    if isinstance(policy, TransformerTemporalPolicy):
        trainable_modules += (policy.temporal_encoder,)
    parameters = [
        parameter
        for module in trainable_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    parameter_groups = [{"params": parameters, "lr": config.learning_rate}]
    if config.finetune_backbone:
        if not isinstance(train_features, VisualInputCache):
            raise ValueError("Backbone fine-tuning requires image inputs, not frozen features")
        backbone_parameters = []
        for backbone in (policy.rgb_backbone, policy.depth_backbone):
            backbone.requires_grad_(True)
            backbone_parameters.extend(backbone.parameters())
        parameter_groups.append({"params": backbone_parameters, "lr": config.backbone_learning_rate})
        parameters = parameters + backbone_parameters
    optimizer = AdamW(parameter_groups, weight_decay=config.weight_decay)
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
            features = select_feature_batch(policy, train_features, indices, device)
            targets = train_actions[indices].to(device)
            logits = policy.actor(policy.fused_embedding(features))
            loss = action_losses(logits, targets, config.gripper_loss).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            total_samples += len(indices)

        metrics = evaluate_cached(
            policy,
            validation_features,
            validation_actions,
            batch_size=config.batch_size,
            device=device,
            gripper_loss=config.gripper_loss,
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
            "epoch=%d train_loss=%.6f validation_loss=%.6f validation_mae=%.6f",
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
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-manifest", type=Path, help="reuse a fixed train/validation/test split")
    parser.add_argument("--policy", choices=("history", "single", "transformer"), default="history")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--no-previous-action", action="store_true",
                        help="disable action-history input in transformer training and inference")
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-feedforward-dim", type=int, default=512)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--rgb-backbone", choices=("mobilenet_v3_small", "resnet18", "vc1_vitl"),
                        default="mobilenet_v3_small", help="RGB encoder for transformer policies")
    parser.add_argument("--camera-fusion", choices=("sum", "concat"), default="sum")
    parser.add_argument("--rgb-pool-size", type=int, choices=(1, 2, 4), default=1,
                        help="spatial grid for ResNet; VC-1 uses native CLS at 1, patch grids at 2/4")
    parser.add_argument("--train-rollout-episodes", type=int, default=0,
                        help="diagnostic rollouts from a fixed sample of training seeds")
    parser.add_argument("--rollout-eval-episodes", type=int, default=5, help="fresh-seed simulator evaluations after training; 0 skips")
    parser.add_argument("--rollout-eval-max-steps", type=int, default=900)
    parser.add_argument("--rollout-eval-seed", type=int, default=1000000)
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
    parser.add_argument("--finetune-backbone", action="store_true",
                        help="train RGB/depth backbone weights; keep BatchNorm statistics fixed")
    parser.add_argument("--backbone-learning-rate", type=float, default=defaults.backbone_learning_rate)
    parser.add_argument("--gripper-loss", choices=('smooth_l1', 'bce'), default=defaults.gripper_loss)
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
        finetune_backbone=args.finetune_backbone,
        backbone_learning_rate=args.backbone_learning_rate,
        gripper_loss=args.gripper_loss,
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
    logging.getLogger("robosuite_logs").setLevel(logging.WARNING)
    logging.getLogger("OpenGL").setLevel(logging.WARNING)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.training_seed)

    if (args.history_length < 1 or args.rollout_eval_episodes < 0
            or args.train_rollout_episodes < 0 or args.rollout_eval_max_steps < 1):
        raise ValueError("Invalid history length or rollout evaluation settings")
    if args.policy != "transformer" and (args.rgb_backbone != "mobilenet_v3_small"
                                        or args.camera_fusion != "sum" or args.rgb_pool_size != 1):
        raise ValueError("Alternative RGB backbones/fusion require --policy transformer")
    if args.no_previous_action and args.policy != "transformer":
        raise ValueError("--no-previous-action requires --policy transformer")
    manifest = (json.loads(args.split_manifest.read_text()) if args.split_manifest else split_trajectories(
        discover_trajectories(args.trajectories_dir), test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction, seed=args.split_seed,
    ))
    validate_manifest(manifest)
    manifest_path = write_manifest(manifest, run_dir / "split.json")
    run_config = {
        "algorithm": "behavior_cloning",
        "policy_type": args.policy,
        "rollout_evaluation": {"episodes": args.rollout_eval_episodes,
                               "train_episodes": args.train_rollout_episodes,
                               "max_steps": args.rollout_eval_max_steps,
                               "seed": args.rollout_eval_seed},
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
            **({"history_length": args.history_length} if args.policy != "single" else {}),
        },
    }
    if args.policy == "transformer":
        run_config["model"].update(
            use_previous_action=not args.no_previous_action,
            rgb_backbone=args.rgb_backbone,
            camera_fusion=args.camera_fusion,
            rgb_pool_size=args.rgb_pool_size,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            transformer_feedforward_dim=args.transformer_feedforward_dim,
            transformer_dropout=args.transformer_dropout,
        )
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Run directory: %s", run_dir)
    logger.info("Split: %s", manifest["summary"])
    logger.info("Initializing %s visual policy (pretrained=%s)", args.policy, not args.no_pretrained)
    policy_class = POLICY_CLASSES[args.policy]
    policy = policy_class(**run_config["model"], pretrained=not args.no_pretrained).to(device)
    cache_inputs = cache_visual_inputs if config.finetune_backbone else cache_frozen_features
    logger.info("Backbone fine-tuning=%s; head lr=%g; backbone lr=%g; BatchNorm statistics fixed",
                config.finetune_backbone, config.learning_rate, config.backbone_learning_rate)
    datasets = {name: H5TransitionDataset(manifest[name]) for name in ("train", "validation")}
    try:
        logger.info("Preparing training inputs")
        train_features, train_actions = cache_inputs(
            policy, datasets["train"], batch_size=config.feature_batch_size,
        )
        logger.info("Preparing validation inputs")
        validation_features, validation_actions = cache_inputs(
            policy, datasets["validation"], batch_size=config.feature_batch_size,
        )
    finally:
        for dataset in datasets.values():
            dataset.close()
    checkpoint = train_behavior_cloning(
        policy, train_features, train_actions, validation_features, validation_actions,
        config, run_dir, device=device,
    )
    logger.info("Best checkpoint selected on validation loss: %s", checkpoint)
    policy = load_policy(checkpoint, device=device)
    test_dataset = H5TransitionDataset(manifest["test"])
    try:
        logger.info("Caching untouched test trajectories for one final evaluation")
        test_features, test_actions = cache_inputs(
            policy, test_dataset, batch_size=config.feature_batch_size,
        )
    finally:
        test_dataset.close()
    test_metrics = evaluate_cached(policy, test_features, test_actions,
                                   batch_size=config.batch_size, device=device, gripper_loss=config.gripper_loss)
    write_manifest({"checkpoint": str(checkpoint), "metrics": test_metrics}, run_dir / "test_metrics.json")
    logger.info("Final test metrics: %s", test_metrics)
    if args.train_rollout_episodes:
        from v4.evaluate import evaluate_training_seeds
        metrics = evaluate_training_seeds(
            checkpoint, episodes=args.train_rollout_episodes,
            max_steps=args.rollout_eval_max_steps, device=device,
        )
        write_manifest(metrics, run_dir / "train_rollout_metrics.json")
        logger.info("Training-seed rollout results: %s", metrics["summary"])
    if args.rollout_eval_episodes:
        from v4.evaluate import evaluate_checkpoint
        metrics = evaluate_checkpoint(
            checkpoint, episodes=args.rollout_eval_episodes, max_steps=args.rollout_eval_max_steps,
            seed=args.rollout_eval_seed, device=device,
        )
        write_manifest({"output_dir": metrics["output_dir"], "summary": metrics["summary"]},
                       run_dir / "rollout_metrics.json")
        logger.info("Fresh-seed rollout results: %s | videos: %s", metrics["summary"], metrics["output_dir"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Behavior-cloning run failed")
        raise
