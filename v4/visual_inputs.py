"""Preprocessed image caches for differentiable visual history BC."""

import logging
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Mapping
from v1.model import PROPRIO_KEYS, PrivilegedPPOPolicy
from v2.model import TemporalPPOPolicy
from v4.history import add_history

def _normalize_rgb(policy: PrivilegedPPOPolicy, images: Tensor) -> Tensor:
    if getattr(policy, 'rgb_backbone_name', None) == 'vc1_vitl':
        from v4.vc1 import preprocess_vc1
        return preprocess_vc1(images, policy.device)
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


def preprocess_observation_batch(
    policy: PrivilegedPPOPolicy,
    observations: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Match v1 preprocessing; cache pixels without freezing learned features."""
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

    return {
        "front": _normalize_rgb(policy, front),
        "wrist": _normalize_rgb(policy, wrist),
        "depth": depth,
        "proprio": proprio,
    }


class VisualInputCache:
    """Keep each frame once on CPU and form causal windows at minibatch time."""

    def __init__(self, frames, actions, entries, history_length):
        self.frames = frames
        history = add_history(
            {"index": torch.arange(len(actions)).unsqueeze(-1)},
            actions, entries, history_length,
        )
        self.indices = history["index"].squeeze(-1)
        self.previous_action = history["previous_action"]

    def features(self, policy, indices, device):
        window = self.indices[indices]
        # Encode repeated frames only once, preserving gradients through reuse.
        unique, inverse = torch.unique(window.flatten(), return_inverse=True)
        images = {key: value[unique].to(device) for key, value in self.frames.items()}
        encoded = {
            "front": policy.rgb_backbone(images["front"]),
            "wrist": policy.rgb_backbone(images["wrist"]),
            "depth": policy.depth_backbone(images["depth"]),
            "proprio": images["proprio"],
        }
        inverse = inverse.to(device)
        features = {key: value[inverse].reshape(*window.shape, -1)
                    for key, value in encoded.items()}
        if isinstance(policy, TemporalPPOPolicy):
            features["previous_action"] = self.previous_action[indices].to(device)
        else:
            features = {key: value[:, 0] for key, value in features.items()}
        return features


@torch.no_grad()
def cache_visual_inputs(policy, dataset, *, batch_size):
    chunks = {key: [] for key in ("front", "wrist", "depth", "proprio")}
    actions = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for index, batch in enumerate(loader, start=1):
        prepared = preprocess_observation_batch(policy, batch["observations"])
        for key, value in prepared.items():
            chunks[key].append(value.cpu())
        actions.append(batch["action"].float().cpu())
        if index % 10 == 0 or index == len(loader):
            logging.getLogger("v4.train_bc").info("Cached input images for %d/%d batches", index, len(loader))
    actions = torch.cat(actions)
    frames = {key: torch.cat(values) for key, values in chunks.items()}
    return VisualInputCache(frames, actions, dataset.entries,
                            getattr(policy, "history_length", 1)), actions


def select_feature_batch(policy, features, indices, device):
    if isinstance(features, VisualInputCache):
        return features.features(policy, indices, device)
    return {key: value[indices].to(device) for key, value in features.items()}
