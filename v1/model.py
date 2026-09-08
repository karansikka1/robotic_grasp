"""Policy and observation preprocessing for the v1 PPO baseline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


RGB_KEYS = ("frontview_image", "robot0_eye_in_hand_image")
DEPTH_KEY = "frontview_depth"
PROPRIO_KEYS = (
    "robot0_joint_pos",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
PRIVILEGED_POLICY_KEYS = (*PROPRIO_KEYS, *RGB_KEYS, DEPTH_KEY)
PROPRIO_DIM = 16
ACTION_DIM = 7


def privileged_policy_observation(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the RGB, depth, and proprioceptive inputs used by v1."""
    return {key: observation[key] for key in PRIVILEGED_POLICY_KEYS}


class FrozenMobileNetFeatures(nn.Module):
    """ImageNet MobileNetV3-Small trunk with global average pooling."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        network = mobilenet_v3_small(weights=weights)
        self.features = network.features
        self.output_dim = network.classifier[0].in_features
        self.requires_grad_(False)

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)


class PrivilegedPPOPolicy(nn.Module):
    """Frozen visual encoders plus summed embeddings and PPO actor/critic heads.

    A shared RGB backbone processes the front and wrist cameras. A second
    pretrained backbone processes the metric depth image replicated to three
    channels. Their projections and a proprioceptive MLP produce equally sized
    embeddings which are summed before the policy and value MLPs.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        image_size: int = 128,
        max_depth_m: float = 2.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0 or image_size <= 0:
            raise ValueError("network dimensions must be positive")
        if max_depth_m <= 0:
            raise ValueError("max_depth_m must be positive")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.max_depth_m = max_depth_m

        self.rgb_backbone = FrozenMobileNetFeatures(pretrained=pretrained)
        self.depth_backbone = FrozenMobileNetFeatures(pretrained=pretrained)
        backbone_dim = self.rgb_backbone.output_dim

        self.rgb_projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim), nn.ReLU()
        )
        self.depth_projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim), nn.ReLU()
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(PROPRIO_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.fusion_norm = nn.LayerNorm(embedding_dim)
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )
        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.5))

        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        modules = (
            self.rgb_projection,
            self.depth_projection,
            self.proprio_encoder,
            self.actor,
            self.critic,
        )
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def train(self, mode: bool = True) -> "PrivilegedPPOPolicy":
        super().train(mode)
        # Frozen BatchNorm statistics must also remain frozen during PPO updates.
        self.rgb_backbone.eval()
        self.depth_backbone.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.log_std.device

    def _rgb_tensor(self, image: Any) -> Tensor:
        tensor = torch.as_tensor(image, device=self.device)
        if tensor.ndim != 3 or tensor.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {tuple(tensor.shape)}")
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        tensor = tensor.flip(-2)
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return (tensor - self.image_mean) / self.image_std

    def _depth_tensor(self, depth: Any) -> Tensor:
        tensor = torch.as_tensor(depth, device=self.device).float()
        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        if tensor.ndim != 2:
            raise ValueError(f"Expected HW or HW1 depth, got shape {tuple(tensor.shape)}")
        tensor = torch.nan_to_num(
            tensor,
            nan=self.max_depth_m,
            posinf=self.max_depth_m,
            neginf=0.0,
        )
        tensor = tensor.clamp_(0.0, self.max_depth_m).div_(self.max_depth_m)
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).flip(-2)
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return (tensor - self.image_mean) / self.image_std

    def _proprio_tensor(self, observation: Mapping[str, Any]) -> Tensor:
        values = [
            torch.as_tensor(observation[key], device=self.device).flatten()
            for key in PROPRIO_KEYS
        ]
        proprio = torch.cat(values).float().unsqueeze(0)
        if proprio.shape[-1] != PROPRIO_DIM:
            raise ValueError(
                f"Expected {PROPRIO_DIM} proprioceptive values, "
                f"got {proprio.shape[-1]}"
            )
        return proprio

    @torch.no_grad()
    def extract_frozen_features(
        self, observation: Mapping[str, Any]
    ) -> dict[str, Tensor]:
        """Preprocess one observation and return compact rollout features."""
        missing = [key for key in PRIVILEGED_POLICY_KEYS if key not in observation]
        if missing:
            raise KeyError(f"Observation is missing v1 fields: {missing}")
        self.rgb_backbone.eval()
        self.depth_backbone.eval()
        return {
            "front": self.rgb_backbone(
                self._rgb_tensor(observation["frontview_image"])
            ).squeeze(0),
            "wrist": self.rgb_backbone(
                self._rgb_tensor(observation["robot0_eye_in_hand_image"])
            ).squeeze(0),
            "depth": self.depth_backbone(
                self._depth_tensor(observation[DEPTH_KEY])
            ).squeeze(0),
            "proprio": self._proprio_tensor(observation).squeeze(0),
        }

    def fused_embedding(self, features: Mapping[str, Tensor]) -> Tensor:
        front = self.rgb_projection(features["front"])
        wrist = self.rgb_projection(features["wrist"])
        depth = self.depth_projection(features["depth"])
        proprio = self.proprio_encoder(features["proprio"])
        return self.fusion_norm(front + wrist + depth + proprio)

    def distribution_and_value(
        self, features: Mapping[str, Tensor]
    ) -> tuple[Normal, Tensor]:
        embedding = self.fused_embedding(features)
        mean = self.actor(embedding)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std), self.critic(embedding).squeeze(-1)

    @staticmethod
    def _squashed_log_prob(distribution: Normal, raw_action: Tensor) -> Tensor:
        action = torch.tanh(raw_action)
        correction = torch.log(1.0 - action.square() + 1e-6)
        return (distribution.log_prob(raw_action) - correction).sum(-1)

    def act(
        self,
        features: Mapping[str, Tensor],
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution, value = self.distribution_and_value(features)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(distribution, raw_action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        features: Mapping[str, Tensor],
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution, value = self.distribution_and_value(features)
        bounded_actions = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_actions = torch.atanh(bounded_actions)
        log_prob = self._squashed_log_prob(distribution, raw_actions)
        entropy = distribution.entropy().sum(-1)
        return log_prob, entropy, value

    @torch.no_grad()
    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Policy-function adapter suitable for ``harness.evaluate_policy``."""
        was_training = self.training
        self.eval()
        features = self.extract_frozen_features(observation)
        batched = {key: value.unsqueeze(0) for key, value in features.items()}
        action, _, _ = self.act(batched, deterministic=deterministic)
        if was_training:
            self.train()
        return action.squeeze(0).cpu().numpy()

    def get_model_config(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "image_size": self.image_size,
            "max_depth_m": self.max_depth_m,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> "PrivilegedPPOPolicy":
        """Restore a trained policy without downloading pretrained weights."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        policy = cls(pretrained=False, **checkpoint["model_config"])
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.to(device).eval()
        return policy
