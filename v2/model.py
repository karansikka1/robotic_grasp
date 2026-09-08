"""Three-observation feature history plus the previously executed action."""

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from v1.model import ACTION_DIM, PrivilegedPPOPolicy


class FeatureHistory:
    """Episode-local cached frozen features, ordered oldest to newest.

    Re-reading the same observation for PPO bootstrapping does not advance history.
    Training and prediction use separate instances so evaluation cannot overwrite
    the training episode's history.
    """

    def __init__(self, policy: "TemporalPPOPolicy") -> None:
        self.policy = policy
        self.reset()

    def reset(self) -> None:
        self.frames: deque[dict[str, Tensor]] = deque(maxlen=self.policy.history_length)
        self.previous_action = torch.zeros(ACTION_DIM, device=self.policy.device)
        self._observation = None
        self._features = None

    @torch.no_grad()
    def features(self, observation: Mapping[str, Any]) -> dict[str, Tensor]:
        if observation is self._observation:
            return self._features
        current = self.policy.extract_frozen_features(observation)
        if not self.frames:
            # Repeat the initial observation for unavailable past frames.
            self.frames.extend([current] * self.policy.history_length)
        else:
            self.frames.append(current)
        result = {
            name: torch.stack([frame[name] for frame in self.frames])
            for name in current
        }
        result["previous_action"] = self.previous_action.clone()
        self._observation = observation
        self._features = result
        return result

    def record_action(self, action: Tensor) -> None:
        self.previous_action = action.detach().to(self.policy.device).clone()


class TemporalPPOPolicy(PrivilegedPPOPolicy):
    """The v1 encoders/heads with learned fusion of temporal features."""

    def __init__(self, *, history_length: int = 3, **kwargs) -> None:
        if not isinstance(history_length, int) or history_length < 1:
            raise ValueError("history_length must be a positive integer")
        super().__init__(**kwargs)
        self.history_length = history_length
        self.temporal_fusion = nn.Sequential(
            nn.Linear(history_length * self.embedding_dim + ACTION_DIM, self.embedding_dim),
            nn.Tanh(),
        )
        nn.init.orthogonal_(self.temporal_fusion[0].weight, gain=2**0.5)
        nn.init.zeros_(self.temporal_fusion[0].bias)
        self._prediction_history: FeatureHistory | None = None

    def get_model_config(self) -> dict[str, Any]:
        return {**super().get_model_config(), "history_length": self.history_length}

    def fused_embedding(self, features: Mapping[str, Tensor]) -> Tensor:
        per_frame = super().fused_embedding(features)
        # Supports both [time, feature] and [batch, time, feature].
        joined = torch.cat((per_frame.flatten(-2), features["previous_action"]), dim=-1)
        return self.temporal_fusion(joined)

    def reset_history(self) -> None:
        self._prediction_history = FeatureHistory(self)

    @torch.no_grad()
    def predict(self, observation: Mapping[str, Any], *, deterministic: bool = True) -> np.ndarray:
        if self._prediction_history is None:
            self.reset_history()
        was_training = self.training
        self.eval()
        try:
            features = self._prediction_history.features(observation)
            action, _, _ = self.act(features, deterministic=deterministic)
            self._prediction_history.record_action(action)
            return action.cpu().numpy()
        finally:
            self.train(was_training)


def load_policy(checkpoint_path, *, device="cpu") -> PrivilegedPPOPolicy:
    """Restore either old single-observation or new temporal v2 checkpoints."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    policy_class = TemporalPPOPolicy if "history_length" in config else PrivilegedPPOPolicy
    policy = policy_class(pretrained=False, **config)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    return policy.to(device).eval()
