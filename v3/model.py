"""Small MLP actor and critic consuming privileged state, without vision."""

import math

import torch
from torch import nn
from torch.distributions import Normal

from v3.state import STATE_DIM, STATE_SCHEMA, state_vector

ACTION_DIM = 7


class StatePPOPolicy(nn.Module):
    uses_privileged_depth = False
    uses_privileged_state = True

    def __init__(self, *, hidden_dim=128, state_schema=STATE_SCHEMA):
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if state_schema != STATE_SCHEMA:
            raise ValueError(f"Unsupported state schema: {state_schema}")
        self.hidden_dim = hidden_dim
        self.actor = self._head(hidden_dim, ACTION_DIM)
        self.critic = self._head(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.5))
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    @staticmethod
    def _head(hidden_dim, output_dim):
        head = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        for layer in head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        return head

    @property
    def device(self):
        return self.log_std.device

    def extract_frozen_features(self, observation):
        # Shared PPO calls this feature hook; here it stores raw state, with no
        # frozen encoder. Both MLPs are trained on the replayed state minibatches.
        return {"state": torch.as_tensor(state_vector(observation), device=self.device)}

    def distribution_and_value(self, features):
        state = features["state"]
        mean = self.actor(state)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std), self.critic(state).squeeze(-1)

    @staticmethod
    def _squashed_log_prob(distribution, raw_action):
        correction = torch.log(1.0 - torch.tanh(raw_action).square() + 1e-6)
        return (distribution.log_prob(raw_action) - correction).sum(-1)

    def act(self, features, *, deterministic=False):
        distribution, value = self.distribution_and_value(features)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        return torch.tanh(raw_action), self._squashed_log_prob(distribution, raw_action), value

    def evaluate_actions(self, features, actions):
        distribution, value = self.distribution_and_value(features)
        raw_action = torch.atanh(actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        return self._squashed_log_prob(distribution, raw_action), distribution.entropy().sum(-1), value

    @torch.no_grad()
    def predict(self, observation, *, deterministic=True):
        action, _, _ = self.act(self.extract_frozen_features(observation), deterministic=deterministic)
        return action.cpu().numpy()

    def get_model_config(self):
        return {"hidden_dim": self.hidden_dim, "state_schema": STATE_SCHEMA}

    @classmethod
    def from_checkpoint(cls, checkpoint_path, *, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not checkpoint.get("uses_privileged_state", False):
            raise ValueError("Expected a privileged state-based checkpoint")
        policy = cls(**checkpoint["model_config"])
        policy.load_state_dict(checkpoint["policy_state_dict"])
        return policy.to(device).eval()
