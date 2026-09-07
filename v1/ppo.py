"""Minimal single-simulator PPO implementation for the v1 baseline."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from motion_planning.simulator import Simulator
from v1.model import PrivilegedPPOPolicy


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 100_000
    rollout_steps: int = 1_024
    update_epochs: int = 10
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    max_episode_steps: int = 500
    success_reward: float = 10.0
    step_penalty: float = 0.001
    training_seed: int = 10_000
    checkpoint_interval: int = 10

    def validate(self) -> None:
        positive_ints = {
            "total_timesteps": self.total_timesteps,
            "rollout_steps": self.rollout_steps,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            "max_episode_steps": self.max_episode_steps,
            "checkpoint_interval": self.checkpoint_interval,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.minibatch_size > self.rollout_steps:
            raise ValueError("minibatch_size cannot exceed rollout_steps")


class RolloutBuffer:
    """CPU buffer of frozen visual features and PPO transition values."""

    def __init__(self) -> None:
        self.features: dict[str, list[Tensor]] = defaultdict(list)
        self.actions: list[Tensor] = []
        self.log_probs: list[Tensor] = []
        self.values: list[Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        features: Mapping[str, Tensor],
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: float,
        done: bool,
    ) -> None:
        for name, feature in features.items():
            self.features[name].append(feature.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.values.append(value.detach().cpu())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def tensors(self) -> dict[str, Any]:
        return {
            "features": {
                name: torch.stack(values) for name, values in self.features.items()
            },
            "actions": torch.stack(self.actions),
            "log_probs": torch.stack(self.log_probs),
            "values": torch.stack(self.values),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
        }


def compute_gae(
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    last_value: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute generalized advantage estimates and value targets on CPU."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros((), dtype=torch.float32)
    last_value = last_value.detach().cpu().float()

    for index in reversed(range(len(rewards))):
        next_value = last_value if index == len(rewards) - 1 else values[index + 1]
        next_nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * next_nonterminal - values[index]
        last_advantage = (
            delta + gamma * gae_lambda * next_nonterminal * last_advantage
        )
        advantages[index] = last_advantage
    return advantages, advantages + values


def save_checkpoint(
    path: Path,
    policy: PrivilegedPPOPolicy,
    optimizer: Adam,
    config: PPOConfig,
    *,
    global_step: int,
    update: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "version": 1,
            "global_step": global_step,
            "update": update,
            "config": asdict(config),
            "model_config": {
                "embedding_dim": policy.embedding_dim,
                "hidden_dim": policy.hidden_dim,
                "image_size": policy.image_size,
                "max_depth_m": policy.max_depth_m,
            },
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "uses_privileged_depth": True,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _reset_episode(
    simulator: Simulator,
    seed: int,
) -> Mapping[str, Any]:
    np.random.seed(seed)
    simulator.reset()
    action_min, _ = simulator.action_spec
    return simulator.step(np.zeros_like(action_min))


def _episode_policy_step_limit(simulator: Simulator, requested: int) -> int:
    """Keep the reset bootstrap plus policy actions within robosuite's horizon."""
    environment = getattr(simulator, "env", None)
    if environment is None or bool(getattr(environment, "ignore_done", False)):
        return requested
    horizon = getattr(environment, "horizon", None)
    timestep = getattr(environment, "timestep", None)
    if horizon is None or timestep is None:
        return requested
    return min(requested, max(int(horizon) - int(timestep), 0))


def _batch_to_device(
    features: Mapping[str, Tensor], indices: Tensor, device: torch.device
) -> dict[str, Tensor]:
    return {name: value[indices].to(device) for name, value in features.items()}


def train_ppo(
    policy: PrivilegedPPOPolicy,
    config: PPOConfig,
    run_dir: Path,
    *,
    device: torch.device,
) -> Path:
    """Train ``policy`` and return the final checkpoint path."""
    config.validate()
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    optimizer = Adam(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        eps=1e-5,
    )
    policy.to(device)
    policy.train()

    np.random.seed(config.training_seed)
    torch.manual_seed(config.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training_seed)

    simulator = Simulator(has_renderer=False)
    episode_seed = config.training_seed
    observation = _reset_episode(simulator, episode_seed)
    episode_step_limit = _episode_policy_step_limit(
        simulator, config.max_episode_steps
    )
    episode_steps = 0
    episode_return = 0.0
    completed_episodes = 0
    recent_successes: deque[float] = deque(maxlen=100)
    recent_lengths: deque[int] = deque(maxlen=100)
    global_step = 0
    update = 0

    try:
        while global_step < config.total_timesteps:
            update += 1
            steps_this_rollout = min(
                config.rollout_steps, config.total_timesteps - global_step
            )
            buffer = RolloutBuffer()
            positive_reward_seen = False

            for _ in range(steps_this_rollout):
                features = policy.extract_frozen_features(observation)
                with torch.no_grad():
                    action, log_prob, value = policy.act(features)
                next_observation = simulator.step(action.cpu().numpy())
                success = bool(next_observation["task_complete"])
                reward = config.success_reward * float(success) - config.step_penalty
                positive_reward_seen = positive_reward_seen or reward > 0
                episode_steps += 1
                episode_return += reward
                horizon_reached = episode_steps >= episode_step_limit
                done = success or horizon_reached

                buffer.add(
                    features,
                    action,
                    log_prob,
                    value,
                    reward,
                    done,
                )
                global_step += 1

                if done:
                    completed_episodes += 1
                    recent_successes.append(float(success))
                    recent_lengths.append(episode_steps)
                    writer.add_scalar(
                        "episode/return", episode_return, completed_episodes
                    )
                    writer.add_scalar(
                        "episode/length", episode_steps, completed_episodes
                    )
                    writer.add_scalar(
                        "episode/success", float(success), completed_episodes
                    )
                    episode_seed += 1
                    observation = _reset_episode(simulator, episode_seed)
                    episode_step_limit = _episode_policy_step_limit(
                        simulator, config.max_episode_steps
                    )
                    episode_steps = 0
                    episode_return = 0.0
                else:
                    observation = next_observation

            with torch.no_grad():
                final_features = policy.extract_frozen_features(observation)
                _, last_value = policy.distribution_and_value(final_features)

            rollout = buffer.tensors()
            advantages, returns = compute_gae(
                rollout["rewards"],
                rollout["dones"],
                rollout["values"],
                last_value,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

            metric_totals: dict[str, float] = defaultdict(float)
            minibatches = 0
            early_stop = False
            batch_size = len(rollout["rewards"])
            for _ in range(config.update_epochs):
                permutation = torch.randperm(batch_size)
                for start in range(0, batch_size, config.minibatch_size):
                    indices = permutation[start : start + config.minibatch_size]
                    batch_features = _batch_to_device(
                        rollout["features"], indices, device
                    )
                    actions = rollout["actions"][indices].to(device)
                    old_log_probs = rollout["log_probs"][indices].to(device)
                    batch_advantages = advantages[indices].to(device)
                    batch_returns = returns[indices].to(device)
                    old_values = rollout["values"][indices].to(device)

                    new_log_probs, entropy, new_values = policy.evaluate_actions(
                        batch_features, actions
                    )
                    log_ratio = new_log_probs - old_log_probs
                    ratio = log_ratio.exp()
                    unclipped_loss = -batch_advantages * ratio
                    clipped_loss = -batch_advantages * ratio.clamp(
                        1.0 - config.clip_coefficient,
                        1.0 + config.clip_coefficient,
                    )
                    policy_loss = torch.maximum(unclipped_loss, clipped_loss).mean()

                    value_delta = new_values - old_values
                    clipped_values = old_values + value_delta.clamp(
                        -config.clip_coefficient, config.clip_coefficient
                    )
                    value_loss = 0.5 * torch.maximum(
                        (new_values - batch_returns).square(),
                        (clipped_values - batch_returns).square(),
                    ).mean()
                    entropy_loss = entropy.mean()
                    loss = (
                        policy_loss
                        + config.value_coefficient * value_loss
                        - config.entropy_coefficient * entropy_loss
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), config.max_grad_norm
                    )
                    optimizer.step()

                    with torch.no_grad():
                        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_fraction = (
                            (ratio - 1.0).abs() > config.clip_coefficient
                        ).float().mean()
                    metrics = {
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "entropy": entropy_loss,
                        "approximate_kl": approximate_kl,
                        "clip_fraction": clip_fraction,
                        "gradient_norm": gradient_norm,
                    }
                    for name, metric in metrics.items():
                        metric_totals[name] += float(metric.detach().cpu())
                    minibatches += 1

                    if (
                        float(approximate_kl.detach().cpu())
                        > config.target_kl
                    ):
                        early_stop = True
                        break
                if early_stop:
                    break

            for name, total in metric_totals.items():
                writer.add_scalar(f"ppo/{name}", total / minibatches, global_step)
            writer.add_scalar(
                "rollout/positive_reward_seen",
                float(positive_reward_seen),
                global_step,
            )
            if recent_successes:
                writer.add_scalar(
                    "rollout/success_rate_100",
                    float(np.mean(recent_successes)),
                    global_step,
                )
                writer.add_scalar(
                    "rollout/mean_episode_length_100",
                    float(np.mean(recent_lengths)),
                    global_step,
                )
            writer.add_scalar("charts/learning_rate", config.learning_rate, global_step)
            writer.add_scalar("charts/completed_episodes", completed_episodes, global_step)
            writer.flush()

            if update % config.checkpoint_interval == 0:
                save_checkpoint(
                    run_dir / "checkpoints" / f"step_{global_step:09d}.pt",
                    policy,
                    optimizer,
                    config,
                    global_step=global_step,
                    update=update,
                )
    finally:
        simulator.close()
        writer.close()

    final_checkpoint = run_dir / "checkpoint_final.pt"
    save_checkpoint(
        final_checkpoint,
        policy,
        optimizer,
        config,
        global_step=global_step,
        update=update,
    )
    return final_checkpoint
