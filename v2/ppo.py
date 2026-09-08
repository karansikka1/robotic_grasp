"""Reuse the PPO optimizer and v1 model with the v2 task adapter."""

from dataclasses import dataclass, field
from v1.ppo import PPOConfig as BasePPOConfig, train_ppo as train_base_ppo
from v2.model import FeatureHistory, TemporalPPOPolicy
from v2.task import GreenLiftSimulator, LiftTaskConfig


@dataclass(frozen=True)
class PPOConfig(BasePPOConfig):
    # v1's success_reward/step_penalty fields are unused by this task.
    success_reward: float = 0.0
    step_penalty: float = 0.0
    task: LiftTaskConfig = field(default_factory=LiftTaskConfig)
    task_name: str = "v2-green-lift"

    def validate(self) -> None:
        super().validate()
        self.task.validate()


def train_ppo(policy, config, run_dir, *, device, mini_evaluation_fn=None):
    return train_base_ppo(
        policy, config, run_dir, device=device,
        mini_evaluation_fn=mini_evaluation_fn,
        simulator_factory=lambda: GreenLiftSimulator(config.task),
        reward_fn=lambda observation: float(observation["task_reward"]),
        episode_metrics_fn=lambda observation: observation["task_metrics"],
        rollout_context=FeatureHistory(policy) if isinstance(policy, TemporalPPOPolicy) else None,
    )
