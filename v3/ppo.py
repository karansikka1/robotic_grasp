"""Use the existing PPO update and green-lift reward with state observations."""

from dataclasses import dataclass

from v1.ppo import train_ppo as train_base_ppo
from v2.ppo import PPOConfig as LiftPPOConfig
from v3.task import StateGreenLiftSimulator


@dataclass(frozen=True)
class PPOConfig(LiftPPOConfig):
    initialization: dict | None = None
    num_envs: int = 1
    max_episode_steps: int = 1_000
    eval_max_steps: int = 500
    total_timesteps: int = 200_000
    update_epochs: int = 8
    minibatch_size: int = 256


    def validate(self):
        super().validate()
        if not isinstance(self.num_envs, int) or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if self.num_envs > self.rollout_steps:
            raise ValueError("num_envs cannot exceed rollout_steps")
        if self.eval_max_steps <= 0:
            raise ValueError("eval_max_steps must be greater than zero")


def train_ppo(policy, config, run_dir, *, device, mini_evaluation_fn=None):
    config.validate()
    if config.num_envs > 1:
        from v3.parallel_ppo import train_parallel_ppo
        return train_parallel_ppo(
            policy, config, run_dir, device=device, mini_evaluation_fn=mini_evaluation_fn,
        )
    return train_base_ppo(
        policy, config, run_dir, device=device,
        mini_evaluation_fn=mini_evaluation_fn,
        simulator_factory=lambda: StateGreenLiftSimulator(config.task, horizon=config.max_episode_steps + 1),
        reward_fn=lambda observation: float(observation["task_reward"]),
        episode_metrics_fn=lambda observation: observation["task_metrics"],
    )
