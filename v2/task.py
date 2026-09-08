"""External grasp-and-lift task adapter; distance progress plus grasp/lift milestones."""

from dataclasses import dataclass
import logging
import math
from typing import Any

from motion_planning.simulator import Simulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiftTaskConfig:
    grasp_reward: float = 1.0
    lift_reward: float = 5.0
    lift_height_m: float = 0.05
    hold_steps: int = 5
    reach_reward_scale: float = 1.0

    def validate(self) -> None:
        for name in ("grasp_reward", "lift_reward", "lift_height_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.reach_reward_scale) or self.reach_reward_scale < 0:
            raise ValueError("reach_reward_scale must be finite and non-negative")
        if not isinstance(self.hold_steps, int) or self.hold_steps <= 0:
            raise ValueError("hold_steps must be a positive integer")

    @classmethod
    def from_saved_config(cls, config: dict[str, Any]) -> "LiftTaskConfig":
        # Checkpoints predating reaching rewards used only the two milestones.
        return cls(**{"reach_reward_scale": 0.0, **config})


class LiftEpisode:
    """Track one-time milestones from contact and height measurements."""

    def __init__(
        self, config: LiftTaskConfig, initial_height: float, initial_distance_m: float
    ) -> None:
        config.validate()
        self.config = config
        self.initial_height = initial_height
        if not math.isfinite(initial_distance_m) or initial_distance_m < 0:
            raise ValueError("Gripper distance must be finite and non-negative")
        self.initial_distance_m = initial_distance_m
        self.previous_distance_m = initial_distance_m
        self.min_distance_m = initial_distance_m
        self.reach_return = 0.0
        self.steps = 0
        self.steps_to_grasp: int | None = None
        self.steps_to_lift: int | None = None
        self.held_steps = 0
        self.max_lift_height_m = 0.0
        self.grasp_return = 0.0
        self.lift_return = 0.0
        self.success = False

    def advance(self, grasped: bool, height: float, distance_m: float) -> float:
        if not math.isfinite(height):
            raise ValueError("Green object height is not finite")
        if not math.isfinite(distance_m) or distance_m < 0:
            raise ValueError("Gripper distance must be finite and non-negative")
        self.steps += 1
        lift_height = height - self.initial_height
        self.max_lift_height_m = max(self.max_lift_height_m, lift_height)
        # Signed progress: approach earns reward, retreat loses it, hovering
        # earns zero. Undiscounted rewards telescope to scale * (start - end).
        reward = self.config.reach_reward_scale * (self.previous_distance_m - distance_m)
        self.reach_return += reward
        self.previous_distance_m = distance_m
        self.min_distance_m = min(self.min_distance_m, distance_m)
        if grasped and self.steps_to_grasp is None:
            self.steps_to_grasp = self.steps
            self.grasp_return = self.config.grasp_reward
            reward += self.config.grasp_reward
            logger.info("Green grasp achieved at step %d (+%.1f)", self.steps, self.config.grasp_reward)
        if grasped and lift_height >= self.config.lift_height_m:
            self.held_steps += 1
        else:
            self.held_steps = 0
        if not self.success and self.held_steps >= self.config.hold_steps:
            self.success = True
            self.steps_to_lift = self.steps
            self.lift_return = self.config.lift_reward
            reward += self.config.lift_reward
            logger.info("Green held lift achieved at step %d (+%.1f)", self.steps, self.config.lift_reward)
        return reward

    def metrics(self) -> dict[str, Any]:
        return {
            "grasped": self.steps_to_grasp is not None,
            "lift_success": self.success,
            "max_lift_height_m": self.max_lift_height_m,
            "steps_to_grasp": self.steps_to_grasp,
            "steps_to_lift": self.steps_to_lift,
            "initial_gripper_distance_m": self.initial_distance_m,
            "final_gripper_distance_m": self.previous_distance_m,
            "min_gripper_distance_m": self.min_distance_m,
            "reach_return": self.reach_return,
            "grasp_return": self.grasp_return,
            "lift_return": self.lift_return,
        }


class GreenLiftSimulator:
    """Use the same scene/cameras, with green-lift rewards and success externally.

    The first step following reset is the harness/PPO zero-action bootstrap.
    It sets the resting height and initial distance, earning no reward or hold credit.
    Extra task fields are excluded by privileged_policy_observation.
    """

    def __init__(self, config: LiftTaskConfig = LiftTaskConfig()) -> None:
        config.validate()
        self.config = config
        self.simulator = Simulator(has_renderer=False)
        self.episode: LiftEpisode | None = None

    @property
    def env(self):
        return self.simulator.env

    @property
    def action_spec(self):
        return self.simulator.action_spec

    def reset(self) -> None:
        self.simulator.reset()
        self.episode = None

    def step(self, action):
        observation = dict(self.simulator.step(action))
        height = float(self.env.sim.data.body_xpos[self.env.cubeB_body_id][2])
        distance_m = float(self.env._gripper_to_target(
            gripper=self.env.robots[0].gripper, target=self.env.cubeB,
            target_type="body", return_distance=True,
        ))
        if self.episode is None:
            self.episode = LiftEpisode(self.config, height, distance_m)
            reward = 0.0
        else:
            grasped = bool(self.env._check_grasp(
                gripper=self.env.robots[0].gripper, object_geoms=self.env.cubeB,
            ))
            reward = self.episode.advance(grasped, height, distance_m)
        observation["task_complete"] = self.episode.success
        observation["task_reward"] = reward
        observation["task_metrics"] = self.episode.metrics()
        return observation

    def close(self) -> None:
        self.simulator.close()
