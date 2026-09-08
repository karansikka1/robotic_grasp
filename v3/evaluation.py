"""Evaluate the state policy with v2's grasp/lift metrics and optional videos."""

from v2.evaluation import evaluate_policy as evaluate_lift_policy
from v2.task import LiftTaskConfig
from v3.state import state_policy_observation
from v3.task import StateGreenLiftSimulator


def evaluate_policy(policy_fn, evaluation_dir, *, task_config=LiftTaskConfig(), **kwargs):
    record_video = kwargs.get("record_video", True)
    kwargs.setdefault("max_steps", 500)
    return evaluate_lift_policy(
        policy_fn, evaluation_dir, task_config=task_config,
        simulator_factory=lambda: StateGreenLiftSimulator(
            task_config, record_video=record_video, horizon=kwargs["max_steps"] + 1,
        ),
        observation_adapter=state_policy_observation,
        **kwargs,
    )
