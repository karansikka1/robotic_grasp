"""Evaluate held green lifts with the same task definition as training."""

from dataclasses import asdict
from pathlib import Path
import numpy as np

from harness import evaluate_policy as evaluate_base_policy, _write_json_atomically
from v1.model import privileged_policy_observation
from v2.task import GreenLiftSimulator, LiftTaskConfig


def evaluate_policy(policy_fn, evaluation_dir, *, task_config=LiftTaskConfig(), **kwargs):
    metrics = evaluate_base_policy(
        policy_fn, evaluation_dir,
        simulator_factory=lambda: GreenLiftSimulator(task_config),
        observation_adapter=privileged_policy_observation,
        episode_metrics_fn=lambda observation: observation["task_metrics"],
        **kwargs,
    )
    metrics["task_name"] = "v2-green-lift"
    metrics["config"]["task"] = asdict(task_config)
    episodes = [episode["task_metrics"] for episode in metrics["episodes"]]
    grasp_steps = [episode["steps_to_grasp"] for episode in episodes if episode["grasped"]]
    metrics["summary"].update({
        "grasp_rate": float(np.mean([episode["grasped"] for episode in episodes])),
        "lift_success_rate": metrics["summary"]["success_rate"],
        "mean_max_lift_height_m": float(np.mean([episode["max_lift_height_m"] for episode in episodes])),
        "mean_steps_to_grasp": float(np.mean(grasp_steps)) if grasp_steps else None,
        "mean_grasp_return": float(np.mean([episode["grasp_return"] for episode in episodes])),
        "mean_lift_return": float(np.mean([episode["lift_return"] for episode in episodes])),
    })
    _write_json_atomically(Path(metrics["output_dir"]) / "metrics.json", metrics)
    return metrics
