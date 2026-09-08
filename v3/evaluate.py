"""Evaluate a v3 state checkpoint using its saved grasp-and-lift task settings."""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from v3.model import StatePPOPolicy
from v1.train import select_device
from v3.evaluation import evaluate_policy
from v2.task import LiftTaskConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/v3"))
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("robosuite_logs").setLevel(logging.WARNING)
    logging.getLogger("OpenGL").setLevel(logging.WARNING)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("task_name") != "v2-green-lift":
        raise ValueError("Expected a v2-green-lift checkpoint")
    task = LiftTaskConfig.from_saved_config(config["task"])
    policy = StatePPOPolicy.from_checkpoint(args.checkpoint, device=select_device(args.device))
    metrics = evaluate_policy(
        policy.predict, args.output_dir,
        task_config=task,
        run_name=args.name or args.checkpoint.stem,
        episodes=args.episodes,
        max_steps=args.max_steps if args.max_steps is not None else config.get("eval_max_steps", config["max_episode_steps"]),
        seed=args.seed, record_video=not args.no_video,
    )
    print(f"Evaluation artifacts: {metrics['output_dir']}")
    print(json.dumps(metrics["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
