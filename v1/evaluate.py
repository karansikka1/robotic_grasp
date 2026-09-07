"""Evaluate a saved v1 checkpoint on the fixed evaluation initializations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness import (
    DEFAULT_EVALUATION_EPISODES,
    DEFAULT_MAX_STEPS,
    evaluate_policy,
)
from v1.model import PrivilegedPPOPolicy, privileged_policy_observation
from v1.train import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--name", help="evaluation run name")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVALUATION_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    policy = PrivilegedPPOPolicy.from_checkpoint(args.checkpoint, device=device)
    metrics = evaluate_policy(
        policy.predict,
        args.output_dir,
        run_name=args.name or args.checkpoint.stem,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        observation_adapter=privileged_policy_observation,
        record_video=not args.no_video,
    )
    print(f"Evaluation artifacts: {metrics['output_dir']}")
    print(json.dumps(metrics["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
