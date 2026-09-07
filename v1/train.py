"""Train and evaluate the privileged-depth v1 vanilla PPO baseline."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from harness import MINI_EVALUATION_EPISODES, evaluate_policy
from v1.model import PrivilegedPPOPolicy, privileged_policy_observation
from v1.ppo import PPOConfig, train_ppo


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_training_run_dir(root: Path, experiment_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", experiment_name).strip("-._")
    if not safe_name:
        raise ValueError("experiment name must contain at least one letter or number")
    run_dir = root.expanduser().resolve() / f"{safe_name}-{uuid.uuid4()}"
    run_dir.mkdir(parents=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    defaults = PPOConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="v1-vanilla-ppo")
    parser.add_argument("--runs-dir", type=Path, default=Path("v1/runs"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--rollout-steps", type=int, default=defaults.rollout_steps)
    parser.add_argument("--update-epochs", type=int, default=defaults.update_epochs)
    parser.add_argument("--minibatch-size", type=int, default=defaults.minibatch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--max-episode-steps", type=int, default=defaults.max_episode_steps
    )
    parser.add_argument("--training-seed", type=int, default=defaults.training_seed)
    parser.add_argument(
        "--mini-eval-interval-updates",
        type=int,
        default=defaults.mini_eval_interval_updates,
        help="run a mini evaluation every N PPO updates; 0 disables it",
    )
    parser.add_argument(
        "--mini-eval-episodes",
        type=int,
        default=MINI_EVALUATION_EPISODES,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="disable ImageNet weights for offline architecture smoke tests",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="disable both periodic mini evaluation and final evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    config = replace(
        PPOConfig(),
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        max_episode_steps=args.max_episode_steps,
        training_seed=args.training_seed,
        mini_eval_interval_updates=args.mini_eval_interval_updates,
    )
    config.validate()
    if args.mini_eval_episodes <= 0:
        raise ValueError("--mini-eval-episodes must be greater than zero")
    run_dir = create_training_run_dir(args.runs_dir, args.exp_name)
    run_config = {
        "experiment_name": args.exp_name,
        "device": str(device),
        "pretrained": not args.no_pretrained,
        "uses_privileged_depth": True,
        "evaluation": {
            "enabled": not args.skip_evaluation,
            "mini_episodes": args.mini_eval_episodes,
            "mini_seed": 0,
        },
        "ppo": asdict(config),
        "model": {
            "image_size": args.image_size,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "max_depth_m": args.max_depth_m,
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Training artifacts: {run_dir}")
    print(f"Device: {device}")

    np.random.seed(config.training_seed)
    torch.manual_seed(config.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training_seed)
    policy = PrivilegedPPOPolicy(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        image_size=args.image_size,
        max_depth_m=args.max_depth_m,
        pretrained=not args.no_pretrained,
    )
    mini_evaluation_fn = None
    if not args.skip_evaluation and config.mini_eval_interval_updates > 0:

        def mini_evaluation_fn(
            current_policy: PrivilegedPPOPolicy,
            update: int,
            global_step: int,
        ) -> dict[str, Any]:
            print(
                f"Running {args.mini_eval_episodes}-episode mini evaluation "
                f"after update {update} at step {global_step}"
            )
            return evaluate_policy(
                current_policy.predict,
                args.evaluation_dir / "mini",
                run_name=f"{args.exp_name}-step-{global_step:09d}",
                episodes=args.mini_eval_episodes,
                max_steps=config.max_episode_steps,
                seed=0,
                observation_adapter=privileged_policy_observation,
                record_video=True,
            )

    checkpoint_path = train_ppo(
        policy,
        config,
        run_dir,
        device=device,
        mini_evaluation_fn=mini_evaluation_fn,
    )
    print(f"Final checkpoint: {checkpoint_path}")

    if not args.skip_evaluation:
        metrics = evaluate_policy(
            policy.predict,
            args.evaluation_dir,
            run_name=args.exp_name,
            max_steps=config.max_episode_steps,
            observation_adapter=privileged_policy_observation,
        )
        print(f"Evaluation artifacts: {metrics['output_dir']}")
        print(json.dumps(metrics["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
