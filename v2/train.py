"""Train and evaluate the privileged-depth v2 green grasp-and-lift policy."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from harness import MINI_EVALUATION_EPISODES
from v2.evaluation import evaluate_policy
from v2.task import LiftTaskConfig
from v1.model import PrivilegedPPOPolicy
from v1.train import select_device, create_training_run_dir
from v2.ppo import PPOConfig, train_ppo


logger = logging.getLogger("v2.train")


def parse_args() -> argparse.Namespace:
    defaults = PPOConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="v2-green-lift")
    parser.add_argument("--runs-dir", type=Path, default=Path("v2/runs"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("evaluation/v2"))
    parser.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--rollout-steps", type=int, default=defaults.rollout_steps)
    parser.add_argument("--update-epochs", type=int, default=defaults.update_epochs)
    parser.add_argument("--minibatch-size", type=int, default=defaults.minibatch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--kl-coefficient",
        type=float,
        default=defaults.kl_coefficient,
        help="weight of the KL(old || new) loss penalty; 0 disables it",
    )
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
    parser.add_argument("--grasp-reward", type=float, default=defaults.task.grasp_reward)
    parser.add_argument("--lift-reward", type=float, default=defaults.task.lift_reward)
    parser.add_argument("--lift-height-m", type=float, default=defaults.task.lift_height_m)
    parser.add_argument("--hold-steps", type=int, default=defaults.task.hold_steps)
    parser.add_argument("--final-eval-episodes", type=int, default=25)
    parser.add_argument(
        "--evaluate-untrained", action="store_true",
        help="evaluate the initialized policy on the final evaluation seeds before training",
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
        task=LiftTaskConfig(
            grasp_reward=args.grasp_reward, lift_reward=args.lift_reward,
            lift_height_m=args.lift_height_m, hold_steps=args.hold_steps,
        ),
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        kl_coefficient=args.kl_coefficient,
        max_episode_steps=args.max_episode_steps,
        training_seed=args.training_seed,
        mini_eval_interval_updates=args.mini_eval_interval_updates,
    )
    config.validate()
    if args.final_eval_episodes <= 0:
        raise ValueError("--final-eval-episodes must be greater than zero")
    if args.mini_eval_episodes <= 0:
        raise ValueError("--mini-eval-episodes must be greater than zero")
    run_uuid = str(uuid.uuid4())
    run_dir = create_training_run_dir(args.runs_dir, args.exp_name, run_uuid=run_uuid)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
        ],
        force=True,
    )
    # Route named training loggers and plain logging.info calls to stdout.
    run_logger = logging.getLogger("v1")
    for handler in list(run_logger.handlers):
        run_logger.removeHandler(handler)
        handler.close()
    run_logger.setLevel(logging.INFO)
    run_logger.propagate = True
    # Robosuite uses this separate logger, not the "robosuite" namespace.
    robosuite_logger = logging.getLogger("robosuite_logs")
    robosuite_logger.setLevel(logging.WARNING)
    robosuite_logger.propagate = False
    logging.getLogger("OpenGL").setLevel(logging.WARNING)
    run_config = {
        "task_name": config.task_name,
        "experiment_name": args.exp_name,
        "run_uuid": run_uuid,
        "device": str(device),
        "pretrained": not args.no_pretrained,
        "uses_privileged_depth": True,
        "evaluation": {
            "enabled": not args.skip_evaluation,
            "mini_episodes": args.mini_eval_episodes,
            "mini_seed": 0,
            "final_episodes": args.final_eval_episodes,
            "untrained_baseline": args.evaluate_untrained and not args.skip_evaluation,
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
    logger.info(f"Training artifacts: {run_dir}")
    logger.info(f"Device: {device}")

    np.random.seed(config.training_seed)
    torch.manual_seed(config.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training_seed)
    logger.info("Initializing policy (pretrained=%s)", not args.no_pretrained)
    logger.info("PPO config: %s", asdict(config))
    policy = PrivilegedPPOPolicy(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        image_size=args.image_size,
        max_depth_m=args.max_depth_m,
        pretrained=not args.no_pretrained,
    )
    if args.evaluate_untrained and not args.skip_evaluation:
        logger.info("Evaluating untrained policy baseline")
        baseline = evaluate_policy(
            policy.predict, args.evaluation_dir / "untrained",
            task_config=config.task, run_name=args.exp_name, run_uuid=run_uuid,
            episodes=args.final_eval_episodes, max_steps=config.max_episode_steps,
        )
        logger.info("Untrained baseline: %s | artifacts: %s", baseline["summary"], baseline["output_dir"])
    mini_evaluation_fn = None
    if not args.skip_evaluation and config.mini_eval_interval_updates > 0:

        def mini_evaluation_fn(
            current_policy: PrivilegedPPOPolicy,
            update: int,
            global_step: int,
        ) -> dict[str, Any]:
            logger.info(
                f"Running {args.mini_eval_episodes}-episode mini evaluation "
                f"after update {update} at step {global_step}"
            )
            return evaluate_policy(
                current_policy.predict,
                args.evaluation_dir / "mini",
                run_name=f"{args.exp_name}-step-{global_step:09d}",
                run_uuid=run_uuid,
                episodes=args.mini_eval_episodes,
                max_steps=config.max_episode_steps,
                seed=0,
                task_config=config.task,
                record_video=True,
            )

    checkpoint_path = train_ppo(
        policy,
        config,
        run_dir,
        device=device,
        mini_evaluation_fn=mini_evaluation_fn,
    )
    logger.info(f"Final checkpoint: {checkpoint_path}")

    if not args.skip_evaluation:
        logger.info("Starting final evaluation")
        metrics = evaluate_policy(
            policy.predict,
            args.evaluation_dir,
            run_name=args.exp_name,
            run_uuid=run_uuid,
            max_steps=config.max_episode_steps,
            task_config=config.task,
            episodes=args.final_eval_episodes,
        )
        logger.info(f"Evaluation artifacts: {metrics['output_dir']}")
        logger.info(json.dumps(metrics["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training run failed")
        raise
