"""Train and evaluate the privileged-state v3 green grasp-and-lift policy."""

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

from v3.evaluation import evaluate_policy
from v2.task import LiftTaskConfig
from v3.model import StatePPOPolicy
from v3.initialization import initialize_policy
from v1.train import select_device, create_training_run_dir
from v3.ppo import PPOConfig, train_ppo


logger = logging.getLogger("v3.train")


def parse_args() -> argparse.Namespace:
    defaults = PPOConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="v3-state")
    parser.add_argument("--runs-dir", type=Path, default=Path("v3/runs"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("evaluation/v3"))
    parser.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--num-envs", type=int, default=defaults.num_envs,
                        help="parallel rollout simulators; 1 keeps the original single-simulator path")
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
    parser.add_argument("--eval-max-steps", type=int, default=defaults.eval_max_steps,
                        help="policy steps per evaluation episode (independent of training)")
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
        default=10,
    )
    parser.add_argument(
        "--reach-reward-scale", type=float, default=defaults.task.reach_reward_scale,
        help="reward per meter of distance reduction; retreat is negative, 0 disables",
    )
    parser.add_argument("--grasp-reward", type=float, default=defaults.task.grasp_reward)
    parser.add_argument("--lift-reward", type=float, default=defaults.task.lift_reward)
    parser.add_argument("--lift-height-m", type=float, default=defaults.task.lift_height_m)
    parser.add_argument("--grasp-hold-steps", type=int, default=defaults.task.grasp_hold_steps,
                        help="consecutive bilateral-contact steps before the one-time grasp bonus")
    parser.add_argument("--hold-steps", type=int, default=defaults.task.hold_steps)
    parser.add_argument("--final-eval-episodes", type=int, default=25)
    parser.add_argument(
        "--evaluate-untrained", action="store_true",
        help="evaluate the initialized policy on the final evaluation seeds before training",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint", type=Path, help="initialize policy weights; start fresh optimizer and step counters")
    parser.add_argument("--hidden-dim", type=int, help="MLP width; default 128 or the checkpoint architecture")
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
            reach_reward_scale=args.reach_reward_scale,
            grasp_reward=args.grasp_reward, lift_reward=args.lift_reward,
            lift_height_m=args.lift_height_m, hold_steps=args.hold_steps,
            grasp_hold_steps=args.grasp_hold_steps,
        ),
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        kl_coefficient=args.kl_coefficient,
        max_episode_steps=args.max_episode_steps,
        eval_max_steps=args.eval_max_steps,
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
        "policy_type": "state",
        "task_name": config.task_name,
        "experiment_name": args.exp_name,
        "run_uuid": run_uuid,
        "device": str(device),
        "uses_privileged_depth": False,
        "uses_privileged_state": True,
        "evaluation": {
            "enabled": not args.skip_evaluation,
            "mini_episodes": args.mini_eval_episodes,
            "mini_seed": 0,
            "max_steps": config.eval_max_steps,
            "final_episodes": args.final_eval_episodes,
            "untrained_baseline": args.evaluate_untrained and not args.skip_evaluation,
        },
        "ppo": asdict(config),
        "model": {"hidden_dim": args.hidden_dim},
    }
    logger.info(f"Training artifacts: {run_dir}")
    logger.info(f"Device: {device}")

    np.random.seed(config.training_seed)
    torch.manual_seed(config.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training_seed)
    logger.info("Initializing state MLP policy; training cameras disabled")
    logger.info("PPO config: %s", asdict(config))
    policy, initialization = initialize_policy(
        args.init_checkpoint, hidden_dim=args.hidden_dim, device=device,
    )
    config = replace(config, initialization=initialization)
    run_config["ppo"] = asdict(config)
    run_config["initialization"] = initialization
    if initialization is not None:
        logger.info("Initialized policy from %s at source step %d; new optimizer and counters",
                    initialization["checkpoint"], initialization["global_step"])
    run_config["model"] = policy.get_model_config()
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    logger.info("Policy: state | model config: %s", policy.get_model_config())
    if args.evaluate_untrained and not args.skip_evaluation:
        logger.info("Evaluating initialized policy under this run's reward definition")
        baseline = evaluate_policy(
            policy.predict, args.evaluation_dir / "untrained",
            task_config=config.task, run_name=args.exp_name, run_uuid=run_uuid,
            episodes=args.final_eval_episodes, max_steps=config.eval_max_steps,
        )
        logger.info("Initial baseline: %s | artifacts: %s", baseline["summary"], baseline["output_dir"])
    mini_evaluation_fn = None
    if not args.skip_evaluation and config.mini_eval_interval_updates > 0:

        def mini_evaluation_fn(
            current_policy: StatePPOPolicy,
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
                max_steps=config.eval_max_steps,
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
            max_steps=config.eval_max_steps,
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
