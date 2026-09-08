"""Batched policy inference and parallel simulator collection for v3 PPO."""

from collections import deque
import logging
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from v1.ppo import RolloutBuffer, compute_gae, optimize_ppo, save_checkpoint
from v3.vector_env import ParallelSimulators

logger = logging.getLogger(__name__)


def combine_rollouts(buffers, last_values, config):
    """Compute GAE along each simulator's timeline before flattening the batch."""
    rollouts, advantages, returns = [], [], []
    for env_id, buffer in enumerate(buffers):
        if not buffer.rewards:
            continue
        rollout = buffer.tensors()
        adv, ret = compute_gae(
            rollout['rewards'], rollout['dones'], rollout['values'], last_values[env_id],
            gamma=config.gamma, gae_lambda=config.gae_lambda,
        )
        rollouts.append(rollout)
        advantages.append(adv)
        returns.append(ret)
    combined = {
        'features': {
            name: torch.cat([r['features'][name] for r in rollouts])
            for name in rollouts[0]['features']
        },
        **{name: torch.cat([r[name] for r in rollouts])
           for name in ('actions', 'log_probs', 'values', 'rewards', 'dones')},
    }
    return combined, torch.cat(advantages), torch.cat(returns)


def train_parallel_ppo(policy, config, run_dir, *, device, mini_evaluation_fn=None):
    config.validate()
    run_dir.mkdir(parents=True, exist_ok=True)
    policy.to(device).train()
    optimizer = Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
    np.random.seed(config.training_seed)
    torch.manual_seed(config.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training_seed)
    writer = SummaryWriter(log_dir=str(run_dir / 'tensorboard'))
    simulators = None
    global_step = completed_episodes = update = next_env = 0
    recent_successes = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    total_updates = (config.total_timesteps + config.rollout_steps - 1) // config.rollout_steps
    try:
        logger.info('Starting %d simulator processes; %d total rollout steps per update',
                    config.num_envs, config.rollout_steps)
        simulators = ParallelSimulators(config.num_envs, config.task, config.max_episode_steps,
                                       config.training_seed)
        training_started = time.monotonic()
        while global_step < config.total_timesteps:
            update += 1
            steps_this_rollout = min(config.rollout_steps, config.total_timesteps - global_step)
            logger.info('Update %d/%d: collecting %d total steps across %d simulators',
                        update, total_updates, steps_this_rollout, config.num_envs)
            rollout_started = last_progress = time.monotonic()
            buffers = [RolloutBuffer() for _ in range(config.num_envs)]
            collected = 0
            positive_reward_seen = False
            while collected < steps_this_rollout:
                count = min(config.num_envs, steps_this_rollout - collected)
                # Partial final batches preserve the exact requested step budget.
                # Rotation distributes uneven rollout lengths across workers.
                env_ids = [(next_env + i) % config.num_envs for i in range(count)]
                next_env = (next_env + count) % config.num_envs
                features = {'state': torch.as_tensor(simulators.states[env_ids], device=device)}
                with torch.no_grad():
                    actions, log_probs, values = policy.act(features)
                results = simulators.step(env_ids, actions.cpu().numpy())
                for i, (env_id, result) in enumerate(zip(env_ids, results)):
                    _, reward, done, episode = result
                    buffers[env_id].add({'state': features['state'][i]}, actions[i],
                                        log_probs[i], values[i], reward, done)
                    positive_reward_seen = positive_reward_seen or reward > 0
                    if episode is not None:
                        completed_episodes += 1
                        recent_successes.append(float(episode['success']))
                        recent_lengths.append(episode['length'])
                        logger.info('Episode %d | env=%d seed=%d return=%.3f length=%d success=%s success_rate_100=%.1f%%',
                                    completed_episodes, env_id, episode['seed'], episode['return'],
                                    episode['length'], episode['success'], 100 * np.mean(recent_successes))
                        for name in ('return', 'length', 'success'):
                            writer.add_scalar(f'episode/{name}', float(episode[name]), completed_episodes)
                        logger.info('Episode %d task metrics: %s', completed_episodes, episode['task_metrics'])
                        for name, value in episode['task_metrics'].items():
                            if value is not None:
                                writer.add_scalar(f'episode/task/{name}', float(value), completed_episodes)
                collected += count
                global_step += count
                if time.monotonic() - last_progress >= 30:
                    logger.info('Rollout %d/%d | total steps %d/%d (%.1f%%)',
                                collected, steps_this_rollout, global_step, config.total_timesteps,
                                100 * global_step / config.total_timesteps)
                    last_progress = time.monotonic()
            with torch.no_grad():
                _, last_values = policy.distribution_and_value(
                    {'state': torch.as_tensor(simulators.states, device=device)})
            rollout, advantages, returns = combine_rollouts(buffers, last_values, config)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            rollout_seconds = time.monotonic() - rollout_started
            optimization_started = time.monotonic()
            logger.info('Update %d: optimizing PPO (rollout took %.1fs)', update, rollout_seconds)
            totals, minibatches = optimize_ppo(policy, optimizer, config, rollout, advantages, returns, device)
            metrics = {name: total / minibatches for name, total in totals.items()}
            logger.info('Update %d/%d | steps=%d/%d | policy_loss=%.4f value_loss=%.4f entropy=%.4f '
                        'kl=%.5f kl_penalty=%.5f clip_fraction=%.3f | minibatches=%d '
                        '| optimization=%.1fs elapsed=%.1fs steps/s=%.2f',
                        update, total_updates, global_step, config.total_timesteps,
                        metrics['policy_loss'], metrics['value_loss'], metrics['entropy'],
                        metrics['approximate_kl'], metrics['kl_penalty'], metrics['clip_fraction'],
                        minibatches, time.monotonic() - optimization_started,
                        time.monotonic() - training_started,
                        global_step / max(time.monotonic() - training_started, 1e-9))
            for name, value in metrics.items():
                writer.add_scalar(f'ppo/{name}', value, global_step)
            writer.add_scalar('rollout/positive_reward_seen', float(positive_reward_seen), global_step)
            writer.add_scalar('charts/num_envs', config.num_envs, global_step)
            writer.add_scalar('charts/learning_rate', config.learning_rate, global_step)
            writer.add_scalar('charts/completed_episodes', completed_episodes, global_step)
            writer.add_scalar('charts/rollout_steps_per_second', collected / max(rollout_seconds, 1e-9), global_step)
            if recent_successes:
                writer.add_scalar('rollout/success_rate_100', np.mean(recent_successes), global_step)
                writer.add_scalar('rollout/mean_episode_length_100', np.mean(recent_lengths), global_step)
            writer.flush()
            if update % config.checkpoint_interval == 0:
                save_checkpoint(run_dir / 'checkpoints' / f'step_{global_step:09d}.pt', policy,
                                optimizer, config, global_step=global_step, update=update)
            if mini_evaluation_fn is not None and config.mini_eval_interval_updates > 0 and update % config.mini_eval_interval_updates == 0:
                policy.eval()
                try:
                    evaluation = mini_evaluation_fn(policy, update, global_step)
                finally:
                    policy.train()
                logger.info('Mini evaluation complete: %s | artifacts: %s', evaluation['summary'], evaluation['output_dir'])
                for name, value in evaluation['summary'].items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'evaluation/{name}', value, global_step)
                writer.add_text('evaluation/latest_output_dir', str(evaluation['output_dir']), global_step)
                writer.flush()
    finally:
        try:
            if simulators is not None:
                simulators.close()
        finally:
            writer.close()
    checkpoint = run_dir / 'checkpoint_final.pt'
    save_checkpoint(checkpoint, policy, optimizer, config, global_step=global_step, update=update)
    return checkpoint
