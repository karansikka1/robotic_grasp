"""Evaluate BC policies on fresh simulator seeds, with history reset and videos."""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

from harness import evaluate_policy, _write_json_atomically
from motion_planning.simulator import Simulator
from v1.model import privileged_policy_observation
from v4.model import load_policy
from v1.train import select_device
from v4.teacher import read_oracle_state


class StableStackSimulator(Simulator):
    """Require consecutive official successes, matching the teacher's settle check."""

    def __init__(self, *, success_hold_steps=10):
        self.success_hold_steps = success_hold_steps
        self.success_steps = 0
        super().__init__(has_renderer=False)

    def reset(self):
        self.success_steps = 0
        return super().reset()

    def step(self, action):
        observation = super().step(action)
        self.success_steps = self.success_steps + 1 if observation['task_complete'] else 0
        observation['task_complete'] = self.success_steps >= self.success_hold_steps
        return observation


class DiagnosticStackSimulator(StableStackSimulator):
    """Training-only telemetry is filtered out before the policy sees observations."""

    def reset(self):
        self.initial_heights = None
        self.contact_steps = np.zeros(3, dtype=int)
        self.lift_streak = np.zeros(3, dtype=int)
        self.held_lift = np.zeros(3, dtype=bool)
        self.max_lift = np.zeros(3)
        self.stack_streak = np.zeros(2, dtype=int)
        self.placed = np.zeros(2, dtype=bool)
        return super().reset()

    def step(self, action):
        observation = super().step(action)
        state = read_oracle_state(self)
        if self.initial_heights is None:
            self.initial_heights = state.positions[:, 2].copy()
        else:
            lift = state.positions[:, 2] - self.initial_heights
            self.contact_steps += state.grasped
            self.max_lift = np.maximum(self.max_lift, lift)
            self.lift_streak = np.where(state.grasped & (lift >= .05), self.lift_streak + 1, 0)
            self.held_lift |= self.lift_streak >= 5
            for pair, (lower, upper) in enumerate(((0, 1), (1, 2))):
                stacked = (state.object_contacts[pair] and not state.grasped[upper]
                           and state.positions[upper, 2] > state.positions[lower, 2] + .02)
                self.stack_streak[pair] = self.stack_streak[pair] + 1 if stacked else 0
            self.placed |= self.stack_streak >= 5
        observation['evaluation_metrics'] = {
            'bilateral_contact_steps_red_green_blue': self.contact_steps.tolist(),
            'held_lift_red_green_blue': self.held_lift.tolist(),
            'max_lift_m_red_green_blue': self.max_lift.tolist(),
            'green_on_red_observed': bool(self.placed[0]),
            'blue_on_green_observed': bool(self.placed[1]),
        }
        return observation


def _evaluation_policy(path, device, zero_rotation=False):
    policy = load_policy(path, device=device)
    if zero_rotation:
        # The demonstrated controller never rotates; constrain these outputs before
        # prediction so action history contains exactly the executed command.
        with torch.no_grad():
            policy.actor[-1].weight[3:6].zero_()
            policy.actor[-1].bias[3:6].zero_()
    return policy


def evaluate_checkpoint(checkpoint_path, *, episodes=5, max_steps=900, seed=1000000,
                        device='cpu', output_dir=Path('evaluation/v4'), record_video=True,
                        split_manifest=None, zero_rotation=False):
    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('algorithm') != 'behavior_cloning':
        raise ValueError('Expected a behavior-cloning checkpoint')
    split_path = Path(split_manifest) if split_manifest else path.parent / 'split.json'
    manifest = json.loads(split_path.read_text())
    used_seeds = {int(entry['seed']) for name in ('train', 'validation', 'test')
                  for entry in manifest.get(name, [])}
    if used_seeds.intersection(range(seed, seed + episodes)):
        raise ValueError('Rollout evaluation seeds overlap the demonstration dataset; choose fresh seeds')
    policy = _evaluation_policy(path, device, zero_rotation)
    reset = getattr(policy, 'reset_history', None)
    metrics = evaluate_policy(
        policy.predict, output_dir, run_name=path.parent.name + '-' + path.stem
        + ('-zero-rotation' if zero_rotation else ''),
        episodes=episodes, max_steps=max_steps, seed=seed,
        observation_adapter=privileged_policy_observation,
        policy_reset_fn=reset, simulator_factory=DiagnosticStackSimulator,
        episode_metrics_fn=lambda observation: observation['evaluation_metrics'],
        record_video=record_video,
    )
    metrics['task_name'] = 'full-stack'
    metrics['config']['success_hold_steps'] = 10
    metrics['config']['checkpoint'] = str(path)
    metrics['config']['split_manifest'] = str(split_path.resolve())
    metrics['config']['model'] = checkpoint['model_config']
    metrics['config']['fresh_seeds_verified'] = True
    metrics['config']['zero_rotation'] = zero_rotation
    _write_json_atomically(Path(metrics['output_dir']) / 'metrics.json', metrics)
    return metrics


def evaluate_training_seeds(checkpoint_path, *, episodes=10, max_steps=900,
                            device='cpu', output_dir=Path('evaluation/v4'), record_video=True,
                            zero_rotation=False):
    """Training diagnostics are explicitly separate from fresh-seed evaluation."""
    path = Path(checkpoint_path).resolve()
    manifest = json.loads((path.parent / 'split.json').read_text())
    entries = sorted(manifest['train'], key=lambda item: (item['seed'], item['path']))
    by_seed = {entry['seed']: entry for entry in entries}
    entries = [by_seed[seed] for seed in sorted(by_seed)]
    if episodes < 1:
        raise ValueError('episodes must be positive')
    selected = np.random.default_rng(0).permutation(len(entries))[:episodes]
    policy = _evaluation_policy(path, device, zero_rotation)
    results = []
    for index in selected:
        entry = entries[int(index)]
        with h5py.File(entry['path'], 'r') as demonstration:
            initial = {key: np.asarray(value[0]) for key, value in demonstration['observations'].items()}
        first_observation = True
        initial_matches = None

        def predict(observation):
            nonlocal first_observation, initial_matches
            if first_observation:
                initial_matches = all(np.array_equal(observation[key], initial[key])
                                      for key in observation)
                first_observation = False
            return policy.predict(observation)

        metrics = evaluate_policy(
            predict, output_dir, run_name=f'{path.parent.name}-train-{entry["seed"]}'
            + ('-zero-rotation' if zero_rotation else ''),
            episodes=1, max_steps=max_steps, seed=int(entry['seed']),
            observation_adapter=privileged_policy_observation,
            policy_reset_fn=getattr(policy, 'reset_history', None),
            simulator_factory=DiagnosticStackSimulator, record_video=record_video,
            episode_metrics_fn=lambda observation: observation['evaluation_metrics'],
        )
        result = {**metrics['episodes'][0], 'output_dir': metrics['output_dir'],
                  'initial_observation_matches_demonstration': initial_matches}
        metrics['config'].update(checkpoint=str(path), evaluation_partition='train',
                                 fresh_seeds_verified=False,
                                 zero_rotation=zero_rotation,
                                 initial_observation_matches_demonstration=initial_matches)
        _write_json_atomically(Path(metrics['output_dir']) / 'metrics.json', metrics)
        results.append(result)
        logging.info('Training seed %s: success=%s; initial observation match=%s',
                     entry['seed'], result['success'], initial_matches)
    successful = [item for item in results if item['success']]
    return {'checkpoint': str(path), 'evaluation_partition': 'train', 'episodes': results,
            'zero_rotation': zero_rotation,
            'summary': {'episodes': len(results), 'successes': len(successful),
                        'success_rate': len(successful) / len(results),
                        'mean_steps_to_completion': (float(np.mean([item['steps_to_completion']
                                                                   for item in successful]))
                                                     if successful else None),
                        'all_initial_observations_match': all(
                            item['initial_observation_matches_demonstration'] for item in results)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--split-manifest', type=Path)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=900)
    parser.add_argument('--seed', type=int, default=1000000)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', type=Path, default=Path('evaluation/v4'))
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--training-seeds', action='store_true')
    parser.add_argument('--zero-rotation', action='store_true', help='diagnostic: constrain rotation outputs to teacher zeros')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('robosuite_logs').setLevel(logging.WARNING)
    if args.training_seeds:
        metrics = evaluate_training_seeds(
            args.checkpoint, episodes=args.episodes, max_steps=args.max_steps,
            device=select_device(args.device), output_dir=args.output_dir,
            record_video=not args.no_video,
            zero_rotation=args.zero_rotation,
        )
        filename = 'train_rollout_metrics_zero_rotation.json' if args.zero_rotation else 'train_rollout_metrics.json'
        _write_json_atomically(args.checkpoint.resolve().parent / filename, metrics)
        print(json.dumps(metrics['summary'], indent=2))
        return
    metrics = evaluate_checkpoint(
        args.checkpoint, episodes=args.episodes, max_steps=args.max_steps,
        seed=args.seed, device=select_device(args.device), output_dir=args.output_dir,
        record_video=not args.no_video, split_manifest=args.split_manifest,
        zero_rotation=args.zero_rotation,
    )
    print(json.dumps({'output_dir': metrics['output_dir'], 'summary': metrics['summary']}, indent=2))


if __name__ == '__main__':
    main()
