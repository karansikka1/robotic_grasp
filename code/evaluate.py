"""Evaluate packaged BC policies on fixed validation layouts, with optional videos."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch

from motion_planning.simulator import Simulator
from bc_data.recording import VideoWriter, sha256, write_manifest
from bc_data.state_policy import PROPRIO_KEYS, state_features
from bc_data.teacher import read_oracle_state
from bc_policy.data import RGB_KEYS
from bc_policy.models import STATE_TYPES, load_policy

DEFAULT_LAYOUTS = Path(__file__).resolve().parent / 'plans/evaluation_validation.json'
INPUT_KEYS = (*RGB_KEYS, *PROPRIO_KEYS)


def evaluate_one(policy, kind, seed, output, video=False):
    """Count policy actions only; stop after ten consecutive official successes."""
    np.random.seed(seed)
    simulator = Simulator(has_renderer=False)
    video_path = output / f'validation-{seed}.mp4'
    try:
        np.random.seed(seed)
        simulator.reset()
        observation = simulator.step(np.zeros(7, dtype=np.float32))
        success_steps = int(observation['task_complete'])
        reset = getattr(policy, 'reset_history', None)
        if reset:
            reset()
        steps = 0
        with (VideoWriter(video_path, 20) if video else nullcontext()) as writer:
            if writer:
                writer.add_observation(observation)
            with torch.inference_mode():
                while steps < 900 and success_steps < 10:
                    if kind in STATE_TYPES:
                        state = read_oracle_state(simulator)
                        proprio = np.concatenate([observation[key] for key in PROPRIO_KEYS])
                        features = state_features(state.positions, state.eef_position,
                                                  proprio, [None], use_stage=False)
                        values = torch.as_tensor(features, device=next(policy.parameters()).device)
                        action = policy.predict_state(values)[0].cpu().numpy()
                    else:
                        action = policy.predict({key: observation[key] for key in INPUT_KEYS})
                    action = np.asarray(action, dtype=np.float32)
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError('Expected seven finite action values')
                    if action[3:6].any() or np.abs(action).max() > 1.000001:
                        raise ValueError('Expected zero rotations and actions in [-1, 1]')
                    observation = simulator.step(action)
                    success_steps = success_steps + 1 if observation['task_complete'] else 0
                    steps += 1
                    if writer:
                        writer.add_observation(observation)
        success = success_steps >= 10
        return {'seed': seed, 'partition': 'validation', 'success': success,
                'episode_steps': steps, 'simulator_steps': steps + 1,
                'seconds_to_completion': steps / 20 if success else None,
                'video': video_path.name if video else None}
    finally:
        simulator.close()


def summarize(episodes):
    successful = [row for row in episodes if row['success']]
    return {'episodes': len(episodes), 'successes': len(successful),
            'success_rate': len(successful) / len(episodes),
            'mean_completion_actions': (float(np.mean([r['episode_steps'] for r in successful]))
                                        if successful else None),
            'mean_completion_seconds': (float(np.mean([r['seconds_to_completion'] for r in successful]))
                                        if successful else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy', choices=(*STATE_TYPES, 'rgb_lstm', 'rgb_spatial'), required=True)
    parser.add_argument('--layouts', type=Path, default=DEFAULT_LAYOUTS,
                        help='JSON containing a validation_seeds list')
    parser.add_argument('--seeds', type=int, nargs='+', help='Evaluate a subset of the validation seeds')
    parser.add_argument('--output', type=Path, required=True, help='New directory for results and videos')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    available = json.loads(args.layouts.read_text())['validation_seeds']
    seeds = args.seeds if args.seeds is not None else available
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(available):
        parser.error('Choose distinct seeds from the validation layout manifest')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    digest = sha256(args.checkpoint)
    policy = load_policy(args.checkpoint, args.policy, args.device)
    if sha256(args.checkpoint) != digest:
        raise RuntimeError('Checkpoint changed while loading')
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'complete': False, 'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': digest, 'policy_type': args.policy,
              'validation_seeds': seeds, 'success_hold_steps': 10,
              'max_policy_actions': 900, 'control_frequency_hz': 20,
              'uses_privileged_state': args.policy in STATE_TYPES, 'episodes': []}
    write_manifest(result, args.output / 'results.json')
    for seed in seeds:
        row = evaluate_one(policy, args.policy, seed, args.output, args.video)
        result['episodes'].append(row)
        result['summary'] = summarize(result['episodes'])
        write_manifest(result, args.output / 'results.json')
        print(f"Validation {seed}: success={row['success']}, actions={row['episode_steps']}", flush=True)
    result['complete'] = True
    write_manifest(result, args.output / 'results.json')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
