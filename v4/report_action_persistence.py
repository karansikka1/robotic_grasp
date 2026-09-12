"""Measure teacher-action persistence without loading a policy or simulator.

This is an offline baseline using the preceding demonstration action. It is
not an executable controller and does not estimate closed-loop success.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def summarize(entries):
    targets, previous_actions = [], []
    for entry in entries:
        with h5py.File(entry['path'], 'r') as trajectory:
            actions = np.asarray(trajectory['actions'], dtype=np.float64)
        if actions.shape != (entry['steps'], 7) or len(actions) == 0:
            raise ValueError(f'Invalid action shape: {entry["path"]}')
        if not np.isfinite(actions).all() or not np.isin(actions[:, 6], [-1, 1]).all():
            raise ValueError('Expected finite actions and binary -1/+1 gripper labels')
        targets.append(actions)
        # Match BC history padding; never carry actions across trajectories.
        previous_actions.append(np.concatenate([np.zeros_like(actions[:1]), actions[:-1]]))
    targets = np.concatenate(targets)
    previous = np.concatenate(previous_actions)
    error = np.abs(previous - targets)
    switches = (previous[:, 6] != 0) & ((previous[:, 6] > 0) != (targets[:, 6] > 0))
    correct = (previous[:, 6] > 0) == (targets[:, 6] > 0)
    return {
        'trajectories': len(entries),
        'samples': len(targets),
        'mae': float(error.mean()),
        'action_mae': error.mean(axis=0).tolist(),
        'translation_mae': float(error[:, :3].mean()),
        'gripper_sign_accuracy': float(correct.mean()),
        'gripper_switch_samples': int(switches.sum()),
        'gripper_switch_accuracy': float(correct[switches].mean()) if switches.any() else None,
        'gripper_switch_fraction': float(switches.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split_manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.split_manifest.read_text())
    report = {
        'baseline': 'previous demonstration action; zero at each trajectory start',
        'closed_loop_evaluation': False,
        'split_manifest': str(args.split_manifest.resolve()),
        'partitions': {name: summarize(manifest[name]) for name in ('train', 'validation', 'test')},
    }
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
