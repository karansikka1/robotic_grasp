"""Summarize demonstration placement coverage and action/stage balance."""

import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from v4.data import write_manifest, validate_manifest


def report(manifest_path, reference_path=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    reference = json.loads(Path(reference_path).read_text()) if reference_path else {}
    old_seeds = {entry['seed'] for name in ('train', 'validation', 'test')
                 for entry in reference.get(name, [])}
    positions, quaternions, seeds, lengths = [], [], [], []
    stages, grippers = Counter(), Counter()
    partitions = {}
    for partition in ('train', 'validation', 'test'):
        partitions[partition] = []
        for entry in manifest[partition]:
            with h5py.File(entry['path'], 'r') as trajectory:
                positions.append(trajectory['oracle/object_positions'][0])
                quaternions.append(trajectory['oracle/object_quaternions'][0])
                stages.update(trajectory['stage'].asstr()[:].tolist())
                grippers.update(trajectory['actions'][:, -1].tolist())
            seeds.append(entry['seed'])
            lengths.append(entry['steps'])
            partitions[partition].append(entry['seed'])
    positions = np.asarray(positions)
    quaternions = np.asarray(quaternions)
    prior = np.array([seed in old_seeds for seed in seeds])
    # MuJoCo stores body quaternions in wxyz order.
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    summary = {
        'split_manifest': str(manifest_path.resolve()), 'trajectories': len(seeds),
        'unique_seeds': len(set(seeds)), 'transitions': sum(lengths),
        'reference_trajectories': int(prior.sum()), 'partitions': partitions,
        'steps': {'min': min(lengths), 'max': max(lengths), 'mean': float(np.mean(lengths))},
        'stage_transitions': dict(stages),
        'gripper_counts': {str(key): value for key, value in grippers.items()},
        'objects': {},
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for index, name in enumerate(('red', 'green', 'blue')):
        xy = positions[:, index, :2]
        occupancy, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=4, range=[[-.2, .2], [-.2, .2]])
        summary['objects'][name] = {
            'position_min': positions[:, index].min(axis=0).tolist(),
            'position_max': positions[:, index].max(axis=0).tolist(),
            'yaw_min_rad': float(yaw[:, index].min()), 'yaw_max_rad': float(yaw[:, index].max()),
            'occupied_xy_bins_of_16': int(np.count_nonzero(occupancy)),
        }
        axes[index].scatter(xy[~prior, 0], xy[~prior, 1], label='New demonstrations', s=18, alpha=.65)
        axes[index].scatter(xy[prior, 0], xy[prior, 1], label='Original demonstrations', marker='x', c='black', s=35)
        axes[index].set(title=name.title(), xlabel='World X (m)', ylabel='World Y (m)',
                        xlim=(-.23, .23), ylim=(-.23, .23), aspect='equal')
    axes[0].legend(fontsize=8)
    output = manifest_path.with_suffix('.coverage.json')
    write_manifest(summary, output)
    fig.savefig(manifest_path.with_suffix('.coverage.png'), dpi=150)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--reference-manifest', type=Path)
    args = parser.parse_args()
    print(report(args.manifest, args.reference_manifest))


if __name__ == '__main__':
    main()
