"""Collect successful teacher demonstrations on distinct reproducible random seeds."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import uuid

import numpy as np

from v4.data import discover_trajectories, write_manifest
from v4.record_trajectory import generate_one_success
from v4.teacher import TeacherConfig


def _attempt(root, episode_seed):
    try:
        result = generate_one_success(root, name='bc-demo', seed=episode_seed, max_attempts=1,
                                      max_steps=900, teacher_config=TeacherConfig())
    except RuntimeError as error:
        if not str(error).startswith('Teacher did not produce a successful trajectory'):
            raise
        return {'seed': episode_seed, 'success': False, 'error': str(error)}
    return result['result']


def collect(root, *, count=12, seed=0, max_attempts=None, workers=1, target_total=None):
    if workers < 1:
        raise ValueError('workers must be positive')
    root = Path(root)
    existing = discover_trajectories(root) if list(root.glob('*/trajectory.h5')) else []
    used = {item.seed for item in existing}
    if target_total is not None:
        if target_total < 1:
            raise ValueError('target_total must be positive')
        count = max(0, target_total - len(used))
    for path in sorted(root.glob('collection-*.json')):
        previous = json.loads(path.read_text())
        used.update(int(item['seed']) for item in previous['attempts'])
    if target_total is not None and count == 0:
        print(f'Already have {target_total} distinct successful seeds', flush=True)
        return None
    if count < 1:
        raise ValueError('count must be positive')
    limit = max_attempts if max_attempts is not None else count * 2
    if limit < count:
        raise ValueError('max_attempts cannot be less than count')
    rng = np.random.default_rng(seed)
    manifest_path = root / f'collection-{uuid.uuid4()}.json'
    manifest = {'collection_seed': seed, 'requested_successes': count,
                'existing_successes': len({item.seed for item in existing}),
                'target_total': target_total, 'workers': workers, 'attempts': []}
    successes = 0
    write_manifest(manifest, manifest_path)
    # Spawn isolates OpenGL contexts; fixed batches keep seed selection reproducible.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        while successes < count and len(manifest['attempts']) < limit:
            batch_size = min(workers, count - successes, limit - len(manifest['attempts']))
            seeds = []
            for _ in range(batch_size):
                episode_seed = int(rng.integers(0, 2**31 - 1))
                while episode_seed in used:
                    episode_seed = int(rng.integers(0, 2**31 - 1))
                used.add(episode_seed)
                seeds.append(episode_seed)
            futures = [pool.submit(_attempt, root, episode_seed) for episode_seed in seeds]
            for future in futures:
                result = future.result()
                manifest['attempts'].append(result)
                successes += int(result['success'])
                manifest['successes'] = successes
                write_manifest(manifest, manifest_path)
                print(f'Collected {successes}/{count} successes; manifest={manifest_path}', flush=True)
    if successes != count:
        raise RuntimeError(f'Collected only {successes}/{count}; inspect {manifest_path}')
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('v4/trajectories'))
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-attempts', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--target-total', type=int, help='fill dataset to this many successful seeds')
    args = parser.parse_args()
    print(collect(args.output_dir, count=args.count, seed=args.seed, max_attempts=args.max_attempts,
                  workers=args.workers, target_total=args.target_total))


if __name__ == '__main__':
    main()
