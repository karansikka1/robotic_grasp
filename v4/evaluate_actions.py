"""Report offline action and gripper-switch errors for a saved BC checkpoint."""

import argparse
import json
import logging
from pathlib import Path

from v4.data import H5TransitionDataset, validate_manifest, write_manifest
from v4.model import load_policy
from v4.train_bc import cache_frozen_features, evaluate_cached, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--partition', choices=('train', 'validation', 'test'), default='validation')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    path = args.checkpoint.resolve()
    manifest = json.loads((path.parent / 'split.json').read_text())
    validate_manifest(manifest)
    device = select_device(args.device)
    policy = load_policy(path, device=device)
    config_path = path.parent / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    dataset = H5TransitionDataset(manifest[args.partition])
    try:
        features, actions = cache_frozen_features(policy, dataset, batch_size=32)
    finally:
        dataset.close()
    metrics = evaluate_cached(policy, features, actions, batch_size=64, device=device,
                              gripper_loss=config.get('bc', {}).get('gripper_loss', 'smooth_l1'))
    output = write_manifest({'checkpoint': str(path), 'partition': args.partition, 'metrics': metrics},
                            path.parent / f'{args.partition}_action_diagnostics.json')
    print(json.dumps({'output': str(output), 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
