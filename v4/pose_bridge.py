"""Replace exact object positions with learned RGB positions in a fixed oracle BC controller.

Teacher stage and exact grip-site position remain privileged. This diagnostic
isolates object localization; it is not a legal final actor.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from v1.model import PROPRIO_KEYS
from v4.control_diagnostics import StateBC, rollout_state, selected_training_entries, teacher_config
from v4.data import validate_manifest, write_manifest
from v4.model import FrozenResNetFeatures
from v4.train_bc import create_run_dir


class RGBPositions(nn.Module):
    def __init__(self, *, pretrained=True):
        super().__init__()
        self.backbone = FrozenResNetFeatures(pretrained=pretrained, pool_size=4).eval()
        self.head = nn.Sequential(nn.Linear(16400, 128), nn.ReLU(), nn.Linear(128, 9))
        self.register_buffer('feature_mean', torch.zeros(16400))
        self.register_buffer('feature_scale', torch.ones(16400))
        self.register_buffer('target_mean', torch.zeros(9))
        self.register_buffer('target_scale', torch.ones(9))
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

    @torch.no_grad()
    def encode(self, observations):
        self.backbone.eval()
        cameras = []
        for key in ('frontview_image', 'robot0_eye_in_hand_image'):
            value = torch.as_tensor(observations[key], device=self.image_mean.device)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            value = value.permute(0, 3, 1, 2).float().div(255).flip(-2)
            value = F.interpolate(value, size=(128, 128), mode='bilinear', align_corners=False, antialias=True)
            cameras.append(self.backbone((value - self.image_mean) / self.image_std))
        proprio = [torch.as_tensor(observations[key], device=self.image_mean.device).reshape(len(cameras[0]), -1)
                   for key in PROPRIO_KEYS]
        return torch.cat(cameras + proprio, dim=1).float()

    def standardized_prediction(self, features):
        return self.head((features - self.feature_mean) / self.feature_scale)

    def forward(self, features):
        return self.standardized_prediction(features) * self.target_scale + self.target_mean

    @torch.no_grad()
    def predict(self, observation):
        return self(self.encode(observation))[0].reshape(3, 3).cpu().numpy()


def cache(model, entries):
    features, targets = [], []
    for index, entry in enumerate(entries):
        with h5py.File(entry['path'], 'r') as demo:
            for start in range(0, entry['steps'], 32):
                obs = {key: demo['observations'][key][start:start + 32] for key in
                       (*PROPRIO_KEYS, 'frontview_image', 'robot0_eye_in_hand_image')}
                features.append(model.encode(obs).cpu())
                targets.append(torch.tensor(demo['oracle/object_positions'][start:start + 32]).float().flatten(1))
        if (index + 1) % 10 == 0:
            print(f'Cached {index + 1}/{len(entries)} trajectories', flush=True)
    return torch.cat(features), torch.cat(targets)


@torch.no_grad()
def metrics(model, features, targets):
    predictions = torch.cat([model(batch.to(model.feature_mean.device)).cpu() for batch in features.split(128)])
    error = predictions - targets
    distances = error.reshape(-1, 3, 3).norm(dim=2)
    return {'coordinate_mae_m': float(error.abs().mean()),
            'mean_object_distance_error_m': distances.mean(0).tolist(),
            'p95_object_distance_error_m': distances.quantile(.95, dim=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controller', type=Path)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    manifest = json.loads((args.controller.parent / 'split.json').read_text())
    validate_manifest(manifest)
    used = {entry['seed'] for name in ('train', 'validation', 'test') for entry in manifest[name]}
    if used.intersection(range(1000000, 1000005)):
        raise ValueError('Fresh rollout seeds overlap training data')
    np.random.seed(0)
    torch.manual_seed(0)
    output = create_run_dir(Path('v4/runs'), 'v4-diagnostic-rgb-positions')
    print(f'Run directory: {output}', flush=True)
    write_manifest({'controller': str(args.controller.resolve()), 'epochs': args.epochs,
                    'seed': 0, 'learning_rate': 3e-4, 'device': args.device,
                    'remaining_privileged_inputs': ['live teacher stage', 'exact grip-site position'],
                    'pose_inputs': ['frontview_image', 'robot0_eye_in_hand_image', *PROPRIO_KEYS]}, output / 'config.json')
    write_manifest(manifest, output / 'split.json')
    model = RGBPositions().to(args.device)
    train_x, train_y = cache(model, manifest['train'])
    val_x, val_y = cache(model, manifest['validation'])
    with torch.no_grad():
        model.feature_mean.copy_(train_x.mean(0))
        model.feature_scale.copy_(train_x.std(0).clamp_min(.01))
        model.target_mean.copy_(train_y.mean(0))
        model.target_scale.copy_(train_y.std(0).clamp_min(.01))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=3e-4, weight_decay=1e-5)
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        for indices in torch.randperm(len(train_y)).split(64):
            prediction = model.standardized_prediction(train_x[indices].to(args.device))
            target = (train_y[indices].to(args.device) - model.target_mean) / model.target_scale
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        measured = metrics(model, val_x, val_y)
        if measured['coordinate_mae_m'] < best:
            best = measured['coordinate_mae_m']
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'validation': measured}, output / 'best.pt')
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps({'epoch': epoch, **measured}) + '\n')
        print(f'epoch={epoch} validation_coordinate_mae_m={measured["coordinate_mae_m"]:.6f}', flush=True)
    saved = torch.load(output / 'best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    test_x, test_y = cache(model, manifest['test'])
    write_manifest({'selected_epoch': saved['epoch'],
                    'train': metrics(model, train_x, train_y), 'validation': metrics(model, val_x, val_y),
                    'test': metrics(model, test_x, test_y)}, output / 'pose_metrics.json')
    state = torch.load(args.controller, map_location=args.device, weights_only=False)
    if not state.get('use_stage', True):
        raise ValueError('Pose bridge requires the controller trained with teacher stage')
    controller = StateBC(state['state_dict']['mean'], state['state_dict']['scale']).to(args.device)
    controller.load_state_dict(state['state_dict'])
    controller.eval()
    selected = selected_training_entries(manifest)
    results = {'stage_source': 'live privileged teacher; transitions use exact state',
               'object_positions_source': 'RGB regressor', 'train': [], 'fresh': []}
    config = teacher_config(selected[0])
    for partition, entries in [('train', selected), ('fresh', [{'seed': seed} for seed in range(1000000, 1000005)])]:
        for entry in entries:
            result = rollout_state(controller, entry['seed'], config, output,
                                   entry=entry if partition == 'train' else None,
                                   position_predictor=model.predict)
            results[partition].append(result)
            write_manifest(results, output / 'rollouts.json')
            print(f'{partition} seed={entry["seed"]} success={result["success"]} failure={result["failure"]}', flush=True)


if __name__ == '__main__':
    main()
