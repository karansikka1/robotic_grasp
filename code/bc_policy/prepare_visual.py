"""Initialize spatial RGB BC from saved localization weights and a new controller."""
import argparse
import hashlib
from pathlib import Path

import torch

from .data import read_manifest, read_targets
from .visual import VisionStateLSTM


def prepare(localizer_path, encoder_path, kind, *, geometry_data=None, seed=0):
    localizer = torch.load(localizer_path, map_location='cpu', weights_only=False)
    encoder = torch.load(encoder_path, map_location='cpu', weights_only=False)
    cnn = {k.removeprefix('features.'): v for k, v in encoder['state_dict'].items()}
    head = localizer['state_dict']
    if kind == 'finetuned':
        suffix = {k.removeprefix('suffix.'): v for k, v in head.items() if k.startswith('suffix.')}
        if not suffix or localizer.get('train_suffix') is False:
            raise ValueError('Expected localization weights with a fine-tuned layer2 suffix')
        cnn.update({'5.' + k: v for k, v in suffix.items()})
        head = {k.removeprefix('heads.'): v for k, v in head.items() if k.startswith('heads.')}
    config = {'backbone': 'resnet18', 'prefix_half_quantization': kind == 'finetuned',
              'localizer': {'arm': 'spatial_softmax', 'feature_channels': 128, 'grid_size': 16},
              'controller': {'input_dim': 54, 'hidden_dim': 128, 'lstm_layers': 1,
                             'stage_classes': 17, 'action_dim': 4, 'recurrent_activation': 'tanh'},
              'predicted_geometry_inputs': geometry_data is not None}
    # Construct the shared controller before adding geometry so both arms start alike.
    torch.manual_seed(seed)
    base = VisionStateLSTM({**config, 'predicted_geometry_inputs': False})
    base.vision.features.load_state_dict(cnn, strict=True)
    base.spatial_head.load_state_dict({k.removeprefix('heatmap_head.'): v for k, v in head.items()
                                      if k.startswith('heatmap_head.')}, strict=True)
    for name in ('proprio_mean', 'proprio_scale', 'position_mean', 'position_scale'):
        getattr(base, name).copy_(head[name])
    if geometry_data is None:
        return base
    policy = VisionStateLSTM(config)
    state = policy.state_dict()
    state.update(base.state_dict())
    policy.load_state_dict(state)
    policy.input_localizer.load_state_dict(head, strict=True)
    positions = geometry_data['state'][:, :9].reshape(-1, 3, 3)
    offsets = positions - geometry_data['proprio'][:, None, 7:10]
    policy.offset_mean.copy_(offsets.mean(0))
    policy.offset_scale.copy_(offsets.std(0, correction=int(len(offsets) > 1)).clamp_min(1e-3))
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--localizer', required=True, help='Selected localization head checkpoint')
    parser.add_argument('--encoder', required=True, help='Frozen encoder, or frozen prefix for finetuned kind')
    parser.add_argument('--kind', choices=('frozen', 'finetuned'), required=True)
    parser.add_argument('--output', required=True, help='New initialization .pt file')
    parser.add_argument('--predicted-geometry', action='store_true')
    parser.add_argument('--split', help='Training split for geometry offset normalization')
    parser.add_argument('--corrections', help='Optional correction collection added to the training split')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output exists; preserve it and choose a new filename')
    if args.predicted_geometry and not args.split:
        parser.error('--predicted-geometry requires --split')
    if not args.predicted_geometry and (args.split or args.corrections):
        parser.error('--split/--corrections are only needed with --predicted-geometry')
    torch.set_num_threads(1)
    data = None
    if args.predicted_geometry:
        entries = read_manifest(args.split, args.corrections)['train']
        data = read_targets(entries, state=True)
    policy = prepare(args.localizer, args.encoder, args.kind, geometry_data=data, seed=args.seed)
    payload = {'policy_type': 'rgb_spatial', 'model_config': policy.config,
               'policy_state_dict': policy.state_dict(), 'epoch': 0,
               'uses_privileged_state': False, 'uses_depth': False, 'uses_teacher_stage': False,
               'initialization_seed': args.seed,
               'source_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for name, path in [('localizer', args.localizer), ('encoder', args.encoder)]}}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f'Saved {output}; controller is untrained, localization weights are preserved.')


if __name__ == '__main__':
    main()
