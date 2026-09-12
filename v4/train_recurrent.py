"""Train full-episode LSTM BC with ordered, truncated-backpropagation chunks."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v4.control_diagnostics import STAGES, read_state_data, selected_training_entries, teacher_config, rollout_state
from v4.data import H5TransitionDataset, validate_manifest, write_manifest
from v4.evaluate import evaluate_training_seeds, evaluate_checkpoint
from v4.model import load_policy
from v4.recurrent import SpatialLSTMPolicy, StateLSTMPolicy, load_state_lstm
from v4.train_bc import cache_frozen_features, create_run_dir


def trajectory_groups(entries, order, batch_trajectories):
    starts = np.cumsum([0] + [entry['steps'] for entry in entries[:-1]])
    for offset in range(0, len(order), batch_trajectories):
        chosen = order[offset:offset + batch_trajectories]
        yield [(int(starts[index]), entries[index]['steps']) for index in chosen]


def chunks(features, actions, group, chunk_length, device):
    """End padding is masked; no chunk crosses a trajectory boundary."""
    starts = torch.tensor([start for start, _ in group])[:, None]
    lengths = torch.tensor([length for _, length in group])[:, None]
    for time in range(0, int(lengths.max()), chunk_length):
        times = torch.arange(time, min(time + chunk_length, int(lengths.max())))[None, :]
        mask = times < lengths
        indices = starts + torch.minimum(times, lengths - 1)
        values = {key: value[indices].to(device) for key, value in features.items()}
        yield values, actions[indices].to(device), mask.to(device), indices


def masked_loss(logits, targets, mask):
    return F.smooth_l1_loss(logits.tanh()[mask], targets[mask])


@torch.no_grad()
def measure(policy, features, actions, entries, device, batch_trajectories=8, chunk_length=32):
    policy.eval()
    predictions = torch.empty_like(actions)
    for group in trajectory_groups(entries, list(range(len(entries))), batch_trajectories):
        hidden = None
        for values, targets, mask, indices in chunks(features, actions, group, chunk_length, device):
            logits, hidden = policy.sequence(values, hidden)
            predictions[indices[mask.cpu()]] = logits.tanh()[mask].cpu()
    error = (predictions - actions).abs()
    correct = (predictions[:, 6] > 0) == (actions[:, 6] > 0)
    switches = torch.zeros(len(actions), dtype=torch.bool)
    start = 0
    for entry in entries:
        stop = start + entry['steps']
        switches[start + 1:stop] = (actions[start + 1:stop, 6] > 0) != (actions[start:stop - 1, 6] > 0)
        start = stop
    return {'loss': float(F.smooth_l1_loss(predictions, actions)),
            'mae': float(error.mean()), 'translation_mae': float(error[:, :3].mean()),
            'action_mae': error.mean(0).tolist(), 'gripper_sign_accuracy': float(correct.float().mean()),
            'gripper_switch_samples': int(switches.sum()),
            'gripper_switch_accuracy': float(correct[switches].float().mean()) if switches.any() else None}


def save(path, policy, optimizer, epoch, metrics, args):
    temporary = path.with_suffix('.tmp')
    torch.save({'algorithm': 'behavior_cloning', 'policy_type': 'spatial_lstm' if args.mode == 'visual' else 'state_lstm',
                'policy_state_dict': policy.state_dict(), 'model_config': policy.get_model_config(),
                'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'metrics': metrics,
                'uses_privileged_depth': args.mode == 'visual', 'uses_privileged_state_and_stage': args.mode == 'state',
                'training': {'chunk_length': args.chunk_length, 'batch_trajectories': args.batch_trajectories,
                             'learning_rate': args.learning_rate, 'seed': 0, 'hidden_state_reset': 'trajectory boundary'}}, temporary)
    temporary.replace(path)


def fit(policy, data, manifest, args, output):
    parameters = [value for name, value in policy.named_parameters()
                  if value.requires_grad and not name.startswith('critic.') and name != 'log_std']
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(0)
    best = float('inf')
    features, actions = data['train']
    for epoch in range(1, args.epochs + 1):
        policy.train()
        order = torch.randperm(len(manifest['train']), generator=generator).tolist()
        total_loss, total_samples = 0., 0
        for group in trajectory_groups(manifest['train'], order, args.batch_trajectories):
            hidden = None
            for values, targets, mask, _ in chunks(features, actions, group, args.chunk_length, args.device):
                logits, hidden = policy.sequence(values, hidden)
                loss = masked_loss(logits, targets, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                # Carry memory forward; truncate only the gradient graph.
                hidden = tuple(value.detach() for value in hidden)
                count = int(mask.sum())
                total_loss += float(loss.detach()) * count
                total_samples += count
        metrics = measure(policy, *data['validation'], manifest['validation'], args.device)
        metrics.update(epoch=epoch, train_loss=total_loss / total_samples)
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(metrics) + '\n')
        if metrics['loss'] < best:
            best = metrics['loss']
            save(output / 'best.pt', policy, optimizer, epoch, metrics, args)
        print(f'epoch={epoch} train_loss={metrics["train_loss"]:.6f} validation_translation_mae={metrics["translation_mae"]:.6f}', flush=True)
    save(output / 'final.pt', policy, optimizer, epoch, metrics, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('visual', 'state'))
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--chunk-length', type=int, default=32)
    parser.add_argument('--batch-trajectories', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--split-manifest', type=Path, default=Path('v4/splits/diverse100.json'))
    args = parser.parse_args()
    args.epochs = args.epochs if args.epochs is not None else (30 if args.mode == 'visual' else 200)
    args.learning_rate = args.learning_rate if args.learning_rate is not None else (3e-4 if args.mode == 'visual' else 1e-3)
    args.batch_trajectories = args.batch_trajectories if args.batch_trajectories is not None else (2 if args.mode == 'visual' else 8)
    if min(args.epochs, args.chunk_length, args.batch_trajectories) < 1 or not 0 < args.learning_rate < float('inf'):
        raise ValueError('Training settings must be positive and finite')
    np.random.seed(0)
    torch.manual_seed(0)
    manifest = json.loads(args.split_manifest.read_text())
    validate_manifest(manifest)
    used = {entry['seed'] for part in ('train', 'validation', 'test') for entry in manifest[part]}
    if used.intersection(range(1000000, 1000005)):
        raise ValueError('Fresh seeds overlap demonstrations')
    output = create_run_dir(Path('v4/runs'), f'v4-lstm-{args.mode}')
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(output / 'train.log')])
    logging.getLogger('robosuite_logs').setLevel(logging.WARNING)
    print(f'Run directory: {output}', flush=True)
    write_manifest(manifest, output / 'split.json')
    write_manifest({**vars(args), 'split_manifest': str(args.split_manifest.resolve()),
                    'seed': 0, 'memory': 'persistent h/c; reset at trajectory/episode boundaries',
                    'previous_action_input': False, 'lstm_layers': 1,
                    'selection_partition': 'validation'}, output / 'config.json')
    data = {}
    if args.mode == 'state':
        for part in ('train', 'validation'):
            x, y, _ = read_state_data(manifest[part])
            data[part] = ({'state': x}, y)
        values = data['train'][0]['state']
        mean, scale = values.mean(0), values.std(0).clamp_min(.01)
        mean[-len(STAGES):], scale[-len(STAGES):] = 0, 1
        policy = StateLSTMPolicy(mean, scale).to(args.device)
    else:
        policy = SpatialLSTMPolicy(pretrained=True).to(args.device)
        for part in ('train', 'validation'):
            dataset = H5TransitionDataset(manifest[part])
            try:
                print(f'Caching {part}', flush=True)
                data[part] = cache_frozen_features(policy, dataset, batch_size=32)
            finally:
                dataset.close()
    fit(policy, data, manifest, args, output)
    policy = load_policy(output / 'best.pt', device=args.device) if args.mode == 'visual' else load_state_lstm(output / 'best.pt', args.device)
    metrics = {part: measure(policy, *data[part], manifest[part], args.device) for part in ('train', 'validation')}
    if args.mode == 'state':
        x, y, _ = read_state_data(manifest['test'])
        test = ({'state': x}, y)
    else:
        dataset = H5TransitionDataset(manifest['test'])
        try:
            test = cache_frozen_features(policy, dataset, batch_size=32)
        finally:
            dataset.close()
    metrics['test'] = measure(policy, *test, manifest['test'], args.device)
    write_manifest(metrics, output / 'action_metrics.json')
    if args.mode == 'visual':
        train = evaluate_training_seeds(output / 'best.pt', episodes=5, device=args.device)
        write_manifest(train, output / 'train_rollout_metrics.json')
        print(f'Training rollouts: {train["summary"]}', flush=True)
        fresh = evaluate_checkpoint(output / 'best.pt', episodes=5, device=args.device)
        write_manifest({'summary': fresh['summary'], 'output_dir': fresh['output_dir']}, output / 'rollout_metrics.json')
        print(f'Fresh rollouts: {fresh["summary"]}', flush=True)
    else:
        selected = selected_training_entries(manifest)
        results = {'stage_source': 'live privileged teacher; action discarded', 'train': [], 'fresh': []}
        for part, entries in [('train', selected), ('fresh', [{'seed': seed} for seed in range(1000000, 1000005)])]:
            for entry in entries:
                result = rollout_state(policy, entry['seed'], teacher_config(selected[0]), output,
                                       entry=entry if part == 'train' else None)
                results[part].append(result)
                write_manifest(results, output / 'rollouts.json')
                print(f'{part} seed={entry["seed"]} success={result["success"]} steps={result["episode_steps"]}', flush=True)


if __name__ == '__main__':
    main()
