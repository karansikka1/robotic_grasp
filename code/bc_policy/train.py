"""Train BC from collected HDF5 episodes; select checkpoints by validation action loss."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_manifest, read_targets, sequence_batches
from .models import STATE_TYPES, make_state_policy, load_state_checkpoint


def losses(logits, actions, stage_logits, stages, mask, stage_weight):
    """Average over the seven simulator controls, including three structural zeros."""
    if not mask.any():
        return logits.sum() * 0, logits.sum() * 0
    action = F.smooth_l1_loss(logits.tanh()[mask], actions[mask])
    stage = F.cross_entropy(stage_logits[mask], stages[mask]) if stage_weight else action.new_zeros(())
    return action, stage


def run_epoch(policy, data, entries, args, *, optimizer=None, generator=None, pixels=None):
    training = optimizer is not None
    policy.train(training)
    totals = dict(action_loss=0., stage_loss=0., translation_mae=0., gripper_accuracy=0.)
    count = 0
    if args.policy == 'state_mlp':
        supervised = data['mask'].nonzero().flatten()
        order = torch.randperm(len(supervised), generator=generator) if training else torch.arange(len(supervised))
        batches = ((supervised[i], torch.ones(len(i), dtype=torch.bool), True)
                   for i in order.split(args.batch_size))
    else:
        order = torch.randperm(len(entries), generator=generator).tolist() if training else list(range(len(entries)))
        batches = sequence_batches(entries, order, args.chunk_length, args.batch_trajectories)
    hidden = None
    with torch.set_grad_enabled(training):
        for indices, valid, reset in batches:
            batch = {key: value[indices].to(args.device) for key, value in data.items()}
            mask = valid.to(args.device) & batch['mask']
            if reset:
                hidden = None
            if args.policy == 'state_mlp':
                logits, stage = policy(batch['state']), None
            elif args.policy in STATE_TYPES:
                if args.stage_weight:
                    logits, hidden, stage = policy.sequence_with_stage(batch, hidden)
                else:
                    logits, hidden = policy.sequence(batch, hidden)
                    stage = None
            else:
                images = pixels.read(indices)
                logits, hidden, stage = policy.sequence_images(images, batch['proprio'], hidden)
            action_loss, stage_loss = losses(logits, batch['actions'], stage, batch['stages'], mask, args.stage_weight)
            loss = action_loss + args.stage_weight * stage_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite BC loss')
            if training and mask.any():
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.policy != 'state_mlp':
                    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],
                                                   1., error_if_nonfinite=True)
                optimizer.step()
            if hidden is not None:
                hidden = tuple(value.detach() for value in hidden)
            n = int(mask.sum())
            if n:
                prediction, target = logits.detach().tanh()[mask], batch['actions'][mask]
                totals['action_loss'] += float(action_loss.detach()) * n
                totals['stage_loss'] += float(stage_loss.detach()) * n
                totals['translation_mae'] += float((prediction[:, :3] - target[:, :3]).abs().mean()) * n
                totals['gripper_accuracy'] += float(((prediction[:, 6] > 0) == (target[:, 6] > 0)).float().mean()) * n
                count += n
    if not count:
        raise ValueError('No expert targets in this partition')
    return {**{key: value / count for key, value in totals.items()}, 'supervised_actions': count}


def save_checkpoint(path, policy, args, epoch, metrics):
    payload = {'algorithm': 'behavior_cloning', 'policy_type': args.policy,
               'policy_state_dict': policy.state_dict(), 'model_config': policy.get_model_config(),
               'epoch': epoch, 'metrics': metrics, 'training': vars(args),
               'uses_privileged_state': args.policy in STATE_TYPES, 'uses_depth': False,
               'uses_teacher_stage': False, 'use_stage': False, 'uses_previous_action': False}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=(*STATE_TYPES, 'rgb_lstm', 'rgb_spatial'), required=True)
    parser.add_argument('--split', required=True, help='Completed demonstration split.json')
    parser.add_argument('--corrections', help='Optional completed correction collection.json')
    parser.add_argument('--output', required=True, help='New directory for this training run')
    parser.add_argument('--init-checkpoint', help='Initialize from saved weights, with a fresh optimizer')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--batch-size', type=int, default=256, help='MLP frames per batch')
    parser.add_argument('--batch-trajectories', type=int)
    parser.add_argument('--chunk-length', type=int, default=32)
    parser.add_argument('--finetune-cnn', action='store_true', help='Update layer1/layer2 during spatial RGB BC')
    parser.add_argument('--cnn-learning-rate', type=float, default=1e-5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    is_state = args.policy in STATE_TYPES
    default_lr = (1e-3 if not args.init_checkpoint else 1e-4) if is_state else (3e-4 if args.policy == 'rgb_lstm' else 1e-4)
    args.learning_rate = args.learning_rate if args.learning_rate is not None else default_lr
    args.batch_trajectories = args.batch_trajectories if args.batch_trajectories is not None else (8 if is_state else 2)
    args.stage_weight = .01 if args.policy in ('state_lstm_aux', 'rgb_lstm', 'rgb_spatial') else 0.
    if min(args.epochs, args.batch_size, args.batch_trajectories, args.chunk_length) < 1:
        parser.error('Epochs and batch/chunk sizes must be positive')
    if any(not math.isfinite(lr) or lr <= 0 for lr in (args.learning_rate, args.cnn_learning_rate)):
        parser.error('Learning rates must be positive and finite')
    if args.finetune_cnn and args.policy != 'rgb_spatial':
        parser.error('--finetune-cnn requires rgb_spatial')
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; use a new directory to preserve earlier runs')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    manifest = read_manifest(args.split, args.corrections)
    data = {part: read_targets(manifest[part], state=is_state) for part in ('train', 'validation')}
    pixels = {}
    if is_state:
        if args.init_checkpoint:
            policy = load_state_checkpoint(args.init_checkpoint, args.policy, args.device)
        else:
            state = data['train']['state']
            mean, scale = state.mean(0), state.std(0, correction=0).clamp_min(.01)
            if len(state) > 1:
                scale = state.std(0).clamp_min(.01)
            mean[-17:], scale[-17:] = 0, 1
            policy = make_state_policy(args.policy, mean, scale).to(args.device)
    else:
        from .visual import make_visual_policy, PixelReader
        policy = make_visual_policy(args.policy, data['train']['proprio'], args.init_checkpoint,
                                    args.finetune_cnn).to(args.device)
        pixels = {part: PixelReader(manifest[part]) for part in data}
    parameters = [p for p in policy.parameters() if p.requires_grad]
    if args.finetune_cnn:
        cnn = [p for p in policy.vision.parameters() if p.requires_grad]
        cnn_ids = {id(p) for p in cnn}
        groups = [{'params': [p for p in parameters if id(p) not in cnn_ids]},
                  {'params': cnn, 'lr': args.cnn_learning_rate}]
    else:
        groups = parameters
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=1e-5)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    (output / 'split.json').write_text(json.dumps(manifest, indent=2) + '\n')
    generator = torch.Generator().manual_seed(args.seed)
    best = float('inf')
    try:
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(policy, data['train'], manifest['train'], args,
                              optimizer=optimizer, generator=generator, pixels=pixels.get('train'))
            validation = run_epoch(policy, data['validation'], manifest['validation'], args,
                                   pixels=pixels.get('validation'))
            metrics = {'epoch': epoch, 'train': train, 'validation': validation}
            with (output / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(metrics) + '\n')
            if validation['action_loss'] < best:
                best = validation['action_loss']
                save_checkpoint(output / 'best.pt', policy, args, epoch, metrics)
            print(f'epoch={epoch} train_loss={train["action_loss"]:.6f} validation_loss={validation["action_loss"]:.6f}', flush=True)
        save_checkpoint(output / 'final.pt', policy, args, epoch, metrics)
    finally:
        for reader in pixels.values():
            reader.close()


if __name__ == '__main__':
    main()
