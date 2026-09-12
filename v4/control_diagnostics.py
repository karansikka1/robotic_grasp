"""Perception/control diagnostics; all oracle and stage inputs are diagnostic only."""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from harness import _VideoWriter
from v1.model import PROPRIO_KEYS
from v4.data import H5TransitionDataset, validate_manifest, write_manifest
from v4.evaluate import DiagnosticStackSimulator, evaluate_training_seeds
from v4.model import TransformerTemporalPolicy, load_policy
from v4.teacher import PrivilegedStackTeacher, TeacherConfig, TeacherFailure, read_oracle_state
from v4.train_bc import (BCConfig, cache_frozen_features, create_run_dir,
                         evaluate_cached, train_behavior_cloning)


STAGES = tuple(f'{moving}_{stage}_over_{support}'
               for moving, support in (('green', 'red'), ('blue', 'green'))
               for stage in ('move_above', 'descend', 'close', 'lift', 'transfer',
                             'lower', 'release', 'retreat')) + ('settle',)


def state_features(positions, eef, proprio, stages, *, use_stage=True):
    """Shared offline/online packing. Relative positions are derived, not targets."""
    positions = np.asarray(positions).reshape(-1, 3, 3)
    eef = np.asarray(eef).reshape(-1, 3)
    proprio = np.asarray(proprio).reshape(-1, 16)
    one_hot = (np.eye(len(STAGES))[np.asarray([STAGES.index(stage) for stage in stages])]
               if use_stage else np.zeros((len(positions), len(STAGES))))
    return np.concatenate((positions.reshape(-1, 9), eef, proprio,
                           (positions - eef[:, None, :]).reshape(-1, 9), one_hot), axis=1).astype(np.float32)


def read_state_data(entries):
    values, actions, labels = [], [], []
    for entry in entries:
        with h5py.File(entry['path'], 'r') as demo:
            stages = demo['stage'].asstr()[:].tolist()
            proprio = np.concatenate([demo['observations'][key][:] for key in PROPRIO_KEYS], axis=1)
            values.append(state_features(demo['oracle/object_positions'][:],
                                         demo['oracle/eef_position'][:], proprio, stages))
            actions.append(demo['actions'][:])
            labels.extend(stages)
    return torch.from_numpy(np.concatenate(values)), torch.from_numpy(np.concatenate(actions)), labels


class StateBC(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('scale', scale.clone())
        self.network = nn.Sequential(nn.Linear(len(mean), 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 7))

    def forward(self, values):
        return self.network((values - self.mean) / self.scale).tanh()


@torch.no_grad()
def action_metrics(prediction, targets, labels):
    error = (prediction - targets).abs()
    result = {'loss': float(F.smooth_l1_loss(prediction, targets)),
              'mae': float(error.mean()), 'translation_mae': float(error[:, :3].mean()),
              'action_mae': error.mean(0).tolist(),
              'gripper_sign_accuracy': float(((prediction[:, 6] > 0) == (targets[:, 6] > 0)).float().mean()),
              'per_stage': {}}
    for stage in sorted(set(labels)):
        mask = torch.tensor([label == stage for label in labels], device=targets.device)
        result['per_stage'][stage] = {'samples': int(mask.sum()),
                                     'translation_mae': float(error[mask, :3].mean())}
    return result


def selected_training_entries(manifest, count=5):
    by_seed = {entry['seed']: entry for entry in sorted(manifest['train'], key=lambda e: (e['seed'], e['path']))}
    entries = [by_seed[seed] for seed in sorted(by_seed)]
    return [entries[int(i)] for i in np.random.default_rng(0).permutation(len(entries))[:count]]


def teacher_config(entry):
    with h5py.File(entry['path'], 'r') as demo:
        return TeacherConfig(**json.loads(demo.attrs['teacher_config']))


def rollout_state(policy, seed, config, output, max_steps=900, entry=None, use_stage=True, position_predictor=None):
    """Teacher decides stage transitions on student states; teacher actions are discarded."""
    np.random.seed(seed)
    simulator = DiagnosticStackSimulator()
    teacher = PrivilegedStackTeacher(config)
    if hasattr(policy, 'reset_history'):
        policy.reset_history()
    steps, failure, stages = 0, None, []
    initial_matches = None
    try:
        np.random.seed(seed)
        simulator.reset()
        observation = simulator.step(np.zeros(7, dtype=np.float32))
        if entry is not None:
            with h5py.File(entry['path'], 'r') as demo:
                initial_matches = all(np.array_equal(observation[key], demo['observations'][key][0])
                                      for key in demo['observations'])
        with _VideoWriter(output / f'seed_{seed}.mp4', 20) as writer:
            writer.add_observation(observation)
            while steps < max_steps and not observation['task_complete']:
                stage = None
                if use_stage:
                    try:
                        decision = teacher.decide(simulator)
                    except TeacherFailure as error:
                        failure = str(error)
                        break
                    if decision.done:
                        failure = 'Teacher finished before stable simulator success'
                        break
                    stage = decision.stage
                oracle = read_oracle_state(simulator)
                proprio = np.concatenate([observation[key] for key in PROPRIO_KEYS])
                positions = oracle.positions if position_predictor is None else position_predictor(observation)
                features = state_features(positions, oracle.eef_position, proprio, [stage], use_stage=use_stage)
                with torch.no_grad():
                    values = torch.from_numpy(features).to(policy.mean.device)
                    prediction = policy.predict_state(values) if hasattr(policy, 'predict_state') else policy(values)
                    action = prediction[0].cpu().numpy()
                stages.append(stage)
                observation = simulator.step(action)
                steps += 1
                writer.add_observation(observation)
        success = bool(observation['task_complete'])
        return {'seed': seed, 'success': success, 'episode_steps': steps,
                'simulator_steps': steps + 1, 'steps_to_completion': steps if success else None,
                'seconds_to_completion': steps / 20 if success else None,
                'failure': failure or (None if success else 'max_steps'),
                'last_stage': stages[-1] if stages else None,
                'visited_stages': list(dict.fromkeys(stages)),
                'initial_observation_matches_demonstration': initial_matches,
                'task_metrics': observation['evaluation_metrics'],
                'video': str(output / f'seed_{seed}.mp4')}
    finally:
        simulator.close()


def audit_replay(entry, output):
    """Replay logged actions and compare labels against a live teacher, pre-step."""
    with h5py.File(entry['path'], 'r') as demo:
        actions = demo['actions'][:]
        labels = demo['stage'].asstr()[:]
        positions = demo['oracle/object_positions'][:]
        eef = demo['oracle/eef_position'][:]
        initial = {key: dataset[0] for key, dataset in demo['observations'].items()}
    config = teacher_config(entry)
    np.random.seed(entry['seed'])
    simulator = DiagnosticStackSimulator()
    errors, position_errors, eef_errors, mismatches = [], [], [], 0
    try:
        np.random.seed(entry['seed'])
        simulator.reset()
        obs = simulator.step(np.zeros(7, dtype=np.float32))
        initial_matches = all(np.array_equal(obs[key], value) for key, value in initial.items())
        teacher = PrivilegedStackTeacher(config)
        for index, action in enumerate(actions):
            decision = teacher.decide(simulator)
            state = read_oracle_state(simulator)
            errors.append(np.abs(decision.action - action))
            position_errors.append(np.abs(state.positions - positions[index]).max())
            eef_errors.append(np.abs(state.eef_position - eef[index]).max())
            mismatches += decision.stage != labels[index]
            obs = simulator.step(action)
        result = {'seed': entry['seed'], 'steps': len(actions),
                  'initial_observations_match': initial_matches,
                  'max_teacher_action_error': float(np.max(errors)),
                  'stage_mismatches': mismatches,
                  'max_pre_action_object_position_error_m': float(np.max(position_errors)),
                  'max_pre_action_eef_position_error_m': float(np.max(eef_errors)),
                  'stable_success_after_replay': bool(obs['task_complete']),
                  'action_min': actions.min(axis=0).tolist(), 'action_max': actions.max(axis=0).tolist(),
                  'teacher_translation_scale_m': config.action_translation_m}
        write_manifest(result, output / 'replay_audit.json')
        return result
    finally:
        simulator.close()


def run_privileged(args, manifest, output):
    data = {name: read_state_data(manifest[name]) for name in ('train', 'validation', 'test')}
    if args.without_stage:
        for values, _, _ in data.values():
            values[:, -len(STAGES):] = 0
    x, y, _ = data['train']
    # Standardize only continuous features; stage one-hots remain in [0, 1].
    mean, scale = x.mean(0), x.std(0).clamp_min(.01)
    mean[-len(STAGES):] = 0
    scale[-len(STAGES):] = 1
    policy = StateBC(mean, scale).to(args.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-5)
    tensors = {name: (a.to(args.device), b.to(args.device), labels) for name, (a, b, labels) in data.items()}
    x, y, _ = tensors['train']
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        policy.train()
        for indices in torch.randperm(len(y), device=args.device).split(256):
            loss = F.smooth_l1_loss(policy(x[indices]), y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        policy.eval()
        vx, vy, vl = tensors['validation']
        metrics = action_metrics(policy(vx), vy, vl)
        if metrics['loss'] < best:
            best = metrics['loss']
            torch.save({'algorithm': 'privileged_state_bc_diagnostic', 'epoch': epoch,
                        'state_dict': policy.state_dict(), 'input_dim': len(mean),
                        'stages': STAGES, 'use_stage': not args.without_stage, 'validation': metrics}, output / 'best.pt')
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps({'epoch': epoch, 'loss': metrics['loss'], 'translation_mae': metrics['translation_mae']}) + '\n')
        if epoch % 10 == 0:
            print(f'epoch={epoch} validation_translation_mae={metrics["translation_mae"]:.6f}', flush=True)
    checkpoint = torch.load(output / 'best.pt', map_location=args.device, weights_only=False)
    policy.load_state_dict(checkpoint['state_dict'])
    metrics = {name: action_metrics(policy(a), b, labels) for name, (a, b, labels) in tensors.items()}
    write_manifest({'selected_epoch': checkpoint['epoch'], 'partitions': metrics}, output / 'action_metrics.json')
    selected = selected_training_entries(manifest)
    config = teacher_config(selected[0])
    results = {'stage_source': ('none; no teacher used during rollout' if args.without_stage else
                                'live privileged teacher state machine; teacher actions discarded'),
               'teacher_guard_failure_terminates_episode': not args.without_stage, 'train': [], 'fresh': []}
    for partition, entries in [('train', selected), ('fresh', [{'seed': seed} for seed in range(1000000, 1000005)])]:
        for entry in entries:
            result = rollout_state(policy, entry['seed'], config, output,
                                   entry=entry if partition == 'train' else None, use_stage=not args.without_stage)
            results[partition].append(result)
            write_manifest(results, output / 'rollouts.json')
            print(f'{partition} seed={entry["seed"]} success={result["success"]} last_stage={result["last_stage"]} failure={result["failure"]}', flush=True)


def run_one_demo(args, manifest, output):
    entry = selected_training_entries(manifest, 1)[0]
    # Explicit training-only selection; never disguise this as a validation split.
    write_manifest({'diagnostic': 'one_demo_overfit', 'train': [entry], 'validation': [], 'test': [],
                    'selection_partition': 'train'}, output / 'split.json')
    if args.warm_start:
        source = json.loads((args.warm_start.parent / 'split.json').read_text())
        if source.get('diagnostic') != 'one_demo_overfit' or source['train'] != [entry]:
            raise ValueError('Warm start must use the same single demonstration')
        policy = load_policy(args.warm_start, device=args.device)
        if policy.use_previous_action:
            raise ValueError('Single-demo diagnostic requires no previous-action input')
    else:
        policy = TransformerTemporalPolicy(pretrained=True, rgb_backbone='resnet18',
                                           camera_fusion='concat', rgb_pool_size=4,
                                           use_previous_action=False).to(args.device)
    dataset = H5TransitionDataset([entry])
    try:
        features, actions = cache_frozen_features(policy, dataset, batch_size=32)
    finally:
        dataset.close()
    config = BCConfig(epochs=args.epochs, learning_rate=args.learning_rate)
    checkpoint = train_behavior_cloning(policy, features, actions, features, actions,
                                        config, output, device=torch.device(args.device))
    policy = load_policy(checkpoint, device=args.device)
    metrics = evaluate_cached(policy, features, actions, batch_size=64, device=torch.device(args.device))
    write_manifest({'selection_partition': 'train', 'metrics': metrics,
                    'note': 'Trainer validation_loss log is resubstitution training error for this diagnostic.'},
                   output / 'train_action_metrics.json')
    rollouts = evaluate_training_seeds(checkpoint, episodes=1, device=args.device)
    write_manifest(rollouts, output / 'train_rollout_metrics.json')
    print(json.dumps(rollouts['summary']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('privileged', 'one-demo', 'replay'))
    parser.add_argument('--split-manifest', type=Path, default=Path('v4/splits/diverse100.json'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--warm-start', type=Path, help='one-demo fitting refinement from the same demo checkpoint')
    parser.add_argument('--without-stage', action='store_true', help='privileged diagnostic: remove teacher stage consistently')
    args = parser.parse_args()
    if args.warm_start and args.mode != 'one-demo':
        raise ValueError('--warm-start is only supported for one-demo')
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    np.random.seed(0)
    torch.manual_seed(0)
    manifest = json.loads(args.split_manifest.read_text())
    validate_manifest(manifest)
    if set(range(1000000, 1000005)) & {entry['seed'] for name in ('train', 'validation', 'test') for entry in manifest[name]}:
        raise ValueError('Fresh seeds overlap demonstrations')
    if args.without_stage and args.mode != 'privileged':
        raise ValueError('--without-stage is only supported for privileged mode')
    output = create_run_dir(Path('v4/runs'), f'v4-diagnostic-{args.mode}' + ('-no-stage' if args.without_stage else ''))
    write_manifest({'mode': args.mode, 'epochs': args.epochs, 'device': args.device,
                    'use_stage': not args.without_stage, 'seed': 0, 'source_split': str(args.split_manifest.resolve()),
                    'diagnostic_only': True, 'learning_rate': args.learning_rate if args.mode == 'one-demo' else 1e-3,
                    'warm_start': str(args.warm_start.resolve()) if args.warm_start else None}, output / 'config.json')
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(output / 'train.log')])
    logging.getLogger('robosuite_logs').setLevel(logging.WARNING)
    print(f'Run directory: {output}', flush=True)
    if args.mode == 'privileged':
        write_manifest(manifest, output / 'split.json')
        run_privileged(args, manifest, output)
    elif args.mode == 'one-demo':
        run_one_demo(args, manifest, output)
    else:
        print(json.dumps(audit_replay(selected_training_entries(manifest, 1)[0], output), indent=2), flush=True)


if __name__ == '__main__':
    main()
