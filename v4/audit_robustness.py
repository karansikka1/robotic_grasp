"""Audit categorized BC artifacts and optionally replay one episode per category."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import av
import h5py
import numpy as np

from v4.data import inspect_trajectory, validate_manifest, write_manifest
from v4.perturbations import ActionPerturber, CATEGORIES, PerturbationConfig
from v4.record_trajectory import OBSERVATION_KEYS


def audit_trajectory(entry, expected_config):
    item = inspect_trajectory(Path(entry['path']))
    assert (item.seed, item.steps) == (entry['seed'], entry['steps'])
    with h5py.File(item.path, 'r') as f:
        assert f.attrs['format_version'] == 2
        assert f.attrs['category'] == entry['category']
        assert f.attrs['steps'] == item.steps
        assert f.attrs['perturbation_seed'] == entry['perturbation_seed']
        config = json.loads(f.attrs['perturbation_config'])
        assert config == expected_config
        intended, executed = f['actions'][:], f['executed_actions'][:]
        assert executed.shape == intended.shape == (item.steps, 7)
        assert np.isfinite(intended).all() and np.isfinite(executed).all()
        assert np.abs(intended).max() <= 1 and np.abs(executed).max() <= 1
        np.testing.assert_array_equal(intended[:, 3:], executed[:, 3:])
        metadata = {key: value[:] for key, value in f['perturbation'].items()}
        assert all(len(value) == item.steps for value in metadata.values())
        perturber = ActionPerturber(PerturbationConfig(**config), entry['perturbation_seed'])
        for step, action in enumerate(intended):
            expected_action, expected_meta = perturber.apply(action)
            np.testing.assert_array_equal(executed[step], expected_action)
            for key, value in expected_meta.items():
                np.testing.assert_array_equal(metadata[key][step], value)
        assert (perturber.changed_steps == 0) == (entry['category'] == 'clean')
        assert f['terminal_observation'].attrs['task_complete']
        assert f['next_task_complete'][-10:].all() and item.steps >= 10
        assert f['stage'].shape == f['phase'].shape == (item.steps,)
        for name, dataset in f['oracle'].items():
            assert len(dataset) == item.steps
            assert np.isfinite(dataset[:]).all(), name
        for key in OBSERVATION_KEYS:
            dataset = f[f'observations/{key}']
            assert len(dataset) == item.steps
            # Read every proprioceptive value and three full camera/depth frames.
            frames = dataset[[0, item.steps // 2, item.steps - 1]] if dataset.ndim > 2 else dataset[:]
            assert np.isfinite(frames).all(), key
            assert np.isfinite(f[f'terminal_observation/{key}'][:]).all(), key
        return {'path': item.path, 'seed': item.seed, 'steps': item.steps,
                'category': entry['category'], 'changed_steps': perturber.changed_steps,
                'scheduled_steps': int(metadata['active'].sum()),
                'max_translation_delta': float(np.abs(executed[:, :3] - intended[:, :3]).max()),
                'bytes': Path(item.path).stat().st_size, 'passed': True}


def replay_trajectory(entry):
    from motion_planning.simulator import Simulator
    from v4.teacher import PrivilegedStackTeacher, TeacherConfig, read_oracle_state
    np.random.seed(entry['seed'])
    simulator = Simulator(has_renderer=False)
    try:
        np.random.seed(entry['seed'])
        simulator.reset()
        observation = simulator.step(np.zeros(7, dtype=np.float32))
        with h5py.File(entry['path'], 'r') as f:
            teacher = PrivilegedStackTeacher(TeacherConfig(**json.loads(f.attrs['teacher_config'])))
            actions, executed = f['actions'][:], f['executed_actions'][:]
            stages, phases = f['stage'].asstr()[:], f['phase'][:]
            positions, eef = f['oracle/object_positions'][:], f['oracle/eef_position'][:]
            for key in OBSERVATION_KEYS:
                np.testing.assert_array_equal(observation[key], f[f'observations/{key}'][0])
            for step in range(len(actions)):
                oracle = read_oracle_state(simulator)
                np.testing.assert_array_equal(oracle.positions, positions[step])
                np.testing.assert_array_equal(oracle.eef_position, eef[step])
                decision = teacher.decide(simulator)
                assert not decision.done
                assert decision.stage == stages[step] and decision.phase == phases[step]
                np.testing.assert_array_equal(decision.action, actions[step])
                observation = simulator.step(executed[step])
            assert teacher.decide(simulator).done
            assert observation['task_complete']
            for key in OBSERVATION_KEYS:
                np.testing.assert_array_equal(observation[key], f[f'terminal_observation/{key}'][:])
        return {'path': entry['path'], 'category': entry['category'], 'seed': entry['seed'],
                'steps': len(actions), 'initial_and_terminal_observations_exact': True,
                'all_pre_action_positions_exact': True, 'all_teacher_labels_and_stages_exact': True,
                'stable_official_success': True}
    finally:
        simulator.close()


def audit(root, split_path, *, replay=False, workers=5):
    root = Path(root)
    manifest = json.loads(Path(split_path).read_text())
    validate_manifest(manifest)
    plan = json.loads((root / 'plan.json').read_text())
    entries = manifest['train'] + manifest['validation']
    assert len(entries) == len(plan['slots'])
    assert not {e['seed'] for e in entries} & set(plan['excluded_seeds'])
    expected = Counter((s['partition'], s['category']) for s in plan['slots'])
    actual = Counter((part, e['category']) for part in ('train', 'validation') for e in manifest[part])
    assert actual == expected
    records = [audit_trajectory(entry, plan['perturbations'][entry['category']]) for entry in entries]
    videos = json.loads((root / 'videos.json').read_text())['videos']
    assert len(videos) == 10
    assert Counter(v['category'] for v in videos) == Counter({c: 2 for c in CATEGORIES})
    video_checks = []
    for video in videos:
        with av.open(video['video']) as container:
            decoded = sum(1 for _ in container.decode(video=0))
        assert decoded == video['steps'] + 1, video
        video_checks.append({**video, 'decoded_frames': decoded, 'passed': True})
    report = {'passed': True, 'trajectories': len(records), 'transitions': sum(r['steps'] for r in records),
              'bytes': sum(r['bytes'] for r in records),
              'checks': ['split counts and exclusions', 'all intended/executed actions and disturbance metadata',
                         'all oracle and proprioceptive values', 'initial/middle/final camera frames',
                         'terminal observation and ten consecutive official success flags', 'ten fully decoded videos'],
              'records': records, 'videos': video_checks}
    write_manifest(report, root / 'validation.json')
    if replay:
        selected = [next(e for e in entries if e['category'] == category) for category in CATEGORIES]
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            replays = list(pool.map(replay_trajectory, selected))
        write_manifest({'passed': True, 'episodes': replays}, root / 'replay_validation.json')
    return {key: value for key, value in report.items() if key not in ('records', 'videos')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('v4/trajectories/robustness300'))
    parser.add_argument('--split', type=Path, default=Path('v4/splits/robustness300.json'))
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.split, replay=args.replay, workers=args.workers), indent=2))


if __name__ == '__main__':
    main()
