"""Preserve finalized recovery300 scratch recordings after a publication failure.

Run with the collector stopped. Copy and hash-check each HDF5 before publishing
its job record and removing its scratch copy. Simulator data is never rewritten;
missing textual recovery reasons and wall-clock timings remain explicitly unknown.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
import numpy as np

from harness import _VideoWriter
from v4.collect_recovery import ROOT, annotate_video, event_measurements, summarize
from v4.data import write_manifest
from v4.recovery_teacher import DisturbanceConfig


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def reconstruct(path, slot):
    with h5py.File(path, 'r') as f:
        config = DisturbanceConfig(**slot['config'])
        seed, noise_seed, steps = (int(f.attrs[k]) for k in ('seed', 'perturbation_seed', 'steps'))
        assert f.attrs['success'] and f['terminal_observation'].attrs['task_complete']
        assert np.all(f['next_task_complete'][-10:]) and steps == len(f['actions'])
        assert f.attrs['collection_slot'] == slot['id'] and f.attrs['category'] == slot['category']
        assert json.loads(f.attrs['perturbation_config']) == slot['config']
        # Every recoverable scratch file in this incident was the first candidate.
        assert slot['candidates'][0] == {'seed': seed, 'noise_seed': noise_seed}
        active = np.flatnonzero(f['perturbation/active'][:])
        assert len(active) == config.duration
        assert np.array_equal(active, np.arange(active[0], active[0] + config.duration))
        start, end = int(active[0]), int(active[-1]) + 1
        assert end + 30 < steps  # All measurements use recorded pre-action grasp flags.
        controller = SimpleNamespace(config=config, event_start=start, event_end=end,
                                     event_stage=f['stage'].asstr()[start])
        event = event_measurements(controller, f['oracle/object_positions'][:],
                                   f['oracle/eef_position'][:], f['oracle/grasped'][:])
        changed = int(f['perturbation/changed'][:].sum())
        assert changed > 0
        if config.category == 'drop_block':
            assert event['grasp_lost'] and event['downward_travel_cm'] >= 3.5
        if config.category == 'gripper_interrupt' and config.trigger == 'reopen':
            assert event['grasp_lost']
        mode = f['supervisor/mode_id'][:]
        assert np.array_equal(mode != 0, f['perturbation/recovery_active'][:])
        edges = np.diff(np.r_[False, mode != 0, False].astype(np.int8))
        recoveries = [{'phase': int(f['phase'][a]), 'start': int(a), 'end': int(b), 'reason': None}
                      for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
        # Read every stored dataset, including image/depth chunks, to catch damaged files.
        def read_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.ndim and len(obj) == steps:
                    for i in range(0, steps, 16):
                        obj[i:i + 16]
                else:
                    obj[()]
        f.visititems(read_dataset)
    run = path.parent
    return {'slot_id': slot['id'], 'category': slot['category'], 'seed': seed,
            'noise_seed': noise_seed, 'config': slot['config'], 'success': True, 'failure': None,
            'steps': steps, 'seconds': steps / 20, 'trajectory': str(path), 'video': None,
            'event': event, 'recoveries': recoveries, 'changed_steps': changed, 'wall_seconds': None,
            'metrics_path': str(run / 'metrics.json'),
            'publication_recovery': {'method': 'Finalized HDF5 recovered after disk quota failure',
                                     'unknown_fields': ['wall_seconds', 'recoveries[].reason'],
                                     'video_source': 'Stored RGB frames' if slot['save_video'] else None}}


def recover(root, inventory_path):
    root = Path(root).resolve()
    inventory = json.loads(Path(inventory_path).read_text())
    plan = json.loads((root / 'plan.json').read_text())
    slots = {s['id']: s for s in plan['slots']}
    write_manifest(inventory, root / 'scratch_inventory.json')
    report_path = root / 'storage_recovery.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {'recovered': []}
    for item in sorted(inventory['valid'], key=lambda i: i['slot']):
        source = Path(item['path'])
        assert source.parent == Path('/tmp/robotic-grasp-recovery300')
        slot = slots[item['slot']]
        job_path = root / 'jobs' / (slot['id'] + '.json')
        run = root / 'episodes' / slot['id'] / f"recovered-{item['seed']}"
        target = run / 'trajectory.h5'
        if job_path.exists():
            state = json.loads(job_path.read_text())
            assert state['result']['seed'] == item['seed'] and Path(state['result']['trajectory']) == target
            expected = next(r['sha256'] for r in report['recovered'] if r['slot'] == slot['id'])
            assert digest(target) == expected
            if source.exists():
                assert digest(source) == expected
                source.unlink()
            continue
        run.mkdir(parents=True, exist_ok=True)
        expected = digest(source)
        pending = target.with_suffix('.h5.pending')
        shutil.copyfile(source, pending)
        assert pending.stat().st_size == item['bytes'] and digest(pending) == expected
        pending.replace(target)
        result = reconstruct(target, slot)
        if slot['save_video']:
            raw, annotated = run / 'trajectory.mp4', run / 'review.mp4'
            keys = ('frontview_image', 'robot0_eye_in_hand_image')
            with h5py.File(target, 'r') as f, _VideoWriter(raw, 20) as writer:
                for i in range(result['steps']):
                    writer.add_observation({k: f['observations/' + k][i] for k in keys})
                writer.add_observation({k: f['terminal_observation/' + k][:] for k in keys})
            annotate_video(raw, annotated, result)
            result.update(raw_video=str(raw), video=str(annotated))
        write_manifest(result, run / 'metrics.json')
        report['recovered'].append({**item, 'destination': str(target), 'sha256': expected,
                                    'all_hdf5_datasets_read': True})
        write_manifest(report, report_path)
        write_manifest({'slot': slot, 'attempts': [result], 'result': result}, job_path)
        source.unlink()
        print(json.dumps({'slot': slot['id'], 'recovered': len(report['recovered'])}), flush=True)
    states = {p.stem: json.loads(p.read_text()) for p in (root / 'jobs').glob('*.json')}
    write_manifest(summarize(states, plan), root / 'summary.json')
    print(json.dumps({'completed': sum(bool(s['result']) for s in states.values()),
                      'remaining': len(plan['slots']) - sum(bool(s['result']) for s in states.values()),
                      'recovered_bytes': sum(r['bytes'] for r in report['recovered'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    recover(args.root, args.inventory)
