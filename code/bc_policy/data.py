"""Read collection manifests and aligned targets; keep learner prefixes unlabelled."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from bc_data.state_policy import PROPRIO_KEYS, STAGES, state_features

RGB_KEYS = ('frontview_image', 'robot0_eye_in_hand_image')


def read_manifest(path, corrections=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('complete') is False:
        raise ValueError('Collection is incomplete; use the completed split.json')
    groups = {}
    for part in ('train', 'validation', 'test'):
        groups[part] = [dict(entry, path=str((path.parent / entry['path']).resolve()))
                        for entry in manifest.get(part, [])]
    if corrections:
        source = Path(corrections).resolve()
        collected = json.loads(source.read_text())
        if collected.get('complete') is not True or not collected.get('selected'):
            raise ValueError('Use a complete correction collection.json with selected episodes')
        for entry in collected['selected']:
            if not entry.get('success'):
                raise ValueError('Correction selection contains an unsuccessful episode')
            groups['train'].append(dict(entry, path=str((source.parent / entry['path']).resolve())))
    if not groups['train'] or not groups['validation']:
        raise ValueError('Nonempty training and validation partitions are required')
    paths, seeds = set(), {}
    for part, entries in groups.items():
        for entry in entries:
            file = Path(entry['path'])
            if not file.is_file() or file in paths:
                raise ValueError(f'Missing or duplicated trajectory: {file}')
            paths.add(file)
            seed = int(entry['seed'])
            if seed in seeds and seeds[seed] != part:
                raise ValueError(f'Seed {seed} crosses {seeds[seed]}/{part} partitions')
            seeds[seed] = part
    return groups


def read_targets(entries, *, state=False):
    rows = []
    for entry in entries:
        with h5py.File(entry['path'], 'r') as demo:
            actions = np.asarray(demo['actions'][:], dtype=np.float32)
            n = len(actions)
            if (n < 1 or actions.shape != (n, 7) or n != entry['steps']
                    or int(demo.attrs['seed']) != entry['seed'] or not demo.attrs.get('success')):
                raise ValueError(f'Invalid trajectory metadata: {entry["path"]}')
            if not np.isfinite(actions).all() or np.abs(actions).max() > 1.000001 or actions[:, 3:6].any():
                raise ValueError('Expected normalized finite actions with zero rotations')
            correction = bool(demo.attrs.get('requires_supervision_mask', False))
            if correction and 'supervision/mask' not in demo:
                raise ValueError('Correction trajectory is missing its supervision mask')
            mask = demo['supervision/mask'][:].astype(bool) if 'supervision/mask' in demo else np.ones(n, bool)
            if mask.shape != (n,) or not mask.any():
                raise ValueError('Expected at least one supervised action per trajectory')
            if correction:
                if not np.array_equal(mask, np.arange(n) >= int(demo.attrs['handoff_step'])):
                    raise ValueError('Correction mask does not match teacher handoff')
            elif not mask.all():
                raise ValueError('Partial supervision requires correction metadata')
            labels = demo['stage'].asstr()[:]
            if len(labels) != n:
                raise ValueError('Stage/action lengths differ')
            stages = np.asarray([STAGES.index(s) if valid else -1 for s, valid in zip(labels, mask)])
            proprio = np.concatenate([demo[f'observations/{key}'][:] for key in PROPRIO_KEYS], axis=1).astype(np.float32)
            if proprio.shape != (n, 16) or not np.isfinite(proprio).all():
                raise ValueError('Invalid robot observations')
            row = {'actions': actions, 'stages': stages, 'mask': mask, 'proprio': proprio}
            if state:
                row['state'] = state_features(demo['oracle/object_positions'][:],
                                              demo['oracle/eef_position'][:], proprio, [None] * n,
                                              use_stage=False)
                if not np.isfinite(row['state']).all():
                    raise ValueError('Nonfinite state features')
            rows.append(row)
    return {key: torch.from_numpy(np.concatenate([row[key] for row in rows])) for key in rows[0]}


def sequence_batches(entries, order, chunk_length=32, batch_trajectories=8):
    """Yield padded indices and masks; a new group starts with fresh LSTM memory."""
    starts = np.cumsum([0] + [entry['steps'] for entry in entries[:-1]])
    for offset in range(0, len(order), batch_trajectories):
        chosen = order[offset:offset + batch_trajectories]
        lengths = torch.tensor([entries[i]['steps'] for i in chosen])[:, None]
        begin = torch.tensor(starts[chosen])[:, None]
        for time in range(0, int(lengths.max()), chunk_length):
            times = torch.arange(time, min(time + chunk_length, int(lengths.max())))[None, :]
            valid = times < lengths
            indices = begin + torch.minimum(times, lengths - 1)
            yield indices, valid, time == 0
