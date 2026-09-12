"""Collect a fixed, categorized BC dataset with reproducible disturbance schedules."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from v4.data import inspect_trajectory, validate_manifest, write_manifest
from v4.perturbations import CATEGORIES, PerturbationConfig
from v4.record_trajectory import generate_one_success
from v4.teacher import TeacherConfig


DEFAULT_ROOT = Path('v4/trajectories/robustness300')
DEFAULT_SPLIT = Path('v4/splits/robustness300.json')


def make_plan(source, *, clean_count=100, other_count=50, seed=20260912, attempts_per_slot=8):
    if any(count < 10 or count % 10 for count in (clean_count, other_count)):
        raise ValueError('Category counts must be positive multiples of ten for exact 10% validation')
    if attempts_per_slot < 1:
        raise ValueError('attempts_per_slot must be positive')
    # Exclude every old partition, plus the repeatedly inspected fresh diagnostics.
    excluded = {entry['seed'] for part in ('train', 'validation', 'test') for entry in source[part]}
    excluded.update(range(1000000, 1000005))
    used = set(excluded)
    rng = np.random.default_rng(seed)
    slots = []
    for category in CATEGORIES:
        count = clean_count if category == 'clean' else other_count
        for index in range(count):
            partition = 'validation' if index % 10 == 0 else 'train'
            candidates = []
            for _ in range(attempts_per_slot):
                episode_seed = int(rng.integers(0, 2**31 - 1))
                while episode_seed in used:
                    episode_seed = int(rng.integers(0, 2**31 - 1))
                used.add(episode_seed)
                candidates.append({'seed': episode_seed,
                                   'perturbation_seed': int(rng.integers(0, 2**31 - 1))})
            slots.append({'id': f'{category}-{index:03d}', 'category': category,
                          'partition': partition, 'save_video': index in (0, 10),
                          'candidates': candidates})
    return {'schema_version': 1, 'collection_seed': seed,
            'counts': {'clean': clean_count, 'each_other': other_count},
            'validation_fraction': 0.1, 'attempts_per_slot': attempts_per_slot,
            'excluded_seeds': sorted(excluded), 'test': source['test'],
            'perturbations': {c: PerturbationConfig(category=c).to_dict() for c in CATEGORIES},
            'teacher_config': TeacherConfig().__dict__, 'slots': slots}


def collect_slot(root, slot, config, scratch_dir):
    root = Path(root)
    state_path = root / 'jobs' / f"{slot['id']}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {'slot': slot, 'attempts': [], 'result': None}
    if state['result'] is not None:
        inspect_trajectory(Path(state['result']['trajectory']))
        return state
    slot_root = root / 'episodes' / slot['id']
    for candidate in slot['candidates'][len(state['attempts']):]:
        start = time.monotonic()
        # Recover an attempt published just before an interrupted ledger write.
        metrics = None
        for path in sorted(slot_root.glob('*/metrics.json')):
            saved = json.loads(path.read_text())
            if saved['requested_seed'] == candidate['seed']:
                metrics = saved
                break
        if metrics is None:
            try:
                metrics = generate_one_success(
                    slot_root, name='demo', seed=candidate['seed'], max_attempts=1,
                    max_steps=900, teacher_config=TeacherConfig(),
                    perturbation_config=PerturbationConfig(**config),
                    perturbation_seed=candidate['perturbation_seed'],
                    save_video=slot['save_video'], scratch_dir=Path(scratch_dir), verbose=False)
            except RuntimeError as error:
                if not str(error).startswith('Teacher did not produce a successful trajectory'):
                    raise
                for path in sorted(slot_root.glob('*/metrics.json')):
                    saved = json.loads(path.read_text())
                    if saved['requested_seed'] == candidate['seed']:
                        metrics = saved
                        break
                if metrics is None:
                    raise
        state['attempts'].append({**candidate, **metrics['attempts'][0],
                                  'metrics_path': str(Path(metrics['run_dir']) / 'metrics.json'),
                                  'wall_seconds': time.monotonic() - start})
        state['result'] = metrics['result']
        write_manifest(state, state_path)
        if state['result'] is not None:
            break
    return state


def summarize(root, plan):
    counts = {c: {'successes': 0, 'attempts': 0, 'failed_attempts': 0,
                  'train_trajectories': 0, 'validation_trajectories': 0,
                  'transitions': 0, 'changed_transitions': 0} for c in CATEGORIES}
    states = []
    failures = Counter()
    for slot in plan['slots']:
        path = root / 'jobs' / f"{slot['id']}.json"
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        states.append(state)
        item = counts[slot['category']]
        item['attempts'] += len(state['attempts'])
        for attempt in state['attempts']:
            if not attempt['success']:
                item['failed_attempts'] += 1
                failures[attempt['failure_reason']] += 1
        if state['result'] is not None:
            item['successes'] += 1
            item[f"{slot['partition']}_trajectories"] += 1
            item['transitions'] += state['result']['steps']
            item['changed_transitions'] += state['result']['perturbed_steps']
    summary = {'categories': counts, 'failure_reasons': dict(failures),
               'successful_trajectories': sum(c['successes'] for c in counts.values()),
               'target_trajectories': len(plan['slots']),
               'transitions': sum(c['transitions'] for c in counts.values())}
    write_manifest(summary, root / 'summary.json')
    return summary, states


def publish_split(root, plan, destination):
    summary, states = summarize(root, plan)
    if summary['successful_trajectories'] != len(plan['slots']):
        raise RuntimeError(f"Dataset incomplete: {summary['successful_trajectories']}/{len(plan['slots'])}")
    groups = {'train': [], 'validation': [],
              'test': [{**row, 'category': 'clean', 'provenance': 'legacy_diverse100_test'} for row in plan['test']]}
    videos = []
    for state in states:
        slot, result = state['slot'], state['result']
        entry = {'path': result['trajectory'], 'seed': result['seed'], 'steps': result['steps'],
                 'category': slot['category'], 'perturbation_seed': result['perturbation_seed'],
                 'slot_id': slot['id']}
        groups[slot['partition']].append(entry)
        if result['video'] is not None:
            videos.append({**entry, 'partition': slot['partition'], 'video': result['video']})
    manifest = {'schema_version': 2, 'split_seed': plan['collection_seed'],
                'validation_fraction': 0.1, 'collection_plan': str((root / 'plan.json').resolve()),
                'test_provenance': 'Existing 10 clean diverse100 test demonstrations; no new test collection',
                'action_target': 'actions (unperturbed teacher command at the actual visited state)',
                'perturbations': plan['perturbations'], **groups,
                'summary': {**{f'{p}_trajectories': len(rows) for p, rows in groups.items()},
                            **{f'{p}_samples': sum(r['steps'] for r in rows) for p, rows in groups.items()}}}
    validate_manifest(manifest)
    write_manifest(manifest, destination)
    write_manifest({'videos': videos}, root / 'videos.json')
    return manifest


def collect(root, *, source_split, split_output, workers=8, pilot=False,
            clean_count=100, other_count=50, seed=20260912, attempts_per_slot=8,
            scratch_dir=Path('/tmp/robotic-grasp-robustness')):
    if workers < 1:
        raise ValueError('workers must be positive')
    root = Path(root).resolve()
    if (root / 'retirement.json').exists():
        raise RuntimeError('This mild dataset was retired after storage cleanup; resume v4.collect_recovery instead')
    source = json.loads(Path(source_split).read_text())
    desired = make_plan(source, clean_count=clean_count, other_count=other_count,
                        seed=seed, attempts_per_slot=attempts_per_slot)
    desired['source_split'] = str(Path(source_split).resolve())
    plan_path = root / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != desired:
            raise ValueError('Existing frozen collection plan differs; use a new output directory')
    else:
        write_manifest(desired, plan_path)
    plan = desired
    selected = [s for s in plan['slots'] if not pilot or s['id'].endswith('-001')]
    pending = []
    for slot in selected:
        path = root / 'jobs' / f"{slot['id']}.json"
        if not path.exists() or json.loads(path.read_text())['result'] is None:
            pending.append(slot)
    print(f'Collecting {len(pending)} remaining slots ({len(selected)} selected); plan={plan_path}', flush=True)
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(collect_slot, root, slot, plan['perturbations'][slot['category']],
                               scratch_dir): slot for slot in pending}
        for future in as_completed(futures):
            state = future.result()
            summary, _ = summarize(root, plan)
            result = state['result']
            print(json.dumps({'slot': state['slot']['id'], 'success': result is not None,
                              'attempts': len(state['attempts']),
                              'steps': result['steps'] if result else None,
                              'completed': summary['successful_trajectories'],
                              'target': len(plan['slots'])}), flush=True)
    summary, states = summarize(root, plan)
    if pilot:
        if any(not any(st['slot']['id'] == slot['id'] and st['result'] for st in states) for slot in selected):
            raise RuntimeError('Pilot failed; inspect job ledgers before full collection')
    else:
        publish_split(root, plan, Path(split_output))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--source-split', type=Path, default=Path('v4/splits/diverse100.json'))
    parser.add_argument('--split-output', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--clean-count', type=int, default=100)
    parser.add_argument('--other-count', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--attempts-per-slot', type=int, default=8)
    parser.add_argument('--scratch-dir', type=Path, default=Path('/tmp/robotic-grasp-robustness'))
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    print(json.dumps(collect(args.output_dir, source_split=args.source_split, split_output=args.split_output,
                             workers=args.workers, pilot=args.pilot, clean_count=args.clean_count,
                             other_count=args.other_count, seed=args.seed,
                             attempts_per_slot=args.attempts_per_slot, scratch_dir=args.scratch_dir), indent=2))


if __name__ == '__main__':
    main()
