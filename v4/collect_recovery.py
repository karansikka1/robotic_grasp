"""Collect the approved stronger 300-trajectory dataset with object/phase variation."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import uuid

import av
import cv2
import h5py
import numpy as np

from motion_planning.simulator import Simulator
from v4.data import inspect_trajectory, validate_manifest, write_manifest
from v4.record_trajectory import AttemptRecorder
from v4.recovery_teacher import CATEGORIES, DisturbanceConfig, RecoveryTeacher, sample_config
from v4.teacher import TeacherFailure, read_oracle_state


ROOT = Path('v4/trajectories/robustness300/recovery300')
SPLIT = Path('v4/splits/recovery300.json')
PILOT = {'clean': {1}, 'gaussian_burst': {1, 3, 5, 7}, 'drop_block': {0, 1},
         'sideways_error': {3}, 'gripper_interrupt': {0, 1, 2, 3}}
SOURCE_FILES = ('v4/recovery_teacher.py', 'v4/collect_recovery.py', 'v4/record_trajectory.py',
                'v4/teacher.py', 'motion_planning/environment.py', 'motion_planning/simulator.py')


def make_plan(seed=20260913):
    source = json.loads(Path('v4/splits/diverse100.json').read_text())
    old_plan = json.loads(Path('v4/trajectories/robustness300/plan.json').read_text())
    excluded = set(old_plan['excluded_seeds'])
    excluded.update(c['seed'] for s in old_plan['slots'] for c in s['candidates'])
    excluded.update(e['seed'] for p in ('train', 'validation', 'test') for e in source[p])
    rng = np.random.default_rng(seed)
    used = set(excluded)
    slots = []
    for category in CATEGORIES:
        for index in range(100 if category == 'clean' else 50):
            validation = index % 10 == 0 if category == 'clean' else index in (0, 11, 22, 33, 44)
            config_seed = int(rng.integers(0, 2**31 - 1))
            config = sample_config(category, index, config_seed)
            candidates = []
            for _ in range(12):
                layout_seed = int(rng.integers(0, 2**31 - 1))
                while layout_seed in used:
                    layout_seed = int(rng.integers(0, 2**31 - 1))
                used.add(layout_seed)
                candidates.append({'seed': layout_seed, 'noise_seed': int(rng.integers(0, 2**31 - 1))})
            slots.append({'id': f'{category}-{index:03d}', 'category': category, 'index': index,
                          'partition': 'validation' if validation else 'train',
                          'config': config.to_dict(), 'config_seed': config_seed,
                          'save_video': index in ((0, 10) if category == 'clean' else (0, 11)),
                          'pilot': index in PILOT[category], 'candidates': candidates})
    return {'schema_version': 1, 'collection_seed': seed, 'slots': slots,
            'excluded_seeds': sorted(excluded), 'test': source['test'], 'validation_fraction': 0.1,
            'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCE_FILES}}


def event_measurements(controller, positions, eef, grasped):
    start, end = controller.event_start, controller.event_end
    if start is None:
        return None
    index = controller.config.target_phase + 1
    end = min(end, len(positions) - 1)
    last = min(len(positions), end + 30)
    loss_indices = np.flatnonzero(~grasped[start:end+1, index]) + start if grasped[start, index] else []
    regrasp = (np.flatnonzero(grasped[loss_indices[0]+1:, index]) + loss_indices[0]+1
               if len(loss_indices) else [])
    return {'target_object': 'green' if index == 1 else 'blue', 'start': start, 'end': controller.event_end,
            'actual_stage': controller.event_stage,
            'eef_displacement_cm': float(np.linalg.norm(eef[start:end+1] - eef[start], axis=1).max() * 100),
            'object_displacement_cm': float(np.linalg.norm(positions[start:end+1, index] - positions[start, index], axis=1).max() * 100),
            'downward_travel_cm': float((positions[start, index, 2] - positions[start:last, index, 2].min()) * 100),
            'grasped_at_start': bool(grasped[start, index]), 'grasp_lost': bool(len(loss_indices)),
            'regrasp_step': int(regrasp[0]) if len(regrasp) else None}


def annotate_video(raw, output_path, result):
    event = result['event']
    with av.open(str(raw)) as source, av.open(str(output_path), 'w', options={'movflags': '+faststart'}) as output:
        stream = output.add_stream('libx264', rate=20)
        stream.width, stream.height, stream.pix_fmt = 512, 320, 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast'}
        for i, frame in enumerate(source.decode(video=0)):
            pixels = np.zeros((320, 512, 3), dtype=np.uint8)
            pixels[64:] = frame.to_ndarray(format='rgb24')
            active = event is not None and event['start'] <= i < event['end']
            recovering = any(r['start'] <= i and (r['end'] is None or i < r['end']) for r in result['recoveries'])
            status = 'DISTURBANCE ACTIVE' if active else 'REGRASP RECOVERY' if recovering else 'NORMAL TEACHER'
            color = (255, 180, 90) if active else (125, 226, 175) if recovering else (160, 198, 225)
            title = result['category'].replace('_', ' ').upper()
            if event:
                title += ' / ' + event['target_object'].upper()
            cv2.putText(pixels, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 244, 250), 1, cv2.LINE_AA)
            cv2.putText(pixels, f'{status} | {i / 20:.2f} s', (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            encoded = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            for packet in stream.encode(encoded):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def generate_attempt(root, slot, candidate):
    started = time.monotonic()
    run = Path(root).resolve() / 'episodes' / slot['id'] / f'demo-{uuid.uuid4()}'
    run.mkdir(parents=True)
    config = DisturbanceConfig(**slot['config'])
    np.random.seed(candidate['seed'])
    simulator = Simulator(has_renderer=False)
    success, failure, steps = False, None, 0
    trajectory_path = video_path = None
    positions, eef, grasps = [], [], []
    controller = None
    event = None
    try:
        np.random.seed(candidate['seed'])
        simulator.reset()
        observation = simulator.step(np.zeros(7, dtype=np.float32))
        controller = RecoveryTeacher(config, candidate['noise_seed'], read_oracle_state(simulator))
        with AttemptRecorder(run, seed=candidate['seed'], teacher=controller.teacher,
                             perturbation=controller, save_video=slot['save_video'],
                             scratch_dir=Path('/tmp/robotic-grasp-recovery300')) as recorder:
            recorder.h5_file.attrs['supervisor_mode_vocabulary'] = json.dumps(['normal', 'recovery_retreat'])
            recorder.h5_file.attrs['collection_slot'] = slot['id']
            recorder.h5_file.create_group('supervisor')
            recorder.add_initial_observation(observation)
            try:
                while steps < 900:
                    state = read_oracle_state(simulator)
                    decision, executed, info = controller.command(simulator, state, steps)
                    if decision.done:
                        success = bool(observation['task_complete'])
                        break
                    positions.append(state.positions.copy())
                    eef.append(state.eef_position.copy())
                    grasps.append(state.grasped.copy())
                    next_observation = simulator.step(executed)
                    recorder.add_transition(observation, decision.action, stage=decision.stage, phase=decision.phase,
                                            oracle=state, next_observation=next_observation,
                                            executed_action=executed, perturbation_info=info)
                    recorder._append_array('supervisor/mode_id', np.int8(controller.last_mode != 'normal'))
                    observation = next_observation
                    steps += 1
            except TeacherFailure as error:
                failure = str(error)
            terminal = read_oracle_state(simulator)
            positions.append(terminal.positions.copy())
            eef.append(terminal.eef_position.copy())
            grasps.append(terminal.grasped.copy())
            event = event_measurements(controller, np.asarray(positions), np.asarray(eef), np.asarray(grasps))
            if success and config.category != 'clean':
                if event is None or controller.changed_steps == 0:
                    success, failure = False, 'Planned intervention did not occur'
                elif config.category == 'drop_block' and (not event['grasp_lost'] or event['downward_travel_cm'] < 3.5):
                    success, failure = False, 'Release did not produce a physical block drop'
                elif config.category == 'gripper_interrupt' and config.trigger == 'reopen' and not event['grasp_lost']:
                    success, failure = False, 'Reopening did not interrupt the grasp'
            if success:
                recorder.add_terminal_observation(observation, simulator)
                trajectory_path, video_path = recorder.finish_successfully()
            else:
                failure = failure or 'episode step limit reached'
                recorder.discard()
    finally:
        simulator.close()
    result = {'slot_id': slot['id'], 'category': config.category, 'seed': candidate['seed'],
              'noise_seed': candidate['noise_seed'], 'config': config.to_dict(),
              'success': success, 'failure': failure, 'steps': steps, 'seconds': steps / 20,
              'trajectory': str(trajectory_path) if trajectory_path else None,
              'video': str(video_path) if video_path else None, 'event': event,
              'recoveries': controller.recoveries, 'changed_steps': controller.changed_steps,
              'wall_seconds': time.monotonic() - started, 'metrics_path': str(run / 'metrics.json')}
    if video_path is not None:
        annotated = run / 'review.mp4'
        annotate_video(video_path, annotated, result)
        result['raw_video'] = str(video_path)
        result['video'] = str(annotated)
    write_manifest(result, run / 'metrics.json')
    return result


def collect_slot(root, slot):
    root = Path(root)
    state_path = root / 'jobs' / (slot['id'] + '.json')
    state = json.loads(state_path.read_text()) if state_path.exists() else {'slot': slot, 'attempts': [], 'result': None}
    if state['result'] is not None:
        inspect_trajectory(Path(state['result']['trajectory']))
        return state
    for candidate in slot['candidates'][len(state['attempts']):]:
        result = None
        for path in sorted((root / 'episodes' / slot['id']).glob('*/metrics.json')):
            saved = json.loads(path.read_text())
            if saved['seed'] == candidate['seed']:
                result = saved
                break
        if result is None:
            result = generate_attempt(root, slot, candidate)
        state['attempts'].append(result)
        if result['success']:
            state['result'] = result
        write_manifest(state, state_path)
        if state['result'] is not None:
            break
    return state


def summarize(states, plan):
    counts = {c: {'successes': 0, 'attempts': 0, 'train': 0, 'validation': 0, 'transitions': 0,
                  'green': 0, 'blue': 0, 'with_recovery': 0} for c in CATEGORIES}
    failures = Counter()
    for state in states.values():
        slot = state['slot']
        item = counts[slot['category']]
        item['attempts'] += len(state['attempts'])
        failures.update(r['failure'] for r in state['attempts'] if not r['success'])
        result = state['result']
        if result:
            item['successes'] += 1
            item[slot['partition']] += 1
            item['transitions'] += result['steps']
            item['with_recovery'] += int(bool(result['recoveries']))
            if result['event']:
                item[result['event']['target_object']] += 1
    return {'categories': counts, 'failure_reasons': dict(failures),
            'successful_trajectories': sum(c['successes'] for c in counts.values()),
            'target_trajectories': len(plan['slots']),
            'transitions': sum(c['transitions'] for c in counts.values())}


def publish(root, states, plan):
    groups = {'train': [], 'validation': [], 'test': [{**e, 'category': 'clean'} for e in plan['test']]}
    videos = []
    for slot in plan['slots']:
        result = states[slot['id']]['result']
        if result is None:
            raise RuntimeError('Incomplete slot: ' + slot['id'])
        entry = {'path': result['trajectory'], 'seed': result['seed'], 'steps': result['steps'],
                 'category': result['category'], 'noise_seed': result['noise_seed'],
                 'config': result['config'], 'slot_id': slot['id']}
        groups[slot['partition']].append(entry)
        if result['video']:
            videos.append({**entry, 'partition': slot['partition'], 'video': result['video'],
                           'event': result['event'], 'recoveries': result['recoveries'],
                           'metrics_path': result['metrics_path']})
    manifest = {'schema_version': 2, 'split_seed': plan['collection_seed'], 'validation_fraction': 0.1,
                'collection_plan': str((root / 'plan.json').resolve()),
                'test_provenance': 'Existing ten clean diverse100 test demonstrations; no new test collection',
                'action_target': 'Unperturbed recovery-supervisor command at the actual visited state',
                **groups, 'summary': {**{f'{p}_trajectories': len(e) for p,e in groups.items()},
                                     **{f'{p}_samples': sum(r['steps'] for r in e) for p,e in groups.items()}}}
    validate_manifest(manifest)
    write_manifest(manifest, SPLIT)
    write_manifest({'videos': videos}, root / 'videos.json')


def collect(root=ROOT, workers=8, pilot=False):
    if workers < 1:
        raise ValueError('workers must be positive')
    root = Path(root).resolve()
    desired = make_plan()
    plan_path = root / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != desired:
            raise ValueError('Frozen plan or source differs; use a new dataset version')
    else:
        write_manifest(desired, plan_path)
    plan = desired
    states = {p.stem: json.loads(p.read_text()) for p in (root / 'jobs').glob('*.json')}
    selected = [s for s in plan['slots'] if not pilot or s['pilot']]
    pending = [s for s in selected if s['id'] not in states or states[s['id']]['result'] is None]
    print(f'Selected {len(selected)} slots; {len(pending)} remaining. Plan: {plan_path}', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(collect_slot, root, slot) for slot in pending]
        for future in as_completed(futures):
            state = future.result()
            states[state['slot']['id']] = state
            summary = summarize(states, plan)
            write_manifest(summary, root / 'summary.json')
            r = state['result']
            print(json.dumps({'slot': state['slot']['id'], 'success': bool(r), 'attempts': len(state['attempts']),
                              'event': r['event'] if r else None,
                              'recoveries': len(r['recoveries']) if r else None,
                              'completed': summary['successful_trajectories']}), flush=True)
    summary = summarize(states, plan)
    write_manifest(summary, root / 'summary.json')
    if any(states[s['id']]['result'] is None for s in selected):
        raise RuntimeError('Some slots exhausted attempts; inspect ledgers')
    if not pilot:
        publish(root, states, plan)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    print(json.dumps(collect(workers=args.workers, pilot=args.pilot), indent=2))


if __name__ == '__main__':
    main()
