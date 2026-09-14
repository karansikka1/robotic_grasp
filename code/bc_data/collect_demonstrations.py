"""Collect/resume frozen clean and disturbed expert slots; publish a whole-trajectory split."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
from pathlib import Path
import time
import uuid

import numpy as np

from motion_planning.simulator import Simulator
from .recording import AttemptRecorder
from .recovery import DisturbanceConfig, RecoveryTeacher
from .teacher import TeacherFailure, read_oracle_state

from .recording import inspect_trajectory, prepare_output, relative_entry, sha256, source_hashes, write_manifest

DEFAULT_PLAN = Path(__file__).resolve().parents[1] / 'plans/demonstrations.json'


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
                             scratch_dir=Path(root)/'scratch') as recorder:
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
    write_manifest(result, run / 'metrics.json')
    return result


def collect_slot(job):
    root, slot, videos = job
    root = Path(root)
    ledger = root / 'jobs' / f"{slot['id']}.json"
    state = json.loads(ledger.read_text()) if ledger.exists() else {'slot': slot, 'attempts': [], 'result': None}
    if state['slot'] != slot:
        raise ValueError('Saved slot differs from the requested plan')
    if state['result']:
        inspect_trajectory(root / state['result']['trajectory'])
        return state
    for candidate in slot['candidates'][len(state['attempts']):]:
        run_slot = {**slot, 'save_video': videos or slot['save_video']}
        result = generate_attempt(root, run_slot, candidate)
        # Ledgers move with their output directory; no source-host paths required.
        for key in ('trajectory', 'video', 'metrics_path'):
            if result.get(key):
                result[key] = str(Path(result[key]).relative_to(root))
        state['attempts'].append(result)
        if result['success']:
            state['result'] = result
        write_manifest(state, ledger)
        if result['success']:
            break
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--slots', nargs='+', help='Optional exact slot IDs for a small collection')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--videos', action='store_true', help='Record every selected slot, not only plan review slots')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    plan = json.loads(args.plan.read_text())
    slots = {slot['id']: slot for slot in plan['slots']}
    if args.slots and (len(set(args.slots)) != len(args.slots) or set(args.slots) - slots.keys()):
        parser.error('slots must be unique IDs present in the frozen plan')
    selected = [slots[name] for name in args.slots] if args.slots else list(slots.values())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_output(root, {'kind': 'demonstrations', 'plan_sha256': sha256(args.plan),
                              'all_videos': args.videos, 'source_sha256': source_hashes()})
        jobs = [(str(root), slot, args.videos) for slot in selected]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for state in pool.map(collect_slot, jobs):
                print(f"{state['slot']['id']}: success={bool(state['result'])}, attempts={len(state['attempts'])}", flush=True)
        groups = {'train': [], 'validation': [], 'test': []}
        attempts = 0
        for slot in slots.values():
            file = root / 'jobs' / f"{slot['id']}.json"
            if not file.exists():
                continue
            state = json.loads(file.read_text())
            attempts += len(state['attempts'])
            if state['result']:
                groups[slot['partition']].append(relative_entry(root / state['result']['trajectory'], root / 'split.json',
                                                               category=slot['category'], slot_id=slot['id']))
        count = sum(len(rows) for rows in groups.values())
        complete = count == len(slots)
        result = {'split_type': 'train_validation_only', 'complete': complete, 'planned_trajectories': len(slots),
                  'attempts': attempts, 'successful_trajectories': count, **groups}
        target = root / ('split.json' if complete else 'split.partial.json')
        write_manifest(result, target)
        print(f"Saved {target}: {count}/{len(slots)} trajectories", flush=True)
        failed = [slot['id'] for slot in selected
                  if not json.loads((root / 'jobs' / f"{slot['id']}.json").read_text())['result']]
        if failed:
            raise SystemExit(f'No accepted demonstration for slots: {failed}; inspect their job ledgers')


if __name__ == '__main__':
    main()
