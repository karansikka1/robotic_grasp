"""Run the frozen learner until a detected error, then record an expert recovery."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from .recovery import FailureDetector, KINDS, PolicyRecoveryTeacher
from .recording import inspect_trajectory, prepare_output, sha256, source_hashes, write_manifest
from .recording import AttemptRecorder
from motion_planning.simulator import Simulator
from .state_policy import PROPRIO_KEYS, load_state_lstm, state_features
from .teacher import PrivilegedStackTeacher, TeacherFailure, read_oracle_state

DEFAULT_PLAN = Path(__file__).resolve().parents[1] / 'plans/corrections.json'


class StableStackSimulator(Simulator):
    """Require consecutive official successes, matching the teacher's settle check."""

    def __init__(self, *, success_hold_steps=10):
        self.success_hold_steps = success_hold_steps
        self.success_steps = 0
        super().__init__(has_renderer=False)

    def reset(self):
        self.success_steps = 0
        return super().reset()

    def step(self, action):
        observation = super().step(action)
        self.success_steps = self.success_steps + 1 if observation['task_complete'] else 0
        observation['task_complete'] = self.success_steps >= self.success_hold_steps
        return observation


def policy_action(policy, observation, state):
    proprio = np.concatenate([observation[key] for key in PROPRIO_KEYS])
    features = state_features(state.positions, state.eef_position, proprio, [None], use_stage=False)
    return policy.predict_state(torch.from_numpy(features))[0].numpy()


def collect_attempt(job):
    root, candidate, eligible, checkpoint, plan, videos = job
    root = Path(root)
    seed = candidate['seed']
    output = root / 'attempts' / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / 'result.json'
    if ledger.exists():
        result = json.loads(ledger.read_text())
        if result['eligible_kinds'] != eligible:
            raise ValueError('Resume eligibility changed; use the same worker count and protocol')
        if result['success']:
            inspect_trajectory(root / result['path'])
        return result

    torch.set_num_threads(1)
    policy = load_state_lstm(checkpoint)
    np.random.seed(seed)
    simulator = StableStackSimulator(success_hold_steps=plan['success_hold_steps'])
    steps, event, controller = 0, None, None
    started = time.monotonic()
    result = {'seed': seed, 'eligible_kinds': eligible, 'success': False, 'event': None}
    try:
        np.random.seed(seed)
        simulator.reset()
        observation = simulator.step(np.zeros(7, dtype=np.float32))
        initial = read_oracle_state(simulator)
        detector = FailureDetector(initial)
        policy.reset_history()
        with AttemptRecorder(output, seed=seed, teacher=PrivilegedStackTeacher(),
                             save_video=videos, scratch_dir=root / 'scratch') as recorder:
            recorder.h5_file.attrs.update({
                'requires_supervision_mask': True,
                'format_version': 3,
                'action_semantics': 'actions are teacher labels only where supervision/mask is true; executed_actions are actual commands',
                'source_checkpoint_sha256': plan['checkpoint_sha256'],
                'success_hold_steps': plan['success_hold_steps'],
            })
            recorder.h5_file.create_group('supervision')
            recorder.add_initial_observation(observation)
            while steps < plan['max_total_steps'] and not observation['task_complete']:
                state = read_oracle_state(simulator)
                expert = controller is not None
                if expert:
                    decision = controller.decide(simulator, state, steps)
                    if decision.done:
                        break
                    action, label = decision.action, decision.action
                    stage, phase = decision.stage, decision.phase
                else:
                    if steps >= plan['max_policy_steps']:
                        result['failure'] = 'No requested trigger within the policy action budget'
                        break
                    action = policy_action(policy, observation, state)
                    label, stage, phase = np.zeros(7, dtype=np.float32), '', -1
                if not np.isfinite(action).all() or action[3:6].any():
                    raise ValueError('Invalid controller action')
                next_observation = simulator.step(action)
                recorder.add_transition(observation, label, stage=stage, phase=phase,
                                        oracle=state, next_observation=next_observation)
                recorder._append_array('executed_actions', action)
                recorder._append_array('supervision/mask', np.bool_(expert))
                recorder._append_array('supervision/source', np.int8(expert))
                recorder._append_array('supervision/recovery_retreat',
                                       np.bool_(expert and controller.last_mode == 'recovery_retreat'))
                steps += 1
                observation = next_observation
                if not expert:
                    detected = detector.update(read_oracle_state(simulator), action, steps)
                    if detected is not None and detected['kind'] in eligible:
                        event = detected
                        result['event'] = event
                        recorder.h5_file.attrs['handoff_step'] = steps
                        recorder.h5_file.attrs['failure_event'] = json.dumps(event)
                        controller = PolicyRecoveryTeacher(initial, read_oracle_state(simulator), steps, event, seed)
            success = event is not None and bool(observation['task_complete'])
            result.update(success=success, steps=steps, policy_steps=event['step'] if event else steps,
                          expert_steps=steps-event['step'] if event else 0,
                          recoveries=controller.recoveries if controller else [],
                          official_stable_success=bool(observation['task_complete']))
            if success:
                recorder.add_terminal_observation(observation, simulator)
                trajectory, video = recorder.finish_successfully()
                result.update(path=str(trajectory.relative_to(root)),
                              video=str(video.relative_to(root)) if video else None)
                inspect_trajectory(trajectory)
            elif 'failure' not in result:
                result['failure'] = ('Policy succeeded without a requested error' if observation['task_complete']
                                     else 'Correction exceeded action budget')
    except TeacherFailure as error:
        result.update(failure=str(error), steps=steps, event=event,
                      recoveries=controller.recoveries if controller else [])
    finally:
        simulator.close()
    result['wall_seconds'] = time.monotonic() - started
    write_manifest(result, ledger)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Original state-BC checkpoint used for correction collection')
    parser.add_argument('--per-kind', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seeds', nargs='+', type=int, help='Optional subset of the frozen candidate seeds')
    parser.add_argument('--no-videos', action='store_true')
    args = parser.parse_args()
    if min(args.per_kind, args.workers) < 1:
        parser.error('per-kind and workers must be positive')
    plan = json.loads(args.plan.read_text())
    checkpoint = args.checkpoint.resolve()
    if sha256(checkpoint) != plan['checkpoint_sha256']:
        raise ValueError('Correction checkpoint does not match the frozen plan')
    candidates = plan['candidates']
    allowed = {candidate['seed'] for candidate in candidates}
    if args.seeds:
        if len(set(args.seeds)) != len(args.seeds) or set(args.seeds) - allowed:
            parser.error('seeds must be unique members of the frozen candidate plan')
        candidates = [candidate for candidate in candidates if candidate['seed'] in args.seeds]
    if {candidate['seed'] for candidate in candidates} & set(plan['validation_seeds_excluded']):
        raise ValueError('Correction candidates overlap the excluded validation seeds')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_output(root, {'kind': 'corrections', 'plan_sha256': sha256(args.plan),
                              'checkpoint_sha256': plan['checkpoint_sha256'], 'candidates': candidates,
                              'per_kind': args.per_kind, 'workers': args.workers,
                              'videos': not args.no_videos, 'source_sha256': source_hashes()})
        counts, results, selected = Counter(), [], []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for offset in range(0, len(candidates), args.workers):
                eligible = [kind for kind in KINDS if counts[kind] < args.per_kind]
                if not eligible:
                    break
                batch = candidates[offset:offset+args.workers]
                # These two original training layouts intentionally replay through the drop.
                jobs = [(str(root), candidate,
                         ['green_drop'] if candidate['seed'] in (991004301, 1142630323) else eligible,
                         str(checkpoint), plan, not args.no_videos) for candidate in batch]
                for result in pool.map(collect_attempt, jobs):
                    results.append(result)
                    if result['success'] and counts[result['event']['kind']] < args.per_kind:
                        counts[result['event']['kind']] += 1
                        selected.append(result)
                    complete = all(counts[kind] >= args.per_kind for kind in KINDS)
                    write_manifest({'attempts': len(results), 'counts': dict(counts),
                                    'target_per_kind': args.per_kind, 'selected': selected,
                                    'results': results, 'complete': complete}, root / 'collection.json')
                    print(f"seed={result['seed']} success={result['success']} counts={dict(counts)}", flush=True)
        if not all(counts[kind] >= args.per_kind for kind in KINDS):
            raise SystemExit(f'Correction quotas not met: {dict(counts)}; inspect collection.json')
        print(f'Saved {root / "collection.json"}: {dict(counts)}', flush=True)


if __name__ == '__main__':
    main()
