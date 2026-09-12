"""Three review-only disturbance examples, including physical block release/regrasp.

These stage-triggered previews are kept outside BC splits. They intentionally
use an external hold/recovery controller and stronger disturbances than the
step-scheduled robustness300 dataset. No supplied simulator or teacher is edited.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
import multiprocessing
from pathlib import Path
import shutil
import uuid

import av
import cv2
import h5py
import numpy as np

from motion_planning.simulator import Simulator
from v4.data import write_manifest
from v4.record_trajectory import AttemptRecorder
from v4.teacher import PrivilegedStackTeacher, Stage, TeacherDecision, TeacherFailure, read_oracle_state


CATEGORIES = ('gaussian_burst', 'drop_block', 'sideways_error')
ROOT = Path('v4/trajectories/robustness300/visible_pilots')


@dataclass(frozen=True)
class PreviewConfig:
    category: str
    gaussian_std: float = 0.7
    gaussian_clip: float = 0.9
    gaussian_hold_steps: int = 5
    gaussian_duration: int = 20
    release_duration: int = 12
    bias_magnitude: float = 0.95
    bias_duration: int = 10

    def to_dict(self):
        return asdict(self)


class PreviewController:
    def __init__(self, category, seed, initial_state):
        if category not in CATEGORIES:
            raise ValueError(category)
        self.config = PreviewConfig(category)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.teacher = PrivilegedStackTeacher()
        self.initial_green_z = float(initial_state.positions[1, 2])
        self.trigger_delay = int(self.rng.integers(0, 3)) if category != 'gaussian_burst' else 0
        self.eligible_steps = 0
        self.event_start = None
        self.event_end = None
        self.anchor = None
        self.noise = np.zeros(2, dtype=np.float32)
        self.bias_sign = 1.0
        self.recovering = False
        self.recovery_started = None
        self.recovery_finished = None
        self.recovery_checked = False
        self.changed_steps = 0

    def _hold(self, state, *, gripper, stage, clear=False):
        target = self.anchor.copy()
        if clear:
            target[2] += 0.06
        return TeacherDecision(self.teacher._translation_action(state, target, gripper), stage, 0)

    def command(self, simulator, state, step):
        category = self.config.category
        if self.event_start is None:
            if category == 'gaussian_burst':
                target = state.positions[1] + np.array([0, 0, 0.12])
                eligible = self.teacher.stage == Stage.MOVE_ABOVE and np.linalg.norm(state.eef_position - target) < 0.045
            else:
                eligible = (self.teacher.phase == 0 and bool(state.grasped[1])
                            and state.positions[1, 2] - self.initial_green_z > 0.10)
            if eligible:
                self.eligible_steps += 1
                if self.eligible_steps > self.trigger_delay:
                    self.event_start = step
                    duration = (self.config.gaussian_duration if category == 'gaussian_burst' else
                                self.config.release_duration if category == 'drop_block' else self.config.bias_duration)
                    self.event_end = step + duration
                    self.anchor = state.eef_position.copy()
                    self.bias_sign = -1.0 if state.positions[1, 1] > 0 else 1.0

        active = self.event_start is not None and self.event_start <= step < self.event_end
        if active:
            offset = step - self.event_start
            if category == 'drop_block':
                # Once contact is lost, a recovery supervisor would also open and clear.
                decision = self._hold(state, gripper=1.0 if state.grasped[1] else -1.0,
                                      stage='preview_release_green', clear=not state.grasped[1])
                executed = decision.action.copy()
                executed[6] = -1.0
            elif category == 'gaussian_burst':
                decision = self._hold(state, gripper=-1.0, stage='preview_hold_above_green')
                if offset % self.config.gaussian_hold_steps == 0:
                    self.noise = np.clip(self.rng.normal(0, self.config.gaussian_std, 2),
                                         -self.config.gaussian_clip, self.config.gaussian_clip).astype(np.float32)
                executed = decision.action.copy()
                executed[:2] += self.noise
            else:
                decision = self._hold(state, gripper=1.0, stage='preview_hold_lifted_green')
                executed = decision.action.copy()
                executed[1] += self.bias_sign * self.config.bias_magnitude
            executed[:3] = np.clip(executed[:3], -1, 1)
        else:
            if self.event_end is not None and step >= self.event_end and not self.recovery_checked:
                self.recovery_checked = True
                self.recovering = category == 'drop_block' or (category == 'sideways_error' and not state.grasped[1])
                if self.recovering:
                    self.recovery_started = step
            if self.recovering:
                elapsed = step - self.recovery_started
                settled = (abs(state.positions[1, 2] - self.initial_green_z) < 0.02
                           and np.linalg.norm(state.body_velocities[1, 3:]) < 0.025)
                if elapsed >= 25 and settled:
                    # Restart the green pick from its actual landed position.
                    self.teacher = PrivilegedStackTeacher()
                    self.recovering = False
                    self.recovery_finished = step
                elif elapsed >= 100:
                    raise TeacherFailure('Dropped block did not settle for recovery')
            if self.recovering:
                decision = self._hold(state, gripper=-1.0, stage='recovery_clear_and_wait', clear=True)
            else:
                decision = self.teacher.decide(simulator)
            executed = decision.action.copy()
        delta = executed - decision.action
        changed = bool(np.any(delta))
        self.changed_steps += int(changed)
        info = {'active': np.bool_(active), 'changed': np.bool_(changed), 'delta': delta,
                'event_id': np.int32(0 if active else -1),
                'event_step': np.int32(step - self.event_start if active else 0),
                'event_duration': np.int32(self.event_end - self.event_start if active else 0),
                'recovery_active': np.bool_(self.recovering)}
        return decision, executed, info


class PreviewRecorder(AttemptRecorder):
    def finish_preview(self, success):
        """Retain failed preview videos too, without marking failed data successful."""
        self.h5_file.attrs['steps'] = self.steps
        self.h5_file.attrs['success'] = success
        self.h5_file.attrs['review_only'] = True
        self.h5_file.flush()
        self.h5_file.close()
        self.h5_file = None
        self.video_writer.__exit__(None, None, None)
        for source, name in ((self.h5_path, 'trajectory.h5'), (self.video_path, 'raw.mp4')):
            target = self.run_dir / name
            pending = target.with_suffix(target.suffix + '.pending')
            shutil.copyfile(source, pending)
            pending.replace(target)
            source.unlink()


def make_annotated_video(run, result):
    with av.open(str(run / 'raw.mp4')) as source, av.open(str(run / 'review.mp4'), 'w', options={'movflags': '+faststart'}) as output:
        stream = output.add_stream('libx264', rate=20)
        stream.width, stream.height, stream.pix_fmt = 512, 320, 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast'}
        for i, frame in enumerate(source.decode(video=0)):
            pixels = np.zeros((320, 512, 3), dtype=np.uint8)
            pixels[64:] = frame.to_ndarray(format='rgb24')
            start, end = result['event_start'], result['event_end']
            active = start is not None and start <= i < end
            if active:
                status, color = 'DISTURBANCE ACTIVE', (255, 180, 90)
            elif end is not None and i >= end:
                status, color = 'RECOVERY / RESUME', (125, 226, 175)
            else:
                status, color = 'NORMAL TEACHER', (160, 198, 225)
            cv2.putText(pixels, result['category'].replace('_', ' ').upper(), (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 244, 250), 1, cv2.LINE_AA)
            cv2.putText(pixels, f'{status}  |  {i / 20:.2f} s', (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            encoded = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            for packet in stream.encode(encoded):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def generate(category, root, layout_seed=220769969, noise_seed=34):
    root = Path(root).resolve()
    run = root / category / f'preview-{uuid.uuid4()}'
    run.mkdir(parents=True)
    np.random.seed(layout_seed)
    simulator = Simulator(has_renderer=False)
    success, failure, steps = False, None, 0
    controller = None
    try:
        np.random.seed(layout_seed)
        simulator.reset()
        observation = simulator.step(np.zeros(7, dtype=np.float32))
        controller = PreviewController(category, noise_seed, read_oracle_state(simulator))
        with PreviewRecorder(run, seed=layout_seed, teacher=controller.teacher,
                             perturbation=controller, scratch_dir=Path('/tmp/robotic-grasp-visible-pilots')) as recorder:
            recorder.add_initial_observation(observation)
            try:
                while steps < 900:
                    oracle = read_oracle_state(simulator)
                    decision, executed, info = controller.command(simulator, oracle, steps)
                    if decision.done:
                        success = bool(observation['task_complete'])
                        break
                    next_observation = simulator.step(executed)
                    recorder.add_transition(observation, decision.action, stage=decision.stage,
                                            phase=decision.phase, oracle=oracle,
                                            next_observation=next_observation,
                                            executed_action=executed, perturbation_info=info)
                    observation = next_observation
                    steps += 1
            except TeacherFailure as error:
                failure = str(error)
            if not success and failure is None:
                failure = 'step limit' if steps == 900 else 'teacher finished without success'
            recorder.add_terminal_observation(observation, simulator)
            recorder.finish_preview(success)
    finally:
        simulator.close()
    result = {'category': category, 'layout_seed': layout_seed, 'noise_seed': noise_seed,
              'config': controller.config.to_dict(), 'trigger_delay_eligible_steps': controller.trigger_delay,
              'success': success, 'failure': failure, 'steps': steps, 'seconds': steps / 20,
              'event_start': controller.event_start, 'event_end': controller.event_end,
              'recovery_started': controller.recovery_started, 'recovery_finished': controller.recovery_finished,
              'trajectory': str(run / 'trajectory.h5'), 'video': str(run / 'review.mp4'),
              'review_only': True}
    with h5py.File(run / 'trajectory.h5', 'r') as f:
        positions = np.concatenate([f['oracle/object_positions'][:], f['terminal_observation/object_positions'][:][None]], axis=0)
        eef = np.concatenate([f['oracle/eef_position'][:], f['terminal_observation/eef_position'][:][None]], axis=0)
        grasped = f['oracle/grasped'][:, 1]
        start, end = result['event_start'], result['event_end']
        if start is not None:
            end = min(end, len(eef) - 1)
            result['eef_event_displacement_cm'] = float(np.linalg.norm(eef[start:end+1] - eef[start], axis=1).max() * 100)
            result['green_event_displacement_cm'] = float(np.linalg.norm(positions[start:end+1, 1] - positions[start, 1], axis=1).max() * 100)
            result['green_downward_travel_cm'] = float((positions[start, 1, 2] - positions[start:min(len(positions), end+30), 1, 2].min()) * 100)
            lost = np.flatnonzero(~grasped[start:end]) + start if grasped[start] else []
            result['grasped_at_event_start'] = bool(grasped[start])
            result['grasp_lost_during_event'] = bool(len(lost))
            regrasp = np.flatnonzero(grasped[lost[0]+1:]) + lost[0]+1 if len(lost) else []
            result['regrasp_step'] = int(regrasp[0]) if len(regrasp) else None
    make_annotated_video(run, result)
    with av.open(result['video']) as video:
        result['decoded_frames'] = sum(1 for _ in video.decode(video=0))
    assert result['decoded_frames'] == steps + 1
    write_manifest(result, run / 'metrics.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--category', choices=CATEGORIES)
    args = parser.parse_args()
    categories = [args.category] if args.category else CATEGORIES
    with ProcessPoolExecutor(max_workers=len(categories), mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(generate, category, args.output_dir) for category in categories]
        results = [future.result() for future in futures]
    write_manifest({'review_only': True, 'examples': results}, args.output_dir / 'examples.json')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
