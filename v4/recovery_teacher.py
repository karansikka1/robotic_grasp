"""External disturbance/recovery teacher with canonical stage labels for BC.

The original waypoint teacher is unchanged. Intended actions come from its
closed-loop control (or a retreat after failed contact); only executed commands
are perturbed. One event per trajectory is assigned to green or blue.
"""

from dataclasses import asdict, dataclass

import numpy as np

from v4.teacher import PrivilegedStackTeacher, Stage, TeacherDecision, TeacherFailure


CATEGORIES = ('clean', 'gaussian_burst', 'drop_block', 'sideways_error', 'gripper_interrupt')
TRANSPORT_STAGES = (Stage.LIFT, Stage.TRANSFER, Stage.LOWER)


@dataclass(frozen=True)
class DisturbanceConfig:
    category: str
    target_phase: int = 0
    trigger: str = 'approach'
    duration: int = 18
    eligible_delay: int = 0
    gaussian_std: float = 0.7
    gaussian_clip: float = 0.9
    sample_hold_steps: int = 4
    bias_magnitude: float = 0.9
    bias_angle: float = 0.0
    lift_threshold_m: float = 0.10
    recovery_clearance_m: float = 0.10
    recovery_min_steps: int = 20
    recovery_timeout_steps: int = 100
    max_recoveries: int = 3

    def __post_init__(self):
        if self.category not in CATEGORIES or self.target_phase not in (0, 1):
            raise ValueError('Invalid category or phase')
        if self.trigger not in ('approach', 'descent', 'lift', 'transfer', 'delay_close', 'reopen'):
            raise ValueError('Invalid trigger')
        if not 1 <= self.duration <= 40 or not 0 <= self.eligible_delay <= 4:
            raise ValueError('Invalid event duration/delay')
        if not 1 <= self.sample_hold_steps <= 8:
            raise ValueError('Invalid noise hold duration')
        if not 0 < self.gaussian_std <= 1 or not 0 < self.gaussian_clip <= 1 or not 0 < self.bias_magnitude <= 1:
            raise ValueError('Invalid command disturbance magnitude')
        if not np.isfinite(self.bias_angle) or not 0.05 <= self.lift_threshold_m <= 0.16:
            raise ValueError('Invalid direction or lift trigger')
        if not 0 < self.recovery_clearance_m <= 0.2:
            raise ValueError('Invalid recovery clearance')
        if not 0 < self.recovery_min_steps < self.recovery_timeout_steps or self.max_recoveries < 1:
            raise ValueError('Invalid recovery limits')

    def to_dict(self):
        return asdict(self)


def sample_config(category, index, seed):
    """Balance object and phase bins; vary severity/duration within each bin."""
    rng = np.random.default_rng(seed)
    phase = index % 2
    trigger = 'approach'
    duration = int(rng.integers(14, 25))
    if category == 'gaussian_burst':
        trigger = ('approach', 'descent', 'lift', 'transfer')[(index // 2) % 4]
    elif category == 'sideways_error':
        trigger = ('lift', 'transfer')[(index // 2) % 2]
        duration = int(rng.integers(8, 14))
    elif category == 'drop_block':
        trigger, duration = 'lift', int(rng.integers(8, 15))
    elif category == 'gripper_interrupt':
        trigger = ('delay_close', 'reopen')[(index // 2) % 2]
        duration = int(rng.integers(4, 11))
    return DisturbanceConfig(category=category, target_phase=phase, trigger=trigger,
                             duration=duration, eligible_delay=0 if trigger in ('descent', 'reopen', 'delay_close') else int(rng.integers(0, 3)),
                             gaussian_std=float(rng.uniform(0.5, 0.8)),
                             gaussian_clip=float(rng.uniform(0.85, 1.0)),
                             sample_hold_steps=int(rng.integers(3, 6)),
                             bias_magnitude=float(rng.uniform(0.8, 1.0)),
                             bias_angle=float(rng.uniform(-np.pi, np.pi)),
                             lift_threshold_m=float(rng.uniform(0.08, 0.12)))


class RecoveryTeacher:
    def __init__(self, config, seed, initial_state):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.teacher = PrivilegedStackTeacher()
        self.rest_z = initial_state.positions[:, 2].copy()
        self.event_start = None
        self.event_end = None
        self.event_stage = None
        self.event_phase = None
        self.eligible_steps = 0
        self.noise = np.zeros(2, dtype=np.float32)
        self.bias = None
        self.changed_steps = 0
        self.recovery_start = None
        self.recovery_anchor = None
        self.recovery_phase = None
        self.recoveries = []
        self.last_mode = 'normal'
        self.supervisor_modes = []

    def _start_recovery(self, state, step, reason):
        if len(self.recoveries) >= self.config.max_recoveries:
            raise TeacherFailure('Recovery limit reached: ' + reason)
        phase = self.teacher.phase
        if phase == 1 and not state.object_contacts[0]:
            raise TeacherFailure('Green-red base disturbed during blue recovery')
        self.recovery_phase = phase
        self.recovery_start = step
        self.recovery_anchor = state.eef_position.copy()
        self.recovery_anchor[2] = min(1.22, self.recovery_anchor[2] + self.config.recovery_clearance_m)
        self.recoveries.append({'phase': phase, 'start': step, 'reason': reason, 'end': None})

    def _recovery_decision(self, state, step):
        phase = self.recovery_phase
        index = phase + 1
        elapsed = step - self.recovery_start
        if phase == 1 and not state.object_contacts[0]:
            raise TeacherFailure('Green-red base disturbed during blue recovery')
        settled = (not state.grasped[index]
                   and abs(state.positions[index, 2] - self.rest_z[index]) < 0.025
                   and np.linalg.norm(state.body_velocities[index, 3:]) < 0.035)
        event_finished = self.event_end is None or step >= self.event_end
        if elapsed >= self.config.recovery_min_steps and settled and event_finished:
            # Preserve blue phase and the existing green-red base when restarting a blue pick.
            self.teacher = PrivilegedStackTeacher()
            self.teacher.phase = phase
            self.recoveries[-1]['end'] = step
            self.recovery_start = None
            self.last_mode = 'normal'
            return None
        if elapsed >= self.config.recovery_timeout_steps:
            raise TeacherFailure('Object did not settle for a new pick')
        self.last_mode = 'recovery_retreat'
        moving, support = ('green', 'red') if phase == 0 else ('blue', 'green')
        action = self.teacher._translation_action(state, self.recovery_anchor, -1)
        return TeacherDecision(action, f'{moving}_retreat_over_{support}', phase)

    def _intended(self, simulator, state, step):
        if self.recovery_start is not None:
            decision = self._recovery_decision(state, step)
            if decision is not None:
                return decision
        self.last_mode = 'normal'
        index = self.teacher.phase + 1
        if (self.event_start is not None and self.teacher.stage in TRANSPORT_STAGES
                and not state.grasped[index]):
            self._start_recovery(state, step, 'lost grasp')
            return self._recovery_decision(state, step)
        try:
            return self.teacher.decide(simulator)
        except TeacherFailure as error:
            # Retreat and retry failed picks/approaches after an intervention.
            # Placement/base failures remain explicit failures; do not silently rebuild the stack.
            if (self.event_start is None or self.teacher.stage not in
                    (Stage.MOVE_ABOVE, Stage.DESCEND, Stage.CLOSE, Stage.LIFT, Stage.TRANSFER)):
                raise
            self._start_recovery(state, step, str(error))
            return self._recovery_decision(state, step)

    def _eligible(self, decision, state):
        if decision.phase != self.config.target_phase or self.last_mode != 'normal':
            return False
        index = self.config.target_phase + 1
        stage = decision.stage
        trigger = self.config.trigger
        vertical_gap = state.eef_position[2] - state.positions[index, 2]
        if trigger == 'approach':
            return ('_move_above_' in stage and vertical_gap > 0.07
                    and np.linalg.norm(state.eef_position[:2] - state.positions[index, :2]) < 0.10)
        if trigger == 'descent':
            return '_descend_' in stage and 0.025 < vertical_gap < 0.11
        if trigger == 'lift':
            return ('_lift_' in stage and state.grasped[index]
                    and state.positions[index, 2] - self.rest_z[index] > self.config.lift_threshold_m)
        if trigger == 'transfer':
            return '_transfer_' in stage and state.grasped[index]
        if trigger == 'delay_close':
            return '_close_' in stage and decision.action[6] > 0 and not state.grasped[index]
        if trigger == 'reopen':
            return (('_close_' in stage or '_lift_' in stage) and state.grasped[index]
                    and state.positions[index, 2] - self.rest_z[index] < 0.04)
        return False

    def command(self, simulator, state, step):
        decision = self._intended(simulator, state, step)
        if (self.config.category != 'clean' and self.event_start is None and not decision.done
                and self._eligible(decision, state)):
            self.eligible_steps += 1
            if self.eligible_steps > self.config.eligible_delay:
                self.event_start, self.event_end = step, step + self.config.duration
                self.event_stage, self.event_phase = decision.stage, decision.phase
                index = self.config.target_phase + 1
                direction = np.array([np.cos(self.config.bias_angle), np.sin(self.config.bias_angle)])
                if np.dot(direction, state.positions[index, :2]) > 0:
                    direction *= -1
                self.bias = direction.astype(np.float32) * self.config.bias_magnitude
        active = self.event_start is not None and self.event_start <= step < self.event_end
        executed = decision.action.copy()
        if active:
            if self.config.category == 'gaussian_burst':
                if (step - self.event_start) % self.config.sample_hold_steps == 0:
                    self.noise = np.clip(self.rng.normal(0, self.config.gaussian_std, 2),
                                         -self.config.gaussian_clip, self.config.gaussian_clip).astype(np.float32)
                executed[:2] += self.noise
            elif self.config.category == 'sideways_error':
                executed[:2] += self.bias
            elif self.config.category in ('drop_block', 'gripper_interrupt'):
                executed[6] = -1
            executed[:3] = np.clip(executed[:3], -1, 1)
        delta = executed - decision.action
        changed = bool(np.any(delta))
        self.changed_steps += int(changed)
        info = {'active': np.bool_(active), 'changed': np.bool_(changed), 'delta': delta,
                'event_id': np.int32(0 if active else -1),
                'event_step': np.int32(step - self.event_start if active else 0),
                'event_duration': np.int32(self.config.duration if active else 0),
                'recovery_active': np.bool_(self.recovery_start is not None),
                'target_phase': np.int8(self.config.target_phase)}
        self.supervisor_modes.append(self.last_mode)
        return decision, executed, info
