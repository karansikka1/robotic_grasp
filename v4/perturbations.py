"""Reproducible movement disturbances, independent of teacher state and global RNG."""

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


CATEGORIES = ('clean', 'gaussian_noise', 'dropped_commands', 'delayed_commands',
              'sustained_bias')


@dataclass(frozen=True)
class PerturbationConfig:
    category: str = 'clean'
    gaussian_std: float = 0.015
    gaussian_clip: float = 0.045
    drop_probability: float = 0.03
    drop_min_steps: int = 1
    drop_max_steps: int = 3
    delay_steps: int = 1
    bias_probability: float = 0.02
    bias_magnitude: float = 0.02
    bias_min_steps: int = 5
    bias_max_steps: int = 10

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f'Unknown perturbation category: {self.category}')
        for name in ('gaussian_std', 'gaussian_clip', 'bias_magnitude'):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must be finite and in (0, 1]')
        for name in ('drop_probability', 'bias_probability'):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in (0, 1]')
        for name in ('drop_min_steps', 'drop_max_steps', 'delay_steps',
                     'bias_min_steps', 'bias_max_steps'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.drop_min_steps > self.drop_max_steps or self.bias_min_steps > self.bias_max_steps:
            raise ValueError('Minimum burst duration must not exceed maximum')

    def to_dict(self):
        return asdict(self)


class ActionPerturber:
    """One instance per episode; only translation channels can change.

    Schedules depend on control-step count and a dedicated seed, never on oracle
    stages. The same wrapper can therefore perturb a learned policy at evaluation.
    A dropped command executes zero XYZ delta. Delay buffers only XYZ commands;
    missing commands at episode start are zero. All frames remain recorded.
    """

    def __init__(self, config: PerturbationConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.history = deque(maxlen=config.delay_steps + 1)
        self.remaining = 0
        self.duration = 0
        self.event_id = -1
        self.bias = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.changed_steps = 0

    def apply(self, action):
        intended = np.asarray(action, dtype=np.float32)
        if intended.shape != (7,) or not np.isfinite(intended).all() or (np.abs(intended) > 1).any():
            raise ValueError('Expected a finite normalized action of shape (7,)')
        executed = intended.copy()
        category = self.config.category
        active = False
        event_id, event_step, duration = -1, 0, 0
        if category == 'gaussian_noise':
            noise = np.clip(self.rng.normal(0, self.config.gaussian_std, 3),
                            -self.config.gaussian_clip, self.config.gaussian_clip)
            executed[:3] += noise.astype(np.float32)
            active, event_id, event_step, duration = True, self.steps, 0, 1
        elif category == 'delayed_commands':
            self.history.append(intended[:3].copy())
            executed[:3] = self.history[0] if len(self.history) > self.config.delay_steps else 0
            active, event_id, event_step, duration = True, 0, self.steps, -1
        elif category in ('dropped_commands', 'sustained_bias'):
            prefix = 'drop' if category == 'dropped_commands' else 'bias'
            if self.remaining == 0 and self.rng.random() < getattr(self.config, f'{prefix}_probability'):
                self.duration = int(self.rng.integers(getattr(self.config, f'{prefix}_min_steps'),
                                                     getattr(self.config, f'{prefix}_max_steps') + 1))
                self.remaining = self.duration
                self.event_id += 1
                if category == 'sustained_bias':
                    direction = self.rng.normal(size=3)
                    self.bias = (direction / max(np.linalg.norm(direction), 1e-12)
                                 * self.config.bias_magnitude).astype(np.float32)
            if self.remaining:
                active = True
                event_id, event_step, duration = self.event_id, self.duration - self.remaining, self.duration
                if category == 'dropped_commands':
                    executed[:3] = 0
                else:
                    executed[:3] += self.bias
                self.remaining -= 1
        executed[:3] = np.clip(executed[:3], -1, 1)
        delta = executed - intended
        changed = bool(np.any(delta))
        self.changed_steps += int(changed)
        self.steps += 1
        return executed, {'active': np.bool_(active), 'changed': np.bool_(changed),
                          'delta': delta, 'event_id': np.int32(event_id),
                          'event_step': np.int32(event_step), 'event_duration': np.int32(duration)}
