"""Explicit, versioned state inputs shared by training and evaluation."""

import numpy as np

# Order is part of the checkpoint contract. Distances are meters, velocities
# meters/radians per second, joint positions radians, quaternions xyzw.
STATE_FIELDS = (
    ("robot0_joint_pos", 7),
    ("robot0_joint_vel", 7),
    ("robot0_eef_pos", 3),
    ("robot0_eef_quat", 4),
    ("robot0_gripper_qpos", 2),
    ("robot0_gripper_qvel", 2),
    ("green_pos", 3),
    ("green_quat", 4),
    ("gripper_to_green", 3),
    ("eef_linear_velocity", 3),
    ("eef_angular_velocity", 3),
    ("green_linear_velocity", 3),
    ("green_angular_velocity", 3),
)
STATE_DIM = sum(size for _, size in STATE_FIELDS)
STATE_SCHEMA = "green-lift-state-v1"


def state_policy_observation(observation):
    """Exclude images, task rewards, success flags, and unrelated observations."""
    return {name: observation[name] for name, _ in STATE_FIELDS}


def state_vector(observation):
    values = []
    for name, size in STATE_FIELDS:
        value = np.asarray(observation[name], dtype=np.float32).reshape(-1)
        if value.size != size or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain {size} finite state values")
        values.append(value)
    return np.concatenate(values)
