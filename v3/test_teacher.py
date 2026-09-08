"""Fast deterministic tests for the teacher and streaming trajectory format."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

import v3.record_trajectory as record_module
from v3.record_trajectory import generate_one_success
from v3.teacher import TeacherConfig


class _Object:
    def __init__(self, index: int, size: float) -> None:
        self.index = index
        self.size = np.full(3, size, dtype=np.float64)


class FakeSimulator:
    """Small kinematic stand-in that exercises the complete teacher state machine."""

    def __init__(self, has_renderer: bool = False) -> None:
        del has_renderer
        self.action_spec = (np.full(7, -1.0), np.full(7, 1.0))
        self.env = self
        self.horizon = 1_000
        self.ignore_done = False
        self.timestep = 0
        self.cubeA = _Object(0, 0.020)
        self.cubeB = _Object(1, 0.025)
        self.cubeC = _Object(2, 0.025)
        self.cubeA_body_id = 0
        self.cubeB_body_id = 1
        self.cubeC_body_id = 2
        self.robots = [
            SimpleNamespace(
                eef_site_id={"right": 0},
                gripper=object(),
            )
        ]
        self.sim = SimpleNamespace(
            data=SimpleNamespace(
                body_xpos=np.zeros((3, 3), dtype=np.float64),
                body_xquat=np.zeros((3, 4), dtype=np.float64),
                cvel=np.zeros((3, 6), dtype=np.float64),
                site_xpos=np.zeros((1, 3), dtype=np.float64),
            )
        )
        self.grasped_index: int | None = None
        self.grasp_offset = np.zeros(3)
        self.reset()

    def reset(self) -> None:
        self.timestep = 0
        self.sim.data.body_xpos[:] = np.asarray(
            ((0.00, 0.00, 0.82), (0.12, 0.02, 0.825), (-0.12, -0.02, 0.825))
        )
        self.sim.data.body_xquat[:] = np.asarray(
            ((1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0))
        )
        self.sim.data.site_xpos[0] = np.asarray((0.0, -0.1, 1.05))
        self.grasped_index = None

    def _contact(self, first: int, second: int) -> bool:
        positions = self.sim.data.body_xpos
        objects = (self.cubeA, self.cubeB, self.cubeC)
        xy_close = np.linalg.norm(positions[first, :2] - positions[second, :2]) < 0.015
        expected_height = objects[first].size[2] + objects[second].size[2]
        height_close = abs(abs(positions[first, 2] - positions[second, 2]) - expected_height) < 0.008
        return bool(xy_close and height_close)

    def _check_grasp(self, *, gripper, object_geoms) -> bool:
        del gripper
        return self.grasped_index == object_geoms.index

    def check_contact(self, first, second) -> bool:
        return self._contact(first.index, second.index)

    def _check_success(self) -> bool:
        heights = self.sim.data.body_xpos[:, 2]
        return bool(
            heights[0] < heights[1] < heights[2]
            and self._contact(0, 1)
            and self._contact(1, 2)
        )

    def step(self, action):
        action = np.asarray(action)
        self.timestep += 1
        self.sim.data.site_xpos[0] += action[:3] * 0.05
        if self.grasped_index is not None:
            self.sim.data.body_xpos[self.grasped_index] = (
                self.sim.data.site_xpos[0] - self.grasp_offset
            )

        if action[6] > 0 and self.grasped_index is None:
            distances = np.linalg.norm(
                self.sim.data.body_xpos - self.sim.data.site_xpos[0], axis=1
            )
            closest = int(np.argmin(distances))
            if distances[closest] < 0.015:
                self.grasped_index = closest
                self.grasp_offset = (
                    self.sim.data.site_xpos[0]
                    - self.sim.data.body_xpos[closest]
                )
        elif action[6] < 0:
            self.grasped_index = None

        return {
            "robot0_joint_pos": np.zeros(7, dtype=np.float32),
            "robot0_eef_pos": self.sim.data.site_xpos[0].copy(),
            "robot0_eef_quat": np.asarray((1, 0, 0, 0), dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            "frontview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "frontview_depth": np.ones((256, 256, 1), dtype=np.float32),
            "task_complete": self._check_success(),
        }

    def close(self) -> None:
        return None


class TeacherTrajectoryTest(unittest.TestCase):
    def test_generates_aligned_successful_trajectory(self) -> None:
        original_simulator = record_module.Simulator
        record_module.Simulator = FakeSimulator
        try:
            with tempfile.TemporaryDirectory() as directory:
                metrics = generate_one_success(
                    Path(directory),
                    name="test-teacher",
                    seed=7,
                    max_attempts=1,
                    max_steps=500,
                    teacher_config=TeacherConfig(),
                )
                result = metrics["result"]
                self.assertTrue(result["success"])
                trajectory_path = Path(result["trajectory"])
                self.assertTrue(trajectory_path.exists())
                self.assertGreater(Path(result["video"]).stat().st_size, 0)
                with h5py.File(trajectory_path, "r") as trajectory:
                    steps = int(trajectory.attrs["steps"])
                    self.assertGreater(steps, 0)
                    self.assertEqual(trajectory["actions"].shape, (steps, 7))
                    self.assertEqual(trajectory["stage"].shape, (steps,))
                    self.assertEqual(
                        trajectory["observations/frontview_image"].shape,
                        (steps, 256, 256, 3),
                    )
                    self.assertTrue(
                        trajectory["terminal_observation"].attrs["task_complete"]
                    )
        finally:
            record_module.Simulator = original_simulator


if __name__ == "__main__":
    unittest.main()
