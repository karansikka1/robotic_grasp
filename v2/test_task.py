"""Reward and evaluation tests runnable without starting a real simulator."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from v2.task import GreenLiftSimulator, LiftEpisode, LiftTaskConfig
from v2.evaluation import evaluate_policy


class FakeSimulator:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0
        self.closed = False
        self.grasped = False
        self.green = object()
        self.env = SimpleNamespace(
            sim=SimpleNamespace(data=SimpleNamespace(body_xpos=np.zeros((2, 3)))),
            cubeB_body_id=1, cubeB=self.green,
            robots=[SimpleNamespace(gripper=object())],
            _check_grasp=self.check_grasp,
        )
        self.action_spec = (-np.ones(7), np.ones(7))

    def check_grasp(self, *, gripper, object_geoms):
        assert object_geoms is self.green
        return self.grasped

    def reset(self):
        self.index = 0

    def step(self, action):
        self.grasped, height = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        self.env.sim.data.body_xpos[1, 2] = height
        # Full stack success must not be mistaken for v2 success.
        return dict(
            task_complete=True,
            robot0_joint_pos=np.zeros(7), robot0_eef_pos=np.zeros(3),
            robot0_eef_quat=np.zeros(4), robot0_gripper_qpos=np.zeros(2),
            frontview_image=np.zeros((2, 2, 3), dtype=np.uint8),
            robot0_eye_in_hand_image=np.zeros((2, 2, 3), dtype=np.uint8),
            frontview_depth=np.ones((2, 2, 1)),
        )

    def close(self):
        self.closed = True


class LiftTaskTests(unittest.TestCase):
    def test_grasp_bonus_is_once_per_episode(self):
        episode = LiftEpisode(LiftTaskConfig(), 0.8)
        rewards = [episode.advance(grasp, 0.8) for grasp in (False, True, True, False, True)]
        self.assertEqual(rewards, [0, 1, 0, 0, 0])
        self.assertFalse(episode.success)
        self.assertEqual(episode.steps_to_grasp, 2)

    def test_lift_requires_consecutive_held_height(self):
        episode = LiftEpisode(LiftTaskConfig(), 0.8)
        self.assertEqual(episode.advance(False, 0.9), 0)  # thrown or pushed upward
        self.assertEqual(episode.advance(True, 0.9), 1)
        for _ in range(3):
            self.assertEqual(episode.advance(True, 0.9), 0)
        self.assertFalse(episode.success)
        episode.advance(False, 0.9)  # contact loss resets hold count
        for _ in range(4):
            episode.advance(True, 0.9)
        self.assertFalse(episode.success)
        episode.advance(True, 0.82)  # dipping below target also resets it
        for _ in range(4):
            episode.advance(True, 0.9)
        self.assertEqual(episode.advance(True, 0.9), 5)
        self.assertTrue(episode.success)
        self.assertEqual(episode.advance(True, 0.9), 0)
        self.assertEqual(episode.grasp_return + episode.lift_return, 6)

    def test_bootstrap_and_reset_do_not_award_rewards(self):
        fake = FakeSimulator([(True, 0.8), (True, 0.8)])
        with patch("v2.task.Simulator", return_value=fake):
            simulator = GreenLiftSimulator()
        for _ in range(2):
            simulator.reset()
            bootstrap = simulator.step(np.zeros(7))
            self.assertEqual(bootstrap["task_reward"], 0)
            self.assertFalse(bootstrap["task_complete"])
            self.assertIsNone(bootstrap["task_metrics"]["steps_to_grasp"])
            self.assertEqual(simulator.step(np.zeros(7))["task_reward"], 1)
        simulator.close()
        self.assertTrue(fake.closed)

    def test_evaluation_uses_lift_success_and_filters_privileged_task_fields(self):
        fake = FakeSimulator([(False, 0.8), (True, 0.8)] + [(True, 0.9)] * 5)
        with patch("v2.task.Simulator", return_value=fake):
            simulator = GreenLiftSimulator()

        def policy(observation):
            self.assertNotIn("task_complete", observation)
            self.assertNotIn("task_reward", observation)
            self.assertNotIn("task_metrics", observation)
            return np.zeros(7)

        with tempfile.TemporaryDirectory() as directory:
            with patch("v2.evaluation.GreenLiftSimulator", return_value=simulator):
                metrics = evaluate_policy(policy, directory, episodes=1, max_steps=10, record_video=False)
        self.assertEqual(metrics["summary"]["grasp_rate"], 1)
        self.assertEqual(metrics["summary"]["lift_success_rate"], 1)
        self.assertEqual(metrics["episodes"][0]["episode_steps"], 6)
        self.assertEqual(metrics["episodes"][0]["task_metrics"]["steps_to_grasp"], 1)
        self.assertEqual(metrics["episodes"][0]["task_metrics"]["lift_return"], 5)
        self.assertTrue(fake.closed)

    def test_invalid_task_settings(self):
        for kwargs in (
            {"grasp_reward": float("nan")}, {"lift_reward": -1},
            {"lift_height_m": 0}, {"hold_steps": 0}, {"hold_steps": 1.5},
        ):
            with self.assertRaises(ValueError):
                LiftTaskConfig(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
