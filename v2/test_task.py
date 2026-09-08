"""Reward and evaluation tests runnable without starting a real simulator."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dataclasses import asdict

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
            _gripper_to_target=self.distance,
        )
        self.action_spec = (-np.ones(7), np.ones(7))

    def distance(self, *, gripper, target, target_type, return_distance):
        assert target is self.green and target_type == "body" and return_distance
        return 0.2

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
        episode = LiftEpisode(LiftTaskConfig(), 0.8, 0.2)
        rewards = [episode.advance(grasp, 0.8, 0.2) for grasp in (False, True, True, False, True)]
        self.assertEqual(rewards, [0, 1, 0, 0, 0])
        self.assertFalse(episode.success)
        self.assertEqual(episode.steps_to_grasp, 2)

    def test_lift_requires_consecutive_held_height(self):
        episode = LiftEpisode(LiftTaskConfig(), 0.8, 0.2)
        self.assertEqual(episode.advance(False, 0.9, 0.2), 0)  # thrown or pushed upward
        self.assertEqual(episode.advance(True, 0.9, 0.2), 1)
        for _ in range(3):
            self.assertEqual(episode.advance(True, 0.9, 0.2), 0)
        self.assertFalse(episode.success)
        episode.advance(False, 0.9, 0.2)  # contact loss resets hold count
        for _ in range(4):
            episode.advance(True, 0.9, 0.2)
        self.assertFalse(episode.success)
        episode.advance(True, 0.82, 0.2)  # dipping below target also resets it
        for _ in range(4):
            episode.advance(True, 0.9, 0.2)
        self.assertEqual(episode.advance(True, 0.9, 0.2), 5)
        self.assertTrue(episode.success)
        self.assertEqual(episode.advance(True, 0.9, 0.2), 0)
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
        self.assertAlmostEqual(metrics["summary"]["mean_min_gripper_distance_m"], 0.2)
        self.assertEqual(metrics["summary"]["mean_reach_return"], 0)
        self.assertEqual(metrics["summary"]["grasp_rate"], 1)
        self.assertEqual(metrics["summary"]["lift_success_rate"], 1)
        self.assertEqual(metrics["episodes"][0]["episode_steps"], 6)
        self.assertEqual(metrics["episodes"][0]["task_metrics"]["steps_to_grasp"], 1)
        self.assertEqual(metrics["episodes"][0]["task_metrics"]["lift_return"], 5)
        self.assertTrue(fake.closed)

    def test_signed_progress_rewards_and_no_hover_bonus(self):
        episode = LiftEpisode(LiftTaskConfig(), 0.8, 0.2)
        self.assertAlmostEqual(episode.advance(False, 0.8, 0.15), 0.05)
        self.assertEqual(episode.advance(False, 0.8, 0.15), 0)
        self.assertAlmostEqual(episode.advance(False, 0.8, 0.2), -0.05)
        self.assertAlmostEqual(episode.reach_return, 0)
        self.assertAlmostEqual(episode.metrics()["min_gripper_distance_m"], 0.15)
        self.assertAlmostEqual(episode.advance(False, 0.8, 0.1), 0.1)
        self.assertAlmostEqual(episode.advance(True, 0.8, 0.05), 1.05)
        self.assertFalse(episode.success)  # Reaching is not lift success.
        self.assertAlmostEqual(episode.reach_return, 0.15)

    def test_progress_and_lift_reward_add(self):
        episode = LiftEpisode(LiftTaskConfig(hold_steps=1), 0.8, 0.2)
        self.assertAlmostEqual(episode.advance(True, 0.9, 0.1), 6.1)
        self.assertTrue(episode.success)
        self.assertEqual(episode.grasp_return, 1)
        self.assertEqual(episode.lift_return, 5)

    def test_zero_scale_and_old_checkpoint_keep_sparse_rewards(self):
        old = {"grasp_reward": 1.0, "lift_reward": 5.0, "lift_height_m": 0.05, "hold_steps": 5}
        config = LiftTaskConfig.from_saved_config(old)
        self.assertEqual(config.reach_reward_scale, 0)
        episode = LiftEpisode(config, 0.8, 0.3)
        self.assertEqual(episode.advance(False, 0.8, 0.1), 0)
        self.assertEqual(episode.advance(True, 0.8, 0.05), 1)
        self.assertEqual(episode.reach_return, 0)
        new = LiftTaskConfig(reach_reward_scale=0.4)
        self.assertEqual(LiftTaskConfig.from_saved_config(asdict(new)), new)

    def test_reaching_bootstrap_and_reset(self):
        fake = FakeSimulator([(False, 0.8)] * 5)
        with patch("v2.task.Simulator", return_value=fake):
            simulator = GreenLiftSimulator()
        for _ in range(2):
            simulator.reset()
            with patch.object(fake.env, "_gripper_to_target", side_effect=[0.3, 0.2]):
                bootstrap = simulator.step(np.zeros(7))
                self.assertEqual(bootstrap["task_reward"], 0)
                self.assertEqual(bootstrap["task_metrics"]["reach_return"], 0)
                observation = simulator.step(np.zeros(7))
                self.assertAlmostEqual(observation["task_reward"], 0.1)
                self.assertAlmostEqual(observation["task_metrics"]["min_gripper_distance_m"], 0.2)
        simulator.close()

    def test_invalid_distances(self):
        for distance in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                LiftEpisode(LiftTaskConfig(), 0.8, distance)
            episode = LiftEpisode(LiftTaskConfig(), 0.8, 0.2)
            with self.assertRaises(ValueError):
                episode.advance(False, 0.8, distance)

    def test_invalid_task_settings(self):
        for kwargs in (
            {"reach_reward_scale": -1}, {"reach_reward_scale": float("nan")},
            {"reach_reward_scale": float("inf")}, {"grasp_reward": float("nan")}, {"lift_reward": -1},
            {"lift_height_m": 0}, {"hold_steps": 0}, {"hold_steps": 1.5},
        ):
            with self.assertRaises(ValueError):
                LiftTaskConfig(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
