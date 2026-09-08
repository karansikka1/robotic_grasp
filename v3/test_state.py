"""State input, PPO replay, checkpoint, and task/evaluation regression tests."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from v1.ppo import save_checkpoint
from v2.test_task import FakeSimulator
from v3.evaluation import evaluate_policy
from v3.model import StatePPOPolicy
from v3.ppo import PPOConfig
from v3.state import STATE_DIM, STATE_FIELDS, state_policy_observation, state_vector
from v3.task import StateGreenLiftSimulator, StateSimulator


def observation():
    return {name: np.linspace(0.01, 0.1, size, dtype=np.float32) for name, size in STATE_FIELDS}


class StateTests(unittest.TestCase):
    def test_policy_needs_only_state_and_ignores_task_and_images(self):
        policy = StatePPOPolicy(hidden_dim=16)
        obs = observation()
        expected = policy.predict(obs)
        obs.update(task_reward=float('nan'), task_complete=True, frontview_image=object())
        np.testing.assert_array_equal(policy.predict(obs), expected)
        self.assertEqual(set(state_policy_observation(obs)), {key for key, _ in STATE_FIELDS})
        self.assertEqual(state_vector(obs).shape, (STATE_DIM,))
        self.assertEqual(expected.shape, (7,))
        self.assertTrue(np.all(np.abs(expected) <= 1))

    def test_invalid_state_is_rejected(self):
        for value in (np.zeros(2), np.full(3, np.nan), np.full(3, np.inf)):
            with self.assertRaises(ValueError):
                state_vector(dict(observation(), green_pos=value))
        obs = observation()
        del obs['green_pos']
        with self.assertRaises(KeyError):
            state_vector(obs)

    def test_ppo_action_replay_and_both_heads_receive_gradients(self):
        policy = StatePPOPolicy(hidden_dim=16)
        features = policy.extract_frozen_features(observation())
        with torch.no_grad():
            action, old_log_prob, old_value = policy.act(features)
        batch = {'state': torch.stack([features['state']] * 3)}
        log_prob, entropy, value = policy.evaluate_actions(batch, action.expand(3, -1))
        torch.testing.assert_close(log_prob, old_log_prob.expand(3), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(value, old_value.expand(3))
        loss = -log_prob.mean() + (value - 1).square().mean() - 0.01 * entropy.mean()
        loss.backward()
        for head in (policy.actor, policy.critic):
            self.assertGreater(head[0].weight.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(head[0].weight.grad).all())

    def test_checkpoint_roundtrip_retains_state_contract_and_actions(self):
        policy = StatePPOPolicy(hidden_dim=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            save_checkpoint(path, policy, torch.optim.Adam(policy.parameters()), PPOConfig(), global_step=4, update=1)
            saved = torch.load(path, weights_only=False)
            self.assertFalse(saved['uses_privileged_depth'])
            self.assertTrue(saved['uses_privileged_state'])
            restored = StatePPOPolicy.from_checkpoint(path)
            np.testing.assert_array_equal(policy.predict(observation()), restored.predict(observation()))
        with self.assertRaises(ValueError):
            StatePPOPolicy(state_schema='unknown')

    def test_state_extraction_coordinates_velocities_and_copies(self):
        data = SimpleNamespace(
            body_xpos=np.array([[1., 2., 3.]]),
            body_xquat=np.array([[1., 0., 0., 0.]]),
            site_xpos=np.array([[0.5, 1., 2.]]),
            get_site_xvelp=lambda name: np.array([1., 0., 0.]),
            get_site_xvelr=lambda name: np.array([0., 2., 0.]),
            get_body_xvelp=lambda name: np.array([0., 0., 3.]),
            get_body_xvelr=lambda name: np.array([4., 0., 0.]),
        )
        obs = observation()
        env = SimpleNamespace(
            step=lambda action: (obs, 0, False, {}), cubeB_body_id=0,
            cubeB=SimpleNamespace(root_body='green'),
            robots=[SimpleNamespace(arms=['right'], eef_site_id={'right': 0})],
            sim=SimpleNamespace(data=data, model=SimpleNamespace(site_id2name=lambda index: 'grip'), _render_context_offscreen=None),
        )
        simulator = StateSimulator.__new__(StateSimulator)
        simulator.env = env
        result = simulator.step(np.zeros(7))
        np.testing.assert_allclose(result['gripper_to_green'], [0.5, 1., 1.])
        np.testing.assert_array_equal(result['green_quat'], [0, 0, 0, 1])
        np.testing.assert_array_equal(result['eef_angular_velocity'], [0, 2, 0])
        np.testing.assert_array_equal(result['green_linear_velocity'], [0, 0, 3])
        data.body_xpos[:] = 100
        obs['robot0_joint_pos'][:] = 200
        np.testing.assert_array_equal(result['green_pos'], [1, 2, 3])
        self.assertLess(result['robot0_joint_pos'].max(), 1)

    def test_evaluation_uses_same_grasp_lift_reward_with_state_adapter(self):
        fake = FakeSimulator([(False, 0.8), (True, 0.8)] + [(True, 0.9)] * 5)
        original_step = fake.step
        def step(action):
            obs = original_step(action)
            return dict(observation(), **obs)
        fake.step = step
        with patch('v3.task.StateSimulator', return_value=fake):
            simulator = StateGreenLiftSimulator()
        def policy(obs):
            self.assertEqual(set(obs), {key for key, _ in STATE_FIELDS})
            return np.zeros(7)
        with tempfile.TemporaryDirectory() as directory:
            with patch('v3.evaluation.StateGreenLiftSimulator', return_value=simulator):
                metrics = evaluate_policy(policy, directory, episodes=1, max_steps=10, record_video=False)
        self.assertEqual(metrics['summary']['grasp_rate'], 1)
        self.assertEqual(metrics['summary']['lift_success_rate'], 1)
        self.assertEqual(metrics['episodes'][0]['episode_steps'], 6)
        self.assertTrue(fake.closed)


if __name__ == '__main__':
    unittest.main()
