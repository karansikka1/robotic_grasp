"""Parallel timeline isolation, process failures, seeding, and default routing."""

import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from v1.ppo import RolloutBuffer
from v3.parallel_ppo import combine_rollouts
from v3.ppo import PPOConfig, train_ppo
from v3.state import STATE_FIELDS
from v3.vector_env import ParallelSimulators


class FakeStateSimulator:
    def __init__(self, task_config, *, horizon):
        self.action_spec = (-np.ones(7), np.ones(7))
        self.config = task_config
        self.steps = 0
        self.sample = 0

    def reset(self):
        self.steps = 0
        self.sample = np.random.uniform()

    def step(self, action):
        self.steps += 1
        if self.config == 'step_failure' and self.steps > 1:
            raise ValueError('deliberate step failure')
        obs = {name: np.zeros(size, dtype=np.float32) for name, size in STATE_FIELDS}
        obs['robot0_joint_pos'][:2] = [self.sample, self.steps]
        success = self.config == 'success' and self.steps == 2
        obs.update(task_reward=1.0, task_complete=success,
                   task_metrics={'grasped': success, 'lift_success': success})
        return obs

    def close(self):
        pass


def failing_factory(task_config, *, horizon):
    raise ValueError('deliberate startup failure')


class ParallelTests(unittest.TestCase):
    def test_gae_isolated_across_workers_and_episode_resets(self):
        buffers = [RolloutBuffer() for _ in range(3)]
        for env_id, rewards, dones in [(0, [1, 2, 3], [False, True, False]),
                                       (1, [10, 20], [False, False])]:
            for reward, done in zip(rewards, dones):
                buffers[env_id].add({'state': torch.tensor([env_id], dtype=torch.float32)},
                                    torch.zeros(7), torch.tensor(0.), torch.tensor(0.), reward, done)
        rollout, advantages, returns = combine_rollouts(
            buffers, torch.tensor([5., 5., 999.]), SimpleNamespace(gamma=1., gae_lambda=1.))
        torch.testing.assert_close(returns, torch.tensor([3., 2., 8., 35., 25.]))
        torch.testing.assert_close(advantages, returns)
        self.assertEqual(rollout['features']['state'].flatten().tolist(), [0, 0, 0, 1, 1])

    def test_default_uses_original_trainer(self):
        config = PPOConfig()
        self.assertEqual(config.num_envs, 1)
        with patch('v3.ppo.train_base_ppo', return_value='checkpoint') as original:
            result = train_ppo(object(), config, Path('unused'), device='cpu')
        self.assertEqual(result, 'checkpoint')
        original.assert_called_once()

    def test_invalid_worker_counts(self):
        for count in (0, -1, 1.5, 1025):
            with self.assertRaises(ValueError):
                PPOConfig(num_envs=count).validate()

    def test_workers_have_independent_resets_and_seed_streams(self):
        pool = ParallelSimulators(2, None, 2, 10, simulator_factory=FakeStateSimulator)
        try:
            initial = pool.states.copy()
            np.testing.assert_allclose(initial[:, 0], [np.random.RandomState(i).uniform() for i in (10, 11)])
            # Advance only worker zero; worker one must remain unchanged.
            first = pool.step([0], [np.zeros(7)])[0]
            self.assertFalse(first[2])
            np.testing.assert_array_equal(pool.states[1], initial[1])
            result = pool.step([0], [np.zeros(7)])[0]
            self.assertTrue(result[2])
            self.assertEqual(result[3]['seed'], 10)
            self.assertEqual(result[3]['length'], 2)  # bootstrap earns no return
            self.assertEqual(result[3]['return'], 2)
            self.assertAlmostEqual(pool.states[0, 0], np.random.RandomState(12).uniform(), places=6)
            self.assertEqual(pool.states[0, 1], 1)  # next episode bootstrap
            np.testing.assert_array_equal(pool.states[1], initial[1])
        finally:
            pool.close()
        self.assertTrue(all(not p.is_alive() for p in pool.processes))

    def test_success_autoresets_before_time_limit(self):
        pool = ParallelSimulators(2, 'success', 1000, 20, simulator_factory=FakeStateSimulator)
        try:
            results = pool.step([0, 1], np.zeros((2, 7)))
            for result in results:
                self.assertTrue(result[2])
                self.assertTrue(result[3]['success'])
                self.assertEqual(result[3]['length'], 1)
            np.testing.assert_array_equal(pool.states[:, 1], [1, 1])
        finally:
            pool.close()

    def test_worker_errors_reach_parent_and_processes_close(self):
        before = {child.pid for child in mp.active_children()}
        with self.assertRaisesRegex(RuntimeError, 'deliberate startup failure'):
            ParallelSimulators(2, None, 2, 10, simulator_factory=failing_factory)
        self.assertEqual({child.pid for child in mp.active_children()}, before)
        pool = ParallelSimulators(2, 'step_failure', 2, 10, simulator_factory=FakeStateSimulator)
        try:
            with self.assertRaisesRegex(RuntimeError, 'deliberate step failure'):
                pool.step([0, 1], np.zeros((2, 7)))
        finally:
            pool.close()
        self.assertTrue(all(not p.is_alive() for p in pool.processes))


if __name__ == '__main__':
    unittest.main()
