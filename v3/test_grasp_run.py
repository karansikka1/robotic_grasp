"""Sustained-contact reward semantics and checkpoint initialization checks."""

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from v1.ppo import save_checkpoint
from v2.task import LiftEpisode, LiftTaskConfig
from v3.initialization import initialize_policy
from v3.model import StatePPOPolicy
from v3.ppo import PPOConfig
from v3.state import STATE_FIELDS


class GraspRunTests(unittest.TestCase):
    def test_brief_contacts_do_not_earn_grasp_and_streak_resets(self):
        episode = LiftEpisode(LiftTaskConfig(grasp_hold_steps=5), 0.8, 0.2)
        for contact in [True] * 4 + [False] + [True] * 4:
            self.assertEqual(episode.advance(contact, 0.8, 0.2), 0)
        self.assertFalse(episode.metrics()['grasped'])
        self.assertEqual(episode.advance(True, 0.8, 0.2), 1)
        self.assertEqual(episode.steps_to_grasp, 10)
        self.assertEqual(episode.metrics()['max_contact_streak_steps'], 5)
        self.assertEqual(episode.metrics()['contact_loss_count'], 1)
        episode.advance(False, 0.8, 0.2)
        self.assertFalse(episode.metrics()['bilateral_contact'])
        self.assertTrue(episode.metrics()['grasped'])  # episode milestone remains recorded
        self.assertEqual(episode.metrics()['contact_loss_count'], 2)
        for _ in range(6):
            self.assertEqual(episode.advance(True, 0.8, 0.2), 0)  # no regrasp farming
        self.assertEqual(episode.metrics()['max_contact_streak_steps'], 6)

    def test_new_episode_starts_without_contact_credit(self):
        episode = LiftEpisode(LiftTaskConfig(grasp_hold_steps=5), 0.8, 0.2)
        self.assertFalse(episode.metrics()['contact_seen'])
        self.assertFalse(episode.metrics()['bilateral_contact'])
        self.assertEqual(episode.metrics()['contact_steps'], 0)
        self.assertEqual(episode.metrics()['contact_streak_steps'], 0)

    def test_lift_reward_still_requires_held_height(self):
        episode = LiftEpisode(LiftTaskConfig(grasp_hold_steps=5), 0.8, 0.2)
        for _ in range(4):
            self.assertEqual(episode.advance(True, 0.8, 0.2), 0)
        self.assertEqual(episode.advance(True, 0.8, 0.2), 1)
        self.assertFalse(episode.success)
        for _ in range(4):
            self.assertEqual(episode.advance(True, 0.9, 0.2), 0)
        self.assertEqual(episode.advance(True, 0.9, 0.2), 5)
        self.assertTrue(episode.success)

    def test_old_checkpoints_keep_single_contact_semantics(self):
        old = asdict(LiftTaskConfig())
        del old['grasp_hold_steps']
        config = LiftTaskConfig.from_saved_config(old)
        self.assertEqual(config.grasp_hold_steps, 1)
        self.assertEqual(LiftEpisode(config, 0.8, 0.2).advance(True, 0.8, 0.2), 1)
        current = LiftTaskConfig(grasp_hold_steps=5)
        self.assertEqual(LiftTaskConfig.from_saved_config(asdict(current)), current)
        for value in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                LiftTaskConfig(grasp_hold_steps=value).validate()

    def test_warm_start_preserves_all_weights_and_architecture(self):
        policy = StatePPOPolicy(hidden_dim=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save_checkpoint(path, policy, torch.optim.Adam(policy.parameters()), PPOConfig(),
                            global_step=204800, update=100)
            restored, source = initialize_policy(path)
            self.assertEqual(restored.hidden_dim, 16)
            self.assertEqual(source['global_step'], 204800)
            self.assertFalse(source['optimizer_restored'])
            self.assertEqual(len(source['sha256']), 64)
            for key, value in policy.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))
            obs = {key: np.zeros(size, dtype=np.float32) for key, size in STATE_FIELDS}
            np.testing.assert_array_equal(policy.predict(obs), restored.predict(obs))
            with self.assertRaisesRegex(ValueError, 'hidden-dim'):
                initialize_policy(path, hidden_dim=128)
            config = PPOConfig(task=LiftTaskConfig(grasp_hold_steps=5), initialization=source)
            new_path = Path(directory) / 'new.pt'
            fresh_optimizer = torch.optim.Adam(restored.parameters())
            self.assertEqual(len(fresh_optimizer.state), 0)
            save_checkpoint(new_path, restored, fresh_optimizer, config, global_step=0, update=0)
            saved = torch.load(new_path, weights_only=False)
            self.assertEqual(saved['config']['task']['grasp_hold_steps'], 5)
            self.assertEqual(saved['config']['initialization'], source)
            self.assertEqual(saved['global_step'], 0)


if __name__ == '__main__':
    unittest.main()
