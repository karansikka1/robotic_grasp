"""Causal history, split isolation, temporal learning, and rollout semantics."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from v2.model import FeatureHistory, TemporalPPOPolicy, load_policy
from v4.data import TrajectoryInfo, discover_trajectories, split_trajectories, validate_manifest
from v4.evaluate import (StableStackSimulator, DiagnosticStackSimulator,
                         evaluate_checkpoint, evaluate_training_seeds)
from v4.history import add_history
from v4.test_bc import _write_trajectory
from v4.train_bc import BCConfig, train_behavior_cloning


class HistoryBCTests(unittest.TestCase):
    def test_rollout_diagnostics_distinguish_contact_from_held_lift(self):
        from types import SimpleNamespace
        state = SimpleNamespace(positions=np.array([[0., 0., .8], [0., 0., .8], [0., 0., .8]]),
                                grasped=np.array([False, False, False]),
                                object_contacts=np.array([False, False, False]))
        simulator = DiagnosticStackSimulator.__new__(DiagnosticStackSimulator)
        simulator.success_hold_steps = 10
        with patch('motion_planning.simulator.Simulator.reset'), \
                patch('motion_planning.simulator.Simulator.step',
                      side_effect=lambda action: {'task_complete': False}), \
                patch('v4.evaluate.read_oracle_state', return_value=state):
            simulator.reset()
            simulator.step(np.zeros(7))
            state.grasped[1] = True
            state.positions[1, 2] += .06
            for _ in range(4):
                result = simulator.step(np.zeros(7))
                self.assertFalse(result['evaluation_metrics']['held_lift_red_green_blue'][1])
            state.grasped[1] = False
            simulator.step(np.zeros(7))
            state.grasped[1] = True
            for _ in range(5):
                result = simulator.step(np.zeros(7))
            self.assertTrue(result['evaluation_metrics']['held_lift_red_green_blue'][1])
            self.assertFalse(result['task_complete'])
            simulator.reset()
            result = simulator.step(np.zeros(7))
            self.assertEqual(result['evaluation_metrics']['bilateral_contact_steps_red_green_blue'], [0, 0, 0])

    def test_windows_are_causal_and_reset_at_trajectory_boundaries(self):
        raw = {'frame': torch.arange(5).float().unsqueeze(1)}
        actions = torch.arange(35).float().reshape(5, 7)
        entries = [{'steps': 3}, {'steps': 2}]
        result = add_history(raw, actions, entries, 3)
        self.assertEqual(result['frame'].squeeze(-1).tolist(),
                         [[0, 0, 0], [0, 0, 1], [0, 1, 2], [3, 3, 3], [3, 3, 4]])
        torch.testing.assert_close(result['previous_action'][0], torch.zeros(7))
        torch.testing.assert_close(result['previous_action'][3], torch.zeros(7))
        torch.testing.assert_close(result['previous_action'][2], actions[1])
        # Changing a target or a future action cannot alter the input at t=1.
        changed = actions.clone()
        changed[1:] += 100
        updated = add_history(raw, changed, entries, 3)
        torch.testing.assert_close(updated['previous_action'][1], result['previous_action'][1])

    def test_cached_inputs_match_online_history_exactly(self):
        class StubPolicy:
            history_length = 3
            device = 'cpu'
            def extract_frozen_features(self, obs):
                return {'frame': torch.tensor([float(obs['t'])])}
        policy = StubPolicy()
        actions = torch.linspace(-1, 1, 35).reshape(5, 7)
        cached = add_history({'frame': torch.arange(5).float().unsqueeze(1)}, actions,
                             [{'steps': 3}, {'steps': 2}], 3)
        history = FeatureHistory(policy)
        for t in range(5):
            if t == 3:
                history.reset()
            online = history.features({'t': t})
            for key in online:
                torch.testing.assert_close(online[key], cached[key][t])
            history.record_action(actions[t])

    def test_split_groups_repeated_seeds_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, seed in enumerate([10, 10, 20, 30, 40]):
                _write_trajectory(root / str(i) / 'trajectory.h5', seed=seed, steps=2)
            trajectories = discover_trajectories(root)
            split = split_trajectories(trajectories, test_fraction=.2, validation_fraction=.2, seed=0)
            validate_manifest(split)
            self.assertEqual(split, split_trajectories(trajectories, test_fraction=.2, validation_fraction=.2, seed=0))
            groups = [set(e['seed'] for e in split[name]) for name in ('train', 'validation', 'test')]
            self.assertFalse(groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
            duplicate = json.loads(json.dumps(split))
            duplicate['test'].append(duplicate['train'][0])
            with self.assertRaises(ValueError):
                validate_manifest(duplicate)
        with self.assertRaises(ValueError):
            split_trajectories([TrajectoryInfo('a', 1, 1), TrajectoryInfo('b', 1, 1)],
                               test_fraction=.2, validation_fraction=.2, seed=0)

    def test_temporal_fusion_is_trained_and_checkpoint_reloads(self):
        policy = TemporalPPOPolicy(embedding_dim=16, hidden_dim=16, image_size=32, pretrained=False)
        width = policy.rgb_backbone.output_dim
        raw = {k: torch.randn(10, width) for k in ('front', 'wrist', 'depth')}
        raw['proprio'] = torch.randn(10, 16)
        actions = torch.randn(10, 7).tanh()
        features = add_history(raw, actions, [{'steps': 10}], 3)
        temporal_before = policy.temporal_fusion[0].weight.detach().clone()
        critic_before = policy.critic[0].weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = train_behavior_cloning(
                policy, {k:v[:7] for k,v in features.items()}, actions[:7],
                {k:v[7:] for k,v in features.items()}, actions[7:],
                BCConfig(epochs=2, batch_size=4), Path(directory), device=torch.device('cpu'))
            self.assertFalse(torch.equal(temporal_before, policy.temporal_fusion[0].weight))
            self.assertTrue(torch.equal(critic_before, policy.critic[0].weight))
            restored = load_policy(path)
            self.assertIsInstance(restored, TemporalPPOPolicy)
            self.assertEqual(restored.history_length, 3)
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint['policy_type'], 'history')

    def test_evaluation_rejects_demonstration_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({'algorithm': 'behavior_cloning'}, root / 'best.pt')
            (root / 'split.json').write_text(json.dumps({'train': [{'seed': 1000000}], 'validation': [], 'test': []}))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                evaluate_checkpoint(root / 'best.pt', seed=1000000, episodes=1)

    def test_closed_loop_success_requires_consecutive_official_success(self):
        sim = StableStackSimulator.__new__(StableStackSimulator)
        sim.success_steps = 0
        sim.success_hold_steps = 10
        with patch('motion_planning.simulator.Simulator.step') as base:
            for flag in [True] * 9 + [False] + [True] * 9:
                base.return_value = {'task_complete': flag}
                self.assertFalse(sim.step(np.zeros(7))['task_complete'])
            base.return_value = {'task_complete': True}
            self.assertTrue(sim.step(np.zeros(7))['task_complete'])

    def test_training_rollouts_use_training_seeds_and_verify_initial_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (10, 20, 30):
                _write_trajectory(root / str(seed) / 'trajectory.h5', seed=seed, steps=2)
            entries = [asdict(entry) for entry in discover_trajectories(root)]
            (root / 'split.json').write_text(json.dumps({'train': entries[:1],
                                                        'validation': entries[1:2], 'test': entries[2:]}))
            import h5py
            from v1.model import PRIVILEGED_POLICY_KEYS
            with h5py.File(entries[0]['path']) as trajectory:
                initial = {key: trajectory[f'observations/{key}'][0] for key in PRIVILEGED_POLICY_KEYS}

            def evaluate(predict, output_dir, **kwargs):
                self.assertEqual(kwargs['seed'], 10)
                predict(initial)
                return {'output_dir': str(root), 'config': {},
                        'episodes': [{'success': False, 'seed': 10, 'steps_to_completion': None}]}

            with patch('v4.evaluate.load_policy'), patch('v4.evaluate.evaluate_policy', side_effect=evaluate):
                result = evaluate_training_seeds(root / 'best.pt', episodes=5)
            self.assertEqual(result['evaluation_partition'], 'train')
            self.assertEqual(result['summary']['episodes'], 1)
            self.assertTrue(result['summary']['all_initial_observations_match'])


if __name__ == '__main__':
    unittest.main()
