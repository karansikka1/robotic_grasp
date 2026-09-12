"""Temporal causality, frozen encoders, history, and checkpoint compatibility."""

import tempfile
import unittest
from pathlib import Path

import torch

from v1.model import PrivilegedPPOPolicy
from v2.model import TemporalPPOPolicy, FeatureHistory
from v4.history import add_history
from v4.model import FrozenResNetFeatures, TemporalTransformer, TransformerTemporalPolicy, load_policy
from v4.train_bc import BCConfig, _predict_actions, train_behavior_cloning
from v4.train_bc import extract_frozen_feature_batch
from v1.model import PRIVILEGED_POLICY_KEYS
from v4.test_bc import _write_trajectory
from v4.data import H5TransitionDataset, discover_trajectories
from dataclasses import asdict
from torch.utils.data import DataLoader
from v4.evaluate import _evaluation_policy


class TransformerBCTest(unittest.TestCase):
    def test_disabled_action_history_is_invariant_and_survives_reload(self):
        policy = self.make_policy(use_previous_action=False).eval()
        raw = self.features(policy, 3)
        features = add_history(raw, torch.randn(3, 7), [{'steps': 3}], 3)
        changed = {**features, 'previous_action': torch.randn(3, 7) * 100}
        with torch.no_grad():
            torch.testing.assert_close(_predict_actions(policy, features),
                                       _predict_actions(policy, changed), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pt'
            torch.save({'policy_type': 'transformer', 'model_config': policy.get_model_config(),
                        'policy_state_dict': policy.state_dict()}, path)
            restored = load_policy(path)
            self.assertFalse(restored.use_previous_action)
            with torch.no_grad():
                torch.testing.assert_close(_predict_actions(policy, features),
                                           _predict_actions(restored, changed), rtol=0, atol=0)
            # Old checkpoints omit this flag and retain the old behavior.
            config = policy.get_model_config()
            config.pop('use_previous_action')
            torch.save({'policy_type': 'transformer', 'model_config': config,
                        'policy_state_dict': policy.state_dict()}, path)
            self.assertTrue(load_policy(path).use_previous_action)

    def test_concat_retains_camera_identity(self):
        for fusion in ('sum', 'concat'):
            policy = self.make_policy(camera_fusion=fusion).eval()
            raw = self.features(policy, 3)
            features = add_history(raw, torch.zeros(3, 7), [{'steps': 3}], 3)
            swapped = {**features, 'front': features['wrist'], 'wrist': features['front']}
            with torch.no_grad():
                original = policy.fused_embedding(features)
                changed = policy.fused_embedding(swapped)
            if fusion == 'sum':
                torch.testing.assert_close(original, changed)
            else:
                self.assertFalse(torch.allclose(original, changed))

    def test_spatial_pool_keeps_locations_and_rejects_upsampling(self):
        for size in (2, 4):
            backbone = FrozenResNetFeatures(pretrained=False, pool_size=size)
            backbone.features = torch.nn.Identity()
            images = torch.zeros(1, 3, 8, 8)
            images[:, :, :2, :2] = 1
            first = backbone(images)
            shifted = backbone(images.flip(-1))
            self.assertEqual(first.shape, (1, 3 * size * size))
            torch.testing.assert_close(first.mean(), shifted.mean())
            self.assertFalse(torch.equal(first, shifted))
            with self.assertRaisesRegex(ValueError, 'too small'):
                backbone(torch.zeros(1, 3, 1, 1))

    def test_spatial_concat_cache_inference_and_checkpoint_agree(self):
        for size in (2, 4):
            policy = TransformerTemporalPolicy(
                pretrained=False, embedding_dim=16, hidden_dim=16, image_size=128,
                transformer_feedforward_dim=32, rgb_backbone='resnet18',
                camera_fusion='concat', rgb_pool_size=size,
            ).eval()
            self.assertEqual(policy.rgb_backbone.output_dim, 512 * size * size)
            self.assertEqual(policy.rgb_projection[0].in_features, 1024 * size * size)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_trajectory(root / 'demo' / 'trajectory.h5', seed=42, steps=2)
                dataset = H5TransitionDataset([asdict(item) for item in discover_trajectories(root)])
                try:
                    batch = next(iter(DataLoader(dataset, batch_size=2)))
                    cached = extract_frozen_feature_batch(policy, batch['observations'])
                    for index in range(2):
                        online = policy.extract_frozen_features({key: batch['observations'][key][index]
                                                                 for key in PRIVILEGED_POLICY_KEYS})
                        for key in online:
                            torch.testing.assert_close(cached[key][index], online[key], atol=1e-5, rtol=1e-4)
                    features = add_history(cached, batch['action'], dataset.entries, policy.history_length)
                    path = train_behavior_cloning(policy, features, batch['action'], features, batch['action'],
                                                   BCConfig(epochs=1, batch_size=2), root,
                                                   device=torch.device('cpu'))
                    restored = load_policy(path)
                    policy.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(_predict_actions(policy, features),
                                                   _predict_actions(restored, features))
                    self.assertEqual(restored.get_model_config(), policy.get_model_config())
                finally:
                    dataset.close()

    def test_zero_rotation_probe_preserves_other_actions_and_executed_history(self):
        policy = self.make_policy().eval()
        with torch.no_grad():
            policy.actor[-1].bias[3:6].fill_(.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pt'
            torch.save({'policy_type': 'transformer', 'model_config': policy.get_model_config(),
                        'policy_state_dict': policy.state_dict()}, path)
            constrained = _evaluation_policy(path, 'cpu', zero_rotation=True)
            raw = {key: value[0] for key, value in self.features(policy, 1).items()}
            policy.extract_frozen_features = lambda observation: raw
            constrained.extract_frozen_features = lambda observation: raw
            original = policy.predict({})
            action = constrained.predict({})
            self.assertTrue((action[3:6] == 0).all())
            torch.testing.assert_close(torch.tensor(original[[0, 1, 2, 6]]),
                                       torch.tensor(action[[0, 1, 2, 6]]))
            torch.testing.assert_close(constrained._prediction_history.previous_action,
                                       torch.tensor(action))

    def make_policy(self, **kwargs):
        return TransformerTemporalPolicy(pretrained=False, embedding_dim=16, hidden_dim=16,
                                         image_size=32, transformer_feedforward_dim=32, **kwargs)

    def features(self, policy, count):
        size = policy.rgb_backbone.output_dim
        return {**{key: torch.randn(count, size) for key in ('front', 'wrist', 'depth')},
                'proprio': torch.randn(count, 16)}

    def test_attention_is_causal_and_batched_inference_agrees(self):
        torch.manual_seed(11)
        encoder = TemporalTransformer(16, 4, 4, 2, 32, 0).eval()
        frames = torch.randn(2, 4, 16)
        changed = frames.clone()
        changed[:, 2:] += torch.randn_like(changed[:, 2:]) * 10
        with torch.no_grad():
            original = encoder(frames)
            modified = encoder(changed)
            torch.testing.assert_close(original[:, :2], modified[:, :2])
            self.assertFalse(torch.allclose(original[:, -1], modified[:, -1]))
            torch.testing.assert_close(original[0], encoder(frames[0]), atol=1e-5, rtol=1e-5)

    def test_online_history_matches_cached_windows_and_reset(self):
        policy = self.make_policy(history_length=4).eval()
        raw = self.features(policy, 6)
        actions = torch.randn(6, 7).tanh()
        cached = add_history(raw, actions, [{'steps': 3}, {'steps': 3}], 4)
        policy.extract_frozen_features = lambda obs: {key: value[obs['index']] for key, value in raw.items()}
        history = FeatureHistory(policy)
        with torch.no_grad():
            for index in range(6):
                if index == 3:
                    history.reset()
                actual = history.features({'index': index})
                expected = {key: value[index] for key, value in cached.items()}
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key])
                batched = {key: value.unsqueeze(0) for key, value in expected.items()}
                torch.testing.assert_close(_predict_actions(policy, actual),
                                           _predict_actions(policy, batched)[0], atol=1e-5, rtol=1e-5)
                history.record_action(actions[index])

    def test_training_updates_transformer_but_not_backbones_and_reloads(self):
        torch.manual_seed(1)
        policy = self.make_policy()
        raw = self.features(policy, 8)
        actions = torch.randn(8, 7).tanh()
        cached = add_history(raw, actions, [{'steps': 4}, {'steps': 4}], 3)
        before = {key: value.clone() for key, value in policy.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = train_behavior_cloning(policy, cached, actions, cached, actions,
                                          BCConfig(epochs=2, batch_size=4), Path(directory),
                                          device=torch.device('cpu'))
            after = policy.state_dict()
            for prefix in ('temporal_encoder.layers.', 'temporal_encoder.positions.', 'temporal_fusion.'):
                self.assertTrue(any(not torch.equal(value, before[key])
                                    for key, value in after.items() if key.startswith(prefix)))
            for key, value in after.items():
                if key.startswith(('rgb_backbone.', 'depth_backbone.', 'critic.')) or key == 'log_std':
                    torch.testing.assert_close(before[key], value, rtol=0, atol=0)
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved['policy_type'], 'transformer')
            restored = load_policy(path)
            self.assertIsInstance(restored, TransformerTemporalPolicy)
            self.assertEqual(restored.get_model_config(), policy.get_model_config())
            # Compare directly against saved weights, which may precede the last epoch.
            policy.load_state_dict(saved['policy_state_dict'])
            policy.eval()
            with torch.no_grad():
                torch.testing.assert_close(_predict_actions(policy, cached), _predict_actions(restored, cached))

    def test_loader_retains_previous_bc_architectures(self):
        with tempfile.TemporaryDirectory() as directory:
            for cls in (PrivilegedPPOPolicy, TemporalPPOPolicy):
                policy = cls(pretrained=False, embedding_dim=16, hidden_dim=16, image_size=32)
                path = Path(directory) / 'old.pt'
                torch.save({'model_config': policy.get_model_config(),
                            'policy_state_dict': policy.state_dict()}, path)
                self.assertIsInstance(load_policy(path), cls)

    def test_resnet_rgb_cache_matches_online_and_checkpoint_reloads(self):
        policy = self.make_policy(rgb_backbone='resnet18').eval()
        self.assertEqual(policy.rgb_backbone.output_dim, 512)
        self.assertEqual(policy.depth_backbone.output_dim, 576)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_trajectory(root / 'demo' / 'trajectory.h5', seed=42, steps=2)
            dataset = H5TransitionDataset([asdict(item) for item in discover_trajectories(root)])
            try:
                batch = next(iter(DataLoader(dataset, batch_size=2)))
                cached = extract_frozen_feature_batch(policy, batch['observations'])
                for index in range(2):
                    online = policy.extract_frozen_features({key: batch['observations'][key][index]
                                                             for key in PRIVILEGED_POLICY_KEYS})
                    for key in online:
                        torch.testing.assert_close(cached[key][index], online[key], atol=1e-5, rtol=1e-4)
                features = add_history(cached, batch['action'], dataset.entries, policy.history_length)
                before = {key: value.clone() for key, value in policy.rgb_backbone.state_dict().items()}
                path = train_behavior_cloning(policy, features, batch['action'], features, batch['action'],
                                               BCConfig(epochs=1, batch_size=2), root,
                                               device=torch.device('cpu'))
                for key, value in policy.rgb_backbone.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                restored = load_policy(path)
                policy.eval()
                with torch.no_grad():
                    torch.testing.assert_close(_predict_actions(policy, features),
                                               _predict_actions(restored, features))
                self.assertEqual(restored.rgb_backbone_name, 'resnet18')
            finally:
                dataset.close()


if __name__ == '__main__':
    unittest.main()
