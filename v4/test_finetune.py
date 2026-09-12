"""Check that visual BC learns encoder weights and reloads the same policy."""

import tempfile
import unittest
from pathlib import Path

import torch

from v1.model import PrivilegedPPOPolicy
from v2.model import TemporalPPOPolicy, load_policy
from v4.data import H5TransitionDataset, inspect_trajectory
from dataclasses import asdict
from v4.test_bc import _write_trajectory
from v4.train_bc import BCConfig, cache_frozen_features, train_behavior_cloning, _predict_actions
from v4.visual_inputs import cache_visual_inputs, select_feature_batch


class BackboneFineTuningTest(unittest.TestCase):
    def test_history_pixels_match_cached_features_and_update_backbones(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (1, 2):
                _write_trajectory(root / str(seed) / 'trajectory.h5', seed=seed, steps=3)
            dataset = H5TransitionDataset([
                asdict(inspect_trajectory(root / str(seed) / 'trajectory.h5')) for seed in (1, 2)
            ])
            try:
                policy = TemporalPPOPolicy(pretrained=False, image_size=32, embedding_dim=16, hidden_dim=16)
                frozen, targets = cache_frozen_features(policy, dataset, batch_size=3)
                pixels, pixel_targets = cache_visual_inputs(policy, dataset, batch_size=3)
                self.assertTrue(torch.equal(targets, pixel_targets))
                direct = select_feature_batch(policy, pixels, slice(None), 'cpu')
                for key in frozen:
                    torch.testing.assert_close(direct[key], frozen[key], atol=1e-5, rtol=1e-5)
                # Check an episode boundary including repeated reset images and zero action.
                self.assertEqual(pixels.indices[3].tolist(), [3, 3, 3])
                self.assertTrue(torch.equal(pixels.previous_action[3], torch.zeros(7)))
                weights = {key: value.detach().clone() for key, value in policy.state_dict().items()}
                checkpoint = train_behavior_cloning(
                    policy, pixels, targets, pixels, targets,
                    BCConfig(epochs=1, batch_size=3, finetune_backbone=True), root,
                    device=torch.device('cpu'),
                )
                for name in ('rgb_backbone', 'depth_backbone'):
                    self.assertTrue(any(not torch.equal(value, weights[key])
                                        for key, value in policy.state_dict().items()
                                        if key.startswith(name) and key.endswith('weight')))
                for key, value in policy.state_dict().items():
                    if 'running_' in key or 'num_batches_tracked' in key or key.startswith('critic.') or key == 'log_std':
                        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
                restored = load_policy(checkpoint)
                self.assertTrue(torch.load(checkpoint, weights_only=False)['config']['finetune_backbone'])
                a = _predict_actions(policy, select_feature_batch(policy, pixels, slice(None), 'cpu'))
                b = _predict_actions(restored, select_feature_batch(restored, pixels, slice(None), 'cpu'))
                torch.testing.assert_close(a, b)
            finally:
                dataset.close()

    def test_single_frame_finetuning_inputs_match_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'demo' / 'trajectory.h5'
            _write_trajectory(path, seed=3, steps=2)
            dataset = H5TransitionDataset([asdict(inspect_trajectory(path))])
            try:
                policy = PrivilegedPPOPolicy(pretrained=False, image_size=32, embedding_dim=16, hidden_dim=16)
                frozen, _ = cache_frozen_features(policy, dataset, batch_size=2)
                pixels, _ = cache_visual_inputs(policy, dataset, batch_size=2)
                direct = select_feature_batch(policy, pixels, slice(None), 'cpu')
                for key in frozen:
                    torch.testing.assert_close(frozen[key], direct[key])
            finally:
                dataset.close()


if __name__ == '__main__':
    unittest.main()
