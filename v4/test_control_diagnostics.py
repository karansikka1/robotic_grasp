"""Check privileged feature packing and trajectory-local diagnostic selection."""
import unittest
import numpy as np
import torch
from v4.control_diagnostics import STAGES, StateBC, state_features, selected_training_entries
from v4.pose_bridge import RGBPositions
from v1.model import PROPRIO_KEYS


class ControlDiagnosticTest(unittest.TestCase):
    def test_offline_and_online_features_agree(self):
        rng = np.random.default_rng(3)
        positions = rng.normal(size=(4, 3, 3))
        eef = rng.normal(size=(4, 3))
        proprio = rng.normal(size=(4, 16))
        labels = list(STAGES[:4])
        batched = state_features(positions, eef, proprio, labels)
        for index in range(4):
            single = state_features(positions[index], eef[index], proprio[index], [labels[index]])
            np.testing.assert_array_equal(single[0], batched[index])
        np.testing.assert_allclose(batched[:, 28:37].reshape(4, 3, 3),
                                   positions - eef[:, None, :], rtol=1e-6)
        np.testing.assert_array_equal(batched[:, -len(STAGES):].sum(1), np.ones(4))
        with self.assertRaises(ValueError):
            state_features(positions[:1], eef[:1], proprio[:1], ['invalid'])

    def test_stage_removal_does_not_require_or_reveal_stage(self):
        values = state_features(np.ones((3, 3)), np.ones(3), np.ones(16), [None], use_stage=False)
        np.testing.assert_array_equal(values[0, -len(STAGES):], np.zeros(len(STAGES)))
        other = state_features(np.ones((3, 3)), np.ones(3), np.ones(16), [STAGES[-1]], use_stage=False)
        np.testing.assert_array_equal(values, other)

    def test_normalized_policy_roundtrip(self):
        model = StateBC(torch.zeros(54), torch.ones(54))
        x = torch.randn(5, 54)
        other = StateBC(torch.zeros(54), torch.ones(54))
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x), other(x))
        self.assertTrue(bool((model(x).abs() <= 1).all()))

    def test_rgb_position_encoder_online_matches_batch_without_privileged_inputs(self):
        torch.manual_seed(0)
        model = RGBPositions(pretrained=False).eval()
        obs = {key: torch.randn(2, size) for key, size in zip(PROPRIO_KEYS, (7, 3, 4, 2))}
        obs.update({key: torch.randint(256, (2, 128, 128, 3), dtype=torch.uint8)
                    for key in ('frontview_image', 'robot0_eye_in_hand_image')})
        encoded = model.encode(obs)
        for index in range(2):
            one = {key: value[index] for key, value in obs.items()}
            # Batched convolution accumulation can differ by a few float32 ULPs.
            online = model.encode(one)[0]
            torch.testing.assert_close(encoded[index], online, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(model(encoded[index:index + 1]), model(online.unsqueeze(0)),
                                       atol=1e-5, rtol=1e-4)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        other = RGBPositions(pretrained=False).eval()
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(encoded), other(encoded))

    def test_training_sample_is_order_independent(self):
        entries = [{'seed': seed, 'path': str(seed)} for seed in range(80)]
        a = selected_training_entries({'train': entries})
        b = selected_training_entries({'train': entries[::-1]})
        self.assertEqual(a, b)
        self.assertEqual(len({entry['seed'] for entry in a}), 5)


if __name__ == '__main__':
    unittest.main()
