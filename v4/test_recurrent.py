"""Persistent memory, causal sequence equivalence, padding, and checkpoint tests."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from v4.model import load_policy
from v4.recurrent import SpatialLSTMPolicy, StateLSTMPolicy, load_state_lstm
from v4.train_recurrent import chunks, trajectory_groups, masked_loss, fit


class RecurrentTest(unittest.TestCase):
    def test_state_sequence_matches_streaming_and_reset(self):
        torch.manual_seed(3)
        policy = StateLSTMPolicy(torch.zeros(54), torch.ones(54), hidden_dim=16).eval()
        values = torch.randn(1, 9, 54)
        with torch.no_grad():
            full, _ = policy.sequence({'state': values})
            online = torch.cat([policy.predict_state(values[:, t]) for t in range(9)])
            torch.testing.assert_close(full[0].tanh(), online, atol=1e-6, rtol=1e-5)
            changed = values.clone()
            changed[:, :6] += 5
            different, _ = policy.sequence({'state': changed})
            self.assertFalse(torch.allclose(full[:, -1], different[:, -1]))
            # Future observations cannot affect earlier actions.
            changed = values.clone()
            changed[:, 5:] += 10
            different, _ = policy.sequence({'state': changed})
            torch.testing.assert_close(full[:, :5], different[:, :5])
            policy.reset_history()
            torch.testing.assert_close(policy.predict_state(values[:, 0]), full[:, 0].tanh())

    def test_visual_streaming_and_checkpoint(self):
        torch.manual_seed(2)
        policy = SpatialLSTMPolicy(pretrained=False, embedding_dim=16, hidden_dim=16).eval()
        features = {'front': torch.randn(1, 6, 8192), 'wrist': torch.randn(1, 6, 8192),
                    'depth': torch.randn(1, 6, policy.depth_backbone.output_dim),
                    'proprio': torch.randn(1, 6, 16)}
        policy.extract_frozen_features = lambda observation: {k: v[0, observation['index']] for k, v in features.items()}
        with torch.no_grad():
            logits, _ = policy.sequence(features)
            online = torch.stack([torch.tensor(policy.predict({'index': t})) for t in range(6)])
            torch.testing.assert_close(logits[0].tanh(), online, atol=1e-6, rtol=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pt'
            torch.save({'policy_type': 'spatial_lstm', 'model_config': policy.get_model_config(),
                        'policy_state_dict': policy.state_dict()}, path)
            restored = load_policy(path)
            self.assertIsNone(restored._hidden)
            with torch.no_grad():
                expected, _ = restored.sequence(features)
            torch.testing.assert_close(expected, logits)

    def test_chunk_coverage_padding_and_hidden_carry(self):
        entries = [{'steps': 3}, {'steps': 7}, {'steps': 2}]
        x = torch.randn(12, 4)
        y = torch.randn(12, 7).tanh()
        covered = []
        model = StateLSTMPolicy(torch.zeros(4), torch.ones(4), hidden_dim=8).eval()
        with torch.no_grad():
            for group in trajectory_groups(entries, [1, 0, 2], 2):
                hidden = None
                outputs = []
                for features, target, mask, indices in chunks({'state': x}, y, group, 2, 'cpu'):
                    self.assertTrue(torch.equal(target[mask], y[indices[mask]]))
                    covered.extend(indices[mask].tolist())
                    logits, hidden = model.sequence(features, hidden)
                    hidden = tuple(value.detach() for value in hidden)
                    outputs.append(logits)
                joined = torch.cat(outputs, dim=1)
                for row, (start, length) in enumerate(group):
                    full, _ = model.sequence({'state': x[start:start + length].unsqueeze(0)})
                    torch.testing.assert_close(joined[row, :length], full[0], atol=1e-6, rtol=1e-5)
        self.assertEqual(sorted(covered), list(range(12)))
        logits = torch.randn(2, 3, 7, requires_grad=True)
        targets = torch.randn(2, 3, 7)
        mask = torch.tensor([[True, True, True], [True, False, False]])
        masked_loss(logits, targets, mask).backward()
        self.assertTrue(bool((logits.grad[~mask] == 0).all()))

    def test_recurrent_training_and_state_reload(self):
        torch.manual_seed(1)
        policy = StateLSTMPolicy(torch.zeros(4), torch.ones(4), hidden_dim=8)
        features = {'state': torch.randn(12, 4)}
        actions = torch.randn(12, 7).tanh()
        data = {part: (features, actions) for part in ('train', 'validation')}
        entries = [{'steps': 5}, {'steps': 7}]
        manifest = {'train': entries, 'validation': entries}
        args = SimpleNamespace(learning_rate=.001, epochs=2, chunk_length=3,
                               batch_trajectories=2, device='cpu', mode='state')
        before = policy.lstm.weight_ih_l0.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit(policy, data, manifest, args, root)
            self.assertFalse(torch.equal(before, policy.lstm.weight_ih_l0))
            restored = load_state_lstm(root / 'final.pt')
            with torch.no_grad():
                a, _ = policy.sequence({'state': features['state'].unsqueeze(0)})
                b, _ = restored.sequence({'state': features['state'].unsqueeze(0)})
            torch.testing.assert_close(a, b)
            self.assertIsNone(restored._hidden)


if __name__ == '__main__':
    unittest.main()
