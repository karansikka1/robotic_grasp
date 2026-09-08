"""Temporal input ordering, PPO replay, and checkpoint regression tests."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from v1.model import PrivilegedPPOPolicy
from v1.ppo import PPOConfig, save_checkpoint
from v2.model import FeatureHistory, TemporalPPOPolicy, load_policy
from v2.evaluation import evaluate_policy
from v2.test_task import FakeSimulator
from v2.task import GreenLiftSimulator


class StubPolicy:
    history_length = 3
    device = torch.device("cpu")

    def __init__(self):
        self.calls = 0

    def extract_frozen_features(self, observation):
        self.calls += 1
        return {"frame": torch.tensor([float(observation["value"])])}


class HistoryTests(unittest.TestCase):
    def test_order_previous_action_and_bootstrap_cache(self):
        policy = StubPolicy()
        history = FeatureHistory(policy)
        first = {"value": 1}
        features = history.features(first)
        self.assertEqual(features["frame"].flatten().tolist(), [1, 1, 1])
        self.assertTrue(torch.equal(features["previous_action"], torch.zeros(7)))
        history.record_action(torch.ones(7))
        second = {"value": 2}
        features = history.features(second)
        self.assertEqual(features["frame"].flatten().tolist(), [1, 1, 2])
        self.assertTrue(torch.equal(features["previous_action"], torch.ones(7)))
        self.assertIs(history.features(second), features)  # Bootstrap + next rollout.
        self.assertEqual(policy.calls, 2)
        third = history.features({"value": 3})
        self.assertEqual(third["frame"].flatten().tolist(), [1, 2, 3])
        # Old rollout entries cannot be mutated by later history updates.
        self.assertEqual(features["frame"].flatten().tolist(), [1, 1, 2])

    def test_reset_clears_history_and_previous_action(self):
        history = FeatureHistory(StubPolicy())
        history.features({"value": 1})
        history.record_action(torch.ones(7))
        history.reset()
        features = history.features({"value": 9})
        self.assertEqual(features["frame"].flatten().tolist(), [9, 9, 9])
        self.assertTrue(torch.equal(features["previous_action"], torch.zeros(7)))

    @staticmethod
    def features(policy):
        width = policy.rgb_backbone.output_dim
        return {
            "front": torch.randn(3, width),
            "wrist": torch.randn(3, width),
            "depth": torch.randn(3, width),
            "proprio": torch.randn(3, 16),
            "previous_action": torch.randn(7).tanh(),
        }

    def test_ppo_replays_stored_history_in_shuffled_batches(self):
        policy = TemporalPPOPolicy(pretrained=False, embedding_dim=16, hidden_dim=16)
        features = self.features(policy)
        with torch.no_grad():
            action, old_log_prob, old_value = policy.act(features)
            batch = {key: torch.stack([value, value]) for key, value in features.items()}
            new_log_prob, _, values = policy.evaluate_actions(batch, torch.stack([action, action]))
        torch.testing.assert_close(new_log_prob, old_log_prob.expand(2), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(values, old_value.expand(2))
        changed = dict(features, previous_action=features["previous_action"] + 0.5)
        self.assertFalse(torch.allclose(policy.fused_embedding(features), policy.fused_embedding(changed)))

    def test_checkpoint_round_trip_and_single_frame_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            for cls in (TemporalPPOPolicy, PrivilegedPPOPolicy):
                policy = cls(pretrained=False, embedding_dim=16, hidden_dim=16)
                checkpoint = Path(directory) / (cls.__name__ + ".pt")
                save_checkpoint(checkpoint, policy, torch.optim.Adam(policy.parameters()),
                                PPOConfig(), global_step=1, update=1)
                restored = load_policy(checkpoint)
                self.assertIs(type(restored), cls)
                self.assertEqual(restored.get_model_config(), policy.get_model_config())
                for key, value in policy.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[key])

    def test_evaluation_resets_prediction_without_overwriting_training_history(self):
        policy = TemporalPPOPolicy(pretrained=False, embedding_dim=16, hidden_dim=16)
        raw = {key: value[0] for key, value in self.features(policy).items() if key != "previous_action"}
        with patch.object(policy, "extract_frozen_features", return_value=raw):
            training = FeatureHistory(policy)
            observation = {}
            stored = training.features(observation)
            training.record_action(torch.ones(7))
            fake = FakeSimulator([(False, 0.8)] * 3)
            with patch("v2.task.Simulator", return_value=fake):
                simulator = GreenLiftSimulator()
            with tempfile.TemporaryDirectory() as directory:
                with patch("v2.evaluation.GreenLiftSimulator", return_value=simulator):
                    with patch.object(policy, "reset_history", wraps=policy.reset_history) as reset:
                        evaluate_policy(policy.predict, directory, episodes=2, max_steps=1, record_video=False)
                        self.assertEqual(reset.call_count, 2)
            self.assertIs(training.features(observation), stored)
            self.assertTrue(torch.equal(training.previous_action, torch.ones(7)))

    def test_invalid_history_length(self):
        for length in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                TemporalPPOPolicy(history_length=length, pretrained=False)


if __name__ == "__main__":
    unittest.main()
