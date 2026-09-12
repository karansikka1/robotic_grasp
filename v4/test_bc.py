"""Tests for trajectory splitting and privileged behavior-cloning inputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from v1.model import PRIVILEGED_POLICY_KEYS, PrivilegedPPOPolicy
from v4.data import H5TransitionDataset, discover_trajectories, split_trajectories
from v4.train_bc import (
    BCConfig,
    extract_frozen_feature_batch,
    train_behavior_cloning,
    action_losses,
)


def _write_trajectory(path: Path, *, seed: int, steps: int) -> None:
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    shapes = {
        "robot0_joint_pos": (7,),
        "robot0_eef_pos": (3,),
        "robot0_eef_quat": (4,),
        "robot0_gripper_qpos": (2,),
        "frontview_image": (32, 32, 3),
        "robot0_eye_in_hand_image": (32, 32, 3),
        "frontview_depth": (32, 32, 1),
    }
    with h5py.File(path, "w") as trajectory:
        trajectory.attrs["success"] = True
        trajectory.attrs["seed"] = seed
        trajectory.create_dataset("actions", data=rng.uniform(-1, 1, (steps, 7)).astype(np.float32))
        observations = trajectory.create_group("observations")
        for key in PRIVILEGED_POLICY_KEYS:
            shape = (steps, *shapes[key])
            if "image" in key:
                values = rng.integers(0, 256, shape, dtype=np.uint8)
            else:
                values = rng.random(shape).astype(np.float32)
            observations.create_dataset(key, data=values)


class BehaviorCloningDataTest(unittest.TestCase):
    def test_binary_gripper_loss_recovers_from_saturated_wrong_predictions(self):
        logits = torch.tensor([[0., 0., 0., 0., 0., 0., -15.],
                               [0., 0., 0., 0., 0., 0., 15.]], requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, -1] = torch.tensor([1., -1.])
        loss = action_losses(logits, targets, 'bce').mean()
        loss.backward()
        self.assertLess(float(logits.grad[0, -1]), -.05)
        self.assertGreater(float(logits.grad[1, -1]), .05)
        self.assertTrue(torch.equal(logits.grad[:, :6], torch.zeros(2, 6)))
        torch.testing.assert_close(action_losses(logits, targets, 'smooth_l1'),
                                   torch.nn.functional.smooth_l1_loss(logits.tanh(), targets, reduction='none'))
        targets[:, -1] = .5
        with self.assertRaisesRegex(ValueError, 'exactly -1/\\+1'):
            action_losses(logits, targets, 'bce')

    def test_episode_split_and_batched_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, steps in ((10, 2), (20, 3), (30, 4)):
                _write_trajectory(
                    root / f"episode-{seed}" / "trajectory.h5",
                    seed=seed,
                    steps=steps,
                )
            trajectories = discover_trajectories(root)
            first = split_trajectories(trajectories, test_fraction=1 / 3, seed=7)
            second = split_trajectories(trajectories, test_fraction=1 / 3, seed=7)
            self.assertEqual(first, second)
            train_paths = {entry["path"] for entry in first["train"]}
            test_paths = {entry["path"] for entry in first["test"]}
            self.assertFalse(train_paths & test_paths)
            self.assertEqual(len(train_paths), 2)
            self.assertEqual(len(test_paths), 1)

            dataset = H5TransitionDataset(first["train"])
            try:
                self.assertEqual(len(dataset), first["summary"]["train_samples"])
                batch = next(iter(DataLoader(dataset, batch_size=2)))
                policy = PrivilegedPPOPolicy(
                    embedding_dim=16,
                    hidden_dim=16,
                    image_size=32,
                    pretrained=False,
                ).eval()
                batched = extract_frozen_feature_batch(policy, batch["observations"])
                individual = [
                    policy.extract_frozen_features(
                        {
                            key: batch["observations"][key][index].numpy()
                            for key in PRIVILEGED_POLICY_KEYS
                        }
                    )
                    for index in range(2)
                ]
                for name, values in batched.items():
                    expected = torch.stack([item[name] for item in individual])
                    self.assertTrue(torch.allclose(values, expected, atol=1e-5))
                self.assertEqual(batch["action"].shape, (2, 7))
            finally:
                dataset.close()

    def test_training_loop_saves_compatible_checkpoint(self) -> None:
        policy = PrivilegedPPOPolicy(
            embedding_dim=16,
            hidden_dim=16,
            image_size=32,
            pretrained=False,
        )
        feature_dim = policy.rgb_backbone.output_dim
        generator = torch.Generator().manual_seed(5)

        def features(samples: int) -> dict[str, torch.Tensor]:
            return {
                "front": torch.randn(samples, feature_dim, generator=generator),
                "wrist": torch.randn(samples, feature_dim, generator=generator),
                "depth": torch.randn(samples, feature_dim, generator=generator),
                "proprio": torch.randn(samples, 16, generator=generator),
            }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = train_behavior_cloning(
                policy,
                features(8),
                torch.rand(8, 7, generator=generator) * 2 - 1,
                features(4),
                torch.rand(4, 7, generator=generator) * 2 - 1,
                BCConfig(epochs=1, batch_size=4, feature_batch_size=2),
                Path(directory),
                device=torch.device("cpu"),
            )
            self.assertTrue(checkpoint.exists())
            self.assertTrue((Path(directory) / "final.pt").exists())
            restored = PrivilegedPPOPolicy.from_checkpoint(checkpoint)
            self.assertEqual(restored.get_model_config(), policy.get_model_config())


if __name__ == "__main__":
    unittest.main()
