"""Disturbance semantics, label alignment, frozen splits, and resumable collection."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from v4.collect_robustness import collect_slot, make_plan
from v4.perturbations import ActionPerturber, CATEGORIES, PerturbationConfig
from v4.record_trajectory import generate_one_success
from v4.teacher import TeacherConfig
from v4.test_bc import _write_trajectory
from v4.test_teacher import FakeSimulator


class PerturbationTests(unittest.TestCase):
    def test_reproducible_bounded_translation_only_and_independent_rng(self):
        actions = np.random.default_rng(1).uniform(-1, 1, (400, 7)).astype(np.float32)
        for category in CATEGORIES:
            with self.subTest(category=category):
                first = ActionPerturber(PerturbationConfig(category=category), 12)
                second = ActionPerturber(PerturbationConfig(category=category), 12)
                np.random.seed(15)
                expected_global = np.random.random()
                np.random.seed(15)
                for action in actions:
                    original = action.copy()
                    executed, meta = first.apply(action)
                    other, other_meta = second.apply(action)
                    np.testing.assert_array_equal(executed, other)
                    np.testing.assert_array_equal(action, original)
                    np.testing.assert_array_equal(executed[3:], action[3:])
                    np.testing.assert_array_equal(meta['delta'], executed - action)
                    self.assertTrue(np.isfinite(executed).all())
                    self.assertLessEqual(np.abs(executed).max(), 1)
                self.assertEqual(np.random.random(), expected_global)
                self.assertEqual(first.changed_steps == 0, category == 'clean')

    def test_drop_bursts_and_delay_keep_gripper_current(self):
        dropped = ActionPerturber(PerturbationConfig(category='dropped_commands', drop_probability=1,
                                                    drop_min_steps=3, drop_max_steps=3), 0)
        delayed = ActionPerturber(PerturbationConfig(category='delayed_commands', delay_steps=2), 0)
        actions = [np.array([i / 10, 0.2, 0.3, 0, 0, 0, (-1)**i], dtype=np.float32) for i in range(6)]
        for i, action in enumerate(actions):
            executed, meta = dropped.apply(action)
            np.testing.assert_array_equal(executed[:3], 0)
            self.assertEqual(executed[6], action[6])
            self.assertEqual(meta['event_step'], i % 3)
            self.assertEqual(meta['event_id'], i // 3)
            executed, _ = delayed.apply(action)
            np.testing.assert_array_equal(executed[:3], actions[i-2][:3] if i >= 2 else 0)
            self.assertEqual(executed[6], action[6])
        reset = ActionPerturber(delayed.config, 0)
        np.testing.assert_array_equal(reset.apply(actions[-1])[0][:3], 0)

    def test_bias_is_constant_within_burst_and_gaussian_is_clipped(self):
        bias = ActionPerturber(PerturbationConfig(category='sustained_bias', bias_probability=1,
                                                 bias_min_steps=5, bias_max_steps=5), 1)
        outputs = [bias.apply(np.zeros(7, dtype=np.float32))[0] for _ in range(10)]
        for i in range(1, 5):
            np.testing.assert_array_equal(outputs[0], outputs[i])
        self.assertAlmostEqual(float(np.linalg.norm(outputs[0][:3])), 0.02, places=7)
        self.assertFalse(np.array_equal(outputs[0], outputs[5]))
        gaussian = ActionPerturber(PerturbationConfig(category='gaussian_noise'), 1)
        for _ in range(100):
            self.assertLessEqual(np.abs(gaussian.apply(np.zeros(7))[0][:3]).max(), 0.04500001)

    def test_invalid_settings_and_actions(self):
        for kwargs in ({'category': 'unknown'}, {'gaussian_std': float('nan')},
                       {'drop_probability': 1.1}, {'delay_steps': 0}, {'bias_min_steps': 11}):
            with self.assertRaises(ValueError):
                PerturbationConfig(**kwargs)
        perturber = ActionPerturber(PerturbationConfig(), 0)
        for action in (np.zeros(6), np.full(7, np.nan), np.full(7, 2)):
            with self.assertRaises(ValueError):
                perturber.apply(action)

    def test_recorded_targets_are_teacher_actions_executed_actions_drive_next_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('v4.record_trajectory.Simulator', FakeSimulator):
                result = generate_one_success(root / 'saved', name='alignment', seed=7,
                    max_attempts=1, max_steps=500, teacher_config=TeacherConfig(),
                    perturbation_config=PerturbationConfig(category='gaussian_noise', gaussian_std=0.001),
                    perturbation_seed=11, save_video=False, scratch_dir=root / 'scratch', verbose=False)['result']
            self.assertIsNone(result['video'])
            self.assertGreater(result['perturbed_steps'], 0)
            self.assertFalse(list((root / 'scratch').iterdir()))
            with h5py.File(result['trajectory'], 'r') as f:
                self.assertEqual(f.attrs['category'], 'gaussian_noise')
                intended, executed = f['actions'][:], f['executed_actions'][:]
                np.testing.assert_array_equal(executed - intended, f['perturbation/delta'][:])
                self.assertFalse(np.array_equal(intended, executed))
                np.testing.assert_array_equal(executed[:, 3:], intended[:, 3:])
                positions = f['observations/robot0_eef_pos'][:]
                np.testing.assert_allclose(positions[1:], positions[:-1] + executed[:-1, :3] * 0.05, atol=1e-7)
                np.testing.assert_allclose(f['terminal_observation/robot0_eef_pos'][:],
                                           positions[-1] + executed[-1, :3] * 0.05, atol=1e-7)
                self.assertTrue(f['terminal_observation'].attrs['task_complete'])
                # Normalized movements are bounded by the unchanged teacher limit.
                self.assertLessEqual(np.abs(intended[:, :3]).max(), 0.6000001)

    def test_plan_has_exact_category_splits_no_seed_leakage_and_ten_videos(self):
        source = {'train': [{'seed': 1}], 'validation': [{'seed': 2}], 'test': [{'seed': 3}]}
        plan = make_plan(source)
        self.assertEqual(plan, make_plan(source))
        self.assertEqual(len(plan['slots']), 300)
        self.assertEqual(sum(s['save_video'] for s in plan['slots']), 10)
        seeds = [c['seed'] for s in plan['slots'] for c in s['candidates']]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertFalse(set(seeds) & set(plan['excluded_seeds']))
        for category in CATEGORIES:
            slots = [s for s in plan['slots'] if s['category'] == category]
            self.assertEqual(len(slots), 100 if category == 'clean' else 50)
            self.assertEqual(sum(s['partition'] == 'validation' for s in slots), len(slots) // 10)

    def test_slot_retries_failures_and_recovers_ledger_without_duplicate_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slot = {'id': 'clean-001', 'category': 'clean', 'partition': 'train', 'save_video': False,
                    'candidates': [{'seed': 7, 'perturbation_seed': 1}, {'seed': 8, 'perturbation_seed': 2}]}
            calls = []
            def generate(output, **kwargs):
                seed = kwargs['seed']
                calls.append(seed)
                run = output / str(seed)
                result = None
                if seed == 8:
                    _write_trajectory(run / 'trajectory.h5', seed=seed, steps=2)
                    result = {'trajectory': str(run / 'trajectory.h5'), 'seed': seed, 'success': True}
                else:
                    run.mkdir(parents=True)
                metrics = {'requested_seed': seed, 'run_dir': str(run), 'result': result,
                           'attempts': [{'success': result is not None, 'failure_reason': 'test' if seed == 7 else None}]}
                (run / 'metrics.json').write_text(json.dumps(metrics))
                if result is None:
                    raise RuntimeError('Teacher did not produce a successful trajectory in 1 attempts')
                return metrics
            with patch('v4.collect_robustness.generate_one_success', side_effect=generate):
                first = collect_slot(root, slot, PerturbationConfig().to_dict(), root / 'scratch')
                self.assertEqual(len(first['attempts']), 2)
                self.assertEqual(collect_slot(root, slot, PerturbationConfig().to_dict(), root / 'scratch'), first)
                (root / 'jobs' / 'clean-001.json').unlink()
                recovered = collect_slot(root, slot, PerturbationConfig().to_dict(), root / 'scratch')
                self.assertEqual(recovered['result'], first['result'])
            self.assertEqual(calls, [7, 8])


if __name__ == '__main__':
    unittest.main()
