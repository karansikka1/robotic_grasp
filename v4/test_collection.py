"""Collection restart, seed exclusion, and failure accounting."""

from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v4.collect_trajectories import collect
from v4.test_bc import _write_trajectory


class ImmediatePool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, *args):
        future = Future()
        future.set_result(function(*args))
        return future


class CollectionTests(unittest.TestCase):
    def test_target_resume_excludes_successful_and_failed_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_trajectory(root / 'old' / 'trajectory.h5', seed=1, steps=2)
            calls = []

            def attempt(root, seed):
                calls.append(seed)
                success = len(calls) != 1
                if success:
                    _write_trajectory(root / str(seed) / 'trajectory.h5', seed=seed, steps=2)
                return {'seed': seed, 'success': success}

            with patch('v4.collect_trajectories.ProcessPoolExecutor', ImmediatePool), \
                    patch('v4.collect_trajectories._attempt', attempt):
                path = collect(root, target_total=4, workers=2, seed=9)
                manifest = json.loads(path.read_text())
                self.assertEqual(manifest['successes'], 3)
                self.assertEqual(len(manifest['attempts']), 4)
                self.assertIsNone(collect(root, target_total=4, workers=2, seed=9))
                collect(root, target_total=5, workers=2, seed=9)
            self.assertEqual(len(calls), len(set(calls)))
            self.assertNotIn(1, calls)

    def test_attempt_limit_and_invalid_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                collect(Path(directory), workers=0)
            with patch('v4.collect_trajectories.ProcessPoolExecutor', ImmediatePool), \
                    patch('v4.collect_trajectories._attempt',
                          side_effect=lambda root, seed: {'seed': seed, 'success': False}):
                with self.assertRaisesRegex(RuntimeError, 'Collected only 0/2'):
                    collect(Path(directory), count=2, max_attempts=2, workers=2)


if __name__ == '__main__':
    unittest.main()
