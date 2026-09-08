"""Trajectory-level splits and timestep datasets for v4 demonstrations."""

from __future__ import annotations

import argparse
import bisect
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from torch.utils.data import Dataset

from v1.model import PRIVILEGED_POLICY_KEYS


@dataclass(frozen=True)
class TrajectoryInfo:
    path: str
    seed: int
    steps: int


def inspect_trajectory(path: Path) -> TrajectoryInfo:
    """Validate a successful trajectory and return its split metadata."""
    path = path.expanduser().resolve()
    with h5py.File(path, "r") as trajectory:
        if not bool(trajectory.attrs.get("success", False)):
            raise ValueError(f"Trajectory is not marked successful: {path}")
        if "actions" not in trajectory:
            raise ValueError(f"Trajectory has no actions dataset: {path}")
        steps = int(trajectory["actions"].shape[0])
        if trajectory["actions"].shape != (steps, 7):
            raise ValueError(f"Expected actions shaped (T, 7): {path}")
        missing = [
            key
            for key in PRIVILEGED_POLICY_KEYS
            if f"observations/{key}" not in trajectory
        ]
        if missing:
            raise ValueError(f"Trajectory {path} is missing observations: {missing}")
        mismatched = [
            key
            for key in PRIVILEGED_POLICY_KEYS
            if trajectory[f"observations/{key}"].shape[0] != steps
        ]
        if mismatched:
            raise ValueError(
                f"Trajectory {path} has observation/action length mismatch: "
                f"{mismatched}"
            )
        return TrajectoryInfo(
            path=str(path),
            seed=int(trajectory.attrs["seed"]),
            steps=steps,
        )


def discover_trajectories(root: Path) -> list[TrajectoryInfo]:
    paths = sorted(root.expanduser().resolve().glob("*/trajectory.h5"))
    if not paths:
        raise FileNotFoundError(f"No trajectory.h5 files found under {root}")
    return [inspect_trajectory(path) for path in paths]


def split_trajectories(
    trajectories: list[TrajectoryInfo],
    *,
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Split whole episodes deterministically; never split adjacent timesteps."""
    if len(trajectories) < 2:
        raise ValueError("At least two trajectories are required for train/test")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")

    ordered = sorted(trajectories, key=lambda item: (item.seed, item.path))
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    test_count = min(
        len(ordered) - 1,
        max(1, int(round(len(ordered) * test_fraction))),
    )
    test_indices = set(int(index) for index in permutation[:test_count])
    train = [item for index, item in enumerate(ordered) if index not in test_indices]
    test = [item for index, item in enumerate(ordered) if index in test_indices]
    return {
        "schema_version": 1,
        "split_seed": seed,
        "test_fraction": test_fraction,
        "train": [asdict(item) for item in train],
        "test": [asdict(item) for item in test],
        "summary": {
            "train_trajectories": len(train),
            "test_trajectories": len(test),
            "train_samples": sum(item.steps for item in train),
            "test_samples": sum(item.steps for item in test),
        },
    }


def write_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{output_path.stem}-",
        suffix=".json",
        dir=output_path.parent,
        delete=False,
    ) as temporary_file:
        json.dump(manifest, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(output_path)
    return output_path


class H5TransitionDataset(Dataset[dict[str, Any]]):
    """Lazy map-style dataset where every item is one teacher transition."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._files: dict[str, h5py.File] = {}
        if not entries:
            raise ValueError("Dataset split cannot be empty")
        self.entries = entries
        self.cumulative_steps: list[int] = []
        total = 0
        for entry in entries:
            total += int(entry["steps"])
            self.cumulative_steps.append(total)

    def __len__(self) -> int:
        return self.cumulative_steps[-1]

    def _locate(self, index: int) -> tuple[dict[str, Any], int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        trajectory_index = bisect.bisect_right(self.cumulative_steps, index)
        start = 0 if trajectory_index == 0 else self.cumulative_steps[trajectory_index - 1]
        return self.entries[trajectory_index], index - start

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry, timestep = self._locate(index)
        path = str(entry["path"])
        trajectory = self._files.get(path)
        if trajectory is None:
            trajectory = h5py.File(path, "r")
            self._files[path] = trajectory
        return {
            "observations": {
                key: np.asarray(trajectory[f"observations/{key}"][timestep])
                for key in PRIVILEGED_POLICY_KEYS
            },
            "action": np.asarray(trajectory["actions"][timestep]),
        }

    def close(self) -> None:
        for trajectory in self._files.values():
            trajectory.close()
        self._files.clear()

    def __del__(self) -> None:
        for trajectory in getattr(self, "_files", {}).values():
            trajectory.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", type=Path, default=Path("v4/trajectories"))
    parser.add_argument("--output", type=Path, default=Path("v4/splits/default.json"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = discover_trajectories(args.trajectories_dir)
    manifest = split_trajectories(
        trajectories,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    output = write_manifest(manifest, args.output)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    print(f"split_manifest={output}")


if __name__ == "__main__":
    main()
