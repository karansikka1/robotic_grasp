"""Evaluate a policy in the UltraTask simulator and save rollout artifacts.

The public entry point is :func:`evaluate_policy`. A policy is any callable that
accepts one observation mapping and returns a seven-dimensional action.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import av
import numpy as np

from motion_planning.simulator import IMAGE_HEIGHT, IMAGE_WIDTH, Simulator
from run_sim import compose_video_frame


CONTROL_FREQUENCY_HZ = 20
DEFAULT_MAX_STEPS = 5_00
DEFAULT_EVALUATION_EPISODES = 25
MINI_EVALUATION_EPISODES = 5 # Quick eval
DEFAULT_EVALUATION_SEED = 0
POLICY_OBSERVATION_KEYS = (
    "robot0_joint_pos",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "frontview_image",
    "robot0_eye_in_hand_image",
)


class SimulatorLike(Protocol):
    """The subset of ``Simulator`` used by the evaluation harness."""

    @property
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]: ...

    def reset(self) -> None: ...

    def step(self, action: np.ndarray) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


PolicyFn = Callable[[Mapping[str, Any]], Any]
ObservationAdapter = Callable[[Mapping[str, Any]], Mapping[str, Any]]
SimulatorFactory = Callable[[], SimulatorLike]
PolicyResetFn = Callable[[], None]


def standard_policy_observation(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only fields that the final policy is permitted to consume."""
    missing = [key for key in POLICY_OBSERVATION_KEYS if key not in observation]
    if missing:
        raise KeyError(f"Simulator observation is missing policy fields: {missing}")
    return {key: observation[key] for key in POLICY_OBSERVATION_KEYS}


class _VideoWriter:
    """Write an episode video to a temporary file, then move it into place."""

    def __init__(self, output_path: Path, fps: int) -> None:
        self.output_path = output_path
        self.fps = fps
        self.temporary_path: Path | None = None
        self.container: Any = None
        self.stream: Any = None

    def __enter__(self) -> "_VideoWriter":
        with tempfile.NamedTemporaryFile(
            prefix=f".{self.output_path.stem}.",
            suffix=".mp4",
            dir=self.output_path.parent,
            delete=False,
        ) as temporary_file:
            self.temporary_path = Path(temporary_file.name)

        self.container = av.open(
            str(self.temporary_path),
            mode="w",
            options={"movflags": "+faststart"},
        )
        self.stream = self.container.add_stream("libx264", rate=self.fps)
        self.stream.width = IMAGE_WIDTH * 2
        self.stream.height = IMAGE_HEIGHT
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"preset": "fast", "crf": "20"}
        return self

    def add_observation(self, observation: Mapping[str, Any]) -> None:
        if self.container is None:
            raise RuntimeError("Video writer is not open")
        image = compose_video_frame(dict(observation))
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.container is not None:
                if exc_type is None:
                    for packet in self.stream.encode():
                        self.container.mux(packet)
                self.container.close()
            if exc_type is None and self.temporary_path is not None:
                self.temporary_path.replace(self.output_path)
        finally:
            if self.temporary_path is not None:
                self.temporary_path.unlink(missing_ok=True)


class _NoVideoWriter:
    def __enter__(self) -> "_NoVideoWriter":
        return self

    def add_observation(self, observation: Mapping[str, Any]) -> None:
        del observation

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None


def _validated_action(
    action: Any,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> tuple[np.ndarray, bool]:
    action_array = np.asarray(action, dtype=np.float32)
    if action_array.shape != action_min.shape:
        raise ValueError(
            f"Policy returned action shape {action_array.shape}; "
            f"expected {action_min.shape}"
        )
    if not np.all(np.isfinite(action_array)):
        raise ValueError("Policy returned an action containing NaN or infinity")

    clipped = np.clip(action_array, action_min, action_max)
    return clipped, not np.array_equal(action_array, clipped)


def _remaining_simulator_steps(simulator: SimulatorLike) -> int | None:
    """Return steps remaining before robosuite terminates, when discoverable."""
    environment = getattr(simulator, "env", None)
    if environment is None or bool(getattr(environment, "ignore_done", False)):
        return None

    horizon = getattr(environment, "horizon", None)
    timestep = getattr(environment, "timestep", None)
    if horizon is None or timestep is None:
        return None
    return max(int(horizon) - int(timestep), 0)


def _create_run_directory(
    evaluation_dir: str | Path,
    run_name: str | None,
    run_uuid: str | None = None,
) -> tuple[Path, str]:
    """Create and return a uniquely named directory for one evaluation run."""
    evaluation_path = Path(evaluation_dir).expanduser().resolve()
    evaluation_path.mkdir(parents=True, exist_ok=True)

    name = run_name.strip() if run_name is not None else "run"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    if not safe_name:
        raise ValueError("run_name must contain at least one letter or number")

    shared_uuid = str(uuid.UUID(run_uuid)) if run_uuid is not None else None
    while True:
        run_id = f"{safe_name}-{shared_uuid or uuid.uuid4()}"
        run_path = evaluation_path / run_id
        try:
            run_path.mkdir(exist_ok=False)
        except FileExistsError:
            if shared_uuid is not None:
                raise
            continue
        return run_path, run_id


def _write_json_atomically(path: Path, value: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}.",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            json.dump(value, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def evaluate_policy(
    policy_fn: PolicyFn,
    evaluation_dir: str | Path,
    *,
    run_name: str | None = None,
    run_uuid: str | None = None,
    episodes: int = DEFAULT_EVALUATION_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_EVALUATION_SEED,
    observation_adapter: ObservationAdapter = standard_policy_observation,
    simulator_factory: SimulatorFactory | None = None,
    policy_reset_fn: PolicyResetFn | None = None,
    record_video: bool = True,
    fps: int = CONTROL_FREQUENCY_HZ,
    episode_metrics_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run evaluation episodes in a unique directory under ``evaluation_dir``.

    ``run_name`` supplies the readable portion of the directory name. A UUID is
    appended. Supply run_uuid to reuse a training run's UUID; otherwise a
    fresh UUID is generated. An existing directory with an explicitly supplied
    UUID raises FileExistsError rather than overwriting prior artifacts.

    ``observation_adapter`` controls exactly what the policy sees. The default
    filters out privileged fields such as depth and ``task_complete``. Supply a
    different adapter for training-only privileged policies or alternate
    motion-planning observation formats.

    ``policy_reset_fn`` is called at the start of every episode and is useful
    for recurrent policies or wrappers that maintain action history.

    An optional episode_metrics_fn extracts task-specific metrics from the final
    observation of each episode and saves them under that episode's task_metrics.

    By default, evaluation uses the fixed initialization seeds 0 through 24.
    The simulator's post-reset zero-action bootstrap is recorded in the video,
    but is deliberately excluded from episode step and completion-time metrics.
    """
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")

    output_path, run_id = _create_run_directory(evaluation_dir, run_name, run_uuid)
    make_simulator = simulator_factory or (lambda: Simulator(has_renderer=False))
    episode_seeds = [seed + episode_index for episode_index in range(episodes)]

    # Seed before construction as simulator initialization may consume NumPy RNG.
    np.random.seed(seed)
    simulator = make_simulator()
    episode_metrics: list[dict[str, Any]] = []

    try:
        action_min, action_max = (
            np.asarray(bound, dtype=np.float32)
            for bound in simulator.action_spec
        )
        if action_min.shape != action_max.shape:
            raise ValueError("Simulator action bounds have different shapes")
        zero_action = np.zeros_like(action_min)

        for episode_index, episode_seed in enumerate(episode_seeds):
            np.random.seed(episode_seed)
            if policy_reset_fn is not None:
                policy_reset_fn()
            simulator.reset()
            observation = simulator.step(zero_action)
            remaining_simulator_steps = _remaining_simulator_steps(simulator)
            policy_step_limit = (
                max_steps
                if remaining_simulator_steps is None
                else min(max_steps, remaining_simulator_steps)
            )
            horizon_limited = (
                remaining_simulator_steps is not None
                and remaining_simulator_steps <= max_steps
            )

            video_name = f"episode_{episode_index:03d}.mp4"
            video_path = output_path / video_name
            writer = (
                _VideoWriter(video_path, fps) if record_video else _NoVideoWriter()
            )
            success = bool(observation.get("task_complete", False))
            steps = 0
            clipped_action_count = 0

            with writer:
                writer.add_observation(observation)
                while not success and steps < policy_step_limit:
                    policy_observation = observation_adapter(observation)
                    action, was_clipped = _validated_action(
                        policy_fn(policy_observation), action_min, action_max
                    )
                    clipped_action_count += int(was_clipped)
                    observation = simulator.step(action)
                    steps += 1
                    writer.add_observation(observation)
                    success = bool(observation.get("task_complete", False))

            episode_metrics.append(
                {
                    "episode": episode_index,
                    "seed": episode_seed,
                    "success": success,
                    "episode_steps": steps,
                    "simulator_steps": steps + 1,
                    "policy_step_limit": policy_step_limit,
                    "steps_to_completion": steps if success else None,
                    "seconds_to_completion": (
                        steps / CONTROL_FREQUENCY_HZ if success else None
                    ),
                    "termination_reason": (
                        "success"
                        if success
                        else "simulator_horizon"
                        if horizon_limited
                        else "max_steps"
                    ),
                    "clipped_action_count": clipped_action_count,
                    "video": video_name if record_video else None,
                    **({"task_metrics": dict(episode_metrics_fn(observation))}
                       if episode_metrics_fn is not None else {}),
                }
            )
    finally:
        simulator.close()

    successful_steps = [
        item["steps_to_completion"]
        for item in episode_metrics
        if item["success"]
    ]
    successes = len(successful_steps)
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "run_uuid": str(uuid.UUID(run_id[-36:])),
        "run_name": run_name,
        "output_dir": str(output_path),
        "config": {
            "episodes": episodes,
            "max_steps": max_steps,
            "base_seed": seed,
            "episode_seeds": episode_seeds,
            "control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "video_fps": fps if record_video else None,
            "bootstrap_step_counted": False,
        },
        "summary": {
            "successes": successes,
            "failures": episodes - successes,
            "success_rate": successes / episodes,
            "mean_episode_steps": float(
                np.mean([item["episode_steps"] for item in episode_metrics])
            ),
            "mean_steps_to_completion": (
                float(np.mean(successful_steps)) if successful_steps else None
            ),
            "mean_seconds_to_completion": (
                float(np.mean(successful_steps)) / CONTROL_FREQUENCY_HZ
                if successful_steps
                else None
            ),
        },
        "episodes": episode_metrics,
    }
    _write_json_atomically(output_path / "metrics.json", metrics)
    return metrics


def _zero_policy(observation: Mapping[str, Any]) -> np.ndarray:
    """CLI smoke-test policy; real callers should import ``evaluate_policy``."""
    del observation
    return np.zeros(7, dtype=np.float32)

def _random_policy(observation: Mapping[str, Any]) -> np.ndarray:
    """CLI smoke-test policy; real callers should import ``evaluate_policy``."""
    del observation
    return np.random.normal(0.0, 0.1, size=7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a zero-policy evaluation smoke test."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument(
        "--name",
        dest="run_name",
        help="readable run name; a UUID is always appended",
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EVALUATION_EPISODES
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_policy(
        # _zero_policy,
        _random_policy,
        args.output_dir,
        run_name=args.run_name,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        record_video=not args.no_video,
    )
    print(f"Saved evaluation artifacts to {metrics['output_dir']}")
    print(json.dumps(metrics["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
