"""Generate and save one successful trajectory from the privileged teacher."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from harness import _VideoWriter
from motion_planning.simulator import Simulator
from v4.perturbations import ActionPerturber, PerturbationConfig
from v4.teacher import (
    OBJECT_NAMES,
    PrivilegedStackTeacher,
    OracleState,
    TeacherConfig,
    TeacherFailure,
    read_oracle_state,
)


OBSERVATION_KEYS = (
    "robot0_joint_pos",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "frontview_image",
    "robot0_eye_in_hand_image",
    "frontview_depth",
)
CONTROL_FREQUENCY_HZ = 20


def create_run_directory(root: Path, name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    if not safe_name:
        raise ValueError("name must contain at least one letter or number")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    while True:
        path = root / f"{safe_name}-{uuid.uuid4()}"
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path


class AttemptRecorder:
    """Stream one attempt to temporary HDF5 and MP4 files."""

    def __init__(
        self,
        run_dir: Path,
        *,
        seed: int,
        teacher: PrivilegedStackTeacher,
        perturbation: ActionPerturber | None = None,
        save_video: bool = True,
        scratch_dir: Path | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.seed = seed
        self.teacher = teacher
        self.perturbation = perturbation
        temporary_root = scratch_dir if scratch_dir is not None else run_dir
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.h5_path = temporary_root / f".attempt-{uuid.uuid4()}.h5"
        self.video_path = temporary_root / f".attempt-{uuid.uuid4()}.mp4"
        self.h5_file: h5py.File | None = None
        self.video_writer = _VideoWriter(self.video_path, CONTROL_FREQUENCY_HZ) if save_video else None
        self.datasets: dict[str, h5py.Dataset] = {}
        self.steps = 0

    def __enter__(self) -> "AttemptRecorder":
        self.h5_file = h5py.File(self.h5_path, "w")
        self.h5_file.attrs["format_version"] = 2 if self.perturbation is not None else 1
        self.h5_file.attrs["seed"] = self.seed
        self.h5_file.attrs["control_frequency_hz"] = CONTROL_FREQUENCY_HZ
        self.h5_file.attrs["bootstrap_step_counted"] = False
        self.h5_file.attrs["teacher_config"] = self.teacher.config_json()
        self.h5_file.attrs["object_order"] = json.dumps(OBJECT_NAMES)
        self.h5_file.create_group("observations")
        self.h5_file.create_group("oracle")
        if self.perturbation is not None:
            self.h5_file.attrs['category'] = self.perturbation.config.category
            self.h5_file.attrs['perturbation_seed'] = self.perturbation.seed
            self.h5_file.attrs['perturbation_config'] = json.dumps(self.perturbation.config.to_dict(), sort_keys=True)
            self.h5_file.attrs['action_semantics'] = 'actions=intended_teacher; executed_actions=simulator_command'
            self.h5_file.create_group('perturbation')
        if self.video_writer is not None:
            self.video_writer.__enter__()
        return self

    def _append_array(self, name: str, value: Any) -> None:
        if self.h5_file is None:
            raise RuntimeError("Recorder is not open")
        array = np.asarray(value)
        dataset = self.datasets.get(name)
        if dataset is None:
            dataset = self.h5_file.create_dataset(
                name,
                shape=(0, *array.shape),
                maxshape=(None, *array.shape),
                chunks=(1, *array.shape),
                dtype=array.dtype,
                compression="lzf",
            )
            self.datasets[name] = dataset
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = array

    def _append_stage(self, stage: str) -> None:
        if self.h5_file is None:
            raise RuntimeError("Recorder is not open")
        dataset = self.datasets.get("stage")
        if dataset is None:
            dataset = self.h5_file.create_dataset(
                "stage",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            self.datasets["stage"] = dataset
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = stage

    def add_initial_observation(self, observation: dict[str, Any]) -> None:
        if self.video_writer is not None:
            self.video_writer.add_observation(observation)

    def add_transition(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        *,
        stage: str,
        phase: int,
        oracle: OracleState,
        next_observation: dict[str, Any],
        executed_action: np.ndarray | None = None,
        perturbation_info: dict[str, Any] | None = None,
    ) -> None:
        for key in OBSERVATION_KEYS:
            self._append_array(f"observations/{key}", observation[key])
        self._append_array("actions", np.asarray(action, dtype=np.float32))
        if self.perturbation is not None:
            if executed_action is None or perturbation_info is None:
                raise ValueError('Perturbed trajectories require executed actions and metadata')
            self._append_array('executed_actions', np.asarray(executed_action, dtype=np.float32))
            for key, value in perturbation_info.items():
                self._append_array(f'perturbation/{key}', value)
        self._append_stage(stage)
        self._append_array("phase", np.asarray(phase, dtype=np.int8))
        self._append_array(
            "next_task_complete",
            np.asarray(bool(next_observation["task_complete"]), dtype=np.bool_),
        )
        self._append_array("oracle/object_positions", oracle.positions)
        self._append_array("oracle/object_quaternions", oracle.quaternions)
        self._append_array("oracle/body_velocities", oracle.body_velocities)
        self._append_array("oracle/object_half_sizes", oracle.half_sizes)
        self._append_array("oracle/eef_position", oracle.eef_position)
        self._append_array("oracle/grasped", oracle.grasped)
        self._append_array("oracle/object_contacts", oracle.object_contacts)
        if self.video_writer is not None:
            self.video_writer.add_observation(next_observation)
        self.steps += 1

    def add_terminal_observation(
        self,
        observation: dict[str, Any],
        simulator: Simulator,
    ) -> None:
        if self.h5_file is None:
            raise RuntimeError("Recorder is not open")
        terminal = self.h5_file.create_group("terminal_observation")
        for key in OBSERVATION_KEYS:
            terminal.create_dataset(key, data=observation[key], compression="lzf")
        oracle = read_oracle_state(simulator)
        terminal.create_dataset("object_positions", data=oracle.positions)
        terminal.create_dataset("object_quaternions", data=oracle.quaternions)
        terminal.create_dataset("eef_position", data=oracle.eef_position)
        terminal.attrs["task_complete"] = oracle.task_complete

    def finish_successfully(self) -> tuple[Path, Path | None]:
        if self.h5_file is None:
            raise RuntimeError("Recorder is not open")
        self.h5_file.attrs["steps"] = self.steps
        self.h5_file.attrs["success"] = True
        self.h5_file.flush()
        self.h5_file.close()
        self.h5_file = None
        if self.video_writer is not None:
            self.video_writer.__exit__(None, None, None)
        trajectory_path = self.run_dir / "trajectory.h5"
        video_path = self.run_dir / "trajectory.mp4" if self.video_writer is not None else None
        for source, destination in ((self.h5_path, trajectory_path), (self.video_path, video_path)):
            if destination is not None:
                if source.parent == destination.parent:
                    source.replace(destination)
                else:
                    pending = destination.with_suffix(destination.suffix + '.pending')
                    shutil.copyfile(source, pending)
                    pending.replace(destination)
                    source.unlink()
        return trajectory_path, video_path

    def discard(self, exc_type: Any = None, exc: Any = None) -> None:
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
        if self.video_writer is not None:
            self.video_writer.__exit__(exc_type or RuntimeError, exc, None)
        self.h5_path.unlink(missing_ok=True)
        self.video_path.unlink(missing_ok=True)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.h5_file is not None:
            self.discard(exc_type, exc)


def _remaining_steps(simulator: Simulator, requested: int) -> int:
    environment = simulator.env
    if environment.ignore_done:
        return requested
    return min(requested, max(int(environment.horizon - environment.timestep), 0))


def generate_one_success(
    output_dir: Path,
    *,
    name: str,
    seed: int,
    max_attempts: int,
    max_steps: int,
    teacher_config: TeacherConfig,
    perturbation_config: PerturbationConfig | None = None,
    perturbation_seed: int = 0,
    save_video: bool = True,
    scratch_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Try consecutive seeds until one official successful rollout is saved."""
    if max_attempts <= 0 or max_steps <= 0:
        raise ValueError("max_attempts and max_steps must be positive")
    run_dir = create_run_directory(output_dir, name)
    attempts: list[dict[str, Any]] = []
    np.random.seed(seed)
    simulator = Simulator(has_renderer=False)
    result: dict[str, Any] | None = None

    try:
        for attempt_index in range(max_attempts):
            attempt_seed = seed + attempt_index
            np.random.seed(attempt_seed)
            simulator.reset()
            action_min, _ = simulator.action_spec
            observation = dict(simulator.step(np.zeros_like(action_min)))
            step_limit = _remaining_steps(simulator, max_steps)
            teacher = PrivilegedStackTeacher(teacher_config)
            perturber = (ActionPerturber(perturbation_config, perturbation_seed + attempt_index)
                         if perturbation_config is not None else None)
            failure_reason: str | None = None
            last_stage: str | None = None
            steps = 0

            with AttemptRecorder(
                run_dir, seed=attempt_seed, teacher=teacher, perturbation=perturber,
                save_video=save_video, scratch_dir=scratch_dir,
            ) as recorder:
                recorder.add_initial_observation(observation)
                try:
                    while steps < step_limit:
                        decision = teacher.decide(simulator)
                        if decision.stage != last_stage:
                            if verbose:
                                print(
                                    f"attempt={attempt_index} seed={attempt_seed} "
                                    f"step={steps} stage={decision.stage}",
                                    flush=True,
                                )
                            last_stage = decision.stage
                        if decision.done:
                            if not bool(observation["task_complete"]):
                                raise TeacherFailure(
                                    "Teacher finished without official task success"
                                )
                            if (perturber is not None and perturber.config.category != 'clean'
                                    and perturber.changed_steps == 0):
                                raise TeacherFailure('No movement perturbation occurred')
                            recorder.add_terminal_observation(observation, simulator)
                            trajectory_path, video_path = recorder.finish_successfully()
                            result = {
                                "success": True,
                                "seed": attempt_seed,
                                "attempt": attempt_index,
                                "steps": steps,
                                "simulated_seconds": steps / CONTROL_FREQUENCY_HZ,
                                "trajectory": str(trajectory_path),
                                "video": str(video_path) if video_path is not None else None,
                                "category": perturbation_config.category if perturbation_config is not None else 'clean',
                                "perturbation_seed": perturbation_seed + attempt_index if perturber is not None else None,
                                "perturbed_steps": perturber.changed_steps if perturber is not None else 0,
                            }
                            break

                        pre_observation = observation
                        oracle = read_oracle_state(simulator)
                        executed_action, perturbation_info = (perturber.apply(decision.action)
                                                              if perturber is not None else (decision.action, None))
                        next_observation = dict(simulator.step(executed_action))
                        recorder.add_transition(
                            pre_observation,
                            decision.action,
                            stage=decision.stage,
                            phase=decision.phase,
                            oracle=oracle,
                            next_observation=next_observation,
                            executed_action=executed_action,
                            perturbation_info=perturbation_info,
                        )
                        observation = next_observation
                        steps += 1
                except TeacherFailure as error:
                    failure_reason = str(error)

                if result is None:
                    if failure_reason is None:
                        failure_reason = "episode step limit reached"
                    recorder.discard()

            attempts.append(
                {
                    "attempt": attempt_index,
                    "seed": attempt_seed,
                    "success": result is not None,
                    "steps": steps,
                    "last_stage": last_stage,
                    "failure_reason": failure_reason,
                    "perturbed_steps": perturber.changed_steps if perturber is not None else 0,
                }
            )
            if verbose:
                print(
                    f"attempt={attempt_index} seed={attempt_seed} steps={steps} "
                    f"success={result is not None}"
                    + (f" reason={failure_reason}" if failure_reason else ""),
                    flush=True,
                )
            if result is not None:
                break
    finally:
        simulator.close()

    metrics = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "teacher_config": asdict(teacher_config),
        "perturbation_config": perturbation_config.to_dict() if perturbation_config is not None else None,
        "perturbation_seed": perturbation_seed,
        "requested_seed": seed,
        "max_attempts": max_attempts,
        "max_steps": max_steps,
        "result": result,
        "attempts": attempts,
    }
    metrics_path = run_dir / "metrics.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".metrics-",
        suffix=".json",
        dir=run_dir,
        delete=False,
    ) as temporary_file:
        json.dump(metrics, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(metrics_path)

    if result is None:
        raise RuntimeError(
            f"Teacher did not produce a successful trajectory in {max_attempts} "
            f"attempts; diagnostics saved to {metrics_path}"
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("v4/trajectories"))
    parser.add_argument("--name", default="scripted-teacher")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--max-translation-action", type=float, default=0.6)
    parser.add_argument("--approach-clearance-m", type=float, default=0.12)
    parser.add_argument("--transfer-clearance-m", type=float, default=0.10)
    parser.add_argument("--retreat-clearance-m", type=float, default=0.10)
    parser.add_argument("--grasp-z-offset-m", type=float, default=0.0)
    parser.add_argument("--placement-gap-m", type=float, default=0.002)
    parser.add_argument("--position-tolerance-m", type=float, default=0.008)
    parser.add_argument("--placement-xy-tolerance-m", type=float, default=0.012)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = generate_one_success(
        args.output_dir,
        name=args.name,
        seed=args.seed,
        max_attempts=args.max_attempts,
        max_steps=args.max_steps,
        teacher_config=TeacherConfig(
            max_translation_action=args.max_translation_action,
            approach_clearance_m=args.approach_clearance_m,
            transfer_clearance_m=args.transfer_clearance_m,
            retreat_clearance_m=args.retreat_clearance_m,
            grasp_z_offset_m=args.grasp_z_offset_m,
            placement_gap_m=args.placement_gap_m,
            position_tolerance_m=args.position_tolerance_m,
            placement_xy_tolerance_m=args.placement_xy_tolerance_m,
        ),
    )
    print(json.dumps(metrics["result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
