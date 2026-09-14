"""Stream aligned HDF5 trajectories and optional camera videos; save collection metadata."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import av
import h5py
import numpy as np

from motion_planning.simulator import Simulator, IMAGE_HEIGHT, IMAGE_WIDTH
from .teacher import OBJECT_NAMES, PrivilegedStackTeacher, OracleState, read_oracle_state

def compose_video_frame(observation: dict) -> np.ndarray:
    """Place the front and wrist RGB observations side by side."""
    front_image = np.flipud(observation["frontview_image"])
    wrist_image = np.flipud(observation["robot0_eye_in_hand_image"])
    return np.ascontiguousarray(
        np.concatenate((front_image, wrist_image), axis=1),
        dtype=np.uint8,
    )

class VideoWriter:
    """Write an episode video to a temporary file, then move it into place."""

    def __init__(self, output_path: Path, fps: int) -> None:
        self.output_path = output_path
        self.fps = fps
        self.temporary_path: Path | None = None
        self.container: Any = None
        self.stream: Any = None

    def __enter__(self) -> "VideoWriter":
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


OBSERVATION_KEYS = ('robot0_joint_pos', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'frontview_image', 'robot0_eye_in_hand_image', 'frontview_depth')
CONTROL_FREQUENCY_HZ = 20

class AttemptRecorder:
    """Stream one attempt to temporary HDF5 and MP4 files."""

    def __init__(
        self,
        run_dir: Path,
        *,
        seed: int,
        teacher: PrivilegedStackTeacher,
        perturbation: Any | None = None,
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
        self.video_writer = VideoWriter(self.video_path, CONTROL_FREQUENCY_HZ) if save_video else None
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.json', delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        temporary = Path(handle.name)
    temporary.replace(path)


def relative_entry(path, manifest_path, **extra):
    item = inspect_trajectory(path)
    item['path'] = os.path.relpath(Path(path).resolve(), Path(manifest_path).resolve().parent)
    return {**item, **extra}


def inspect_trajectory(path):
    """Check action/observation alignment and correction masks without loading RGB arrays."""
    path = Path(path).resolve()
    with h5py.File(path, 'r') as demo:
        if not demo.attrs.get('success', False):
            raise ValueError(f'Unsuccessful recording: {path}')
        actions = demo['actions'][:]
        steps = len(actions)
        if not steps or actions.shape != (steps, 7) or int(demo.attrs['steps']) != steps:
            raise ValueError(f'Invalid action shape/step count: {path}')
        for key in OBSERVATION_KEYS:
            if demo[f'observations/{key}'].shape[0] != steps:
                raise ValueError(f'Observation length mismatch: {key}')
            if key not in demo['terminal_observation']:
                raise ValueError(f'Missing terminal observation: {key}')
        if not demo['terminal_observation'].attrs['task_complete']:
            raise ValueError('Terminal oracle state is not successful')
        executed = demo['executed_actions'][:] if 'executed_actions' in demo else actions
        if executed.shape != actions.shape:
            raise ValueError('Executed/target action shape mismatch')
        for values in (actions, executed):
            if not np.isfinite(values).all() or np.abs(values).max() > 1.000001 or values[:, 3:6].any():
                raise ValueError('Actions must be finite, normalized, with zero rotations')
        mask = demo['supervision/mask'][:].astype(bool) if 'supervision/mask' in demo else np.ones(steps, bool)
        if mask.shape != (steps,) or not mask.any():
            raise ValueError('Invalid supervision mask')
        stages = demo['stage'].asstr()[:]
        if len(stages) != steps or len(demo['phase']) != steps or len(demo['next_task_complete']) != steps:
            raise ValueError('Transition metadata length mismatch')
        if demo.attrs.get('requires_supervision_mask', False):
            if 'supervision/mask' not in demo:
                raise ValueError('Correction requires an explicit supervision mask')
            handoff = int(demo.attrs['handoff_step'])
            if not np.array_equal(mask, np.arange(steps) >= handoff):
                raise ValueError('Mask does not match the learner-to-expert handoff')
            if actions[~mask].any() or any(stages[~mask]) or not all(stages[mask]):
                raise ValueError('Unlabelled prefix must have zero target placeholders and empty stages')
            if not np.array_equal(actions[mask], executed[mask]):
                raise ValueError('Expert targets differ from executed expert actions')
        elif not mask.all():
            raise ValueError('Partial supervision requires correction metadata')
        return {'path': str(path), 'seed': int(demo.attrs['seed']), 'steps': steps,
                'supervised_steps': int(mask.sum()), 'format_version': int(demo.attrs['format_version'])}


def prepare_output(root, protocol):
    """Reject accidental reuse of outputs with a different plan or source implementation."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    saved = root / 'collection_protocol.json'
    if saved.exists():
        if json.loads(saved.read_text()) != protocol:
            raise ValueError('Output uses a different collection protocol; choose a new directory')
    else:
        if any(p.name != '.collection.lock' for p in root.iterdir()):
            raise ValueError('Output is nonempty and has no collection protocol; choose a new directory')
        write_manifest(protocol, saved)
    return root


def source_hashes():
    from motion_planning import environment, simulator
    sources = {f'bc_data/{p.name}': sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    for module in (environment, simulator):
        sources[f'motion_planning/{Path(module.__file__).name}'] = sha256(module.__file__)
    return sources
