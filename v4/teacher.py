"""Closed-loop privileged teacher for the red-green-blue stacking task."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from motion_planning.simulator import Simulator


OBJECT_NAMES = ("red", "green", "blue")
MOVES = (("green", "red"), ("blue", "green"))


class TeacherFailure(RuntimeError):
    """Raised when the teacher cannot safely complete the current attempt."""


class Stage(str, Enum):
    MOVE_ABOVE = "move_above"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    TRANSFER = "transfer"
    LOWER = "lower"
    RELEASE = "release"
    RETREAT = "retreat"
    SETTLE = "settle"
    DONE = "done"


@dataclass(frozen=True)
class TeacherConfig:
    """Conservative waypoint-controller settings in meters and control steps."""

    action_translation_m: float = 0.05
    max_translation_action: float = 0.6
    approach_clearance_m: float = 0.12
    transfer_clearance_m: float = 0.10
    retreat_clearance_m: float = 0.10
    grasp_z_offset_m: float = 0.0
    placement_gap_m: float = 0.002
    position_tolerance_m: float = 0.008
    placement_xy_tolerance_m: float = 0.012
    waypoint_hold_steps: int = 3
    close_min_steps: int = 6
    close_timeout_steps: int = 50
    release_steps: int = 12
    settle_steps: int = 10
    stage_timeout_steps: int = 160

    def validate(self) -> None:
        if not np.isfinite(self.grasp_z_offset_m):
            raise ValueError("grasp_z_offset_m must be finite")
        positive_floats = (
            "action_translation_m",
            "max_translation_action",
            "approach_clearance_m",
            "transfer_clearance_m",
            "retreat_clearance_m",
            "placement_gap_m",
            "position_tolerance_m",
            "placement_xy_tolerance_m",
        )
        for name in positive_floats:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        positive_ints = (
            "waypoint_hold_steps",
            "close_min_steps",
            "close_timeout_steps",
            "release_steps",
            "settle_steps",
            "stage_timeout_steps",
        )
        for name in positive_ints:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.max_translation_action <= 1:
            raise ValueError("max_translation_action must be in (0, 1]")


@dataclass(frozen=True)
class OracleState:
    positions: NDArray[np.float64]
    quaternions: NDArray[np.float64]
    body_velocities: NDArray[np.float64]
    half_sizes: NDArray[np.float64]
    eef_position: NDArray[np.float64]
    grasped: NDArray[np.bool_]
    object_contacts: NDArray[np.bool_]
    task_complete: bool


@dataclass(frozen=True)
class TeacherDecision:
    action: NDArray[np.float32]
    stage: str
    phase: int
    done: bool = False


def _object_half_size(obj: Any) -> NDArray[np.float64]:
    size = np.asarray(obj.size, dtype=np.float64)
    if size.ndim == 0:
        return np.repeat(size, 3)
    if size.shape != (3,):
        raise ValueError(f"Unexpected object size shape: {size.shape}")
    return size.copy()


def read_oracle_state(simulator: Simulator) -> OracleState:
    """Read named teacher-only state without changing the supplied simulator."""
    env = simulator.env
    objects = (env.cubeA, env.cubeB, env.cubeC)
    body_ids = (env.cubeA_body_id, env.cubeB_body_id, env.cubeC_body_id)
    positions = np.stack(
        [np.asarray(env.sim.data.body_xpos[index]).copy() for index in body_ids]
    )
    quaternions = np.stack(
        [np.asarray(env.sim.data.body_xquat[index]).copy() for index in body_ids]
    )
    body_velocities = np.stack(
        [np.asarray(env.sim.data.cvel[index]).copy() for index in body_ids]
    )
    half_sizes = np.stack([_object_half_size(obj) for obj in objects])

    eef_sites = env.robots[0].eef_site_id
    if isinstance(eef_sites, dict):
        eef_site_id = next(iter(eef_sites.values()))
    else:
        eef_site_id = int(np.asarray(eef_sites).flatten()[0])
    eef_position = np.asarray(env.sim.data.site_xpos[eef_site_id]).copy()

    grasped = np.asarray(
        [
            env._check_grasp(
                gripper=env.robots[0].gripper,
                object_geoms=obj,
            )
            for obj in objects
        ],
        dtype=np.bool_,
    )
    object_contacts = np.asarray(
        (
            env.check_contact(env.cubeA, env.cubeB),
            env.check_contact(env.cubeB, env.cubeC),
            env.check_contact(env.cubeA, env.cubeC),
        ),
        dtype=np.bool_,
    )
    return OracleState(
        positions=positions,
        quaternions=quaternions,
        body_velocities=body_velocities,
        half_sizes=half_sizes,
        eef_position=eef_position,
        grasped=grasped,
        object_contacts=object_contacts,
        task_complete=bool(env._check_success()),
    )


class PrivilegedStackTeacher:
    """Finite-state oracle that stacks green on red, then blue on green."""

    def __init__(self, config: TeacherConfig = TeacherConfig()) -> None:
        config.validate()
        self.config = config
        self.phase = 0
        self.stage = Stage.MOVE_ABOVE
        self.stage_steps = 0
        self.within_tolerance_steps = 0
        self.grasp_offset: NDArray[np.float64] | None = None
        self.retreat_target: NDArray[np.float64] | None = None
        self.stable_success_steps = 0

    @property
    def moving_name(self) -> str:
        return MOVES[self.phase][0]

    @property
    def support_name(self) -> str:
        return MOVES[self.phase][1]

    @property
    def stage_name(self) -> str:
        if self.stage in (Stage.SETTLE, Stage.DONE):
            return self.stage.value
        return f"{self.moving_name}_{self.stage.value}_over_{self.support_name}"

    def config_json(self) -> str:
        return json.dumps(asdict(self.config), sort_keys=True)

    def _transition(
        self,
        stage: Stage,
        *,
        retreat_target: NDArray[np.float64] | None = None,
    ) -> None:
        self.stage = stage
        self.stage_steps = 0
        self.within_tolerance_steps = 0
        self.retreat_target = retreat_target

    def _object_index(self, name: str) -> int:
        return OBJECT_NAMES.index(name)

    def _translation_action(
        self,
        state: OracleState,
        target: NDArray[np.float64],
        gripper: float,
    ) -> NDArray[np.float32]:
        error = target - state.eef_position
        translation = np.clip(
            error / self.config.action_translation_m,
            -self.config.max_translation_action,
            self.config.max_translation_action,
        )
        action = np.zeros(7, dtype=np.float32)
        action[:3] = translation.astype(np.float32)
        action[6] = np.float32(gripper)
        return action

    def _at_target(
        self,
        current: NDArray[np.float64],
        target: NDArray[np.float64],
        tolerance: float | None = None,
    ) -> bool:
        tolerance = tolerance or self.config.position_tolerance_m
        if np.linalg.norm(current - target) <= tolerance:
            self.within_tolerance_steps += 1
        else:
            self.within_tolerance_steps = 0
        return self.within_tolerance_steps >= self.config.waypoint_hold_steps

    def _placement_center(
        self,
        state: OracleState,
        moving_index: int,
        support_index: int,
    ) -> NDArray[np.float64]:
        support = state.positions[support_index]
        return np.asarray(
            (
                support[0],
                support[1],
                support[2]
                + state.half_sizes[support_index, 2]
                + state.half_sizes[moving_index, 2]
                + self.config.placement_gap_m,
            ),
            dtype=np.float64,
        )

    def _require_grasp(self, state: OracleState, moving_index: int) -> None:
        if not bool(state.grasped[moving_index]):
            raise TeacherFailure(
                f"Lost {self.moving_name} during stage {self.stage_name}"
            )

    def decide(self, simulator: Simulator) -> TeacherDecision:
        """Return the next normalized OSC action from the current oracle state."""
        state = read_oracle_state(simulator)
        decision_stage = self.stage_name
        decision_phase = self.phase
        if self.stage is Stage.DONE:
            return TeacherDecision(
                action=self._translation_action(
                    state, state.eef_position, gripper=-1.0
                ),
                stage=decision_stage,
                phase=decision_phase,
                done=True,
            )

        self.stage_steps += 1
        if self.stage_steps > self.config.stage_timeout_steps:
            raise TeacherFailure(f"Timed out in stage {self.stage_name}")

        moving_index = self._object_index(self.moving_name)
        support_index = self._object_index(self.support_name)
        moving_position = state.positions[moving_index]
        placement_center = self._placement_center(
            state, moving_index, support_index
        )

        if self.stage is Stage.MOVE_ABOVE:
            target = moving_position.copy()
            target[2] += self.config.approach_clearance_m
            if self._at_target(state.eef_position, target):
                self._transition(Stage.DESCEND)
            return TeacherDecision(
                self._translation_action(state, target, -1.0),
                decision_stage,
                decision_phase,
            )

        if self.stage is Stage.DESCEND:
            target = moving_position.copy()
            target[2] += self.config.grasp_z_offset_m
            if self._at_target(state.eef_position, target):
                self._transition(Stage.CLOSE)
            return TeacherDecision(
                self._translation_action(state, target, -1.0),
                decision_stage,
                decision_phase,
            )

        if self.stage is Stage.CLOSE:
            target = moving_position.copy()
            target[2] += self.config.grasp_z_offset_m
            if (
                self.stage_steps >= self.config.close_min_steps
                and bool(state.grasped[moving_index])
            ):
                self.grasp_offset = state.eef_position - moving_position
                self._transition(Stage.LIFT)
            elif self.stage_steps > self.config.close_timeout_steps:
                raise TeacherFailure(f"Failed to grasp {self.moving_name}")
            return TeacherDecision(
                self._translation_action(state, target, 1.0),
                decision_stage,
                decision_phase,
            )

        if self.grasp_offset is None:
            raise TeacherFailure("Teacher reached transport stage without grasp offset")

        if self.stage is Stage.LIFT:
            self._require_grasp(state, moving_index)
            object_target = moving_position.copy()
            object_target[2] = (
                placement_center[2] + self.config.transfer_clearance_m
            )
            target = object_target + self.grasp_offset
            if self._at_target(state.eef_position, target):
                self._transition(Stage.TRANSFER)
            return TeacherDecision(
                self._translation_action(state, target, 1.0),
                decision_stage,
                decision_phase,
            )

        if self.stage is Stage.TRANSFER:
            self._require_grasp(state, moving_index)
            object_target = placement_center.copy()
            object_target[2] += self.config.transfer_clearance_m
            target = object_target + self.grasp_offset
            if self._at_target(state.eef_position, target):
                self._transition(Stage.LOWER)
            return TeacherDecision(
                self._translation_action(state, target, 1.0),
                decision_stage,
                decision_phase,
            )

        if self.stage is Stage.LOWER:
            self._require_grasp(state, moving_index)
            target = placement_center + self.grasp_offset
            xy_error = np.linalg.norm(
                moving_position[:2] - state.positions[support_index, :2]
            )
            support_contact = (
                bool(state.object_contacts[0])
                if self.phase == 0
                else bool(state.object_contacts[1])
            )
            close_to_target = np.linalg.norm(moving_position - placement_center) < (
                self.config.position_tolerance_m * 2
            )
            if (
                xy_error <= self.config.placement_xy_tolerance_m
                and (support_contact or close_to_target)
            ):
                self._transition(Stage.RELEASE)
            return TeacherDecision(
                self._translation_action(state, target, 1.0),
                decision_stage,
                decision_phase,
            )

        if self.stage is Stage.RELEASE:
            target = placement_center + self.grasp_offset
            if (
                self.stage_steps >= self.config.release_steps
                and not bool(state.grasped[moving_index])
            ):
                retreat_target = state.eef_position.copy()
                retreat_target[2] += self.config.retreat_clearance_m
                self._transition(Stage.RETREAT, retreat_target=retreat_target)
            return TeacherDecision(
                self._translation_action(state, target, -1.0),
                decision_stage,
                decision_phase,
            )

        if self.stage is Stage.RETREAT:
            if self.retreat_target is None:
                raise TeacherFailure("Missing retreat waypoint")
            target = self.retreat_target
            if self._at_target(state.eef_position, target):
                support_contact = (
                    bool(state.object_contacts[0])
                    if self.phase == 0
                    else bool(state.object_contacts[1])
                )
                if not support_contact:
                    raise TeacherFailure(
                        f"{self.moving_name} lost contact with {self.support_name}"
                    )
                if self.phase == 0:
                    self.phase = 1
                    self.grasp_offset = None
                    self._transition(Stage.MOVE_ABOVE)
                else:
                    self._transition(Stage.SETTLE)
            return TeacherDecision(
                self._translation_action(state, target, -1.0),
                decision_stage,
                decision_phase,
            )

        if self.stage is Stage.SETTLE:
            if state.task_complete and not bool(state.grasped[2]):
                self.stable_success_steps += 1
            else:
                self.stable_success_steps = 0
            if self.stable_success_steps >= self.config.settle_steps:
                self._transition(Stage.DONE)
                return TeacherDecision(
                    self._translation_action(
                        state, state.eef_position, gripper=-1.0
                    ),
                    decision_stage,
                    decision_phase,
                    done=True,
                )
            return TeacherDecision(
                self._translation_action(
                    state, state.eef_position, gripper=-1.0
                ),
                decision_stage,
                decision_phase,
            )

        raise TeacherFailure(f"Unhandled teacher stage {self.stage}")
