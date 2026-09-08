"""Expose simulator state externally while reusing the exact v2 task reward."""

import numpy as np
import robosuite as suite

from motion_planning.simulator import Simulator
from v2.task import GreenLiftSimulator, LiftTaskConfig


class StateSimulator(Simulator):
    """Same scene and dynamics; cameras are enabled only for evaluation videos."""

    def __init__(self, *, record_video=False, horizon=1000):
        self.env = suite.make(
            env_name="UltraTask", has_renderer=False,
            # Include the zero-action bootstrap in the simulator horizon.
            horizon=horizon,
            has_offscreen_renderer=record_video, use_camera_obs=record_video,
            camera_names=["frontview", "robot0_eye_in_hand"],
            camera_heights=256, camera_widths=256, camera_depths=False,
        )

    def step(self, action):
        self._activate_render_context()
        observation, _, _, _ = self.env.step(action)
        # Copy arrays: later simulator steps must not mutate rollout observations.
        observation = {key: np.array(value, copy=True) for key, value in observation.items()}
        data = self.env.sim.data
        robot = self.env.robots[0]
        arm = robot.arms[0]
        site_id = robot.eef_site_id[arm]
        site_name = self.env.sim.model.site_id2name(site_id)
        body_name = self.env.cubeB.root_body
        green_pos = data.body_xpos[self.env.cubeB_body_id].copy()
        observation.update({
            "green_pos": green_pos,
            # MuJoCo stores wxyz; use the same xyzw convention as robot0_eef_quat.
            "green_quat": data.body_xquat[self.env.cubeB_body_id][[1, 2, 3, 0]].copy(),
            "gripper_to_green": green_pos - data.site_xpos[site_id],
            "eef_linear_velocity": data.get_site_xvelp(site_name).copy(),
            "eef_angular_velocity": data.get_site_xvelr(site_name).copy(),
            "green_linear_velocity": data.get_body_xvelp(body_name).copy(),
            "green_angular_velocity": data.get_body_xvelr(body_name).copy(),
        })
        return observation


class StateGreenLiftSimulator(GreenLiftSimulator):
    def __init__(self, config=LiftTaskConfig(), *, record_video=False, horizon=1000):
        config.validate()
        self.config = config
        self.simulator = StateSimulator(record_video=record_video, horizon=horizon)
        self.episode = None
