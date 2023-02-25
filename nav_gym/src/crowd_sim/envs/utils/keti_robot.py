# -*- coding: utf-8 -*-
import numpy as np

from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.action import ActionRot, ActionXYRot

from dxp2d_env.utils import (
    translation_matrix_from_xyz,
    quaternion_matrix_from_yaw,
    transform_xy
)


class KetiRobot(Robot):
    """
    * 회전중심이 앞바퀴 사이에
    * linear vel=0 인경우에 회전 X
    """
    def __init__(self, config, section):
        super(KetiRobot, self).__init__(config, section)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        # 회전중심
        rot_px, rot_py = transform_xy(
            translation_matrix_from_xyz(
                0.14474 * np.cos(self.theta),
                0.14474 * np.sin(self.theta),
                0
            ),
            quaternion_matrix_from_yaw(0),
            np.array([self.px, self.py])
        )
        theta = self.theta + action.r
        if isinstance(action, ActionRot):
            rot_px = rot_px + np.cos(theta) * action.v * delta_t
            rot_py = rot_py + np.sin(theta) * action.v * delta_t
        elif isinstance(action, ActionXYRot):
            rot_px = rot_px + np.cos(theta) * action.vx * delta_t - np.sin(theta) * action.vy * delta_t
            rot_py = rot_py + np.sin(theta) * action.vx * delta_t + np.cos(theta) * action.vy * delta_t
        px, py = transform_xy(
            translation_matrix_from_xyz(
                -0.14474 * np.cos(theta),
                -0.14474 * np.sin(theta),
                0
            ),
            quaternion_matrix_from_yaw(0),
            np.array([rot_px, rot_py])
        )
        return px, py

    def step(self, action):
        # if isinstance(action, ActionRot):
        #     if action.v == 0:
        #         action = ActionRot(0., 0.)
        # elif isinstance(action, ActionXYRot):
        #     if action.vx == 0 and action.vy == 0:
        #         action = ActionXYRot(0., 0., 0.)
        super(KetiRobot, self).step(action)

    