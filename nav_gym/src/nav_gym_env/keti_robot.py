# -*- coding: utf-8 -*-
import numpy as np

from nav_gym_env.utils import (
    translation_matrix_from_xyz,
    quaternion_matrix_from_yaw,
    transform_xy
)


class KetiRobot(object):
    footprint = [
        [0.3, 0.4],
        [-0.70, 0.4],
        [-0.70, -0.4],
        [0.3, -0.4]
    ]
    threshold_footprint = [
        [0.6, 0.6], 
        [-0.7, 0.6], 
        [-0.7, -0.6], 
        [0.6, -0.6]
    ]
    discomfort_threshold_footprint = [
        [0.6 + 0.5, 0.6 + 0.5], 
        [-0.7, 0.6 + 0.5], 
        [-0.7, -0.6 - 0.5], 
        [0.6 + 0.5, -0.6 - 0.5]
    ]
    # to consider collisions while reversing
    real_threshold_footprint = [
        [0.6, 0.6], 
        [-1.0, 0.6], 
        [-1.0, -0.6], 
        [0.6, -0.6]
    ]
    real_discomfort_threshold_footprint = [
        [0.6 + 1.0, 0.6 + 0.5], 
        [-0.7, 0.6 + 0.5], 
        [-0.7, -0.6 - 0.5], 
        [0.6 + 1.0, -0.6 - 0.5]
    ]
    has_legs = False
    angle_increment = 0.0122718463 
    angle_min = -3.141592 # starting from the back-right of the robot
    angle_max = 3.141592 # until the back-left of the robot
    range_max = 25.
    n_angles = 512
    def __init__(
        self,
        px, py, theta,
        gx, gy,
        time_step,
    ):
        self.px = px
        self.py = py
        self.theta = theta
        self.gx = gx
        self.gy = gy
        self.time_step = time_step

        self.vx, self.vy, self.v, self.r = 0., 0., 0., 0.

    def set_vel(self, linvel, rotvel):
        self.v = linvel
        self.r = rotvel
        self.vx = linvel * np.cos(self.theta)
        self.vy = linvel * np.sin(self.theta)
        # change pose here ....
        # center of rotation
        rot_px, rot_py = transform_xy(
            translation_matrix_from_xyz(
                0.14474 * np.cos(self.theta),
                0.14474 * np.sin(self.theta),
                0
            ),
            quaternion_matrix_from_yaw(0),
            np.array([self.px, self.py])
        )

        theta = self.theta + rotvel * self.time_step
        rot_px = rot_px + np.cos(theta) * linvel * self.time_step
        rot_py = rot_py+ np.sin(theta) * linvel * self.time_step
        self.px, self.py = transform_xy(
            translation_matrix_from_xyz(
                -0.14474 * np.cos(theta),
                -0.14474 * np.sin(theta),
                0
            ),
            quaternion_matrix_from_yaw(0),
            np.array([rot_px, rot_py])
        )
        self.theta = (self.theta + rotvel * self.time_step) % (2 * np.pi)

