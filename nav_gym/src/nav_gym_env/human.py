import numpy as np


class Human(object):
    footprint = [
        [0.22, 0.19],
        [-0.22, 0.19],
        [-0.22, -0.19],
        [0.22, -0.19]
    ]
    has_legs = True
    angle_increment = 0.00613592315
    angle_min = -1.57079632679
    angle_max = 1.57079632679
    range_max = 6.
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
        theta = self.theta + rotvel * self.time_step
        self.px = self.px + np.cos(theta) * linvel * self.time_step
        self.py = self.py + np.sin(theta) * linvel * self.time_step
        self.theta = (self.theta + rotvel * self.time_step) % (2 * np.pi)

