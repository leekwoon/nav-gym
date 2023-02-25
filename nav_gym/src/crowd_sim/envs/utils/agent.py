import abc
import logging

import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.utils.action import ActionRot, ActionXYRot
from crowd_sim.envs.utils.state import ObservableState, FullState


class Agent(object):
    """!
    Base class for robot and human. Have the physical attributes of an agent.
    """

    def __init__(self, config, section):
        self.visible = config.getboolean(section, 'visible')
        self.robot_visible = None
        self.v_pref = config.getfloat(section, 'v_pref')
        self.radius = config.getfloat(section, 'radius')
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

    def print_info(self):
        logging.info('Agent is {} '.format(
            'visible' if self.visible else 'invisible'))

    def set_policy(self, policy):
        self.policy = policy

    def sample_random_attributes(self):
        """
        Samples agent radius and v_pref attribute from certain distribution
        """
        self.v_pref = np.random.uniform(0.8, 1.2)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        next_theta = self.theta + action.r
        next_vx = action.v * np.cos(next_theta)
        next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(
            self.px,
            self.py,
            self.vx,
            self.vy,
            self.radius,
            self.gx,
            self.gy,
            self.v_pref,
            self.theta)

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """!
        Computes state using received observation and passes it to policy.
        """
        return

    def check_validity(self, action):
        assert isinstance(action, ActionRot) or isinstance(action, ActionXYRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        theta = self.theta + action.r
        if isinstance(action, ActionRot):
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t
        elif isinstance(action, ActionXYRot):
            px = self.px + np.cos(theta) * action.vx * delta_t - np.sin(theta) * action.vy * delta_t
            py = self.py + np.sin(theta) * action.vx * delta_t + np.cos(theta) * action.vy * delta_t

        return px, py

    def compute_velocity(self, action):
        self.check_validity(action)
        theta = self.theta + action.r
        if isinstance(action, ActionRot):
            vx = action.v * np.cos(theta)
            vy = action.v * np.sin(theta)
        elif isinstance(action, ActionXYRot):
            vx = action.vx * np.cos(theta) - action.vy * np.sin(theta)
            vy = action.vx * np.sin(theta) + action.vy * np.cos(theta)

        return vx, vy

    def step(self, action):
        """!
        Performs an action and updates the state.
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        vel = self.compute_velocity(action)
        self.px, self.py = pos
        self.vx, self.vy = vel
        self.theta = (self.theta + action.r) % (2 * np.pi)

    def reached_destination(self):
        return norm(np.array(self.get_position()) -
                    np.array(self.get_goal_position())) < self.radius
