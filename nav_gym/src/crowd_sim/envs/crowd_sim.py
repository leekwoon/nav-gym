# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import logging
import math
import configparser
import collections
import os

import numpy as np
import gym
import rvo2
import cv2
import tensorflow as tf
from matplotlib import patches, lines
from numpy.linalg import norm
from PIL import Image

from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.utils.state import ObservableState
# from crowd_sim.envs.utils.action import ActionRot
from crowd_nav.policy.policy_factory import policy_factory

Obstacle = collections.namedtuple(
    "Obstacle", ["location_x", "location_y", "dim", "patch"])


class CrowdSim(gym.Env):
    """!
    Movement simulation for n+1 agents with static obstacles.
    Agent can either be human or robot.
    humans are controlled by a unknown and fixed policy.
    robot is controlled by a known and learnable policy.

    """

    def __init__(self):
        """
        Initializes the simulation.
        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.other_robots = None
        self.global_time = None
        self.human_policy = None

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        self.rotation_penalty_factor = None

        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None

        # for visualization
        self.states = None
        self.attention_weights = None

        # for static map
        self.map = None
        self.map_size_m = None
        self.submap_size_m = None
        self.map_resolution = None
        self.obstacle_vertices = None
        self.local_maps = None

        # for angular map
        self.angular_map_max_range = None
        self.angular_map_dim = None
        self.angular_map_min_angle = None
        self.angular_map_max_angle = None
        self.local_maps_angular = None

    def configure(self, config, silent=False):
        """!
        Configures the simulation.
            @param config: Path to the config file of the environment
        """
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean(
            'env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat(
            'reward', 'discomfort_penalty_factor')
        self.timeout_penalty = config.getfloat('reward', 'timeout_penalty')
        self.rotation_penalty_factor = config.getfloat(
            'reward', 'rotation_penalty_factor')

        self.case_capacity = {
            'train': np.iinfo(
                np.uint32).max - 2000,
            'val': 1000,
            'test': 1000}
        self.case_size = {
            'train': np.iinfo(
                np.uint32).max - 2000,
            'val': config.getint(
                'env',
                'val_size'),
            'test': config.getint(
                'env',
                'test_size')}
        self.train_val_sim = config.get('env', 'train_val_sim')
        self.test_sim = config.get('env', 'test_sim')
        self.square_width = config.getfloat('env', 'square_width')
        self.circle_radius = config.getfloat('env', 'circle_radius_min')
        self.human_num = config.getint('env', 'human_num')
        self.human_policy = config.get('env', 'human_policy')
        self.human_model_dir = os.path.expanduser(config.get('env', 'human_model_dir'))
        self.other_robots_num = config.getint('env', 'robot_num')
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        self.use_grid_map = config.getboolean('map', 'use_grid_map')
        self.map_resolution = config.getfloat('map', 'map_resolution')
        self.map_size_m = config.getfloat('map', 'map_size_m')

        self.num_circles = config.getint('map', 'num_circles')
        self.num_walls = config.getint('map', 'num_walls')

        if self.use_grid_map:
            self.submap_size_m = config.getfloat('map', 'submap_size_m')
        else:
            self.angular_map_max_range = config.getfloat(
                'map', 'angular_map_max_range')
            self.angular_map_dim = config.getint('map', 'angular_map_dim')
            self.angular_map_min_angle = config.getfloat(
                'map', 'angle_min') * np.pi
            self.angular_map_max_angle = config.getfloat(
                'map', 'angle_max') * np.pi

        self.policy = policy_factory[self.human_policy]()
        if self.human_policy == 'sdoadrl':
            policy_config_file = os.path.join(
                self.human_model_dir, 'params.config')
            policy_config = configparser.RawConfigParser()
            policy_config.read(policy_config_file)
            self.policy.configure(tf.Session(), 'global', policy_config)
            self.policy.load_model(
                os.path.join(
                    self.human_model_dir,
                    'rl_model'))
        elif self.human_policy == 'sarl' or self.human_policy == 'cadrl_original':
            policy_config_file = os.path.join(
                self.human_model_dir, 'policy.config')
            policy_config = configparser.RawConfigParser()
            policy_config.read(policy_config_file)
            self.policy.configure(policy_config)
            self.policy.set_env(self)
            self.policy.load_model(
                os.path.join(
                    self.human_model_dir,
                    'rl_model.pth'))

        self.policy.set_phase('test')

        if not silent:
            logging.info('human number: {}'.format(self.human_num))
            logging.info('robot number: {}'.format(self.other_robots_num))

            if self.randomize_attributes:
                logging.info("Randomize human's radius and preferred speed")
            else:
                logging.info("Not randomize human's radius and preferred speed")
            logging.info(
                'Training simulation: {}, test simulation: {}'.format(
                    self.train_val_sim, self.test_sim))
            logging.info(
                'Square width: {}, circle width: {}'.format(
                    self.square_width,
                    self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def generate_static_map_input(self, max_size, phase, config=None):
        """!
        Generates randomly located static obstacles (boxes and walls) in the environment.
            @param max_size: Max size in meters of the map
        """
        if config is not None:
            num_circles = config.getint('general', 'num_circles')
            num_walls = config.getint('general', 'num_walls')
        else:
            num_circles = int(round(np.random.random() * self.num_circles))
            num_walls = int(round(np.random.random() * self.num_walls))

        grid_size = int(round(max_size / self.map_resolution))
        self.map = np.ones((grid_size, grid_size))
        max_locations = int(round(grid_size))
        obstacles = []
        self.obstacle_vertices = []
        if phase == 'test':
            inflation_rate_il = 1
        else:
            inflation_rate_il = 1.25

        for circle_index in range(num_circles):
            while True:
                if config is not None:
                    location_x = config.getfloat(
                        'x_locations_circles', str(circle_index))
                    location_y = config.getfloat(
                        'y_locations_circles', str(circle_index))
                    circle_radius = config.getfloat(
                        'circle_radius', str(circle_index))
                else:
                    location_x = np.random.randint(
                        -max_locations / 2.0, max_locations / 2.0)
                    location_y = np.random.randint(
                        -max_locations / 2.0, max_locations / 2.0)
                    circle_radius = (np.random.random() + 0.5) * 0.7
                dim = (int(round(2 * circle_radius / self.map_resolution)),
                       int(round(2 * circle_radius / self.map_resolution)))
                patch = np.zeros([dim[0], dim[1]])

                location_x_m = location_x * self.map_resolution
                location_y_m = location_y * self.map_resolution

                collide = False
                if norm(
                    (location_x_m - self.robot.px,
                     location_y_m - self.robot.py)) < circle_radius + self.robot.radius + self.discomfort_dist or norm(
                    (location_x_m - self.robot.gx,
                     location_y_m - self.robot.gy)) < circle_radius + self.robot.radius + self.discomfort_dist:
                    collide = True
                if not collide:
                    break
            obstacles.append(Obstacle(int(round(location_x + grid_size / 2.0)),
                                      int(round(location_y + grid_size / 2.0)), dim, patch))
            circle_radius_inflated = inflation_rate_il * circle_radius
            self.obstacle_vertices.append([(location_x_m +
                                            circle_radius_inflated, location_y_m +
                                            circle_radius_inflated), (location_x_m -
                                                                      circle_radius_inflated, location_y_m +
                                                                      circle_radius_inflated), (location_x_m -
                                                                                                circle_radius_inflated, location_y_m -
                                                                                                circle_radius_inflated), (location_x_m +
                                                                                                                          circle_radius_inflated, location_y_m -
                                                                                                                          circle_radius_inflated)])

        for wall_index in range(num_walls):
            while True:
                if config is not None:
                    location_x = config.getfloat(
                        'x_locations_walls', str(wall_index))
                    location_y = config.getfloat(
                        'y_locations_walls', str(wall_index))
                    x_dim = config.getfloat('x_dim', str(wall_index))
                    y_dim = config.getfloat('y_dim', str(wall_index))
                else:
                    location_x = np.random.randint(
                        -max_locations / 2.0, max_locations / 2.0)
                    location_y = np.random.randint(
                        -max_locations / 2.0, max_locations / 2.0)
                    if np.random.random() > 0.5:
                        x_dim = np.random.randint(2, 4)
                        y_dim = 1
                    else:
                        y_dim = np.random.randint(2, 4)
                        x_dim = 1
                dim = (int(round(x_dim / self.map_resolution)),
                       int(round(y_dim / self.map_resolution)))
                patch = np.zeros([dim[0], dim[1]])

                location_x_m = location_x * self.map_resolution
                location_y_m = location_y * self.map_resolution

                collide = False

                if (abs(location_x_m -
                        self.robot.px) < x_dim /
                    2.0 +
                    self.robot.radius +
                    self.discomfort_dist and abs(location_y_m -
                                                 self.robot.py) < y_dim /
                    2.0 +
                    self.robot.radius +
                    self.discomfort_dist) or (abs(location_x_m -
                                                  self.robot.gx) < x_dim /
                                              2.0 +
                                              self.robot.radius +
                                              self.discomfort_dist and abs(location_y_m -
                                                                           self.robot.gy) < y_dim /
                                              2.0 +
                                              self.robot.radius +
                                              self.discomfort_dist):
                    collide = True
                if not collide:
                    break

            obstacles.append(Obstacle(int(round(location_x + grid_size / 2.0)),
                                      int(round(location_y + grid_size / 2.0)), dim, patch))
            x_dim_inflated = inflation_rate_il * x_dim
            y_dim_inflated = inflation_rate_il * y_dim
            self.obstacle_vertices.append([(location_x_m +
                                            x_dim_inflated /
                                            2.0, location_y_m +
                                            y_dim_inflated /
                                            2.0), (location_x_m -
                                                   x_dim_inflated /
                                                   2.0, location_y_m +
                                                   y_dim_inflated /
                                                   2.0), (location_x_m -
                                                          x_dim_inflated /
                                                          2.0, location_y_m -
                                                          y_dim_inflated /
                                                          2.0), (location_x_m +
                                                                 x_dim_inflated /
                                                                 2.0, location_y_m -
                                                                 y_dim_inflated /
                                                                 2.0)])

        for obstacle in obstacles:
            if obstacle.location_x > obstacle.dim[0] / 2.0 and \
                    obstacle.location_x < grid_size - obstacle.dim[0] / 2.0 and \
                    obstacle.location_y > obstacle.dim[1] / 2.0 and \
                    obstacle.location_y < grid_size - obstacle.dim[1] / 2.0:

                start_idx_x = int(
                    round(
                        obstacle.location_x -
                        obstacle.dim[0] /
                        2.0))
                start_idx_y = int(
                    round(
                        obstacle.location_y -
                        obstacle.dim[1] /
                        2.0))
                self.map[start_idx_x:start_idx_x +
                         obstacle.dim[0], start_idx_y:start_idx_y +
                         obstacle.dim[1]] = np.minimum(self.map[start_idx_x:start_idx_x +
                                                                obstacle.dim[0], start_idx_y:start_idx_y +
                                                                obstacle.dim[1]], obstacle.patch)

            else:
                for idx_x in range(obstacle.dim[0]):
                    for idx_y in range(obstacle.dim[1]):
                        shifted_idx_x = idx_x - obstacle.dim[0] / 2.0
                        shifted_idx_y = idx_y - obstacle.dim[1] / 2.0
                        submap_x = int(
                            round(
                                obstacle.location_x +
                                shifted_idx_x))
                        submap_y = int(
                            round(
                                obstacle.location_y +
                                shifted_idx_y))
                        if submap_x > 0 and submap_x < grid_size and submap_y > 0 and submap_y < grid_size:
                            self.map[submap_x,
                                     submap_y] = obstacle.patch[idx_x, idx_y]

        if self.robot.policy.name != 'SDOADRL' and self.robot.policy.name != 'ORCA':
            self.create_observation_from_static_obstacles(obstacles)

    def create_observation_from_static_obstacles(self, obstacles):
        self.static_obstacles_as_pedestrians = []
        for index, obstacle in enumerate(obstacles):
            if obstacle.dim[0] == obstacle.dim[1]:  # Obstacle is a square
                px = (
                    self.obstacle_vertices[index][0][0] + self.obstacle_vertices[index][2][0]) / 2.0
                py = (
                    self.obstacle_vertices[index][0][1] + self.obstacle_vertices[index][2][1]) / 2.0
                radius = (
                    self.obstacle_vertices[index][0][0] - px) * np.sqrt(2)
                self.static_obstacles_as_pedestrians.append(
                    ObservableState(px, py, 0, 0, radius))
            elif obstacle.dim[0] > obstacle.dim[1]:  # Obstacle is rectangle
                py = (
                    self.obstacle_vertices[index][0][1] + self.obstacle_vertices[index][2][1]) / 2.0
                radius = (
                    self.obstacle_vertices[index][0][1] - py) * np.sqrt(2)
                px = self.obstacle_vertices[index][1][0] + radius
                while px < self.obstacle_vertices[index][0][0]:
                    self.static_obstacles_as_pedestrians.append(
                        ObservableState(px, py, 0, 0, radius))
                    px = px + 2 * radius
            else:  # Obstacle is rectangle
                px = (
                    self.obstacle_vertices[index][0][0] + self.obstacle_vertices[index][2][0]) / 2.0
                radius = (
                    self.obstacle_vertices[index][0][0] - px) * np.sqrt(2)
                py = self.obstacle_vertices[index][2][1] + radius
                while py < self.obstacle_vertices[index][0][1]:
                    self.static_obstacles_as_pedestrians.append(
                        ObservableState(px, py, 0, 0, radius))
                    py = py + 2 * radius

    def generate_random_human_position(
            self, human_num, rule, imitation_learning):
        """!
        Generate human position according to certain rule.
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        @param human_num: Number of humans to add
        @param rule: square_crossing or circle_crossing
        """
        if imitation_learning:
            other_robots_num = 0
        else:
            other_robots_num = int(
                round(self.other_robots_num * (0.5 + np.random.random())))
        policy = self.policy
        if rule == 'square_crossing':
            self.humans = []
            self.other_robots = []
            for i in range(human_num):
                self.humans.append(
                    self.generate_square_crossing(
                        'human', policy))
            for i in range(other_robots_num):
                self.other_robots.append(
                    self.generate_square_crossing('robot'))

        elif rule == 'circle_crossing':
            self.humans = []
            self.other_robots = []
            for i in range(human_num):
                self.humans.append(
                    self.generate_circle_crossing(
                        'human', policy))
            for i in range(other_robots_num):
                self.other_robots.append(
                    self.generate_circle_crossing('robot'))
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing(self, agent_type, policy=None):
        """!
        Generate human positions for a circle crossing with randomized staring positions.

        @param agent_type: 'human' for a fixed rule, 'robot' for the learned policy
        """
        if agent_type is 'human':
            agent = Human(self.config, 'humans')
        else:
            agent = Robot(self.config, 'robot')
            agent.set_policy(self.robot.policy)

        if self.randomize_attributes:
            agent.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could
            # meet with human
            px_noise = (np.random.random() - 0.5) * agent.v_pref
            py_noise = (np.random.random() - 0.5) * agent.v_pref
            circle_radius = self.circle_radius * (1 + np.random.random() * 1.5)

            px = circle_radius * np.cos(angle) + px_noise
            py = circle_radius * np.sin(angle) + py_noise

            collide = False
            for other_agent in [self.robot] + self.humans + self.other_robots:
                min_dist = agent.radius + other_agent.radius + self.discomfort_dist
                if norm(
                    (px -
                     other_agent.px,
                     py -
                     other_agent.py)) < min_dist or norm(
                    (px -
                     other_agent.gx,
                     py -
                     other_agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        agent.set(px, py, -px, -py, 0, 0, 0)
        # Randomize if the agent should be visible for the humans
        if np.random.random() > 0.5 and self.robot.visible:
            agent.robot_visible = True
        else:
            agent.robot_visible = False
        if policy is not None:
            agent.set_policy(policy)
        return agent

    def generate_square_crossing(self, agent_type, policy=None):
        """!
        Generate human positions for a square crossing with randomized staring positions.

        @param agent_type: 'human' for a fixed rule, 'robot' for the learned policy
        """
        if agent_type is 'human':
            agent = Human(self.config, 'humans')
        else:
            agent = Robot(self.config, 'robot')
            agent.set_policy(self.robot.policy)

        if self.randomize_attributes:
            agent.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for other_agent in [self.robot] + self.humans + self.other_robots:
                if norm((px - other_agent.px, py - other_agent.py)
                        ) < agent.radius + other_agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for other_agent in [self.robot] + self.humans + self.other_robots:
                if norm((gx - other_agent.gx, gy - other_agent.gy)
                        ) < agent.radius + other_agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        agent.set(px, py, gx, gy, 0, 0, 0)

        # Randomize if the agent should be visible for the humans
        if np.random.random() > 0.5 and self.robot.visible:
            agent.robot_visible = True
        else:
            agent.robot_visible = False
        if policy is not None:
            agent.set_policy(policy)
        return agent

    def generate_static_map_input_from_config(self, config):
        """!
        Generates randomly located static obstacles (boxes and walls) in the environment.
            @param max_size: Max size in meters of the map
        """
        if self.robot.policy.name != 'SDOADRL' and self.robot.policy.name != 'ORCA':
            raise NotImplementedError(
                'Static map can only be created from config for SDOADRL and ORCA.')

        num_obstacles = config.getint('general', 'num_obstacles')

        grid_size = int(round(self.map_size_m / self.map_resolution))
        self.map = np.ones((grid_size, grid_size))
        self.obstacle_vertices = []
        dim = None
        for obstacle in range(num_obstacles):
            num_vertices = config.getint(
                'locations_obstacle_' + str(obstacle), 'num_vertices')
            vertex_list = list()
            for vertex in range(num_vertices):
                location_x_m = config.getfloat(
                    'locations_obstacle_' + str(obstacle), str(vertex) + '_x')
                location_y_m = config.getfloat(
                    'locations_obstacle_' + str(obstacle), str(vertex) + '_y')
                vertex_list.append((location_x_m, location_y_m))

            self.obstacle_vertices.append([vertex for vertex in vertex_list])
            pts = np.array([[int(round(y /
                                       self.map_resolution +
                                       grid_size /
                                       2.0)), int(round(x /
                                                        self.map_resolution +
                                                        grid_size /
                                                        2.0))] for x, y in vertex_list], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(self.map, [pts], 0)

    def load_environment(self, env_config):
        self.phase = 'test'
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        self.global_time = 0
        circle_radius = env_config.getfloat('general', 'circle_radius')
        self.last_circle_radius = circle_radius
        self.robot.set(0, -circle_radius, 0, circle_radius, 0, 0, np.pi / 2)
        human_num = env_config.getint('general', 'human_num')
        np.random.seed(env_config.getint('general', 'seed'))
        self.generate_random_human_position(human_num, self.test_sim, False)
        self.generate_static_map_input_from_config(env_config)

        for agent in [self.robot] + self.humans + self.other_robots:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        self.local_maps = list()
        self.local_maps_angular = list()
        if hasattr(self.robot, 'attention_weights'):
            self.attention_weights = list()

        self.states.append([self.robot.get_full_state(),
                            [human.get_full_state() for human in self.humans],
                            [robot.get_full_state() for robot in self.other_robots]])

        # get current observation
        ob = [human.get_observable_state()
              for human in self.humans + self.other_robots]
        if self.robot.policy.name != 'SDOADRL' and self.robot.policy.name != 'ORCA':
            ob += self.static_obstacles_as_pedestrians

        if self.use_grid_map:
            local_map = self.get_local_map(self.robot.get_full_state())
        else:
            local_map = self.get_local_map_angular(self.robot.get_full_state())

        return ob, local_map

    def reset(self, phase='test', test_case=None, imitation_learning=False, compute_local_map=True):
        """!
        Set px, py, gx, gy, vx, vy, theta for robot and humans.
            @param phase: 'test', 'train' or 'val'
            @param test_case: Number of a specific test case that should be replayed
            @param imitation_learning: If imitation learning is performed
            @return Observation of the environment by the robot
            @return Local map of static obstacles in reference to robot
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        self.phase = phase
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0

        counter_offset = {
            'train': self.case_capacity['val'] + self.case_capacity['test'],
            'val': 0,
            'test': self.case_capacity['val']}

        if self.case_counter[phase] >= 0:
            if phase is 'train':
                np.random.seed(None)
            else:
                np.random.seed(
                    counter_offset[phase] +
                    self.case_counter[phase])
            if self.randomize_attributes:
                self.robot.sample_random_attributes()

            circle_radius = self.circle_radius * \
                min(self.robot.v_pref * 5, 1) * (1 + np.random.random() * 2)
            if circle_radius > 9:
                circle_radius = 9
            self.last_circle_radius = circle_radius
            self.robot.set(
                0, -circle_radius, 0, circle_radius, 0, 0, np.pi / 2)
            if imitation_learning:
                human_num = self.human_num
            else:
                human_num = int(
                    round(self.human_num * (0.5 + np.random.random())))

            if phase in ['train', 'val']:
                self.generate_random_human_position(
                    human_num, self.train_val_sim, imitation_learning)
                self.generate_static_map_input(self.map_size_m, phase)
            else:
                self.generate_random_human_position(
                    human_num, self.test_sim, imitation_learning)
                self.generate_static_map_input(self.map_size_m, phase)

            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (
                self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            raise NotImplementedError

        for agent in [self.robot] + self.humans + self.other_robots:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step
            if agent.policy.name == 'ORCA':
                agent.policy.reset()

        self.states = list()
        self.local_maps = list()
        self.local_maps_angular = list()
        if hasattr(self.robot, 'attention_weights'):
            self.attention_weights = list()

        self.states.append([self.robot.get_full_state(),
                            [human.get_full_state() for human in self.humans],
                            [robot.get_full_state() for robot in self.other_robots]])

        # get current observation
        ob = [human.get_observable_state()
              for human in self.humans + self.other_robots]
        if self.robot.policy.name != 'SDOADRL' and self.robot.policy.name != 'ORCA':
            ob += self.static_obstacles_as_pedestrians

        local_map = None
        if compute_local_map:
            if self.use_grid_map:
                local_map = self.get_local_map(self.robot.get_full_state())
            else:
                local_map = self.get_local_map_angular(self.robot.get_full_state())
        if self.robot.policy.name == 'ORCA':
            return ob, self.obstacle_vertices, local_map
        else:
            return ob, local_map

    def onestep_lookahead(self, action):
        next_human_states, _, reward, done, info = self.step(
            action, update=False)
        return next_human_states, reward, done, info

    def step(self, action, update=True, compute_local_map=True, border=None):
        """!
        Compute actions for all agents, detect collision and update environment.
            @param action: Action chosen by the robot.
            @param update: If the environment should be updated with the agent's actions
            @param compute_local_map: If false, returns None for the local map
            @param border: If not None, collision is checked with border [(min_x, max_x), (min_y, max_y)]
            @return Observation of the environment by the robot
            @return Local map of static obstacles in reference to robot
            @return Reward collected in the step
            @return done: If episode is finished
            @return info: Information about the status of the robot in the environment
        """
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [
                other_human.get_observable_state() for other_human in self.humans +
                self.other_robots if other_human != human]
            if human.robot_visible:
                ob += [self.robot.get_observable_state()]
            if self.human_policy == 'orca':
                human_actions.append(human.act(ob, self.obstacle_vertices))
            elif self.human_policy == 'sdoadrl':
                human_actions.append(
                    human.act(
                        ob,
                        self.get_local_map_angular(
                            human.get_full_state(),
                            append=False)))
            elif self.human_policy == 'random':
                human_actions.append(human.act())
            else:
                human_actions.append(human.act(ob))

        robot_actions = []
        for robot in self.other_robots:
            ob = [
                other_robot.get_observable_state() for other_robot in self.humans +
                self.other_robots if other_robot != robot]
            if robot.robot_visible:
                ob += [self.robot.get_observable_state()]
            if self.use_grid_map:
                robot_actions.append(
                    robot.act(
                        ob,
                        self.get_local_map(
                            robot.get_full_state(),
                            append=False)))
            else:
                robot_actions.append(
                    robot.act(
                        ob,
                        self.get_local_map_angular(
                            robot.get_full_state(),
                            append=False)))

        if self.phase == 'test':
            # collisions that are caused by agents within robot's FOV
            human_states_in_FOV = []
            for agent in self.humans + self.other_robots:
                if self.robot.policy.human_state_in_FOV(self.robot, agent):
                    human_states_in_FOV.append(agent)

            dmin = float('inf')
            collision = False
            for agent in human_states_in_FOV:
                px = agent.px - self.robot.px
                py = agent.py - self.robot.py
                next_vel = self.robot.compute_velocity(action)
                vx = agent.vx - next_vel[0]
                vy = agent.vy - next_vel[1]
                ex = px + vx * self.time_step
                ey = py + vy * self.time_step
                # closest distance between boundaries of two agents
                closest_dist = point_to_segment_dist(
                    px, py, ex, ey, 0, 0) - agent.radius - self.robot.radius
                if closest_dist < 0:
                    collision = True
                    break
                elif closest_dist < dmin:
                    dmin = closest_dist

        # check all collisions
        dmin = float('inf')
        collision_other_agent = False
        for agent in self.humans + self.other_robots:
            px = agent.px - self.robot.px
            py = agent.py - self.robot.py
            next_vel = self.robot.compute_velocity(action)
            vx = agent.vx - next_vel[0]
            vy = agent.vy - next_vel[1]
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(
                px, py, ex, ey, 0, 0) - agent.radius - self.robot.radius
            if closest_dist < 0:
                collision_other_agent = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        if self.phase != 'test':
            collision = collision_other_agent

        # collision with obstacle
        px, py = self.robot.compute_position(action, self.time_step)  # next pos
        robot_idx_map_x = int(
            round((px + self.map_size_m / 2.0) / self.map_resolution))
        robot_idx_map_y = int(
            round((py + self.map_size_m / 2.0) / self.map_resolution))
        robot_size_map = int(
            np.ceil(
                self.robot.radius /
                np.sqrt(2.0) /
                self.map_resolution))

        start_idx_x = robot_idx_map_x - robot_size_map
        end_idx_x = start_idx_x + robot_size_map * 2
        start_idx_x = max(start_idx_x, 0)
        end_idx_x = min(
            end_idx_x, int(
                round(
                    self.map_size_m / self.map_resolution)))
        start_idx_y = robot_idx_map_y - robot_size_map
        end_idx_y = start_idx_y + robot_size_map * 2
        start_idx_y = max(start_idx_y, 0)
        end_idx_y = min(
            end_idx_y, int(
                round(
                    self.map_size_m / self.map_resolution)))
        if end_idx_x > start_idx_x and end_idx_y > start_idx_y:
            map_around_robot = self.map[start_idx_x:end_idx_x,
                                        start_idx_y:end_idx_y]
            if np.sum(map_around_robot) < map_around_robot.size:
                collision = True

        # collision with border
        if border is not None:
            if px <= border[0][0] + self.robot.radius or \
               px >= border[0][1] - self.robot.radius or \
               py <= border[1][0] + self.robot.radius or \
               py >= border[1][1] - self.robot.radius:
                collision = True

        # check if robot is closer to any obstacle than the discomfort dist
        closeToObstacle = False
        robot_size_map = int(
            np.ceil(
                (self.robot.radius +
                 self.discomfort_dist) /
                self.map_resolution))

        start_idx_x = robot_idx_map_x - robot_size_map
        end_idx_x = start_idx_x + robot_size_map * 2
        start_idx_x = max(start_idx_x, 0)
        end_idx_x = min(
            end_idx_x, int(
                round(
                    self.map_size_m / self.map_resolution)))
        start_idx_y = robot_idx_map_y - robot_size_map
        end_idx_y = start_idx_y + robot_size_map * 2
        start_idx_y = max(start_idx_y, 0)
        end_idx_y = min(
            end_idx_y, int(
                round(
                    self.map_size_m / self.map_resolution)))
        if end_idx_x > start_idx_x and end_idx_y > start_idx_y:
            larger_map_around_robot = self.map[start_idx_x:end_idx_x,
                                               start_idx_y:end_idx_y]
            if np.sum(larger_map_around_robot) < larger_map_around_robot.size:
                closeToObstacle = True

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2.0) - \
                    self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into
                    # account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(
            self.robot.compute_position(
                action, self.time_step))
        reaching_goal = norm(
            end_position -
            np.array(
                self.robot.get_goal_position())) < self.robot.radius

        if self.global_time >= self.time_limit:
            reward = self.timeout_penalty
            done = True
            info = Timeout()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif collision_other_agent:
            reward = 0
            done = True
            info = CollisionOtherAgent()
        elif closeToObstacle:
            reward = - self.discomfort_penalty_factor * self.time_step * 0.1
            done = False
            info = Danger(0.1)
        elif dmin < self.discomfort_dist:
            reward = (dmin - self.discomfort_dist) * \
                self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        elif abs(action.r) > 0:
            reward = abs(action.r) * self.rotation_penalty_factor
            done = False
            info = Nothing()
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            if hasattr(self.robot, 'attention_weights'):
                self.attention_weights.append(self.robot.attention_weights)

            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            for i, robot_action in enumerate(robot_actions):
                self.other_robots[i].step(robot_action)
            self.global_time += self.time_step

            # compute the observation
            ob = [agent.get_observable_state()
                  for agent in self.humans + self.other_robots]
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(),
                                [human.get_full_state() for human in self.humans],
                                [robot.get_full_state() for robot in self.other_robots]])
        else:
            assert('SARL' in self.robot.policy.name)
            human_states_in_FOV = []
            human_actions_in_FOV = []
            for agent, action in zip(
                    self.humans + self.other_robots, human_actions + robot_actions):
                if self.robot.policy.human_state_in_FOV(self.robot, agent):
                    human_states_in_FOV.append(agent)
                    human_actions_in_FOV.append(action)
            ob = [
                agent.get_next_observable_state(action) for agent,
                action in zip(
                    human_states_in_FOV,
                    human_actions_in_FOV)]

        if self.robot.policy.name != 'SDOADRL' and self.robot.policy.name != 'ORCA':
            ob += self.static_obstacles_as_pedestrians

        local_map = None
        if compute_local_map:
            if self.use_grid_map:
                local_map = self.get_local_map(self.robot.get_full_state())
            else:
                local_map = self.get_local_map_angular(self.robot.get_full_state())
        return ob, local_map, reward, done, info

    def calculate_angular_map_distances(
            self,
            vertex,
            edge,
            theta,
            radial_dist_vector,
            rad_indeces,
            locations):
        radial_resolution = (self.angular_map_max_angle - \
                             self.angular_map_min_angle) / float(self.angular_map_dim)
        px = (vertex[0] - edge[0]) * np.cos(theta) + \
            (vertex[1] - edge[1]) * np.sin(theta)
        py = (vertex[1] - edge[1]) * np.cos(theta) - \
            (vertex[0] - edge[0]) * np.sin(theta)
        phi = math.atan2(py, px)
        rad_idx = int((phi - self.angular_map_min_angle) /
                      float(radial_resolution))
        distance = np.linalg.norm([px, py])
        if rad_idx >= 0 and rad_idx < self.angular_map_dim:
            radial_dist_vector[rad_idx] = min(
                radial_dist_vector[rad_idx], distance)
        for rad_idx_old, location in zip(rad_indeces, locations):
            if abs(rad_idx - rad_idx_old) > np.pi / radial_resolution:
                wrapped = True
                idx_diff = self.angular_map_dim - rad_idx + \
                    rad_idx_old if rad_idx > rad_idx_old else self.angular_map_dim - rad_idx_old + rad_idx
            else:
                wrapped = False
                idx_diff = abs(rad_idx - rad_idx_old)
            for i in range(idx_diff):
                if (rad_idx < rad_idx_old and not wrapped) or (
                        rad_idx > rad_idx_old and wrapped):
                    if (rad_idx +
                        i) >= 0 and (rad_idx +
                                     i) < self.angular_map_dim:
                        px = (vertex[0] + i / float(idx_diff) * (location[0] - vertex[0]) - edge[0]) * np.cos(theta) + (
                            vertex[1] + i / float(idx_diff) * (location[1] - vertex[1]) - edge[1]) * np.sin(theta)
                        py = (vertex[1] + i / float(idx_diff) * (location[1] - vertex[1]) - edge[1]) * np.cos(theta) - (
                            vertex[0] + i / float(idx_diff) * (location[0] - vertex[0]) - edge[0]) * np.sin(theta)
                        obstacle_value_in_slice = np.linalg.norm([px, py])
                        radial_dist_vector[(rad_idx + i) %
                                           self.angular_map_dim] = min(radial_dist_vector[(rad_idx + i) %
                                                                                          self.angular_map_dim], obstacle_value_in_slice)
                else:
                    if (rad_idx_old +
                        i) >= 0 and (rad_idx_old +
                                     i) < self.angular_map_dim:
                        px = (location[0] + i / float(idx_diff) * (vertex[0] - location[0]) - edge[0]) * np.cos(theta) + (
                            location[1] + i / float(idx_diff) * (vertex[1] - location[1]) - edge[1]) * np.sin(theta)
                        py = (location[1] + i / float(idx_diff) * (vertex[1] - location[1]) - edge[1]) * np.cos(theta) - (
                            location[0] + i / float(idx_diff) * (vertex[0] - location[0]) - edge[0]) * np.sin(theta)
                        obstacle_value_in_slice = np.linalg.norm([px, py])
                        radial_dist_vector[(rad_idx_old + i) %
                                           self.angular_map_dim] = min(radial_dist_vector[(rad_idx_old + i) %
                                                                                          self.angular_map_dim], obstacle_value_in_slice)
        rad_indeces.append(rad_idx)
        locations.append(vertex)

    def get_local_map_angular(self, ob, normalize=True, append=True):
        """
        Compute the distance to surrounding objects in a radially discretized way.
        For each element there will be a floating point distance to the closest object in this sector.
        This allows to preserve the continuous aspect of the distance vs. a standard grid.

        !!! Attention: 0 angle is at the negative x-axis.

        number_elements: radial discretization
        relative_positions: relative positions of the surrounding objects in the local frame
        max_range: maximum range of the distance measurement

        returns:
        radial_dist_vector: contains the distance to the closest object in each sector
        """
        radial_dist_vector = self.angular_map_max_range * \
            np.ones([self.angular_map_dim])
        radial_resolution = (self.angular_map_max_angle - \
                             self.angular_map_min_angle) / float(self.angular_map_dim)
        agent_edges = []
        for s1, s2 in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            agent_edges.append(
                (ob.px + s1 * ob.radius, ob.py + s2 * ob.radius))

        for obstacle in self.obstacle_vertices:
            for edge in agent_edges:
                rad_indeces = []
                locations = []
                for vertex in obstacle:
                    self.calculate_angular_map_distances(
                        vertex, edge, ob.theta, radial_dist_vector, rad_indeces, locations)

        for obstacle in self.obstacle_vertices:
            for vertex in obstacle:
                rad_indeces = []
                locations = []
                for edge in agent_edges:
                    self.calculate_angular_map_distances(
                        vertex, edge, ob.theta, radial_dist_vector, rad_indeces, locations)

        if normalize:
            radial_dist_vector /= float(self.angular_map_max_range)

        if append:
            self.local_maps_angular.append(radial_dist_vector)
        return radial_dist_vector

    def get_local_map(self, ob, append=True):
        """!
        Extract a binary submap around the robot.
        @param ob: Full state of the robot.
        @return Binary submap rotated around robot
        """
        THRESHOLD_VALUE = 0.9
        center_idx_x = int(
            round(
                (ob.px +
                 self.map_size_m /
                 2.0) /
                self.map_resolution))
        center_idx_y = int(
            round(
                (ob.py +
                 self.map_size_m /
                 2.0) /
                self.map_resolution))
        size_submap = int(round(self.submap_size_m / self.map_resolution))

        start_idx_x = int(round(center_idx_x - np.floor(size_submap / 2.0)))
        start_idx_y = int(round(center_idx_y - np.floor(size_submap / 2.0)))
        end_idx_x = start_idx_x + size_submap - 1
        end_idx_y = start_idx_y + size_submap - 1
        grid = np.ones((size_submap, size_submap))
        # Compute end indices (assure size of submap is correct, if out of
        # bounds)
        max_idx_x = self.map.shape[0] - 1
        max_idx_y = self.map.shape[1] - 1

        start_grid_x = 0
        start_grid_y = 0
        end_grid_x = size_submap - 1
        end_grid_y = size_submap - 1

        if start_idx_x < 0:
            start_grid_x = -start_idx_x
            start_idx_x = 0
        elif end_idx_x > max_idx_x:
            end_grid_x = end_grid_x - (end_idx_x - max_idx_x)
            end_idx_x = max_idx_x
        if start_idx_y < 0:
            start_grid_y = -start_idx_y
            start_idx_y = 0
        elif end_idx_y > max_idx_y:
            end_grid_y = end_grid_y - (end_idx_y - max_idx_y)
            end_idx_y = max_idx_y

        if start_grid_y > end_grid_y or start_idx_y > end_idx_y \
                or start_idx_x > end_idx_x or start_grid_x > end_grid_x:
            grid_binary = grid
        else:
            grid[start_grid_x:end_grid_x,
                 start_grid_y:end_grid_y] = self.map[start_idx_x:end_idx_x,
                                                     start_idx_y:end_idx_y]
            grid = self.rotate_grid_around_center(
                grid, (- ob.theta + math.pi / 2) * 180 / math.pi)
            grid_binary = np.zeros_like(grid)
            indeces = grid > THRESHOLD_VALUE
            grid_binary[indeces] = 1
        if append:
            self.local_maps.append(grid_binary)
        return grid_binary

    def rotate_grid_around_center(self, grid, angle):
        '''!
        Rotate grid into direction of robot heading.
            @param grid: Grid to be rotated
            @param angle: Angle to rotate the grid by
            @return The rotated grid
        '''
        grid = grid.copy()
        rows, cols = grid.shape
        M = cv2.getRotationMatrix2D(
            center=(
                rows / 2.0,
                cols / 2.0),
            angle=angle,
            scale=1)
        grid = cv2.warpAffine(grid, M, (rows, cols), borderValue=1)

        return grid

    def render(self, mode, output_file=None, deconv=None):
        '''!
        Visualizes the environment.
            @param mode: Choose 'video' for the rendering
        '''
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = (1, 0.34, 0.114)
        goal_color = 'red'
        arrow_color = 'black'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(8.5, 8.5))
            ax.tick_params(labelsize=20)
            ax.set_xlim(-8, 8)
            ax.set_ylim(-8, 8)
            ax.set_xlabel('x(m)', fontsize=20)
            ax.set_ylabel('y(m)', fontsize=20)

            robot_positions = [self.states[i]
                               [0].position for i in range(len(self.states))]
            human_positions = [[state[1][j].position for j in range(
                len(self.humans))] for state in self.states]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(
                        robot_positions[k],
                        self.robot.radius,
                        fill=True,
                        color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(
                        a)) for i, a in zip(range(len(self.humans)), range(len(self.humans)))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = list()
                    for i in range(len(agents)):
                        if global_time > 0 and norm(
                                [prev_loc[i][0] - agents[i].center[0], prev_loc[i][1] - agents[i].center[1]]) < 0.2:
                            continue
                        else:
                            times.append(
                                plt.text(
                                    agents[i].center[0] -
                                    x_offset,
                                    agents[i].center[1] -
                                    y_offset,
                                    '{:.1f}'.format(global_time),
                                    color='black',
                                    fontsize=16))

                    prev_loc = [[agents[i].center[0] - x_offset,
                                 agents[i].center[1]] for i in range(len(agents))]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px,
                                                    self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py,
                                                    self.states[k][1][i].py),
                                                   color=cmap(i),
                                                   ls='solid') for i in range(len(self.humans))]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            obstacles = [plt.Polygon(obstacle_vertex)
                         for obstacle_vertex in self.obstacle_vertices]

            for obstacle in obstacles:
                ax.add_artist(obstacle)
            goal = lines.Line2D([0],
                                [self.last_circle_radius],
                                color=goal_color,
                                marker='*',
                                linestyle='None',
                                markersize=15,
                                label='Goal')
            ax.add_artist(goal)
            fig2 = plt.figure()
            plt.axis('off')
            plt.legend([goal] +
                       [robot] +
                       [human for human in humans], ['Goal'] +
                       ['Robot'] +
                       ['Human ' +
                        str(i) for i in range(len(humans))], fontsize=20, loc=3)
            plt.show()
        elif mode == 'am':
            frame = 0  # Set the desired frame number

            # Plot environment
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.tick_params(labelsize=22)
            ax.set_xlim(-7.5, 7.5)
            ax.set_ylim(-7.5, 7.5)
            ax.set_xlabel('x(m)', fontsize=22)
            ax.set_ylabel('y(m)', fontsize=22)
            robot_position = self.states[frame][0].position
            robot = plt.Circle(
                robot_position,
                self.robot.radius,
                fill=True,
                color=robot_color)
            ax.add_artist(robot)
            obstacles = [plt.Polygon(obstacle_vertex)
                         for obstacle_vertex in self.obstacle_vertices]
            for obstacle in obstacles:
                ax.add_artist(obstacle)
            orientation = [
                (self.states[frame][0].px,
                 self.states[frame][0].py),
                (self.states[frame][0].px +
                 self.robot.radius *
                 np.cos(
                    self.states[frame][0].theta),
                    self.states[frame][0].py +
                    self.robot.radius *
                    np.sin(
                    self.states[frame][0].theta))]
            arrows = [
                patches.FancyArrowPatch(
                    *orientation,
                    color=arrow_color,
                    arrowstyle=arrow_style)]
            for arrow in arrows:
                ax.add_artist(arrow)

            # Plot angular map
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.tick_params(labelsize=22)
            ax2.set_xlim(-7.5, 7.5)
            ax2.set_ylim(-7.5, 7.5)
            ax2.set_xlabel('x(m)', fontsize=22)
            ax2.set_ylabel('y(m)', fontsize=22)
            angular_resolution = (self.angular_map_max_angle -
                                  self.angular_map_min_angle) / float(self.angular_map_dim)

            cmap = plt.get_cmap('gnuplot')

            for i in range(self.angular_map_dim):
                angle_start = (self.angular_map_min_angle + i *
                               angular_resolution) * 180 / np.pi + 90
                angle_end = (self.angular_map_min_angle + (i + 1)
                             * angular_resolution) * 180 / np.pi + 90

                distance_cone = plt.matplotlib.patches.Wedge(
                    (0.0,
                     0.0),
                    self.local_maps_angular[frame][i] *
                    self.angular_map_max_range,
                    angle_start,
                    angle_end,
                    facecolor=cmap(
                        self.local_maps_angular[frame][i]),
                    alpha=0.5)
                ax2.add_artist(distance_cone)

            plt.show()
        elif mode == 'traj3D':
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.tick_params(labelsize=20)
            ax.set_xlim(-5.4, 5.4)
            ax.set_ylim(-5.4, 5.4)
            ax.set_xlabel('x(m)', fontsize=20)
            ax.set_ylabel('y(m)', fontsize=20)
            ax.set_zlabel('Timestep', fontsize=20)

            robot_positions = [self.states[i]
                               [0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(
                len(self.humans))] for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = (robot_positions[k][0], robot_positions[k][1], k)
                    humans = [(human_positions[k][i][0], human_positions[k][i][1], k)
                              for i, a in zip(range(len(self.humans)), range(len(self.humans)))]
                    ax.scatter(robot[0], robot[1], robot[2], c=cmap(0))
                    for i, human in enumerate(humans):
                        ax.scatter(human[0], human[1], human[2], c=cmap(i + 1))
            goal = lines.Line2D([0],
                                [self.last_circle_radius],
                                color=goal_color,
                                marker='*',
                                linestyle='None',
                                markersize=15,
                                label='Goal')
            ax.add_artist(goal)
            plt.show()

        elif mode == 'og':
            frame = 15  # Set desired frame number
            # Plot environment
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.tick_params(labelsize=22)
            ax.set_xlim(-7.5, 7.5)
            ax.set_ylim(-7.5, 7.5)
            ax.set_xlabel('x(m)', fontsize=22)
            ax.set_ylabel('y(m)', fontsize=22)

            robot_position = self.states[frame][0].position
            robot = plt.Circle(
                robot_position,
                self.robot.radius,
                fill=True,
                color=robot_color)
            ax.add_artist(robot)
            obstacles = [plt.Polygon(obstacle_vertex)
                         for obstacle_vertex in self.obstacle_vertices]
            for obstacle in obstacles:
                ax.add_artist(obstacle)
            orientation = [
                (self.states[frame][0].px,
                 self.states[frame][0].py),
                (self.states[frame][0].px +
                 self.robot.radius *
                 np.cos(
                    self.states[frame][0].theta),
                    self.states[frame][0].py +
                    self.robot.radius *
                    np.sin(
                    self.states[frame][0].theta))]
            arrows = [
                patches.FancyArrowPatch(
                    *orientation,
                    color=arrow_color,
                    arrowstyle=arrow_style)]
            for arrow in arrows:
                ax.add_artist(arrow)
            # Plot occupancy grid
            fig2, ax2 = plt.subplots(figsize=(7.5, 7.5))
            ax2.tick_params(labelsize=22)
            img = Image.fromarray(self.local_maps[i].astype('uint8'))
            img_rotated = img.rotate(90)
            ax2.tick_params(labelsize=16)
            ax2.imshow(img_rotated, cmap="gray", vmin=0, vmax=1)
            middle_idx = int(
                round(
                    self.submap_size_m /
                    2.0 /
                    self.map_resolution))
            robot2 = plt.Circle((middle_idx, middle_idx), int(
                round(self.robot.radius / self.map_resolution)), fill=True, color=robot_color)
            ax2.add_patch(robot2)

            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = lines.Line2D([0],
                                [self.last_circle_radius],
                                color=goal_color,
                                marker='*',
                                linestyle='None',
                                markersize=15,
                                label='Goal')
            robot = plt.Circle(
                robot_positions[0],
                self.robot.radius,
                fill=True,
                color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            human_positions = [[state[1][j].position for j in range(
                len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i],
                                 self.humans[i].radius,
                                 fill=False) for i in range(len(self.humans))]
            human_numbers = [
                plt.text(
                    humans[i].center[0] -
                    x_offset,
                    humans[i].center[1] -
                    y_offset,
                    str(i),
                    color='black',
                    fontsize=12) for i in range(
                    len(
                        self.humans))]

            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            other_robot_positions = [[state[2][j].position for j in range(
                len(self.other_robots))] for state in self.states]
            other_robots = [
                plt.Circle(
                    other_robot_positions[0][i],
                    self.other_robots[i].radius,
                    fill=False,
                    color=robot_color) for i in range(
                    len(
                        self.other_robots))]
            other_robot_numbers = [
                plt.text(
                    other_robots[i].center[0] -
                    x_offset,
                    other_robots[i].center[1] -
                    y_offset,
                    str(i),
                    color='black',
                    fontsize=12) for i in range(
                    len(
                        self.other_robots))]

            for i, other_robot in enumerate(other_robots):
                ax.add_artist(other_robot)
                ax.add_artist(other_robot_numbers[i])

            obstacles = [plt.Polygon(obstacle_vertex)
                         for obstacle_vertex in self.obstacle_vertices]

            for obstacle in obstacles:
                ax.add_artist(obstacle)
            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute orientation in each step and use arrow to show the
            # direction
            radius = self.robot.radius
            orientations_humans = [
                [
                    ((state[1][j].px,
                      state[1][j].py),
                        (state[1][j].px +
                         1.5 *
                         self.humans[j].radius *
                         np.cos(
                            state[1][j].theta),
                         state[1][j].py +
                         1.5 *
                         self.humans[j].radius *
                         np.sin(
                            state[1][j].theta))) for state in self.states] for j in range(
                    len(
                        self.humans))]
            orientations_other_robots = [
                [
                    ((state[2][j].px,
                      state[2][j].py),
                        (state[2][j].px +
                         1.5 *
                         self.other_robots[j].radius *
                         np.cos(
                            state[2][j].theta),
                         state[2][j].py +
                         1.5 *
                         self.other_robots[j].radius *
                         np.sin(
                            state[2][j].theta))) for state in self.states] for j in range(
                    len(
                        self.other_robots))]
            orientation_self = [
                ((state[0].px,
                  state[0].py),
                    (state[0].px +
                     radius *
                     np.cos(
                        state[0].theta),
                     state[0].py +
                     radius *
                     np.sin(
                        state[0].theta))) for state in self.states]
            arrow_self = patches.FancyArrowPatch(
                *orientation_self[0], color=arrow_color, arrowstyle=arrow_style)
            ax.add_artist(arrow_self)
            orientations = orientations_humans
            orientations.extend(orientations_other_robots)
            arrows_others = [
                patches.FancyArrowPatch(
                    *orientation[0],
                    color='red',
                    arrowstyle=arrow_style) for orientation in orientations]
            for arrow in arrows_others:
                ax.add_artist(arrow)
            global_step = 0

            if self.use_grid_map:
                # Plot robot's view
                fig2, ax2 = plt.subplots(figsize=(10, 10))
                img = Image.fromarray(self.local_maps[0].astype('uint8'))
                img_rotated = img.rotate(90)
                ax2.tick_params(labelsize=16)
                ax2.imshow(img_rotated, cmap="gray", vmin=0, vmax=1)
                middle_idx = int(
                    round(
                        self.submap_size_m /
                        2.0 /
                        self.map_resolution))
                robot2 = plt.Circle((middle_idx, middle_idx), int(
                    round(self.robot.radius / self.map_resolution)), fill=True, color='red')
                ax2.add_patch(robot2)
                time2 = plt.text(5, 5, 'Time: {}'.format(0), fontsize=16)
                # ax2.add_artist(time2)
            else:
                # Plot robot's view
                fig2, ax2 = plt.subplots(figsize=(10, 10))
                ax2.tick_params(labelsize=16)
                ax2.set_xlim(-7, 7)
                ax2.set_ylim(-7, 7)
                angular_resolution = (
                    self.angular_map_max_angle - self.angular_map_min_angle) / float(self.angular_map_dim)

                cmap = plt.get_cmap('gnuplot')

                for ii in range(self.angular_map_dim):
                    angle_start = (self.angular_map_min_angle +
                                   ii * angular_resolution) * 180 / np.pi + 90
                    angle_end = (self.angular_map_min_angle + (ii + 1)
                                 * angular_resolution) * 180 / np.pi + 90

                    distance_cone = plt.matplotlib.patches.Wedge(
                        (0.0,
                         0.0),
                        self.local_maps_angular[0][ii] *
                        self.angular_map_max_range,
                        angle_start,
                        angle_end,
                        facecolor=cmap(
                            self.local_maps_angular[0][ii]),
                        alpha=0.5)
                    ax2.add_artist(distance_cone)
                time2 = plt.text(6, 8, 'Time: {}'.format(0), fontsize=16)
                ax2.add_artist(time2)

            if deconv is not None:
                fig3, ax3 = plt.subplots(figsize=(10, 10))
                grid_binary = np.zeros_like(deconv[0])
                THRESHOLD_VALUE = 0.9
                indeces = deconv[0] > THRESHOLD_VALUE
                grid_binary[indeces] = 1
                img = Image.fromarray(grid_binary.astype('uint8'))
                img_rotated = img.rotate(90)
                ax3.imshow(img_rotated, cmap="gray", vmin=0, vmax=1)
                time3 = plt.text(5, 5, 'Time: {}'.format(0), fontsize=16)
                ax3.add_artist(time3)

            def update_env(frame_num):
                nonlocal arrows_others
                nonlocal arrow_self
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position(
                        (human.center[0] - x_offset, human.center[1] - y_offset))
                    # if self.attention_weights is not None:
                    #    human.set_color(str(self.attention_weights[frame_num][i]))
                    #    attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))
                arrow_self.remove()
                for arrow in arrows_others:
                    arrow.remove()
                arrow_self = patches.FancyArrowPatch(
                    *orientation_self[frame_num], color='black', arrowstyle=arrow_style)
                arrows_others = [
                    patches.FancyArrowPatch(
                        *orientation[frame_num],
                        color='red',
                        arrowstyle=arrow_style) for orientation in orientations]
                ax.add_artist(arrow_self)
                for arrow in arrows_others:
                    ax.add_artist(arrow)
                for i, other_robot in enumerate(other_robots):
                    other_robot.center = other_robot_positions[frame_num][i]
                    other_robot_numbers[i].set_position(
                        (other_robot.center[0] - x_offset, other_robot.center[1] - y_offset))

                time.set_text(
                    'Time: {:.2f}'.format(
                        frame_num * self.time_step))

            def update_static_map(frame_num):
                ax2.clear()
                if self.use_grid_map:
                    img = Image.fromarray(
                        self.local_maps[frame_num].astype('uint8'))
                    img_rotated = img.rotate(90)
                    ax2.imshow(img_rotated, cmap="gray", vmin=0, vmax=1)
                    middle_idx = int(
                        round(
                            self.submap_size_m /
                            2.0 /
                            self.map_resolution))
                    robot = plt.Circle((middle_idx, middle_idx), int(
                        round(self.robot.radius / self.map_resolution)), fill=True, color='red')
                    ax2.add_patch(robot)
                else:
                    ax2.set_xlim(-7, 7)
                    ax2.set_ylim(-7, 7)
                    angular_resolution = (
                        self.angular_map_max_angle - self.angular_map_min_angle) / float(self.angular_map_dim)

                    cmap = plt.get_cmap('gnuplot')

                    for ii in range(self.angular_map_dim):
                        angle_start = (
                            self.angular_map_min_angle + ii * angular_resolution) * 180 / np.pi + 90
                        angle_end = (self.angular_map_min_angle + (ii + 1)
                                     * angular_resolution) * 180 / np.pi + 90
                        distance_cone = plt.matplotlib.patches.Wedge(
                            (0.0,
                             0.0),
                            self.local_maps_angular[frame_num][ii] *
                            self.angular_map_max_range,
                            angle_start,
                            angle_end,
                            facecolor=cmap(
                                self.local_maps_angular[frame_num][ii]),
                            alpha=0.5)
                        ax2.add_artist(distance_cone)
                time2.set_text(
                    'Time: {:.2f}'.format(
                        frame_num * self.time_step))
                # ax2.add_artist(time2)

            def update_dec(frame_num):
                ax3.clear()
                grid_binary = np.zeros_like(deconv[frame_num])
                THRESHOLD_VALUE = 0.7
                indeces = deconv[frame_num] > THRESHOLD_VALUE
                grid_binary[indeces] = 1
                img = Image.fromarray(grid_binary.astype('uint8'))
                img_rotated = img.rotate(90)
                ax3.imshow(img_rotated, cmap="gray", vmin=0, vmax=1)
                time3.set_text(
                    'Time: {:.2f}'.format(
                        frame_num * self.time_step))
                ax3.add_artist(time3)

            anim = animation.FuncAnimation(
                fig, update_env, frames=len(
                    self.states), interval=self.time_step * 1000)
            anim.running = True
            anim2 = animation.FuncAnimation(
                fig2, update_static_map, frames=len(
                    self.states), interval=self.time_step * 1000)
            anim2.running = True
            if deconv is not None:
                anim3 = animation.FuncAnimation(
                    fig3, update_dec, frames=len(
                        self.states), interval=self.time_step * 1000)
                anim3.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(
                    fps=8, metadata=dict(
                        artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
                anim2.save(output_file[:-4] + '_map.mp4', writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError
