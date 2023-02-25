# -*- coding: utf-8 -*-
import os
import cv2
import time
import numpy as np
import gym
from gym import utils, spaces
from collections import deque

import torch

import pyastar2d
import range_libc
from CMap2D import flatten_contours, render_contours_in_lidar, CMap2D, CSimAgent
from pose2d import apply_tf_to_vel, inverse_pose2d

import nav_gym_env
from nav_gym_env.utils import (
    angle_correction,
    translation_matrix_from_xyz,
    quaternion_matrix_from_yaw,
    transform_xys
)
from nav_gym_env.map_generator import create_indoor_map, create_outdoor_map
from nav_gym_env.keti_robot import KetiRobot
from nav_gym_env.human import Human
from nav_gym_env.human_policy import HumanPolicy


class NavGymEnv(gym.Env, utils.EzPickle):
    def __init__(
        self,
        robot_type,
        time_step,
        min_turning_radius, # to consider ackermann case. 
        distance_threshold,
        num_scan_stack,
        linvel_range,
        rotvel_range,
        human_v_pref_range,
        human_has_legs_ratio,
        indoor_ratio,
        min_goal_dist,
        max_goal_dist,
        reward_scale,
        reward_success_factor,
        reward_crash_factor,
        reward_progress_factor,
        reward_forward_factor,
        reward_rotation_factor,
        reward_discomfort_factor,
        env_param_range,
    ):
        super(NavGymEnv, self).__init__()

        utils.EzPickle.__init__(
            self,
            robot_type,
            time_step,
            min_turning_radius, # to consider ackermann case. 
            distance_threshold,
            num_scan_stack,
            linvel_range,
            rotvel_range,
            human_v_pref_range,
            human_has_legs_ratio,
            indoor_ratio,
            min_goal_dist,
            max_goal_dist,
            reward_scale,
            reward_success_factor,
            reward_crash_factor,
            reward_progress_factor,
            reward_forward_factor,
            reward_rotation_factor,
            reward_discomfort_factor,
            env_param_range,
        )

        self.robot_type = robot_type
        self.time_step = time_step
        self.min_turning_radius = min_turning_radius
        self.distance_threshold = distance_threshold
        self.num_scan_stack = num_scan_stack
        self.linvel_range = linvel_range
        self.rotvel_range = rotvel_range
        self.human_v_pref_range = human_v_pref_range
        self.human_has_legs_ratio = human_has_legs_ratio
        self.indoor_ratio = indoor_ratio
        self.min_goal_dist = min_goal_dist
        self.max_goal_dist = max_goal_dist
        self.reward_scale = reward_scale
        self.reward_success_factor = reward_success_factor
        self.reward_crash_factor = reward_crash_factor
        self.reward_progress_factor = reward_progress_factor
        self.reward_forward_factor = reward_forward_factor
        self.reward_rotation_factor = reward_rotation_factor
        self.reward_discomfort_factor = reward_discomfort_factor
        self.env_param_range = env_param_range

        # to sumulate human legs
        self.converter_cmap2d = CMap2D()
        self.converter_cmap2d.set_resolution(1.)
        self.distances_travelled_in_base_frame = None

        # to check collision & discomfort
        self.scan_threshold = None
        self.scan_discomfort_threshold = None

        # human simulation related ...
        self.human_policy = HumanPolicy(frames=3, action_space=2)
        self.human_policy.load_state_dict(
            torch.load(
                os.path.join(
                    os.path.dirname(nav_gym_env.__file__), 'human_policy.pth')
                , map_location=torch.device('cpu')
            )
        )

        self.prev_action = np.array([0., 0.])
        self.prev_obs = None
        self.prev_obs_queue = None # deque(maxlen=self.num_scan_stack - 1) 
        self.prev_human_actions = None
        self.prev_humans_obs_queue = None # deque(maxlen=self.num_scan_stack - 1) 
        self.render_obs_txt = ''
        self.render_reward_txt = ''
        self.env_param = None
        self.steps_since_reset = 0

        self._make_scan_threshold()
        self._make_scan_discomfort_threshold()

        self.action_space = spaces.Box(
            low=np.array([self.linvel_range[0], self.rotvel_range[0]]),
            high=np.array([self.linvel_range[1], self.rotvel_range[1]]),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(self.num_scan_stack * self.robot.n_angles + 7,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        })

    def _override_reward_factor(
        self,
        reward_scale=15.,
        reward_success_factor=1,
        reward_crash_factor=1,
        reward_progress_factor=0.001,
        reward_forward_factor=0.0,
        reward_rotation_factor=0.005,
        reward_discomfort_factor=0.01
    ):
        self.reward_scale = reward_scale
        self.reward_success_factor = reward_success_factor
        self.reward_crash_factor = reward_crash_factor
        self.reward_progress_factor = reward_progress_factor
        self.reward_forward_factor = reward_forward_factor
        self.reward_rotation_factor = reward_rotation_factor
        self.reward_discomfort_factor = reward_discomfort_factor

    def _make_scan_threshold(self):
        self.reset()
        self.robot.px = 0.
        self.robot.py = 0.
        self.robot.theta = 0.

        self.contours = [self.robot.threshold_footprint]
        self.scan_threshold = self._compute_scan(self.robot, [], add_scan_noise=False, lidar_legs=False, use_contours=True)
        # print(self.scan_threshold)

    def _make_scan_discomfort_threshold(self):
        self.reset()
        self.robot.px = 0.
        self.robot.py = 0.
        self.robot.theta = 0.

        self.contours = [self.robot.discomfort_threshold_footprint]
        self.scan_discomfort_threshold = self._compute_scan(self.robot, [], add_scan_noise=False, lidar_legs=False, use_contours=True)
        # print(self.scan_discomfort_threshold)

    def _make_render_obs_txt(self, obs):
        observation_dict = observation_to_dict(
            obs['observation'],
            num_scan_stack=self.num_scan_stack,
            n_angles=self.robot.n_angles
        )
        prev_pose = observation_dict['prev_pose']
        pose = observation_dict['pose']
        vel = observation_dict['vel']
        yaw = observation_dict['yaw']
        goal = obs['desired_goal']

        self.render_obs_txt = \
            't: {}\n'.format(self.steps_since_reset) \
            + 'prev_pose: ({:.2f} {:.2f})\n'.format(prev_pose[0], prev_pose[1]) \
            + 'pose: ({:.2f} {:.2f})\n'.format(pose[0], pose[1]) \
            + 'vel: ({:.2f} {:.2f})\n'.format(vel[0], vel[1]) \
            + 'yaw: {:.2f}\n'.format(yaw) \
            + 'goal: ({:.2f} {:.2f})'.format(goal[0], goal[1])

    def _make_render_reward_txt(
        self,
        reward_success, 
        reward_crash,
        reward_progress,
        reward_forward,
        reward_rotation,
        reward_discomfort
    ):
        self.render_reward_txt = \
            'reward_success: {:.5f}\n'.format(reward_success) \
            + 'reward_crash: {:.5f}\n'.format(reward_crash) \
            + 'reward_progress: {:.5f}\n'.format(reward_progress) \
            + 'reward_forward: {:.5f}\n'.format(reward_forward) \
            + 'reward_rotation: {:.5f}\n'.format(reward_rotation) \
            + 'reward_discomfort: {:.5f}'.format(reward_discomfort) 

    def _get_contours(self):
        thresh_occupied = 0.1
        gray = self.map_info['data'].T
        ret, thresh = cv2.threshold(gray.astype(np.float32), thresh_occupied, 1, cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        cv2_output = cv2.findContours(thresh.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if cv2.__version__[0] == '4':
            cont = cv2_output[0]
        elif cv2.__version__[0] == '3':
            cont = cv2_output[1]
        else:
            raise NotImplementedError("cv version {} unsupported".format(cv2.__version__))
        contours = [np.vstack([c[:,0,1], c[:,0,0]]).T for c in cont]
        contours = [batch_ij_to_xy(c, self.map_info) for c in contours]
        contours = [c.tolist() for c in contours]
        return contours

    def _update_dist_travelled(self):
        """ update dist travel var used for animating legs """
        # for each human, get vel in base frame
        for i, human in enumerate(self.humans):
            # dig up rotational velocity from past states log
            vrot = 0.
            if len(self.prev_humans_obs_queue[i]) > 0:
                prev_obs = self.prev_humans_obs_queue[i][-1]
                prev_theta = observation_to_dict(                 
                    prev_obs['observation'], num_scan_stack=3, n_angles=human.n_angles 
                )['yaw']
                vrot = (human.theta
                        - prev_theta) / self.time_step
            # transform world vel to base vel
            baselink_in_world = np.array([human.px, human.py, human.theta])
            world_in_baselink = inverse_pose2d(baselink_in_world)
            vel_in_world_frame = np.array([human.vx, human.vy, vrot])
            vel_in_baselink_frame = apply_tf_to_vel(vel_in_world_frame, world_in_baselink)
            self.distances_travelled_in_base_frame[i, :] += vel_in_baselink_frame * self.time_step

    def _stack_scan(self, obs, prev_obs_queue, num_scan_stack, n_angles):
        # obs: (unstacked)
        scan = obs['observation'][:-7]
        other = obs['observation'][-7:]

        prev_scans = []
        # f the queue is not yet full, replace with the current scan value
        for _ in range(prev_obs_queue.maxlen - len(prev_obs_queue)):
            prev_scans.append(scan)

        for prev_obs in prev_obs_queue:
            # prev_obs: (stacked)
            prev_scan = observation_to_dict(
                prev_obs['observation'],
                num_scan_stack=num_scan_stack,
                n_angles=n_angles
            )['scan']
            prev_scans.append(prev_scan)
        new_obs = obs.copy()
        new_obs['observation'] = np.concatenate(prev_scans + [
            scan, other
        ])
        return new_obs

    def _sample_env_param(self):
        param = dict()
        for key, value in self.env_param_range.items(): 
            if value[1] == 'int':
                param[key] = np.random.choice(
                    np.arange(value[0][0], value[0][1] + 1)
                )
            elif value[1] == 'float':
                param[key] = np.random.uniform(value[0][0], value[0][1])
            else:
                raise NotImplementedError       
        return param

    def _sample_map(self):
        if np.random.random() < self.indoor_ratio:
            self.map_info = create_indoor_map(
                self.env_param['corridor_width'], self.env_param['iterations']
            )
        else:
            self.map_info = create_outdoor_map(
                self.env_param['obstacle_number'], self.env_param['obstacle_width']
            )
        x_min = self.map_info['origin'][0]
        x_max = self.map_info['origin'][0] + self.map_info['width'] * self.map_info['resolution'] 
        y_min = self.map_info['origin'][1]
        y_max = self.map_info['origin'][1] + self.map_info['height'] * self.map_info['resolution'] 
        self.border = [(x_min, x_max), (y_min, y_max)]

        # === define costmap ===
        # =========================================================
        # Define a costmap with a higher resolution to quickly generate start, goal, and path points
        new_resolution = 0.25
        scale = self.map_info['resolution'] / 0.25
        new_height = int(scale * self.map_info['height'])
        new_width = int(scale * self.map_info['width'])
        self.cost_map_info = {
            'data':cv2.resize(
                self.map_info['data'].astype(np.uint8),
                (new_height, new_width),
                interpolation=cv2.INTER_NEAREST
            ),
            'origin': self.map_info['origin'],
            'resolution': new_resolution,
            'width': new_width,
            'height': new_height
        }
        # To avoid generate waypoint near obstacle ...
        kernel = np.ones((9, 9)) # adjacent neighbors in all four directions (up, down, left, and right) 0.25 * 4 = 1m 
        self.cost_map_info['data'] = cv2.filter2D(
            self.cost_map_info['data'].astype(np.uint8), -1, kernel
        ).astype(np.uint8)
        self.cost_map_info['data'][self.cost_map_info['data'] > 0] = 100
        # =========================================================
        # for slow but accurate lidar computations (e.g., collision threshold)
        self.contours = self._get_contours()
        # for fast but inaccurate lidar computations
        MAX_DIST_IJ = self.map_info['data'].shape[0] * self.map_info['data'].shape[1]
        pyomap = range_libc.PyOMap(
            np.ascontiguousarray(self.map_info['data'] >= 0.1))
        self.rmnogpu = range_libc.PyRayMarching(pyomap, MAX_DIST_IJ)

    def _sample_start_goal_path(self, map_info, min_goal_dist, max_goal_dist, start=None, robot_pose=None):
        def find_path(px, py, gx, gy, map_info):  
            grid = np.zeros_like(map_info['data'].T, dtype=np.float32)
            grid[map_info['data'].T == 100] = np.inf
            grid[map_info['data'].T == 0] = 255 

            start_ij = xy_to_ij([px, py], map_info)
            goal_ij = xy_to_ij([gx, gy], map_info)
            path = pyastar2d.astar_path(
                grid, start_ij, goal_ij, allow_diagonal=False)
            if path is not None:
                path = batch_ij_to_xy(path, map_info)
            return path
        fix_start = start is not None
        rs, cs = np.where(map_info['data'].T == 0)
            
        num_try = 0 # debug
        while True:
            num_try += 1
            if num_try > 100:
                print('[sample_start_goal_path] something is wrong...')
                # exit()
                return None, None, None
            if not fix_start:
                start_idx = np.random.choice(np.arange(len(rs)))
                start_i, start_j = rs[start_idx], cs[start_idx]
                start = ij_to_xy([start_i, start_j], map_info)
            # When creating humans, make sure they are not too close to robots from the beginning
            if robot_pose is not None:
                dist_to_robot = np.linalg.norm(robot_pose - start)
                if dist_to_robot < 4:
                    continue

            goal_idx = np.random.choice(np.arange(len(rs)))
            goal_i, goal_j = rs[goal_idx], cs[goal_idx]
            goal = ij_to_xy([goal_i, goal_j], map_info)
            dist = np.linalg.norm(start - goal)
            if min_goal_dist < dist and dist < max_goal_dist:
                path = find_path(start[0], start[1], goal[0], goal[1], map_info)
                if path is not None:
                    break
        return start, goal, path

    def _compute_scan(self, agent, other_agents, add_scan_noise, lidar_legs, use_contours=False):
        lidar_pos = np.array([agent.px, agent.py, agent.theta], dtype=np.float32)
        ranges = np.ones((agent.n_angles,), dtype=np.float32) * agent.range_max
        angles = np.linspace(agent.angle_min,
                             agent.angle_max - agent.angle_increment,
                             agent.n_angles) + lidar_pos[2]
        
        legs = []
        other_agents_contours = []
        for i, other_agent in enumerate(other_agents):
            if agent == other_agent:
                print('something is wrong .. agent = other_agent')
                raise
            if other_agent.has_legs and lidar_legs:
                pos = np.array([other_agent.px, other_agent.py, other_agent.theta], dtype=np.float32)
                dist = self.distances_travelled_in_base_frame[i].astype(np.float32)
                vel = np.array([other_agent.vx, other_agent.vy], dtype=np.float32)
                legs.append(CSimAgent(pos, dist, vel))
            else:
                if other_agent == self.robot:
                    footprint = other_agent.threshold_footprint
                else: 
                    footprint = other_agent.footprint
                transformed_footprint = transform_xys(
                    translation_matrix_from_xyz(other_agent.px, other_agent.py, 0),
                    quaternion_matrix_from_yaw(other_agent.theta),
                    np.concatenate([footprint, [footprint[0]]])
                )
                transformed_footprint = [p.tolist() for p in transformed_footprint]
                other_agents_contours.append(transformed_footprint)

        if use_contours: # slow bug accurate (e.g., collision threshold computation)
            contours = self.contours.copy()
        else:
            origin_ij = xy_to_ij(lidar_pos[:2], self.map_info)
            xythetas = np.zeros((len(ranges), 3))
            xythetas[:, 0] = origin_ij[0]
            xythetas[:, 1] = origin_ij[1]
            xythetas[:, 2] = angles
            xythetas = xythetas.astype(np.float32)
            self.rmnogpu.calc_range_many(xythetas, ranges)        
            ranges *= self.map_info['resolution']
            contours = []
        contours.extend(other_agents_contours)
        if contours:
            flat_contours = flatten_contours(contours)
            render_contours_in_lidar(ranges, angles, flat_contours, lidar_pos[:2])
        self.converter_cmap2d.render_agents_in_lidar(ranges, angles, legs, lidar_pos[:2])

        # cut
        ranges = np.clip(ranges, 0, agent.range_max)

        if add_scan_noise: # do not add noise when we make collision threshold
            ranges[ranges != agent.range_max] = ranges[ranges != agent.range_max] + np.random.normal(
                0, self.env_param['scan_noise_std'], len(ranges[ranges != agent.range_max])
            )
        return ranges

    def _convert_obs(
        self, agent, othet_agents, prev_obs, prev_action, add_scan_noise, lidar_legs
    ):
        scan = self._compute_scan(agent, othet_agents, add_scan_noise, lidar_legs)

        pose = np.array([agent.px, agent.py])
        if prev_obs is None:
            prev_pose = pose
        else:
            prev_pose = prev_obs['achieved_goal']
        vel = np.array(prev_action)
        yaw = angle_correction(agent.theta)
        observation = np.concatenate([scan, prev_pose, pose, vel, [yaw]])

        obs = {
            'observation': observation,
            'achieved_goal': pose, 
            'desired_goal': np.array([agent.gx, agent.gy])
        }
        return obs

    def compute_info(self, obs):
        observation_dict = observation_to_dict(
            obs['observation'],
            num_scan_stack=self.num_scan_stack,
            n_angles=self.robot.n_angles
        )
        scan = observation_dict['scan']
        pose = observation_dict['pose']
        goal = obs['desired_goal']

        distance = np.linalg.norm(pose - goal, axis=-1)
        success = (distance < self.distance_threshold).astype(np.float32)
        crash = np.any(scan < self.scan_threshold).astype(np.float32)
        info = {
            'is_success': success, 
            'is_crash': crash, 
            'distance': distance,
        }
        return info

    def compute_done(self, obs):
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_terminals(next_obs)[0]

    # used in HER
    def compute_terminals(self, obs):
        """batch computation for fast batch sampling in HER
        10x times faster with batch_size>2000 !
        """
        observations = obs['observation']
        observations_dict = observation_batch_to_dict(
            observations,
            num_scan_stack=self.num_scan_stack,
            n_angles=self.robot.n_angles
        )
        desired_goals = obs['desired_goal']

        scan = observations_dict['scan'] 
        pose = observations_dict['pose'] 

        distance = np.linalg.norm(desired_goals - pose, axis=1)

        success = (distance < self.distance_threshold).astype(np.float32)
        crash = scan - self.scan_threshold
        crash = np.any(crash < 0, axis=1).astype(np.float32)
        done = np.logical_or(success, crash)
        return done # batch!

    def compute_reward(self, action, obs, make_render_reward_txt=False):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs, make_render_reward_txt)[0]

    def compute_rewards(self, actions, obs, make_render_reward_txt=False):
        """batch computation for fast batch sampling in HER
        10x times faster with batch_size>2000 !
        """
        observations = obs['observation']
        observations_dict = observation_batch_to_dict(
            observations,
            num_scan_stack=self.num_scan_stack,
            n_angles=self.robot.n_angles
        )
        desired_goals = obs['desired_goal']

        scan = observations_dict['scan'] 
        prev_pose = observations_dict['prev_pose'] 
        pose = observations_dict['pose'] 
        vel = observations_dict['vel'] 
        yaw = observations_dict['yaw'] 

        distance = np.linalg.norm(desired_goals - pose, axis=1)
        prev_distance = np.linalg.norm(desired_goals - prev_pose, axis=1)

        success = (distance < self.distance_threshold).astype(np.float32)
        crash = scan - self.scan_threshold
        crash = np.any(crash < 0, axis=1).astype(np.float32)
        discomfort = scan - self.scan_discomfort_threshold
        discomfort = np.any(discomfort < 0, axis=1).astype(np.float32)
        discomfort = np.logical_and(discomfort, np.logical_not(crash))

        batch_size = observations.shape[0]
        reward_success = np.zeros(batch_size)
        reward_crash = np.zeros(batch_size)
        reward_progress = np.zeros(batch_size)
        reward_forward = np.zeros(batch_size)
        reward_rotation = np.zeros(batch_size)
        reward_discomfort = np.zeros(batch_size)

        reward_success[success.astype(np.bool)] = 1.0 * self.reward_success_factor * self.reward_scale
        reward_crash[crash.astype(np.bool)] = -1.0 * self.reward_crash_factor * self.reward_scale
        reward_progress = (prev_distance - distance) * self.reward_progress_factor * self.reward_scale
        reward_forward = vel[:, 0] * self.reward_forward_factor * self.reward_scale
        # reward_rotation = -np.abs(vel[:, 1]) * self.reward_rotation_factor * self.reward_scale
        reward_rotation = -1.0 * (vel[:, 1] ** 2) * self.reward_rotation_factor * self.reward_scale
        reward_discomfort[discomfort.astype(np.bool)] = -(1.0 - np.min(
            np.divide(
                scan[discomfort.astype(np.bool)] - self.scan_threshold,
                self.scan_discomfort_threshold - self.scan_threshold + 1e-6
            ),
            axis=1
        )) * self.reward_discomfort_factor * self.reward_scale

        reward = (
            reward_success \
            + reward_crash \
            + reward_progress \
            + reward_forward \
            + reward_rotation \
            + reward_discomfort
        )

        if make_render_reward_txt:
            self._make_render_reward_txt(
                reward_success.mean(),
                reward_crash.mean(),
                reward_progress.mean(),
                reward_forward.mean(),
                reward_rotation.mean(),
                reward_discomfort.mean()
            )
        return reward # batch!

    def step(self, action):
        self.steps_since_reset += 1

        action = np.array(action)
        if self.min_turning_radius > 0:
            if action[0] >= 0:
            # Set the minimum linear velocity to satisfy the minimum turning radius
                action[0] = max(action[0], np.abs(action[1]) * self.min_turning_radius)
            else:
                action[0] = min(action[0], -np.abs(action[1]) * self.min_turning_radius)

            # if np.abs(action[0]) < np.abs(action[1]) * self.min_turning_radius:
            #     action[0] = 0.
            #     action[1] = 0.

        if action[0] < self.linvel_range[0] or action[0] > self.linvel_range[1]:
            print('linvel {} is out of range {}'.format(action[0], self.linvel_range))
        if action[1] < self.rotvel_range[0] or action[0] > self.rotvel_range[1]:
            print('rotvel {} is out of range {}'.format(action[1], self.rotvel_range))

        # comment out below to take valid action
        # action[0] = np.clip(action[0], self.linvel_range[0], self.linvel_range[1])
        # action[1] = np.clip(action[1], self.rotvel_range[0], self.rotvel_range[1])

        linvel, rotvel = action

        scan_batch = np.zeros((len(self.humans), 3, 512))
        goal_batch = np.zeros((len(self.humans), 2))
        speed_batch = self.prev_human_actions

        for i, human in enumerate(self.humans):
            observation_dict = observation_to_dict(
                self.prev_humans_obs_queue[i][-1]['observation'],
                num_scan_stack=3, # human use 3 stack rule
                n_angles=human.n_angles
            )
            scan_stack = observation_dict['scan_stack']
            # preprocess for pretrained human policy
            scan_stack = np.clip(scan_stack, 0., 6.)
            scan_stack = scan_stack / 6.0 - 0.5  

            # set goal from waypoints
            while True:
                if len(human.waypoints) == 1:
                    break 
                local_goal_distance = np.linalg.norm(np.array([human.px, human.py]) - np.array(human.waypoints[0]))
                if local_goal_distance < 1.:
                    human.waypoints = human.waypoints[1:]
                else:
                    break
            human.gx = human.waypoints[0][0]
            human.gy = human.waypoints[0][1]
            # local goal (relative positions)
            local_x = (human.gx - human.px) * np.cos(human.theta) + (human.gy - human.py) * np.sin(human.theta)
            local_y = -(human.gx - human.px) * np.sin(human.theta) + (human.gy - human.py) * np.cos(human.theta)

            scan_batch[i, :, :] = scan_stack[-512:] # scan_stack.reshape(3, 512)
            goal_batch[i, :] = np.array([local_x, local_y])

        scan_batch = torch.from_numpy(scan_batch).float()
        goal_batch = torch.from_numpy(goal_batch).float()
        speed_batch = torch.from_numpy(speed_batch).float()

        _, _, _, mean = self.human_policy(scan_batch, goal_batch, speed_batch)
        mean = mean.data.cpu().numpy()
        action_bound = [[0, -1], [1, 1]]
        human_actions = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])
        self.prev_human_actions = human_actions
        for i, human in enumerate(self.humans):
            factor = human.v_pref / 1.0
            human_action = human_actions[i] * factor 
            self.humans[i].set_vel(human_action[0], human_action[1])

        self.robot.set_vel(action[0], action[1])

        # if human reach goal, set new goal
        for human in self.humans:
            pose = np.array([human.px, human.py])
            d = np.linalg.norm(pose - np.array(human.waypoints[-1]))
            if d < 0.5:
                _, _, path = self._sample_start_goal_path(
                    self.cost_map_info, 
                    10,
                    np.inf,
                    start=np.array([human.px, human.py])
                )
                # Only create a new path if the human is not adjacent to a wall!
                if path is not None:
                    waypoints = path_to_waypoints(path, interval=2) 
                    human.waypoints = waypoints

        # human legs informations
        self._update_dist_travelled() 
        # keep human obs
        for i, human in enumerate(self.humans):
            other_agents = [self.robot] + [h for h in self.humans if h != human]
            human_obs = self._convert_obs(
                human, other_agents, self.prev_obs, self.prev_action,
                add_scan_noise=False, lidar_legs=False
            )
            # stack
            human_obs = self._stack_scan(human_obs, self.prev_humans_obs_queue[i], 3, human.n_angles)
            self.prev_humans_obs_queue[i].append(human_obs)

        obs = self._convert_obs(
            self.robot, self.humans, self.prev_obs, self.prev_action,
            add_scan_noise=True, lidar_legs=True
        )
        # stack
        obs = self._stack_scan(obs, self.prev_obs_queue, self.num_scan_stack, self.robot.n_angles)
        reward = self.compute_reward(action, obs, make_render_reward_txt=True)
        done = self.compute_done(obs)
        info = self.compute_info(obs)

        self._make_render_obs_txt(obs)
        
        # if info['is_crash']: # if crash happens, place robot on prev pose
        #     # print(' [!] Crash!')
        #     # replace obs to prev_obs
        #     observation_dict = observation_to_dict(
        #         self.prev_obs['observation'],
        #         num_scan_stack=self.num_scan_stack,
        #         n_angles=self.robot.n_angles
        #     )
        #     self.robot.px = observation_dict['pose'][0]
        #     self.robot.py = observation_dict['pose'][1]
        #     self.robot.theta = observation_dict['yaw']
        #     obs = self._convert_obs(
        #         self.robot, self.humans, self.prev_obs, self.prev_action,
        #         add_scan_noise=True, lidar_legs=True
        #     )
        #     # stack
        #     obs = self._stack_scan(obs, self.prev_obs_queue, self.num_scan_stack, self.robot.n_angles)

        self.prev_action = action
        self.prev_obs = obs
        self.prev_obs_queue.append(obs)
        return obs, reward, done, info

    def reset(self):
        # sample env param every episode to make diverse situations
        self.env_param = self._sample_env_param()

        # to keep track previous information
        self.steps_since_reset = 0
        self.prev_action = np.array([0., 0.])
        self.prev_obs = None
        self.prev_obs_queue = deque(maxlen=self.num_scan_stack - 1) 
        self.prev_human_actions = np.zeros((self.env_param['num_humans'], 2))
        self.prev_humans_obs_queue = [ # human use 3 stack scan
            deque(maxlen=2) for _ in range(self.env_param['num_humans'])
        ]
        self._make_render_reward_txt(0, 0, 0, 0, 0, 0)

        # generate random map
        self._sample_map()
        # === make robot ===
        while True:
            start, goal, path = self._sample_start_goal_path(
                self.cost_map_info, 
                self.min_goal_dist,
                self.max_goal_dist,
            )
            # if path is None:
            #     continue
            waypoints = path_to_waypoints(path, interval=5)
            path_distance = np.linalg.norm(start - waypoints[0])
            for wi in range(len(waypoints) - 1):
                path_distance += np.linalg.norm(waypoints[wi + 1] - waypoints[wi])
            # Ignore goals that would require driving through excessively convoluted paths
            if path_distance > 2.0 * np.linalg.norm(goal - start):
                continue
            robot_theta = np.random.uniform(0, 2 * np.pi)

            # spawn robot with specified positions and orientation 
            if self.robot_type == 'keti':
                self.robot = KetiRobot(
                    start[0], start[1], robot_theta,
                    goal[0], goal[1],
                    self.time_step
                )
            else:
                raise NotImplementedError

            # Initially, the robot is created without a threshold in order to create it first
            if self.scan_discomfort_threshold is None:
                break

            scan = self._compute_scan(self.robot, [], add_scan_noise=True, lidar_legs=True)
            discomfort = scan - self.scan_discomfort_threshold
            discomfort = np.any(discomfort < 0).astype(np.float32)
            if not discomfort:
                break

        # === make human === 
        self.humans = []
        for _ in range(self.env_param['num_humans']):
            start, goal, path = self._sample_start_goal_path(
                self.cost_map_info, 
                10,
                np.inf,
                robot_pose=np.array([self.robot.px, self.robot.py])
            )
            human_theta = np.random.uniform(0, 2 * np.pi)
            human = Human(
                start[0], start[1], human_theta,
                goal[0], goal[1],
                self.time_step
            )
            human.v_pref = np.random.uniform(
                self.human_v_pref_range[0], self.human_v_pref_range[1]
            )
            human.has_legs = np.random.random() < self.human_has_legs_ratio
            waypoints = path_to_waypoints(path, interval=2) 
            human.waypoints = waypoints
            self.humans.append(human)

        # human legs informations
        self.distances_travelled_in_base_frame = np.zeros((len(self.humans), 3))

        # keep human obs
        for i, human in enumerate(self.humans):
            other_agents = [self.robot] + [h for h in self.humans if h != human]
            human_obs = self._convert_obs(
                human, other_agents, self.prev_obs, self.prev_action,
                add_scan_noise=False, lidar_legs=False
            )
            # stack
            human_obs = self._stack_scan(human_obs, self.prev_humans_obs_queue[i], 3, human.n_angles)
            self.prev_humans_obs_queue[i].append(human_obs)

        obs = self._convert_obs(
            self.robot, self.humans, self.prev_obs, self.prev_action,
            add_scan_noise=True, lidar_legs=True
        )
        # stack
        obs = self._stack_scan(obs, self.prev_obs_queue, self.num_scan_stack, self.robot.n_angles)
        # TODO: remove prev_obs
        self.prev_obs = obs
        self.prev_obs_queue.append(obs)
        return obs

    def render(self, mode='human'):
        HEIGHT, WIDTH = 800, 800 # a good size to view

        if mode == 'human':
            img = self.map_info['data'].copy().astype(np.float32)
            img[img == 0] = 1 # free space -> white
            img[img == 100] = 0 # obstacle -> black
            img = cv2.merge([img, img, img])

            # goal
            i,j = xy_to_ij([self.robot.gx, self.robot.gy], self.map_info)
            r = xy_to_ij([1, 0], self.map_info)[0]
            img = cv2.rectangle(
                img,
                (i - r, j - r),
                (i + r, j + r),
                (0, 0, 1),
                -1 # not fill
            )

            # humans local goal
            for human in self.humans:  
                r = xy_to_ij([0.2, 0], self.map_info)[0]
                i,j = xy_to_ij(np.array([human.gx, human.gy]), self.map_info)
                img = cv2.rectangle(
                    img,
                    (i - r, j - r),
                    (i + r, j + r),
                    (1, 1, 0),
                    -1 # not fill
                )

            # draw humans
            for human in self.humans:
                i,j = xy_to_ij([human.px, human.py], self.map_info)
                di, dj = xy_to_ij(
                    [0.6 * np.cos(human.theta), 0.6 * np.sin(human.theta)], 
                    self.map_info,
                    clip_if_outside=False
                )
                img = cv2.arrowedLine(
                    img, (i, j), (i + di, j + dj), 
                    (0, 0, 0), # color
                    xy_to_ij([0.2, 0], self.map_info)[0], # thickness (0.2m)
                )

                transformed_footprint = transform_xys(
                    translation_matrix_from_xyz(human.px, human.py, 0),
                    quaternion_matrix_from_yaw(human.theta),
                    np.concatenate([human.footprint, [human.footprint[0]]])
                )
                for i in range(len(transformed_footprint) - 1):
                    p1 = transformed_footprint[i]
                    p2 = transformed_footprint[i + 1]
                    p1_ij = xy_to_ij(p1, self.map_info)
                    p2_ij = xy_to_ij(p2, self.map_info)
                    cv2.line(
                        img,
                        tuple(p1_ij),
                        tuple(p2_ij),
                        (0, 0, 0), # color
                        1
                    )

            # draw robot arrow
            i,j = xy_to_ij([self.robot.px, self.robot.py], self.map_info)
            di, dj = xy_to_ij(
                [0.8 * np.cos(self.robot.theta), 0.8 * np.sin(self.robot.theta)], 
                self.map_info,
                clip_if_outside=False
            )
            img = cv2.arrowedLine(
                img, (i, j), (i + di, j + dj), 
                (0, 0, 0), # color
                xy_to_ij([0.2, 0], self.map_info)[0], # thickness (0.2m)
            )            

            # draw footprint
            transformed_footprint = transform_xys(
                translation_matrix_from_xyz(self.robot.px, self.robot.py, 0),
                quaternion_matrix_from_yaw(self.robot.theta),
                np.concatenate([self.robot.footprint, [self.robot.footprint[0]]])
            )
            for i in range(len(transformed_footprint) - 1):
                p1 = transformed_footprint[i]
                p2 = transformed_footprint[i + 1]
                p1_ij = xy_to_ij(p1, self.map_info)
                p2_ij = xy_to_ij(p2, self.map_info)
                cv2.line(
                    img,
                    tuple(p1_ij),
                    tuple(p2_ij),
                    (0, 0, 0), # color
                    1
                )
            # threshold footprint (collision check)
            transformed_threshold_footprint = transform_xys(
                translation_matrix_from_xyz(self.robot.px, self.robot.py, 0),
                quaternion_matrix_from_yaw(self.robot.theta),
                np.concatenate([self.robot.threshold_footprint, [self.robot.threshold_footprint[0]]])
            )
            for i in range(len(transformed_threshold_footprint) - 1):
                p1 = transformed_threshold_footprint[i]
                p2 = transformed_threshold_footprint[i + 1]
                p1_ij = xy_to_ij(p1, self.map_info)
                p2_ij = xy_to_ij(p2, self.map_info)
                cv2.line(
                    img,
                    tuple(p1_ij),
                    tuple(p2_ij),
                    (0, 0, 0), # color
                    1
                )
            # discomfort threshold footprint (discomfort check)
            transformed_discomfort_threshold_footprint = transform_xys(
                translation_matrix_from_xyz(self.robot.px, self.robot.py, 0),
                quaternion_matrix_from_yaw(self.robot.theta),
                np.concatenate([self.robot.discomfort_threshold_footprint, [self.robot.discomfort_threshold_footprint[0]]])
            )
            for i in range(len(transformed_discomfort_threshold_footprint) - 1):
                p1 = transformed_discomfort_threshold_footprint[i]
                p2 = transformed_discomfort_threshold_footprint[i + 1]
                p1_ij = xy_to_ij(p1, self.map_info)
                p2_ij = xy_to_ij(p2, self.map_info)
                cv2.line(
                    img,
                    tuple(p1_ij),
                    tuple(p2_ij),
                    (0, 0, 0), # color
                    1
                )

            # lidar
            observation_dict = observation_to_dict(
                self.prev_obs['observation'],
                num_scan_stack=self.num_scan_stack,
                n_angles=self.robot.n_angles
            )
            scan = observation_dict['scan']
            theta = observation_dict['yaw']

            angles = np.linspace(self.robot.angle_min,
                                 self.robot.angle_max - self.robot.angle_increment,
                                 self.robot.n_angles) + theta

            lidar_points_x = self.robot.px + scan * np.cos(angles)
            lidar_points_y = self.robot.py + scan * np.sin(angles)
            lidar_points = np.hstack([
                lidar_points_x[:, None],
                lidar_points_y[:, None]
            ])  
            lidar_ijs = batch_xy_to_ij(lidar_points, self.map_info)
            for ij in lidar_ijs[scan != self.robot.range_max]: # do not plot max distance range
                i,j = ij
                img = cv2.circle(
                    img, 
                    (i, j),
                    xy_to_ij([0.2, 0], self.map_info)[0], # radius on i-j space
                    (0., 1., 0.), # BGR color
                    -1 # -1 means fill the circle
                )

            ## debug (human[0] lidar)
            # human_0 = self.humans[0]
            # observation_dict = observation_to_dict(
            #     self.prev_humans_obs_queue[0][-1]['observation'],
            #     num_scan_stack=3, # human use 3 stack rule
            #     n_angles=human_0.n_angles
            # )
            # scan = observation_dict['scan']
            # theta = observation_dict['yaw']

            # angles = np.linspace(human_0.angle_min,
            #                      human_0.angle_max - human_0.angle_increment,
            #                      human_0.n_angles) + theta

            # lidar_points_x = human_0.px + scan * np.cos(angles)
            # lidar_points_y = human_0.py + scan * np.sin(angles)
            # lidar_points = np.hstack([
            #     lidar_points_x[:, None],
            #     lidar_points_y[:, None]
            # ])  
            # lidar_ijs = batch_xy_to_ij(lidar_points, self.map_info)
            # for ij in lidar_ijs[scan != human_0.range_max]: # do not plot max distance range
            #     i,j = ij
            #     img = cv2.circle(
            #         img, 
            #         (i, j),
            #         xy_to_ij([0.2, 0], self.map_info)[0], # radius on i-j space
            #         (0., 0., 1.), # BGR color
            #         -1 # -1 means fill the circle
            #     )

            """
            Important: Except for the text, it should be drawn above this (flipped and rescaled from below)
            """
            # Flip it vertically to draw it in the world coordinate system
            img = np.flipud(img).copy()

            # resize for ease of viewing
            img = cv2.resize(img, (WIDTH, HEIGHT))

            # debug text
            for i, txt in enumerate(self.render_obs_txt.split('\n') + self.render_reward_txt.split('\n')):
                img = cv2.putText(
                    img, 
                    txt, 
                    (50, 50 + i * 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, # fontscale
                    (0, 0, 1), 
                    2, # thickness
                    cv2.LINE_AA
                ) 

            cv2.imshow("NavGym Env", img)

            cv2.waitKey(1)
        elif mode == 'rgb_array':
            img = self.map_info['data'].copy().astype(np.float32)
            img[img == 0] = 1 # free space -> white
            img[img == 100] = 0 # obstacle -> black
            img = cv2.merge([img, img, img])

            # draw circle, arrow of humans, robot
            for n, agent in enumerate([self.soadrl_sim.robot] + self.soadrl_sim.humans):
                px, py = agent.px, agent.py
                angle = agent.theta
                r = agent.radius
                i,j = xy_to_ij([px, py], self.map_info)
                if n == 0: # robot
                    img = cv2.circle(
                        img, 
                        (i, j),
                        xy_to_ij([self.robot_radius, 0], self.map_info)[0], # i,j space 상에서 radius
                        (0., 0., 0.), # BGR color
                        1 # -1 means fill the circle
                    )
                else:
                    img = cv2.circle(
                        img, 
                        (i, j),
                        xy_to_ij([r, 0], self.map_info)[0], # i,j space 상에서 radius
                        (1., 0., 0.), # BGR color
                        -1 # -1 means fill the circle
                    )
                di, dj = xy_to_ij(
                    [0.8 * np.cos(angle), 0.8 * np.sin(angle)], 
                    self.map_info,
                    clip_if_outside=False
                )
                img = cv2.arrowedLine(
                    img, (i, j), (i + di, j + dj), 
                    (0, 0, 0), # color
                    xy_to_ij([0.2, 0], self.map_info)[0], # thickness (0.2m)
                )

            # draw footprint
            robot = self.soadrl_sim.robot
            transformed_footprint = transform_xys(
                translation_matrix_from_xyz(robot.px, robot.py, 0),
                quaternion_matrix_from_yaw(robot.theta),
                np.concatenate([self.config.footprint, [self.config.footprint[0]]])
            )
            for i in range(len(transformed_footprint) - 1):
                p1 = transformed_footprint[i]
                p2 = transformed_footprint[i + 1]
                p1_ij = xy_to_ij(p1, self.map_info)
                p2_ij = xy_to_ij(p2, self.map_info)
                cv2.line(
                    img,
                    tuple(p1_ij),
                    tuple(p2_ij),
                    (0, 0, 0), # color
                    1
                )
            # threshold footprint (collision check)
            transformed_threshold_footprint = transform_xys(
                translation_matrix_from_xyz(robot.px, robot.py, 0),
                quaternion_matrix_from_yaw(robot.theta),
                np.concatenate([self.config.threshold_footprint, [self.config.threshold_footprint[0]]])
            )
            for i in range(len(transformed_threshold_footprint) - 1):
                p1 = transformed_threshold_footprint[i]
                p2 = transformed_threshold_footprint[i + 1]
                p1_ij = xy_to_ij(p1, self.map_info)
                p2_ij = xy_to_ij(p2, self.map_info)
                cv2.line(
                    img,
                    tuple(p1_ij),
                    tuple(p2_ij),
                    (0, 0, 0), # color
                    1
                )
            # discomfort threshold footprint (discomfort check)
            transformed_discomfort_threshold_footprint = transform_xys(
                translation_matrix_from_xyz(robot.px, robot.py, 0),
                quaternion_matrix_from_yaw(robot.theta),
                np.concatenate([self.config.discomfort_threshold_footprint, [self.config.discomfort_threshold_footprint[0]]])
            )
            for i in range(len(transformed_discomfort_threshold_footprint) - 1):
                p1 = transformed_discomfort_threshold_footprint[i]
                p2 = transformed_discomfort_threshold_footprint[i + 1]
                p1_ij = xy_to_ij(p1, self.map_info)
                p2_ij = xy_to_ij(p2, self.map_info)
                cv2.line(
                    img,
                    tuple(p1_ij),
                    tuple(p2_ij),
                    (0, 0, 0), # color
                    1
                )

            # lidar
            lidar_points_x = robot.px + self.lidar_scan * np.cos(self.lidar_angles)
            lidar_points_y = robot.py + self.lidar_scan * np.sin(self.lidar_angles)
            lidar_points = np.hstack([
                lidar_points_x[:, None],
                lidar_points_y[:, None]
            ])  
            lidar_ijs = batch_xy_to_ij(lidar_points, self.map_info)
            for ij in lidar_ijs[self.lidar_scan != self.range_max]: # do not plot max distance range
                i,j = ij
                img = cv2.circle(
                    img, 
                    (i, j),
                    xy_to_ij([0.2, 0], self.map_info)[0], # i,j space 상에서 radius
                    (0., 1., 0.), # BGR color
                    -1 # -1 means fill the circle
                )

            # goal
            i,j = xy_to_ij([robot.gx, robot.gy], self.map_info)
            r = xy_to_ij([1, 0], self.map_info)[0]
            img = cv2.rectangle(
                img,
                (i - r, j - r),
                (i + r, j + r),
                (0, 0, 1),
                1 # not fill
            )

            # # debug
            # for p in self.debug:
            #     # r = agent.radius
            #     i,j = xy_to_ij([p[0], p[1]], self.map_info)
            #     img = cv2.circle(
            #         img, 
            #         (i, j),
            #         xy_to_ij([0.1, 0], self.map_info)[0], # i,j space 상에서 radius
            #         (0., 0., 0.), # BGR color
            #         -1 # -1 means fill the circle
            #     )      

            """
            중요: 글자 제외하고는 여기보다 위에 그려야 (아래에서 뒤집고 rescale해서)
            """
            # 워래 좌표계로 그려주기위해 위아래 뒤집기
            img = np.flipud(img).copy()

            # 보기 편하게 resize
            img = cv2.resize(img, (WIDTH, HEIGHT))

            # debug text
            for i, txt in enumerate(self.render_obs_txt.split('\n') + self.render_reward_txt.split('\n')):
                img = cv2.putText(
                    img, 
                    txt, 
                    (50, 50 + i * 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, # fontscale
                    (0, 0, 1), 
                    2, # thickness
                    cv2.LINE_AA
                )
            img *= 255
            img = img.astype(np.uint8)
            return img
        else:
            raise NotImplementedError

def batch_ij_to_xy(ij, map_info):
    resolution = map_info['resolution']
    origin = map_info['origin']
    i, j = np.split(np.array(ij), 2, axis=-1)
    x = (i + 0.5) * resolution + origin[0]
    y = (j + 0.5) * resolution + origin[1]
    return np.concatenate([x, y], axis=-1)


def ij_to_xy(ij, map_info):
    ij = np.array(ij)
    return batch_ij_to_xy(ij[None, :], map_info)[0]


def batch_xy_to_ij(xy, map_info, clip_if_outside=True):
    resolution = map_info['resolution'] 
    origin = map_info['origin']
    height = map_info['height'] 
    width = map_info['width']
    if type(xy) is not np.ndarray:
        xy = np.array(xy)
    ij = np.zeros_like(xy, dtype=np.float32)

    if xy.shape[1] != 2:
        raise IndexError("xy should be of shape (n, 2)")
    for k in range(xy.shape[0]):
        ij[k, 0] = (xy[k, 0] - origin[0]) / resolution
        ij[k, 1] = (xy[k, 1] - origin[1]) / resolution
    if clip_if_outside:
        for k in range(xy.shape[0]):
            if ij[k, 0] >= height:
                ij[k, 0] = height - 1
            if ij[k, 1] >= width:
                ij[k, 1] = width - 1
            if ij[k, 0] < 0:
                ij[k, 0] = 0
            if ij[k, 1] < 0:
                ij[k, 1] = 0

    return ij.astype(np.int64)


def xy_to_ij(xy, map_info, clip_if_outside=True):
    xy = np.array(xy)
    return batch_xy_to_ij(xy[None, :], map_info, clip_if_outside)[0]


def path_to_waypoints(path, interval):
    waypoints = []
    tmp_path = path.copy()
    while True:
        d = np.linalg.norm(
            tmp_path[0] - tmp_path,
            axis=-1
        )
        waypoint_idxs = np.where([d > interval])[1]
        if len(waypoint_idxs) > 0:
            waypoints.append(tmp_path[waypoint_idxs[0]])
            tmp_path = tmp_path[waypoint_idxs[0]:]
        else:
            waypoints.append(tmp_path[-1])
            break
    waypoints = np.array(waypoints)
    return np.array(waypoints)


def observation_to_dict(observation, num_scan_stack, n_angles):
    # Return multiple useful information separated from a stacked observation
    scan_stack = observation[:num_scan_stack * n_angles]
    scan = observation[(num_scan_stack - 1) * n_angles:num_scan_stack * n_angles]
    other = observation[num_scan_stack * n_angles:]
    prev_pose = other[:2]
    pose = other[2:4]
    vel = other[4:6]
    yaw = other[6]
    return dict(
        scan_stack=scan_stack,
        scan=scan,
        prev_pose=prev_pose,
        pose=pose,
        vel=vel,
        yaw=yaw        
    )


def observation_batch_to_dict(observation, num_scan_stack, n_angles):
    # Return multiple useful information separated from a stacked observation
    scan_stack = observation[:, :num_scan_stack * n_angles]
    scan = observation[:, (num_scan_stack - 1) * n_angles:num_scan_stack * n_angles]
    other = observation[:, num_scan_stack * n_angles:]
    prev_pose = other[:, :2]
    pose = other[:, 2:4]
    vel = other[:, 4:6]
    yaw = other[:, 6]
    return dict(
        scan_stack=scan_stack,
        scan=scan,
        prev_pose=prev_pose,
        pose=pose,
        vel=vel,
        yaw=yaw        
    )


if __name__ == "__main__":
    env = NavGymEnv(
        robot_type='keti',
        time_step=0.2,
        min_turning_radius=0., # wheel_base / 2
        distance_threshold=0.5, # less then it -> reach goal
        num_scan_stack=1,
        linvel_range=[0, 0.5],
        rotvel_range=[-0.64, 0.64],
        human_v_pref_range=[0., 0.6],
        human_has_legs_ratio=0.5,
        indoor_ratio=0.5,
        min_goal_dist=10, 
        max_goal_dist=20, 
        reward_scale=15.,
        reward_success_factor=1,
        reward_crash_factor=1,
        reward_progress_factor=0.001,
        reward_forward_factor=0.0,
        reward_rotation_factor=0.005,
        reward_discomfort_factor=0.01,
        env_param_range=dict(
            num_humans=([5, 15], 'int'),
            # indoor map param
            corridor_width=([3, 4], 'int'),
            iterations=([80, 150], 'int'),
            # outdoor map param
            obstacle_number=([10, 10], 'int'), # fix
            obstacle_width=([0.3, 1.0], 'float'),
            scan_noise_std=([0., 0.05], 'float'), # fix
        )
    )
    env.reset()

    for _ in range(1000):
        env.step(env.action_space.sample())
        time.sleep(0.01)
        env.render()