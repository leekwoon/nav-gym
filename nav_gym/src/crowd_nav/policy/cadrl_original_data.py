import logging

import torch.nn as nn
import numpy as np

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot
import crowd_nav.cadrl_utils.agent as agent
import crowd_nav.cadrl_utils.util as util
import crowd_nav.cadrl_utils.network as network


class CADRL_ORIGINAL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'CADRL_ORIGINAL'

    def configure(self, config):
        self.possible_actions = network.Actions()
        self.FOV_min_angle = config.getfloat(
            'map', 'angle_min') * np.pi % (2 * np.pi)
        self.FOV_max_angle = config.getfloat(
            'map', 'angle_max') * np.pi % (2 * np.pi)
        self.device = network.Config.DEVICE

        logging.info('Policy: CADRL_Original without occupancy map')

    def load_model(self, model_weights):
        num_actions = self.possible_actions.num_actions
        self.net = network.NetworkVP_rnn(
            network.Config.DEVICE, 'network', num_actions)
        self.net.simple_load('../../checkpoints/network_01900000')

    def predict(self, state):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.reach_destination(state):
            return ActionRot(0, 0)

        host_agent = agent.Agent(
            state.self_state.px,
            state.self_state.py,
            state.self_state.gx,
            state.self_state.gy,
            state.self_state.radius,
            state.self_state.v_pref,
            state.self_state.theta,
            0)
        host_agent.vel_global_frame = np.array([state.self_state.vx,
                                                state.self_state.vy])

        other_agents = []
        for i, human_state in enumerate(state.human_states):
            if self.human_state_in_FOV(state.self_state, human_state):
                x = human_state.px
                y = human_state.py
                v_x = human_state.vx
                v_y = human_state.vy
                heading_angle = np.arctan2(v_y, v_x)
                pref_speed = np.linalg.norm(np.array([v_x, v_y]))
                goal_x = x + 5.0
                goal_y = y + 5.0
                other_agents.append(agent.Agent(x, y, goal_x, goal_y,
                                                human_state.radius, pref_speed,
                                                heading_angle, i + 1))
                other_agents[-1].vel_global_frame = np.array([v_x, v_y])
        obs = host_agent.observe(other_agents)[1:]
        obs = np.expand_dims(obs, axis=0)
        predictions = self.net.predict_p(obs, None)[0]
        raw_action = self.possible_actions.actions[np.argmax(predictions)]
        action = ActionRot(
            host_agent.pref_speed *
            raw_action[0],
            util.wrap(
                raw_action[1]))
        return action

    def human_state_in_FOV(self, self_state, human_state):
        rot = np.arctan2(
            human_state.py -
            self_state.py,
            human_state.px -
            self_state.px)
        angle = (rot - self_state.theta) % (2 * np.pi)
        if angle > self.FOV_min_angle or angle < self.FOV_max_angle or \
                self.FOV_min_angle == self.FOV_max_angle:
            return True
        else:
            return False
