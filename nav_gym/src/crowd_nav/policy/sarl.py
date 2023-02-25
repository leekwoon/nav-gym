import logging
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax

from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.state import ObservableState, FullState


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class ValueNetwork(nn.Module):
    def __init__(
            self,
            input_dim,
            self_state_dim,
            mlp1_dims,
            mlp2_dims,
            mlp3_dims,
            attention_dims,
            with_global_state,
            cell_size,
            cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation
        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(
                size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(
            size[0], size[1], 1).squeeze(dim=2)
        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (
            scores_exp /
            torch.sum(
                scores_exp,
                dim=1,
                keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value


class SARL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'
        self.multiagent_training = None
        self.epsilon = None
        self.gamma = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

    def set_common_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint(
            'action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [
            int(x) for x in config.get(
                'sarl',
                'mlp1_dims').split(', ')]
        mlp2_dims = [
            int(x) for x in config.get(
                'sarl',
                'mlp2_dims').split(', ')]
        mlp3_dims = [
            int(x) for x in config.get(
                'sarl',
                'mlp3_dims').split(', ')]
        attention_dims = [
            int(x) for x in config.get(
                'sarl', 'attention_dims').split(', ')]
        self.FOV_min_angle = config.getfloat(
            'map', 'angle_min') * np.pi % (2 * np.pi)
        self.FOV_max_angle = config.getfloat(
            'map', 'angle_max') * np.pi % (2 * np.pi)
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(
            self.input_dim(),
            self.self_state_dim,
            mlp1_dims,
            mlp2_dims,
            mlp3_dims,
            attention_dims,
            with_global_state,
            self.cell_size,
            self.cell_num)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.multiagent_training = config.getboolean(
            'sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(
            self.name, 'w/' if with_global_state else 'w/o'))

    def load_model(self, model_weights):
        self.model.load_state_dict(
            torch.load(
                model_weights,
                map_location='cpu'))

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_attention_weights(self):
        return self.model.attention_weights

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) /
                  (np.e - 1) * v_pref for i in range(self.speed_samples)]

        rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)

        action_space = [ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)
        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError(
                'Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionRot(0, 0)
        if self.action_space is None or self.v_pref != state.self_state.v_pref:
            self.build_action_space(state.self_state.v_pref)
            self.v_pref = state.self_state.v_pref

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train':
            if probability < self.epsilon:
                max_action = self.action_space[np.random.choice(
                    len(self.action_space))]
        else:
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(
                        action)
                else:
                    next_human_states = [
                        self.propagate(
                            human_state,
                            ActionXY(
                                human_state.vx,
                                human_state.vy)) for human_state in state.human_states]
                    reward = self.compute_reward(
                        next_self_state, next_human_states)
                if len(next_human_states) == 0:
                    next_human_states = [ObservableState(0, 0, 0, 0, 0)]
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(
                    self.device) for next_human_state in next_human_states], dim=0)
                rotated_batch_input = self.rotate(
                    batch_next_states).unsqueeze(0)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(
                            next_human_states, state.self_state).unsqueeze(0)
                    rotated_batch_input = torch.cat(
                        [rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
                # VALUE UPDATE
                next_state_value = self.model(rotated_batch_input).data.item()
                value = reward + \
                    pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError('Value network is not well trained. ')
        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of humans
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(
                next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            # propagate state of current agent
            # perform action without rotation
            next_theta = state.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
            next_px = state.px + next_vx * self.time_step
            next_py = state.py + next_vy * self.time_step
            next_state = FullState(
                next_px,
                next_py,
                next_vx,
                next_vy,
                state.radius,
                state.gx,
                state.gy,
                state.v_pref,
                next_theta)
        else:
            raise ValueError('Type error')

        return next_state

    def compute_reward(self, nav, humans):
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            dist = np.linalg.norm(
                (nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm(
            (nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0

        return reward

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network
        :param state:
        :return: tensor of shape (# of humans, len(state))
        """

        human_states_in_FOV = []
        for human_state in state.human_states:
            if self.human_state_in_FOV(state.self_state, human_state):
                human_states_in_FOV.append(human_state)
        if len(human_states_in_FOV) > 0:
            state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(
                self.device) for human_state in human_states_in_FOV], dim=0)
            if self.with_om:
                occupancy_maps = self.build_occupancy_maps(
                    human_states_in_FOV, state.self_state)
                state_tensor = torch.cat(
                    [self.rotate(state_tensor), occupancy_maps.to(self.device)], dim=1)
            else:
                state_tensor = self.rotate(state_tensor)
        else:
            state_tensor = self.rotate(torch.Tensor(
                [state.self_state + ObservableState(0, 0, 0, 0, 0)]).to(self.device))
            if self.with_om:
                occupancy_maps = self.build_occupancy_maps(
                    [ObservableState(0, 0, 0, 0, 0)], state.self_state)
                state_tensor = torch.cat(
                    [state_tensor, occupancy_maps.to(self.device)], dim=1)

        return state_tensor

    def human_state_in_FOV(self, self_state, human_state):
        rot = np.arctan2(
            human_state.py -
            self_state.py,
            human_state.px -
            self_state.px)
        angle = (rot - self_state.theta) % (2 * np.pi)
        if angle > self.FOV_min_angle or angle < self.FOV_max_angle or self.FOV_min_angle == self.FOV_max_angle:
            return True
        else:
            return False

    def input_dim(self):
        return self.joint_state_dim + \
            (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)
        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        # 0     1      2     3      4        5     6      7         8       9
        # 10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3]
              * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2]
              * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        theta = (state[:, 8] - rot).reshape((batch, -1))
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12]
               * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11]
               * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + \
            (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - \
            (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] -
                                    state[:, 9]).reshape((batch, -
                                                          1)), (state[:, 1] -
                                                                state[:, 10]). reshape((batch, -
                                                                                        1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg,
                               v_pref,
                               theta,
                               radius,
                               vx,
                               vy,
                               px1,
                               py1,
                               vx1,
                               vy1,
                               radius1,
                               da,
                               radius_sum],
                              dim=1)
        return new_state

    def build_occupancy_maps(self, human_states, self_state):
        """
        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        human_states_with_self = list(human_states)
        human_states_with_self.append(self_state)
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                           for other_human in human_states_with_self if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(
                other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(
                other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(
                    other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [
                    list() for _ in range(
                        self.cell_num ** 2 *
                        self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()
