#!/usr/bin/env python3
import copy
import os
import logging
import itertools

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
from keras.layers import Dense

from crowd_sim.envs.utils.action import ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_sim.envs.policy.policy import Policy


class NetworkCore(tf.keras.Model):
    def __init__(self, scope):
        super(NetworkCore, self).__init__()

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            self._create_graph(scope)

            if self.optimizer is not None and 'global' not in scope:
                self._create_losses()
                self._train(scope)
                self._create_visualization(scope)

            if self.use_grid_map:
                self.convnet_saver = tf.train.Saver(
                    var_list={'conv1_weights': self.conv1_weights,
                              'conv1_biases': self.conv1_biases,
                              'conv2_weights': self.conv2_weights,
                              'conv2_biases': self.conv2_biases,
                              'conv3_weights:': self.conv3_weights,
                              'conv3_biases': self.conv3_biases,
                              'fc_grid_weights': self.fc_grid_weights,
                              'fc_grid_biases': self.fc_grid_biases,
                              })

    def _create_visualization(self, scope):
        # Tensorboard
        if self.optimizer is not None:
            tf.summary.scalar('loss_v', self.cost_v)
            tf.summary.scalar('loss_p', self.cost_p)
            tf.summary.scalar('entropy_mean', self.entropy_mean)
            tf.summary.scalar('loss_vp', self.cost_all)
            tf.summary.scalar("max_value", tf.reduce_max(self.value_output))
            tf.summary.scalar("min_value", tf.reduce_min(self.value_output))
            tf.summary.scalar("mean_value", tf.reduce_mean(self.value_output))
            tf.summary.scalar(
                "reward_max", tf.reduce_max(
                    self.output_placeholder))
            tf.summary.scalar(
                "reward_min", tf.reduce_min(
                    self.output_placeholder))
            tf.summary.scalar(
                "reward_mean", tf.reduce_mean(
                    self.output_placeholder))
            tf.summary.histogram('entropy', self.entropy)
            tf.summary.histogram("reward_targets", self.output_placeholder)
            tf.summary.histogram("values", self.value_output)
            tf.summary.histogram("policy_output", self.policy_output)
            for var in tf.trainable_variables(scope=scope):
                tf.summary.histogram(
                    "weights_%s" %
                    var.name.replace(
                        ':', '_'), var)
        elif self.use_grid_map:
            var_list = [
                self.conv1_weights,
                self.conv1_biases,
                self.conv2_weights,
                self.conv2_biases,
                self.conv3_weights,
                self.conv3_biases,
                self.fc_grid_weights,
                self.fc_grid_biases]
            for var in var_list:
                tf.summary.histogram(
                    "weights_%s" %
                    var.name.replace(
                        ':', '_'), var)
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        sumaries = [s for s in summary_ops if scope in s.name]
        self.summary = tf.summary.merge(sumaries)


class NetworkSDOADRL(NetworkCore):
    def __init__(self, config, scope, num_actions, optimizer=None):
        self.with_om = config.getboolean('sarl', 'with_om')
        self.with_global_state = config.getboolean('sarl', 'with_global_state')
        self.self_state_dim = Config.HOST_AGENT_OBSERVATION_LENGTH
        self.human_state_dim = Config.OTHER_AGENT_OBSERVATION_LENGTH
        self.joint_state_dim = self.self_state_dim + self.human_state_dim
        self.gamma = config.getfloat('rl', 'gamma')
        self.cell_num = config.getint('sarl', 'cell_num')
        self.cell_size = config.getfloat('sarl', 'cell_size')
        self.om_channel_size = config.getint('sarl', 'om_channel_size')
        self.input_dim = self.input_dim()
        self.log_epsilon = Config.LOG_EPSILON
        self.global_state_dim = 100
        self.attention_weights = None
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.use_grid_map = config.getboolean('map', 'use_grid_map')
        if self.use_grid_map:
            self.grid_width = config.getint('map', 'map_width')
            self.grid_height = config.getint('map', 'map_height')
        else:
            self.angular_map_dim = config.getint('map', 'angular_map_dim')
        self.batch_size = None
        super(NetworkSDOADRL, self).__init__(scope)

    def _create_graph_inputs(self):
        self.state = tf.placeholder(
            tf.float32, [
                None, None, self.input_dim], name='state')
        self.robot_state = tf.placeholder(
            tf.float32, [None, self.self_state_dim], name='robot_state')
        if self.use_grid_map:
            self.input_grid_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=[self.batch_size, self.grid_width, self.grid_height],
                name='input_grid')
        else:
            self.angular_map_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=[self.batch_size, self.angular_map_dim],
                name='input_angular_map')

    def _create_graph_outputs(self, regularizer, scope):
        if 'only_static' in scope:
            self.concat_output = tf.concat(
                [self.mlp_om_layer_out, self.robot_state], 1, name='concat_output')
        else:
            self.concat_output = tf.concat(
                [self.mlp3_layer3, self.mlp_om_layer_out, self.robot_state], 1, name='concat_output')

        self.mlp5_layer1 = Dense(
            100,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='mlp5_layer1')(
            self.concat_output)
        self.mlp5_layer2 = Dense(
            100,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='mlp5_layer2')(
            self.mlp5_layer1)

        # Value Network
        self.value_output = tf.squeeze(
            Dense(
                1,
                kernel_regularizer=regularizer,
                name='value_output')(
                self.mlp5_layer2))

        # Policy Network
        self.logits_p = Dense(
            self.num_actions,
            name='logits_p',
            activation=None)(
            self.mlp5_layer2)
        self.policy_output = (tf.nn.softmax(
            self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)

        self.output_placeholder = tf.placeholder(
            tf.float32, [None], name='Out')

    def _create_graph(self, scope):
        if Config.USE_REGULARIZATION:
            regularizer = tf.keras.regularizers.l2(0.)
        else:
            regularizer = None

        self._create_graph_inputs()
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        if self.use_grid_map:
            self._create_graph_om()
        else:
            self._create_graph_am()
        if 'only_static' not in scope:
            self._create_graph_ped(regularizer)
        self._create_graph_outputs(regularizer, scope)

    def input_dim(self):
        return self.joint_state_dim + \
            (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def _create_losses(self):
        self.action_index = tf.placeholder(tf.int32, [None], name='action')

        loss = tf.square(self.value_output - self.output_placeholder)
        self.cost_v = 0.5 * tf.reduce_sum(loss, axis=0)

        self.actions_onehot = tf.one_hot(
            self.action_index, self.num_actions, dtype=tf.float32)

        self.selected_action_prob = tf.reduce_sum(
            self.policy_output * self.actions_onehot, axis=1)
        self.entropy = - 0.001 * \
            tf.reduce_sum(self.policy_output * tf.log(self.policy_output + self.log_epsilon), axis=1)
        self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")
        self.policy_loss = tf.log(self.selected_action_prob + self.log_epsilon) * (
            self.output_placeholder - tf.stop_gradient(self.value_output))
        self.cost_p_1 = tf.reduce_sum(self.policy_loss, axis=0)
        self.cost_p_2 = tf.reduce_sum(self.entropy, axis=0)
        self.cost_p = -(self.cost_p_1 + self.cost_p_2)
        self.cost_all = self.cost_p + self.cost_v

    def _train(self, scope):
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        self.gradients = tf.gradients(self.cost_all, local_vars)
        self.var_norms = tf.global_norm(local_vars)

        if 'only_static' not in scope:
            global_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        else:
            global_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'only_static_global')

        # Apply local gradients to global network
        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
        grad_check = tf.check_numerics(grads[0], 'Gradients are invalid')
        with tf.control_dependencies([grad_check]):
            self.apply_grads = self.optimizer.apply_gradients(
                zip(grads, global_vars))

    def _create_graph_am(self):
        ped_grid_out_dim = 128
        self.W_pedestrian_grid = self.get_weight_variable(
            name='w_ped_grid', shape=[self.angular_map_dim, ped_grid_out_dim])
        self.b_pedestrian_grid = self.get_bias_variable(
            name='b_ped_grid', shape=[1, ped_grid_out_dim])
        fc_angular_map = self.fc_layer(
            self.angular_map_placeholder,
            weights=self.W_pedestrian_grid,
            biases=self.b_pedestrian_grid,
            use_activation=False,
            name="fc_ped")

        self.mlp_om_layer_out = Dense(
            100,
            activation=tf.nn.relu,
            name='mlp_om_layer_out')(fc_angular_map)

    def _create_graph_ped(self, regularizer):
        size = tf.shape(self.state)
        self_state = self.state[:, 0, :self.self_state_dim]
        self.mlp1_layer1 = Dense(150,
                                 activation=tf.nn.relu,
                                 kernel_regularizer=regularizer,
                                 name='mlp1_layer1')(tf.reshape(self.state,
                                                                [-1,
                                                                 self.input_dim]))
        self.mlp1_output = Dense(
            100,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='mlp1_output')(
            self.mlp1_layer1)
        self.mlp2_layer1 = Dense(
            100,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='mlp2_layer1')(
            self.mlp1_output)
        self.mlp2_output = Dense(
            50,
            kernel_regularizer=regularizer,
            name='mlp2_output')(
            self.mlp2_layer1)

        if self.with_global_state:
            # compute attention scores
            global_state = tf.reduce_mean(
                tf.reshape(
                    self.mlp1_output, [
                        size[0], size[1], -1]), 1, keepdims=True)
            global_state = tf.tile(global_state, [1, size[1], 1])

            global_state = tf.reshape(
                global_state, [-1, self.global_state_dim])
            self.attention_input = tf.concat(
                [self.mlp1_output, global_state], 1, name='attention_input')
        else:
            self.attention_input = self.mlp1_output
        self.attention_layer1 = Dense(
            100,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='attention_layer1')(
            self.attention_input)
        self.attention_layer2 = Dense(
            100,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='attention_layer2')(
            self.attention_layer1)
        self.attention_score = Dense(
            1,
            kernel_regularizer=regularizer,
            name='attention_score')(
            self.attention_layer2)
        scores = tf.squeeze(
            tf.reshape(
                self.attention_score, [
                    size[0], size[1], 1]), axis=2)

        # masked softmax
        scores_exp = tf.exp(scores) * float((scores != 0))
        weights = tf.expand_dims(
            scores_exp /
            tf.reduce_sum(
                scores_exp,
                1,
                keepdims=True),
            2)
        self.attention_weights = weights[0, :, 0]

        # output feature is a linear combination of input features
        features = tf.reshape(self.mlp2_output, [size[0], size[1], -1])
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = tf.reduce_sum(tf.multiply(weights, features), 1)

        # concatenate agent's state with global weighted humans' state
        joint_state = tf.concat([self_state, weighted_feature], 1)
        self.mlp3_layer1 = Dense(
            150, activation=tf.nn.relu, kernel_regularizer=regularizer, name='mlp3_layer1')(
            tf.reshape(
                joint_state, [
                    size[0], 50 + self.self_state_dim]))
        self.mlp3_layer2 = Dense(
            100,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='mlp3_layer2')(
            self.mlp3_layer1)
        self.mlp3_layer3 = Dense(
            100,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='mlp3_layer3')(
            self.mlp3_layer2)

    def _create_graph_om(self):
        # CNN weights for static occupancy grid processing
        train_grid_encoder_conv = False
        train_grid_encoder_fc = True
        use_summary_convnet = False

        # Convolutional layer 1
        self.conv1_kernel_size = 5
        self.conv1_number_filters = 64
        self.conv1_stride_length = 2
        self.conv1_weights = self.get_weight_variable(
            name="conv1_weights",
            shape=[self.conv1_kernel_size, self.conv1_kernel_size, 1,
                   self.conv1_number_filters],
            trainable=train_grid_encoder_conv,
            summary=use_summary_convnet)
        self.conv1_biases = self.get_bias_variable(
            name="conv1_biases",
            shape=[self.conv1_number_filters],
            trainable=train_grid_encoder_conv,
            summary=use_summary_convnet)

        # Convolutional layer 2
        self.conv2_kernel_size = 3
        self.conv2_number_filters = 32
        self.conv2_stride_length = 2
        self.conv2_weights = self.get_weight_variable(
            name="conv2_weights",
            shape=[self.conv2_kernel_size, self.conv2_kernel_size,
                   self.conv1_number_filters, self.conv2_number_filters],
            trainable=train_grid_encoder_conv,
            summary=use_summary_convnet)
        self.conv2_biases = self.get_bias_variable(
            name="conv2_biases",
            shape=[self.conv2_number_filters],
            trainable=train_grid_encoder_conv,
            summary=use_summary_convnet)

        # Convolutional layer 3
        self.conv3_kernel_size = 3
        self.conv3_number_filters = 8
        self.conv3_stride_length = 2
        self.conv3_weights = self.get_weight_variable(
            name="conv3_weights",
            shape=[self.conv3_kernel_size, self.conv3_kernel_size,
                   self.conv2_number_filters, self.conv3_number_filters],
            trainable=train_grid_encoder_conv,
            summary=use_summary_convnet)
        self.conv3_biases = self.get_bias_variable(
            name="conv3_biases",
            shape=[self.conv3_number_filters],
            trainable=train_grid_encoder_conv,
            summary=use_summary_convnet)

        fc_grid_hidden_dim = 64
        self.fc_grid_weights = self.get_weight_variable(
            shape=[512, fc_grid_hidden_dim], name="fc_grid_weights",
            trainable=train_grid_encoder_fc)
        self.fc_grid_biases = self.get_bias_variable(
            shape=[fc_grid_hidden_dim], name="fc_grid_biases",
            trainable=train_grid_encoder_fc)

        ###################### Assemble Neural Net #####################
        # Process occupancy grid, apply convolutional filters
        self.conv_grid_output = self.process_grid(self.input_grid_placeholder)

        self.mlp_om_layer_out = Dense(
            100,
            activation=tf.nn.relu,
            name='mlp_om_layer_out')(
            self.conv_grid_output)

    def process_grid(self, grid_batch):
        """
        Process occupancy grid with series of convolutional and fc layers.
        input: occupancy grid series (shape: [batch_size, grid_dim_x, grid_dim_y])
        output: convolutional feature vector (shape: list of tbpl elements with [batch_size, feature_vector_size] each)
        """

        conv_feature_vectors = []
        grid_batch = tf.expand_dims(input=grid_batch, axis=3)
        self.conv1 = self.conv_layer(
            input=grid_batch,
            weights=self.conv1_weights,
            biases=self.conv1_biases,
            conv_stride_length=self.conv1_stride_length,
            name="conv1_grid")
        self.conv2 = self.conv_layer(
            input=self.conv1,
            weights=self.conv2_weights,
            biases=self.conv2_biases,
            conv_stride_length=self.conv2_stride_length,
            name="conv2_grid")
        self.conv3 = self.conv_layer(
            input=self.conv2,
            weights=self.conv3_weights,
            biases=self.conv3_biases,
            conv_stride_length=self.conv3_stride_length,
            name="conv3_grid")
        conv_grid_size = 8
        self.conv5_flat = tf.reshape(
            self.conv3, [-1, conv_grid_size * conv_grid_size * self.conv3_number_filters])
        self.fc_final = self.fc_layer(
            input=self.conv5_flat,
            weights=self.fc_grid_weights,
            biases=self.fc_grid_biases,
            use_activation=True,
            name="fc_grid")

        # Flatten to obtain feature vector
        conv_features = tf.contrib.layers.flatten(self.fc_final)

        tf.summary.histogram("fc_activations", self.fc_final)

        return conv_features

    def get_weight_variable(
            self,
            shape,
            name,
            trainable=True,
            regularizer=None,
            summary=True):
        """
        Get weight variable with specific initializer.
        """
        var = tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.truncated_normal_initializer(
                mean=0.0,
                stddev=0.1),
            regularizer=tf.contrib.layers.l2_regularizer(0.01),
            trainable=trainable)
        if summary:
            tf.summary.histogram(name, var)
        return var

    def get_bias_variable(
            self,
            shape,
            name,
            trainable=True,
            regularizer=None,
            summary=True):
        """
        Get bias variable with specific initializer.
        """
        var = tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.constant_initializer(0.1),
            trainable=trainable)
        if summary:
            tf.summary.histogram(name, var)
        return var

    def conv_layer(
            self,
            input,
            weights,
            biases,
            conv_stride_length=1,
            padding="SAME",
            name="conv",
            summary=False):
        """
        Convolutional layer including a ReLU activation but excluding pooling.
        """
        conv = tf.nn.conv2d(
            input,
            filter=weights,
            strides=[
                1,
                conv_stride_length,
                conv_stride_length,
                1],
            padding=padding,
            name=name)
        activations = tf.nn.relu(conv + biases)
        if summary:
            tf.summary.histogram(name, activations)
        return activations

    def fc_layer(
            self,
            input,
            weights,
            biases,
            use_activation=False,
            name="fc",
            summary=False):
        """
        Fully connected layer with given weights and biases.
        Activation and summary can be activated with the arguments.
        """
        affine_result = tf.matmul(input, weights) + biases
        if use_activation:
            activations = tf.nn.sigmoid(affine_result)
        else:
            activations = affine_result
        if summary:
            tf.summary.histogram(name + "_activations", activations)
        return activations


class SDOADRL(
        Policy):  # Static and dynamic obstacle avoidance using deep reinforcement learning
    def __init__(self):
        self.name = 'SDOADRL'
        self.sess = None
        self.action_space = None
        self.model = None
        self.no_human_model = None
        self.use_grid_map = None
        self.with_om = None

        self.phase = None
        self.gamma = None
        self.speed_samples = None
        self.rotation_samples = None
        self.rotation_factor = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.time_step = None
        self.use_grid_map = None
        self.FOV_min_angle = None
        self.FOV_max_angle = None

        logging.info('Policy: SDOADRL')

    def set_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint(
            'action_space', 'rotation_samples')
        self.rotation_factor = config.getfloat(
            'action_space', 'rotation_factor')
        self.cell_num = config.getint('sarl', 'cell_num')
        self.cell_size = config.getfloat('sarl', 'cell_size')
        self.om_channel_size = config.getint('sarl', 'om_channel_size')
        self.time_step = config.getfloat('env', 'time_step')
        self.use_grid_map = config.getboolean('map', 'use_grid_map')
        self.FOV_min_angle = config.getfloat(
            'map', 'angle_min') * np.pi % (2 * np.pi)
        self.FOV_max_angle = config.getfloat(
            'map', 'angle_max') * np.pi % (2 * np.pi)
        self.phase = None

    def configure(self, sess, scope, config, learning_rate=None):
        self.sess = sess
        self.set_parameters(config)
        self.action_space = self.build_action_space()
        if learning_rate is not None:
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, epsilon=1e-04)
            self.model = NetworkSDOADRL(
                config, scope, len(
                    self.action_space), optimizer)
            if not self.use_grid_map:
                self.no_human_model = NetworkSDOADRL(
                    config, 'only_static_{}'.format(scope), len(
                        self.action_space), optimizer)
        else:
            if scope is not None:
                self.model = NetworkSDOADRL(config, scope, len(self.action_space))
                if not self.use_grid_map:
                    self.no_human_model = NetworkSDOADRL(
                        config, 'only_static_{}'.format(scope), len(
                            self.action_space))
        self.with_om = config.getboolean('sarl', 'with_om')

    def predict(self, state, local_map, robot=None):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None:
            raise AttributeError('Phase attributes have to be set!')

        if self.reach_destination(state):
            action = ActionRot(0.0, 0.0)
            if robot is not None:
                robot.action_index = 0
            return action

        occupancy_maps = None
        action = None
        state_tensor, humans_in_FOV = self.transform(state)
        local_map = np.expand_dims(local_map, axis=0)
        self_state = np.expand_dims(state_tensor[0, 0:6], axis=0)
        if humans_in_FOV:
            rotated_batch_input = np.expand_dims(state_tensor, axis=0)
            if self.use_grid_map:
                feed_dict = {self.model.state: rotated_batch_input,
                             self.model.robot_state: self_state,
                             self.model.input_grid_placeholder: local_map}
            else:
                feed_dict = {self.model.state: rotated_batch_input,
                             self.model.robot_state: self_state,
                             self.model.angular_map_placeholder: local_map}
            policy_probs = self.sess.run(self.model.policy_output,
                                         feed_dict=feed_dict)
        else:
            if self.use_grid_map:
                feed_dict = {
                    self.no_human_model.robot_state: self_state,
                    self.no_human_model.input_grid_placeholder: local_map}
            else:
                feed_dict = {
                    self.no_human_model.robot_state: self_state,
                    self.no_human_model.angular_map_placeholder: local_map}

            policy_probs = self.sess.run(self.no_human_model.policy_output,
                                         feed_dict=feed_dict)

        if self.phase == 'train':
            action_index = np.random.choice(
                np.arange(len(policy_probs[0])), p=policy_probs[0])
        else:
            action_index = np.argmax(policy_probs)
        raw_action = copy.deepcopy(self.action_space[action_index])
        if robot is not None:
            action = ActionRot(robot.v_pref * raw_action[0], raw_action[1])
        else:
            action = ActionRot(raw_action[0], raw_action[1])

        if action is None:
            raise ValueError('Value network is not well trained. ')
        if robot is not None and self.phase == 'train':
            robot.last_state, robot.humans_in_FOV = self.transform(state)
            robot.action_index = action_index
        return action

    def load_model(self, model_weights):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_weights)

    def get_reward(self, full_state, self_state, local_map, humans_in_FOV):
        if humans_in_FOV:
            if self.use_grid_map:
                feed_dict = {self.model.state: full_state,
                             self.model.robot_state: self_state,
                             self.model.input_grid_placeholder: local_map}
            else:
                feed_dict = {self.model.state: full_state,
                             self.model.robot_state: self_state,
                             self.model.angular_map_placeholder: local_map}
            reward_discounted = self.sess.run(
                self.model.value_output,
                feed_dict=feed_dict)
        else:
            if self.use_grid_map:
                feed_dict = {
                    self.no_human_model.robot_state: self_state,
                    self.no_human_model.input_grid_placeholder: local_map}
            else:
                feed_dict = {
                    self.no_human_model.robot_state: self_state,
                    self.no_human_model.angular_map_placeholder: local_map}
            reward_discounted = self.sess.run(self.no_human_model.value_output,
                                              feed_dict=feed_dict)
        return reward_discounted

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return tensor of shape (# of humans, len(state))
        """
        human_states_in_FOV = []
        if len(state.human_states) > 0:
            for human_state in state.human_states:
                if self.human_state_in_FOV(state.self_state, human_state):
                    human_states_in_FOV.append(human_state)
        if len(human_states_in_FOV) > 0:
            state_tensor = np.concatenate([([state.self_state + human_state])
                                           for human_state in human_states_in_FOV], 0)
            state_tensor = self.rotate(state_tensor)
            if self.with_om:
                occupancy_maps = self.build_occupancy_maps(
                    human_states_in_FOV, state.self_state)
                state_tensor = np.concatenate(
                    [state_tensor, occupancy_maps], 1)
            humans_in_FOV = True
        else:
            state_tensor = self.rotate(
                np.expand_dims(
                    state.self_state +
                    ObservableState(
                        0,
                        0,
                        0,
                        0,
                        0),
                    axis=0))
            if self.with_om:
                occupancy_maps = self.build_occupancy_maps(
                    [ObservableState(0, 0, 0, 0, 0)], state.self_state)
                state_tensor = np.concatenate(
                    [state_tensor, occupancy_maps], 1)
            humans_in_FOV = False

        return state_tensor, humans_in_FOV

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        # 0     1      2     3      4        5     6      7         8       9
        # 10      11     12       13
        batch = len(state)
        dx = np.reshape((state[:, 5] - state[:, 0]), (batch, -1))
        dy = np.reshape((state[:, 6] - state[:, 1]), [batch, -1])
        rot = np.arctan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])
        dg = LA.norm(np.concatenate([dx, dy], 1), keepdims=True)
        dg = LA.norm(np.concatenate([dx, dy], 1), ord=2, axis=1, keepdims=True)
        v_pref = np.reshape(state[:, 7], [batch, -1])
        vx = np.reshape((state[:, 2] * np.cos(rot) +
                         state[:, 3] * np.sin(rot)), [batch, -1])
        vy = np.reshape((state[:, 3] * np.cos(rot) -
                         state[:, 2] * np.sin(rot)), [batch, -1])

        radius = np.reshape(state[:, 4], [batch, -1])
        theta = np.reshape((state[:, 8] - rot) % (2 * np.pi), [batch, -1])
        vx1 = np.reshape((state[:, 11] * np.cos(rot) +
                          state[:, 12] * np.sin(rot)), [batch, -1])
        vy1 = np.reshape((state[:, 12] * np.cos(rot) -
                          state[:, 11] * np.sin(rot)), [batch, -1])
        px1 = (state[:, 9] - state[:, 0]) * np.cos(rot) + \
            (state[:, 10] - state[:, 1]) * np.sin(rot)
        px1 = np.reshape(px1, [batch, -1])
        py1 = (state[:, 10] - state[:, 1]) * np.cos(rot) - \
            (state[:, 9] - state[:, 0]) * np.sin(rot)
        py1 = np.reshape(py1, [batch, -1])
        radius1 = np.reshape(state[:, 13], [batch, -1])
        radius_sum = radius + radius1
        da = LA.norm(np.concatenate([np.reshape((state[:, 0] - state[:, 9]), [batch, -1]), np.reshape(
            (state[:, 1] - state[:, 10]), [batch, -1])], 1), ord=2, axis=1, keepdims=True)
        new_state = np.concatenate([dg,
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
                                   axis=1)
        return new_state

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

    def build_action_space(self):
        speeds = [(np.exp((i + 1) / float(self.speed_samples)) - 1) /
                  (np.e - 1) for i in range(self.speed_samples)]
        rotations = np.linspace(-np.pi / 4.0, np.pi /
                                4.0, self.rotation_samples)
        action_space = [ActionRot(0.0, 0.0)]
        for rotation, speed in itertools.product(rotations, speeds):
            action_space.append(
                ActionRot(
                    speed,
                    rotation /
                    self.rotation_factor))
        self.speeds = speeds
        self.rotations = rotations
        return action_space

    def reach_destination(self, state):
        self_state = state.self_state
        if np.linalg.norm((self_state.py - self_state.gy,
                           self_state.px - self_state.gx)) < self_state.radius:
            return True
        else:
            return False

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
                other_px /
                float(
                    self.cell_size) +
                self.cell_num /
                2.0)
            other_y_index = np.floor(
                other_py /
                float(
                    self.cell_size) +
                self.cell_num /
                2.0)
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
                            dm[2 * int(index)].append(1)
                            dm[2 * int(index) + 1].append(other_vx[i])
                            dm[2 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return np.concatenate(occupancy_maps, axis=0)

    def get_attention_weights(self):
        return self.attention_weights


class Config:
    #########################################################################
    # GENERAL PARAMETERS
    USE_REGULARIZATION = False

    MIN_POLICY = 1e-6
    LOG_EPSILON = 1e-10

    # dist to goal, heading to goal, pref speed, radius
    HOST_AGENT_OBSERVATION_LENGTH = 6
    # other px, other py, other vx, other vy, other radius, combined radius,
    # distance between
    OTHER_AGENT_OBSERVATION_LENGTH = 7
