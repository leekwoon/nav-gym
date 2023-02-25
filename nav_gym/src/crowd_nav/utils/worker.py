import collections
import itertools
import logging
import math
import os
import random
import threading
import time

import gym
import numpy as np
import tensorflow as tf

from crowd_nav.policy.network_om import SDOADRL
from crowd_sim.envs.utils.info import *

Transition = collections.namedtuple(
    "Transition", [
        "state", "robot_state", "grid", "action", "reward", "humans_in_FOV"])


def update_target_graph(from_scope, to_scope):
    '''!
    Updates the variables of one scope with another

    @param from_scope: The scope of the graph to obtain the weights from.
    @param to_scope: The scope of the graph to copy the weights to.
    @return op_holder: List of TF operations
    '''
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    from_vars = list(sorted(from_vars, key=lambda v: v.name))
    to_vars = list(sorted(to_vars, key=lambda v: v.name))
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class Worker(object):
    """!
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.
    """

    def __init__(
            self,
            name,
            env,
            policy,
            sess,
            global_counter=None,
            learning_rate=None,
            policy_config=None,
            gamma=0.99,
            max_global_steps=None,
            buffer_size=50000,
            imitation_learning=False,
            rl_weight_file=None,
            visualize_rewards=False,
            use_grid_map=False):
        """!
        Initializes the worker.
            @param name: A unique name for this worker
            @param env: The Gym environment used by this worker
            @param policy: Instance of the global policy
            @param sess: Instance of the tensorflow session
            @param global_counter: Counts the episodes of all workers combined
            @param learning_rate: Learning rate that is passed to the optimizer
            @param policy_config: Config file to create a new policy object
            @param gamma: Discount factor for calculating the reward
            @param summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
            @param max_global_steps: If set, stop coordinator when global_counter > max_global_steps
            @param buffer_size: Max size of the buffer for imitation learning
            @param imitation_learning: Indicates if imitation learning is performed
            @param rl_weight_file: Location where the saver should save the model to
        """
        self.name = name
        self.max_global_steps = max_global_steps
        self.global_step = tf.contrib.framework.get_global_step()
        self.global_policy = policy
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.local_t = None
        self.summary_writer = None
        self.env = env
        self.robot = env.robot
        self.phase = self.global_policy.phase
        self.buffer = []
        self.buffer_no_humans = []
        self.buffer_size = buffer_size
        self.sess = sess
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.rl_weight_file = rl_weight_file
        self.il_iteration_counter = -1
        self.use_grid_map = use_grid_map
        if self.phase == 'train':
            self.policy = SDOADRL()
            self.policy.configure(sess, name, policy_config, learning_rate)
            self.policy.set_phase(self.phase)
            if not imitation_learning:
                self.robot.set_policy(self.policy)
            self.update_local_ops = update_target_graph('global', self.name)
            if not self.use_grid_map:  # Not implemented for usage with grid map
                self.update_local_ops_no_humans = update_target_graph(
                    'only_static_global', 'only_static_{}'.format(self.name))
                self.no_human_model = self.policy.no_human_model
            self.model = self.policy.model
        else:
            self.policy = self.global_policy

        self.visualize_rewards = visualize_rewards
        if visualize_rewards:
            self.mean_rewards_goal_list = []
            self.mean_rewards_collision_list = []
            self.mean_rewards_timeout_list = []

    def run(self, coord, t_max):
        """!
        Performs the reinforcement learning steps.

            @param coord: A tf.train.Coordinator() to coordinate between Threads
            @param t_max: Number of episodes to run before using batch for training
        """
        logging.info(
            "Started worker {} on thread {}".format(
                self.name,
                threading.current_thread()))
        success = 0
        collision = 0
        episode_count = 0
        if self.visualize_rewards:
            mean_rewards_goal = []
            num_goal_episodes_added = []
            mean_rewards_collision = []
            num_collision_episodes_added = []
            mean_rewards_timeout = []
            num_timeout_episodes_added = []
        with self.sess.as_default(), self.sess.graph.as_default():
            if self.summary_writer is not None:
                self.saver = tf.train.Saver(max_to_keep=10)
            try:
                while not coord.should_stop():
                    if self.visualize_rewards:
                        new_rewards = []
                    self.sess.run(self.update_local_ops)
                    if not self.use_grid_map:
                        self.sess.run(self.update_local_ops_no_humans)
                    transitions = []
                    accumulated_transitions = []
                    accumulated_transitions_no_humans = []
                    # Initial state
                    ob, local_map = self.env.reset(self.phase)
                    gamma_bar = pow(
                        self.gamma,
                        self.robot.time_step *
                        self.robot.v_pref)
                    done = False
                    while not done:
                        # Take an action using probabilities from policy
                        # network output.
                        action = self.robot.act(ob, local_map)
                        action_index = self.robot.action_index
                        ob, local_map_new, reward, done, info = self.env.step(
                            action)

                        # If the episode hasn't ended, but the experience buffer is full, then we
                        # make an update step using that experience rollout.
                        if len(transitions) == t_max and not done:
                            new_rewards_this_update = []
                            self.local_t = next(self.local_counter)
                            if self.global_counter is not None:
                                self.global_t = next(self.global_counter)
                            # Since we don't know what the true final return is, we "bootstrap" from our current
                            # value estimation.
                            input_grid = np.expand_dims(
                                local_map.tolist(), axis=0)
                            reward_discounted = self.robot.get_reward(
                                input_grid)

                            for transition in transitions[::-1]:
                                reward_discounted = transition.reward + gamma_bar * reward_discounted
                                if transition.humans_in_FOV:
                                    accumulated_transitions.append(Transition(state=transition.state, robot_state=transition.state[0, 0:6], grid=transition.grid.tolist(
                                    ), action=transition.action, reward=reward_discounted, humans_in_FOV=transition.humans_in_FOV))
                                else:
                                    accumulated_transitions_no_humans.append(
                                        Transition(
                                            state=transition.state,
                                            robot_state=transition.state[
                                                0,
                                                0:6],
                                            grid=transition.grid.tolist(),
                                            action=transition.action,
                                            reward=reward_discounted,
                                            humans_in_FOV=transition.humans_in_FOV))
                                if self.visualize_rewards:
                                    new_rewards_this_update.insert(
                                        0, reward_discounted)
                            if self.visualize_rewards:
                                new_rewards.extend(new_rewards_this_update)

                            self.update_transitions(accumulated_transitions)
                            if not self.use_grid_map:
                                self.update_transitions(
                                    accumulated_transitions_no_humans, no_humans=True)
                            transitions = []
                            accumulated_transitions = []
                            accumulated_transitions_no_humans = []
                            self.sess.run(self.update_local_ops)
                            if not self.use_grid_map:
                                self.sess.run(self.update_local_ops_no_humans)
                        # Store transition
                        transitions.append(
                            Transition(
                                state=self.robot.last_state,
                                robot_state=None,
                                grid=local_map,
                                action=action_index,
                                reward=reward,
                                humans_in_FOV=self.robot.humans_in_FOV))
                        local_map = local_map_new

                        if done:
                            if isinstance(info, ReachGoal):
                                success += 1
                            elif isinstance(info, Collision):
                                collision += 1
                            break

                    # Update the network using the episode buffer at the end of
                    # the episode.
                    if len(transitions) != 0:
                        new_rewards_this_update = []
                        self.local_t = next(self.local_counter)
                        if self.global_counter is not None:
                            self.global_t = next(self.global_counter)
                        reward_discounted = 0.0
                        if isinstance(info, Timeout):
                                # When we had a timeout we "bootstrap" from our current
                                # value estimation.
                            input_grid = np.expand_dims(
                                local_map.tolist(), axis=0)
                            reward_discounted = self.robot.get_reward(
                                input_grid)

                        for i, transition in enumerate(transitions[::-1]):
                            if i > 0 or not isinstance(info, Timeout):
                                reward_discounted = transition.reward + gamma_bar * reward_discounted
                            if transition.humans_in_FOV:
                                accumulated_transitions.append(Transition(state=transition.state, robot_state=transition.state[0, 0:6], grid=transition.grid.tolist(
                                ), action=transition.action, reward=reward_discounted, humans_in_FOV=transition.humans_in_FOV))
                            else:
                                accumulated_transitions_no_humans.append(
                                    Transition(
                                        state=transition.state,
                                        robot_state=transition.state[
                                            0,
                                            0:6],
                                        grid=transition.grid.tolist(),
                                        action=transition.action,
                                        reward=reward_discounted,
                                        humans_in_FOV=transition.humans_in_FOV))
                            if self.visualize_rewards:
                                new_rewards_this_update.insert(
                                    0, reward_discounted)
                        if self.visualize_rewards:
                            new_rewards.extend(new_rewards_this_update)
                        self.update_transitions(
                            accumulated_transitions, episode_count)
                        if not self.use_grid_map:
                            self.update_transitions(
                                accumulated_transitions_no_humans, episode_count, no_humans=True)

                    # Periodically save gifs of episodes, model parameters, and
                    # summary statistics.
                    if episode_count % 10 == 0:
                        logging.info(
                            "{}: local Step {}, global step {} with collision_rate {} and success_rate {}".format(
                                self.name, self.local_t, self.global_t, collision / 10.0, success / 10.0))
                        success = 0
                        collision = 0
                    if episode_count % 50 == 0 and self.summary_writer is not None:
                        self.phase = 'val'
                        self.run_k_episodes(self.env.case_size['val'])
                        self.phase = 'train'
                        self.robot.policy.set_phase(self.phase)
                        self.saver.save(self.sess, self.rl_weight_file)

                    if self.visualize_rewards:
                        from itertools import zip_longest
                        if isinstance(info, ReachGoal):
                            mean_rewards_goal[:] = [
                                sum(reward) for reward in zip_longest(
                                    mean_rewards_goal, new_rewards, fillvalue=0)]
                            num_goal_episodes_added[:] = [
                                sum(i) for i in zip_longest(
                                    num_goal_episodes_added, np.ones(
                                        len(new_rewards)), fillvalue=0)]
                        if isinstance(info, Collision):
                            mean_rewards_collision[:] = [
                                sum(reward) for reward in zip_longest(
                                    mean_rewards_collision, new_rewards, fillvalue=0)]
                            num_collision_episodes_added[:] = [
                                sum(i) for i in zip_longest(
                                    num_collision_episodes_added, np.ones(
                                        len(new_rewards)), fillvalue=0)]
                        if isinstance(info, Timeout):
                            mean_rewards_timeout[:] = [
                                sum(reward) for reward in zip_longest(
                                    mean_rewards_timeout, new_rewards, fillvalue=0)]
                            num_timeout_episodes_added[:] = [
                                sum(i) for i in zip_longest(
                                    num_timeout_episodes_added, np.ones(
                                        len(new_rewards)), fillvalue=0)]

                        if episode_count > 0 and episode_count % 50 == 0:
                            mean_rewards_goal[:] = [
                                reward / float(i) for reward,
                                i in zip(
                                    mean_rewards_goal,
                                    num_goal_episodes_added)]
                            mean_rewards_collision[:] = [
                                reward / float(i) for reward,
                                i in zip(
                                    mean_rewards_collision,
                                    num_collision_episodes_added)]
                            mean_rewards_timeout[:] = [
                                reward / float(i) for reward,
                                i in zip(
                                    mean_rewards_timeout,
                                    num_timeout_episodes_added)]
                            self.mean_rewards_goal_list.append(
                                mean_rewards_goal)
                            self.mean_rewards_collision_list.append(
                                mean_rewards_collision)
                            self.mean_rewards_timeout_list.append(
                                mean_rewards_timeout)
                            mean_rewards_goal = []
                            mean_rewards_collision = []
                            mean_rewards_timeout = []
                            num_goal_episodes_added = []
                            num_collision_episodes_added = []
                            num_timeout_episodes_added = []
                            if len(self.mean_rewards_goal_list) == 20:
                                if not os.path.exists(self.rl_weight_file):
                                    os.makedirs(self.rl_weight_file)
                                final_path = str(int(episode_count)) + '.npy'
                                np.save(
                                    os.path.join(
                                        self.rl_weight_file,
                                        'goal' + final_path),
                                    self.mean_rewards_goal_list)
                                np.save(
                                    os.path.join(
                                        self.rl_weight_file,
                                        'collision' + final_path),
                                    self.mean_rewards_collision_list)
                                np.save(
                                    os.path.join(
                                        self.rl_weight_file,
                                        'timeout' + final_path),
                                    self.mean_rewards_timeout_list)
                                self.mean_rewards_goal_list = []
                                self.mean_rewards_collision_list = []
                                self.mean_rewards_timeout_list = []

                    episode_count += 1

                    if self.max_global_steps is not None and self.global_t >= self.max_global_steps:
                        logging.info(
                            "Reached global step {}. Stopping.".format(
                                self.global_t))
                        coord.request_stop()
                        return

            except tf.errors.CancelledError:
                return

    def run_k_episodes(
            self,
            k,
            imitation_learning=False,
            print_failure=False,
            visualize_rewards=False):
        """!
        Runs the agent through the environment for generating experience.

          @param k: Number of episodes to run
          @param imitation_learning: Indicates if imitation learning is performed
          @param print_failure: If more detail on failure cases should be printed
        """
        self.time_diffs = list()
        self.robot.policy.set_phase(self.phase)
        success_times = []
        collision_times = []
        collision_other_agent_times = []
        timeout_times = []
        success = 0
        collision = 0
        collision_other_agent = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        collision_other_agent_cases = []
        timeout_cases = []
        accumulated_transitions = []
        num_episodes = k
        count = 0
        if imitation_learning:
            self.il_iteration_counter += 1
        if visualize_rewards:
            mean_rewards_goal = []
            num_goal_episodes_added = []
            mean_rewards_collision = []
            num_collision_episodes_added = []
        while k > 0:
            if imitation_learning:
                print("Case ", count)
                self.robot.policy.reset()
                ob, global_map, local_map = self.env.reset(
                    self.phase, imitation_learning=True)
            else:
                print("Case ", count)
                if self.robot.policy.name == 'ORCA':
                    self.robot.policy.reset()
                    ob, global_map, local_map = self.env.reset(self.phase)
                else:
                    ob, local_map = self.env.reset(self.phase)
            done = False
            transitions = []
            if visualize_rewards:
                new_rewards = []

            while not done:
                #time_start = time.time()
                if self.robot.policy.name == 'ORCA':
                    action = self.robot.act(ob, global_map)
                elif self.robot.policy.name == 'SDOADRL':
                    action = self.robot.act(ob, local_map)
                else:
                    action = self.robot.act(ob)
                #time_diff = time.time() - time_start
                # self.time_diffs.append(time_diff)
                ob, local_map_new, reward, done, info = self.env.step(action)
                if self.phase == 'train':
                    if imitation_learning:
                        action_index = self.calculate_action_index(action)
                    else:
                        action_index = self.robot.action_index
                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)
                    # Do not use negative reward for being close to obstacles
                    # when doing IL
                    if imitation_learning:
                        reward = 0
                if self.phase == 'train':
                    transitions.append(
                        Transition(
                            state=self.robot.last_state,
                            robot_state=None,
                            grid=local_map,
                            action=action_index,
                            reward=reward,
                            humans_in_FOV=self.robot.humans_in_FOV))
                local_map = local_map_new

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(count)
                collision_times.append(self.env.global_time)
            elif isinstance(info, CollisionOtherAgent):
                collision_other_agent += 1
                collision_other_agent_cases.append(count)
                collision_other_agent_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(count)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')
            if self.phase == 'train':
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    gamma_bar = pow(
                        self.gamma,
                        self.robot.time_step *
                        self.robot.v_pref)
                    self.local_t = next(self.local_counter)
                    if self.global_counter is not None:
                        self.global_t = next(self.global_counter)
                    # only add positive(success) or negative(collision)
                    # experience in experience set
                    reward = 0.0
                    for transition in transitions[::-1]:
                        if imitation_learning:
                            reward = transition.reward + gamma_bar * reward
                            state, humans_in_FOV = self.policy.transform(
                                transition.state)
                            self_state = state[0, 0:6]
                            input_grid = transition.grid.tolist()
                            if humans_in_FOV:
                                self.add_to_buffer(
                                    (state, self_state, input_grid, transition.action, reward))
                            else:
                                self.add_to_buffer(
                                    (state, self_state, input_grid, transition.action, reward), no_humans=True)
                        else:
                            reward = transition.reward + gamma_bar * reward
                            next_state = transition.state
                            self_state = transition.state[0, 0:6]
                            input_grid = transition.grid.tolist()
                        accumulated_transitions.append(
                            Transition(
                                state=transition.state,
                                robot_state=self_state,
                                grid=input_grid,
                                action=transition.action,
                                reward=reward,
                                humans_in_FOV=transition.humans_in_FOV))
                        if visualize_rewards:
                            new_rewards.insert(0, reward)
                    if visualize_rewards:
                        from itertools import zip_longest
                        if isinstance(info, ReachGoal):
                            mean_rewards_goal[:] = [
                                sum(reward) for reward in zip_longest(
                                    mean_rewards_goal, new_rewards, fillvalue=0)]
                            num_goal_episodes_added[:] = [
                                sum(i) for i in zip_longest(
                                    num_goal_episodes_added, np.ones(
                                        len(new_rewards)), fillvalue=0)]
                        if isinstance(info, Collision):
                            mean_rewards_collision[:] = [
                                sum(reward) for reward in zip_longest(
                                    mean_rewards_collision, new_rewards, fillvalue=0)]
                            num_collision_episodes_added[:] = [
                                sum(i) for i in zip_longest(
                                    num_collision_episodes_added, np.ones(
                                        len(new_rewards)), fillvalue=0)]
                elif not imitation_learning:
                    # for RL continue gaining experience until you have a
                    # positive or negative experience
                    k = k + 1
                    num_episodes = num_episodes + 1
            elif self.phase == 'test':
                self.local_t = next(self.local_counter)
            k = k - 1
            count = count + 1
            cumulative_rewards.append(sum([pow(self.gamma,
                                               t * self.robot.time_step * self.robot.v_pref) * transition.reward for t,
                                           transition in enumerate(accumulated_transitions)]))

        if visualize_rewards:
            mean_rewards_goal[:] = [
                reward / float(i) for reward,
                i in zip(
                    mean_rewards_goal,
                    num_goal_episodes_added)]
            mean_rewards_collision[:] = [
                reward / float(i) for reward,
                i in zip(
                    mean_rewards_collision,
                    num_collision_episodes_added)]
            self.mean_rewards_goal_list.append(mean_rewards_goal)
            self.mean_rewards_collision_list.append(mean_rewards_collision)
        success_rate = success / float(num_episodes)
        collision_rate = collision / float(num_episodes)
        collision_other_agent_rate = collision_other_agent / \
            float(num_episodes)
        assert success + collision + timeout + collision_other_agent == num_episodes
        avg_nav_time = sum(success_times) / float(len(success_times)
                                                  ) if success_times else self.env.time_limit

        extra_info = '' if self.local_t is None else 'in episode {} '.format(
            self.local_t)
        if imitation_learning or self.phase in [
                'val', 'test'] or self.local_t % 10 == 0:
            logging.info(
                '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, collision from other agents rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'. format(
                    self.phase.upper(),
                    extra_info,
                    success_rate,
                    collision_rate,
                    collision_other_agent_rate,
                    avg_nav_time,
                    average(cumulative_rewards)))
        if self.phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + \
                             collision_other_agent_times + timeout_times) / self.robot.time_step
            logging.info(
                'Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                too_close / float(total_time),
                average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' +
                         ' '.join([str(x) for x in collision_cases]))
            logging.info('Collision from other agent cases: ' +
                         ' '.join([str(x) for x in collision_other_agent_cases]))
            logging.info('Timeout cases: ' +
                         ' '.join([str(x) for x in timeout_cases]))

    def add_to_buffer(self, item, no_humans=False):
        """!
        Adds the experience to the local buffer while checking that the max size is not exceeded.

           @param item: The item to add
        """
        if no_humans:
            buffer = self.buffer_no_humans
        else:
            buffer = self.buffer
        if len(buffer) + len(item) >= self.buffer_size:
            buffer[0:(len(item) + len(buffer)) - self.buffer_size] = []
        buffer.append(item)

    def update_transitions(
            self,
            transitions,
            episode_number=None,
            no_humans=False):
        """!
        Updates global policy and value networks based on collected experience.

          @param transitions: A list of experience transitions
        """
        states = []
        robot_states = []
        actions = []
        value_targets = []
        input_grids = []
        loss = 0
        if no_humans:
            model = self.no_human_model
        else:
            model = self.model
        if len(transitions) > 0:
            for transition in transitions:
                if len(states) == 0 or len(
                        transition.state) == len(states[-1]):
                    states.append(transition.state)
                    input_grids.append(transition.grid)
                    robot_states.append(transition.robot_state)
                    actions.append(transition.action)
                    value_targets.append(transition.reward)
                    updated = False
                else:
                    network_value, loss_new = self.update(
                        no_humans, states, robot_states, actions, input_grids, value_targets, model, episode_number)
                    loss += loss_new
                    states = [transition.state]
                    robot_states = [transition.robot_state]
                    actions = [transition.action]
                    value_targets = [transition.reward]
                    input_grids = [transition.grid]

            network_value, loss_new = self.update(
                no_humans, states, robot_states, actions, input_grids, value_targets, model, episode_number)
            loss += loss_new
        return loss

    def update(
            self,
            no_humans,
            states,
            robot_states,
            actions,
            input_grids,
            value_targets,
            model,
            episode_number):
        if no_humans:
            feed_dict = {
                model.robot_state: robot_states,
                model.action_index: actions,
                # self.model.input_grid_placeholder: input_grids,
                model.angular_map_placeholder: input_grids,
                model.output_placeholder: value_targets}

        else:
            if self.use_grid_map:
                feed_dict = {model.state: states,
                             model.robot_state: robot_states,
                             model.action_index: actions,
                             model.input_grid_placeholder: input_grids,
                             model.output_placeholder: value_targets}
            else:
                feed_dict = {model.state: states,
                             model.robot_state: robot_states,
                             model.action_index: actions,
                             model.angular_map_placeholder: input_grids,
                             model.output_placeholder: value_targets}

        network_value, loss, _, summary_str = self.sess.run([
            model.value_output,
            model.cost_all,
            model.apply_grads,
            model.summary],
            feed_dict=feed_dict)

        if self.summary_writer is not None and episode_number is not None:
            self.summary_writer.add_summary(summary_str, episode_number)
            self.summary_writer.flush()

        return network_value, loss
# For imitation learning

    def sample_batches(self, buffer, size):
        """!
        Samples batches from the experience buffer for IL.

          @param size: Size of batch
        """
        shuffled_list = buffer.copy()
        random.shuffle(shuffled_list)
        number_batches = len(shuffled_list) // size
        number_entries = number_batches * size
        shortened_list = shuffled_list[0: number_entries]
        return np.reshape(shortened_list, [number_batches, size, -1])

    def optimize_epoch(self, num_epochs, batch_size):
        """!
        Updates global policy and value networks using IL.

          @param num_epochs: Number of epochs to run
          @param batch_size: Size of batch
        """
        for buffer, model in zip(
                (self.buffer, self.buffer_no_humans), (self.model, self.no_human_model)):
            if len(buffer) > 0:
                average_epoch_loss = 0
                for epoch in range(num_epochs):
                    epoch_loss = 0
                    batches = self.sample_batches(buffer, batch_size)
                    for batch in batches:
                        inputs, robot_states, input_grids, actions, values = batch.transpose()
                        inputs = inputs.tolist()
                        input_grids = input_grids.tolist()
                        robot_states = robot_states.tolist()
                        if model == self.model:
                            self.sess.run(self.update_local_ops)
                            no_humans = False
                        else:
                            self.sess.run(self.update_local_ops_no_humans)
                            no_humans = True
                        network_value, loss = self.update(
                            no_humans, inputs, robot_states, actions, input_grids, values, model, self.il_iteration_counter * num_epochs + epoch)

                        epoch_loss += loss

                    average_epoch_loss = epoch_loss / float(len(batches))
                    logging.info(
                        'Average loss in epoch %d/%d: %.2E',
                        epoch,
                        num_epochs,
                        average_epoch_loss)
                logging.info('Experience set size: %d', len(buffer))

    def calculate_action_index(self, action_chosen):
        """!
        Calculates the action index from the action space that is closest to the action chosen by the IL agent.

            @param action_chosen: The action chosen by the IL agent
        """
        diff_v_min = float("inf")
        diff_r_min = float("inf")
        action_index = 0
        for i, action in enumerate(self.policy.action_space):
            if abs(action.v *
                   self.robot.v_pref -
                   action_chosen.v) < diff_v_min or (abs(action.v *
                                                         self.robot.v_pref -
                                                         action_chosen.v) == diff_v_min and abs(action.r -
                                                                                                action_chosen.r) < diff_r_min):
                diff_v_min = abs(action.v - action_chosen.v)
                diff_r_min = abs(action.r - action_chosen.r)
                action_index = i

        #print('Closest action: ', self.policy.action_space[action_index].v, self.policy.action_space[action_index].r, 'action taken:', action_chosen.v, action_chosen.r)
        return action_index


def average(input_list):
    if input_list:
        return sum(input_list) / float(len(input_list))
    else:
        return 0
