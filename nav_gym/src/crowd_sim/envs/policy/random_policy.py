from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionRot
import numpy as np


class RandomPolicy(Policy):
    def __init__(self):
        super().__init__()

    def predict(self, agent):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        action = ActionRot(
            (0.5 + np.random.random() * 0.5) * agent.v_pref,
            (np.random.random() - 0.5) * np.pi)
        return action
