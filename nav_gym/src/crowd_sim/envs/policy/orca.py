import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot


class ORCA(Policy):
    """!
    Executes the ORCA Policy for an agent.
    """

    def __init__(self, safety_space=0):
        """!
        @param safety_space: Safety space for each agent that should be considered
                        when avoiding each other.
        @param timeStep: The time step of the simulation. Must be positive.
        @param neighborDist: The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        @param maxNeighbors: The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        @param timeHorizon: The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        @param timeHorizonObst: The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        @param radius:  The default radius of a new agent.
                        Must be non-negative.
        @param maxSpeed: The default maximum speed of a new agent.
                        Must be non-negative.
        @param velocity: The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step.

        """
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.safety_space = safety_space
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.sim = None
        self.FOV_min_angle = -np.pi % (2 * np.pi)
        self.FOV_max_angle = np.pi % (2 * np.pi)

    def set_phase(self, phase):
        self.phase = phase
        return

    def configure(self, config):
        self.FOV_min_angle = config.getfloat(
            'map', 'angle_min') * np.pi % (2 * np.pi)
        self.FOV_max_angle = config.getfloat(
            'map', 'angle_max') * np.pi % (2 * np.pi)
        self.safety_space = config.getfloat('reward', 'discomfort_dist')

    def reset(self):
        del self.sim
        self.sim = None

    def predict(self, state, global_map, agent):
        """!
        Create a rvo2 simulation at each time step and run one step.
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        @param state: Current state of the environment
        @param global_map: A list of verteces of obstacles in the global environment
        @return Action of the agent
        """
        self_state = state.self_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        # create sim with static obstacles if they don't exist
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, agent.radius, agent.v_pref)
            for obstacle in global_map:
                self.sim.addObstacle(obstacle)
            self.sim.processObstacles()

        self.sim.clearAgents()
        self.sim.addAgent(self_state.position, *params, self_state.radius + 0.01 + self.safety_space,
                          self_state.v_pref, self_state.velocity)
        human_states_in_FOV = []
        for human_state in state.human_states:
            if self.human_state_in_FOV(state.self_state, human_state):
                human_states_in_FOV.append(human_state)
        for human_state in human_states_in_FOV:
            self.sim.addAgent(human_state.position, *params, human_state.radius + 0.01 + self.safety_space,
                              agent.v_pref, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed)
        # in the direction of the goal.
        velocity = np.array(
            (self_state.gx - self_state.px,
             self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(human_states_in_FOV):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        action = ActionRot(np.linalg.norm([action.vx, action.vy]), (np.arctan2(
            action.vy, action.vx) - self_state.theta))
        agent.last_state = state
        self.humans_available = len(state.human_states) > 0

        return action

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
