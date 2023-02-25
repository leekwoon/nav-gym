#!/usr/bin/env python3

# adapted from:
# https://github.com/avidbots/flatland
import rospy

from nav_gym_ros.simulation_manager import SimulationManager


if __name__ == "__main__":
    rospy.init_node('nav_gym_simulator')

    simulation_manager = SimulationManager(
        clock_hz=100, # the physics update rate
        robot_hz=25,
        tf_hz=25,
        human_hz=10
    )
    simulation_manager.run()