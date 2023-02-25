import rospy
from nav_gym_ros.clock_manager import ClockManager
from nav_gym_ros.robot_manager import RobotManager
from nav_gym_ros.service_manager import ServiceManager
from nav_gym_ros.map_manager import MapManager
from nav_gym_ros.tf_manager import TfManager
from nav_gym_ros.human_manager import HumanManager


class SimulationManager(object):
    def __init__(
        self, 
        clock_hz, 
        robot_hz,
        tf_hz,
        human_hz 
    ):
        self.clock_hz = clock_hz
        self.robot_hz = robot_hz
        self.tf_hz = tf_hz
        self.human_hz = human_hz

        self.rate = rospy.Rate(clock_hz)
        self.clock_manager = ClockManager(clock_hz)
        self.robot_manager = RobotManager(self, clock_hz, robot_hz)
        self.tf_manager = TfManager(self, clock_hz, tf_hz)
        self.map_manager = MapManager(self)
        self.service_manager = ServiceManager(self)
        self.human_manager = HumanManager(self, clock_hz, human_hz)

    def run(self):
        while not rospy.is_shutdown():
            self.clock_manager.process()
            self.tf_manager.process()
            self.robot_manager.process()
            self.human_manager.process()
            self.rate.sleep()
