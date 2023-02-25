import rospy
import nav_gym.srv


class ServiceManager(object):
    def __init__(self, simulation_manager):
        self.simulation_manager = simulation_manager

        rospy.Service(
            '/reset_map',
            nav_gym.srv.ResetMap,
            self.reset_map
        )
        rospy.Service(
            '/strict_update', 
            nav_gym.srv.StrictUpdate, 
            self.strict_update
        )

    def reset_map(self, req):
        map_manager = self.simulation_manager.map_manager
        map_manager.map_msg = req.map
        map_manager.process()
        return True

    def strict_update(self, req):
        robot_manager = self.simulation_manager.robot_manager
        robot_manager.pose_msg = req.pose
        robot_manager.footprint_msg = req.footprint
        robot_manager.threshold_footprint_msg = req.threshold_footprint
        robot_manager.discomfort_threshold_footprint_msg = req.discomfort_threshold_footprint
        robot_manager.scan_msg = req.scan

        human_manager = self.simulation_manager.human_manager
        human_manager.humans_msg = req.humans
        return True

