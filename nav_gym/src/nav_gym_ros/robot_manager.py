import numpy as np

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PolygonStamped, PoseStamped


class RobotManager(object):
    def __init__(self, simulation_manager, clock_hz, hz):
        self.simulation_manager = simulation_manager
        self.clock_hz = clock_hz
        self.hz = hz
        self.i = 0

        self.scan_msg = None
        self.pose_msg = None
        self.footprint_msg = None
        self.threshold_footprint_msg = None
        self.discomfort_threshold_footprint_msg = None

        self.pub_pose = rospy.Publisher(
            # '/pose', PoseStamped, queue_size=10
            '/mcl3d/current/pose', PoseStamped, queue_size=10
        )
        self.pub_footprint = rospy.Publisher(
            '/footprint', PolygonStamped, queue_size=10
        )
        self.pub_threshold_footprint = rospy.Publisher(
            '/threshold_footprint', PolygonStamped, queue_size=10
        )
        self.pub_discomfort_threshold_footprint = rospy.Publisher(
            '/discomfort_threshold_footprint', PolygonStamped, queue_size=10
        )
        self.pub_scan_1 = rospy.Publisher(
            # '/scan', LaserScan, queue_size=10
            '/scan_merged', LaserScan, queue_size=10
        )
        # dummy
        self.pub_scan_2 = rospy.Publisher( 
            '/sick_front_scan_1', LaserScan, queue_size=10
        )
        # dummy
        self.pub_scan_3 = rospy.Publisher(
            '/sick_back_scan_2', LaserScan, queue_size=10
        )


    def process(self):
        self.i += 1
        if self.i % (self.clock_hz / self.hz) == 0:
            if self.pose_msg is not None:
                self.pub_pose.publish(self.pose_msg)
            if self.footprint_msg is not None:
                self.pub_footprint.publish(self.footprint_msg)
            if self.threshold_footprint_msg is not None:
                self.pub_threshold_footprint.publish(self.threshold_footprint_msg)
            if self.discomfort_threshold_footprint_msg is not None:
                self.pub_discomfort_threshold_footprint.publish(self.discomfort_threshold_footprint_msg)
            if self.scan_msg is not None:
                self.scan_msg.header.stamp = rospy.Time.now()
                self.pub_scan_1.publish(self.scan_msg)
                self.pub_scan_2.publish(self.scan_msg)
                self.pub_scan_3.publish(self.scan_msg)

