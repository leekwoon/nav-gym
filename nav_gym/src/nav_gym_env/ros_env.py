import os
import time
import copy
import numpy as np

import rospy
import ros_numpy
import rosservice
from pedsim_msgs.msg import TrackedPersons, TrackedPerson
from geometry_msgs.msg import Twist, PolygonStamped, Point32, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

import nav_gym.srv
from nav_gym_env.env import NavGymEnv, observation_to_dict
from nav_gym_env.utils import (
    translation_matrix_from_xyz,
    quaternion_matrix_from_yaw,
    transform_xys
)


class RosEnv(object):
    def __init__(self,  wrapped_env):
        rospy.init_node('ros_env')
        rospy.on_shutdown(self.close)

        self._wrapped_env = wrapped_env
        self.observation_space = self._wrapped_env.observation_space
        self.action_space = self._wrapped_env.action_space

        self.wait_service()
        self.reset_map_service = rospy.ServiceProxy(
            '/reset_map', 
            nav_gym.srv.ResetMap, 
            persistent=True
        )
        self.strict_update_service = rospy.ServiceProxy(
            '/strict_update', 
            nav_gym.srv.StrictUpdate, 
            persistent=True
        )

        self.sub_cmd_vel = rospy.Subscriber(
            'cmd_vel', 
            Twist, 
            self.callback_cmd_vel
        )

        self.cmd_vel = np.array([0., 0.])

    def callback_cmd_vel(self, msg):
        self.cmd_vel = np.array([msg.linear.x, msg.angular.z])

    def wait_service(self):
        while True:
            service_list = rosservice.get_service_list()
            if '/reset_map' in service_list \
                and '/strict_update' in service_list:
                break
            rospy.loginfo("[RosEnv] Not all services ready yet! Wait ...")  
            time.sleep(0.2)

    def reset_map(self):
        """
        update map
        """
        map_info = self._wrapped_env.map_info
        map_msg = ros_numpy.msgify(OccupancyGrid, map_info['data'])
        map_msg.info.resolution = map_info['resolution']
        map_msg.info.width = map_info['width']
        map_msg.info.height = map_info['height']
        map_msg.info.origin.position.x = map_info['origin'][0]
        map_msg.info.origin.position.y = map_info['origin'][1]
        map_msg.info.origin.position.z = 0 
        map_msg.info.origin.orientation.x = 0
        map_msg.info.origin.orientation.y = 0 
        map_msg.info.origin.orientation.z = 0 
        map_msg.info.origin.orientation.w = 1.0 
        response = self.reset_map_service(map_msg)
        
    def strict_update(self):
        """
        update robot pose, human poses, scan 
        """
        robot = self._wrapped_env.robot

        # robot pose
        quaternion = quaternion_from_euler(0, 0, robot.theta)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time(0)
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = robot.px
        pose_msg.pose.position.y = robot.py
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]

        # robot footprint
        footprint_msg = PolygonStamped()
        footprint_msg.header.stamp = rospy.Time.now()
        footprint_msg.header.frame_id = 'map'
        transformed_footprint = transform_xys(
            translation_matrix_from_xyz(robot.px, robot.py, 0),
            quaternion_matrix_from_yaw(robot.theta),
            np.array(robot.footprint)
        )
        for x, y in transformed_footprint:
            p32 = Point32()
            p32.x = x
            p32.y = y
            footprint_msg.polygon.points.append(p32)

        threshold_footprint_msg = PolygonStamped()
        threshold_footprint_msg.header.stamp = rospy.Time.now()
        threshold_footprint_msg.header.frame_id = 'map'
        transformed_threshold_footprint = transform_xys(
            translation_matrix_from_xyz(robot.px, robot.py, 0),
            quaternion_matrix_from_yaw(robot.theta),
            np.array(robot.threshold_footprint)
        )
        for x, y in transformed_threshold_footprint:
            p32 = Point32()
            p32.x = x
            p32.y = y
            threshold_footprint_msg.polygon.points.append(p32)
        
        discomfort_threshold_footprint_msg = PolygonStamped()
        discomfort_threshold_footprint_msg.header.stamp = rospy.Time.now()
        discomfort_threshold_footprint_msg.header.frame_id = 'map'
        transformed_discomfort_threshold_footprint = transform_xys(
            translation_matrix_from_xyz(robot.px, robot.py, 0),
            quaternion_matrix_from_yaw(robot.theta),
            np.array(robot.discomfort_threshold_footprint)
        )
        for x, y in transformed_discomfort_threshold_footprint:
            p32 = Point32()
            p32.x = x
            p32.y = y
            discomfort_threshold_footprint_msg.polygon.points.append(p32)

        # scan
        scan_msg = LaserScan()
        # scan.header.stamp = rospy.Time.now() # will be handled in robot_manager.py
        scan_msg.header.frame_id = 'laser_link'
        scan_msg.angle_min = robot.angle_min
        scan_msg.angle_max = robot.angle_max
        scan_msg.angle_increment = robot.angle_increment
        scan_msg.range_max = robot.range_max
        observation_dict = observation_to_dict(
            self._wrapped_env.prev_obs['observation'],
            num_scan_stack=self._wrapped_env.num_scan_stack,
            n_angles=robot.n_angles
        )
        scan_msg.ranges = observation_dict['scan']

        # humans
        humans_msg = TrackedPersons()
        humans_msg.header.stamp = rospy.Time(0)
        humans_msg.header.frame_id = 'map'
        for i, human in enumerate(self._wrapped_env.humans):
            human_msg = TrackedPerson()
            human_msg.track_id = i
            human_msg.detection_id = i
            human_msg.pose.pose.position.x = human.px
            human_msg.pose.pose.position.y = human.py
            q = quaternion_from_euler(0, 0, human.theta)
            human_msg.pose.pose.orientation.x = q[0]
            human_msg.pose.pose.orientation.y = q[1]
            human_msg.pose.pose.orientation.z = q[2]
            human_msg.pose.pose.orientation.w = q[3]
            human_msg.twist.twist.linear.x = human.vx 
            human_msg.twist.twist.linear.y = human.vy
            humans_msg.tracks.append(human_msg)

        response = self.strict_update_service(
            humans_msg,
            pose_msg, 
            footprint_msg,
            threshold_footprint_msg,
            discomfort_threshold_footprint_msg,
            scan_msg
        )

    def reset(self):
        obs = self._wrapped_env.reset()
        self.reset_map()
        self.strict_update()
        return obs

    def step(self, action):
        obs, reward, done, info = self._wrapped_env.step(action)
        self.strict_update()
        return obs, reward, done, info

    def close(self):
        # self.dxp2d_process.terminate()
        os.kill(os.getpid(), 9) # kill process


def make_circle(c, r, res=10):
    thetas = np.linspace(0, 2*np.pi, res+1)[:-1]
    verts = np.zeros((res, 2))
    verts[:,0] = c[0] + r * np.cos(thetas)
    verts[:,1] = c[1] + r * np.sin(thetas)
    return verts


if __name__ == "__main__":
    import gym
    import nav_gym_env
    env = gym.make('NavGym-v0')
    env = RosEnv(env)

    env.reset()

    for _ in range(1000):
        env.step(env.action_space.sample())
        time.sleep(0.01)
        # env.render()
