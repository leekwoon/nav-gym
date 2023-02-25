import rospy
from tf import TransformBroadcaster
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class TfManager(object):
    def __init__(self, simulation_manager, clock_hz, hz):
        self.simulation_manager = simulation_manager
        self.clock_hz = clock_hz
        self.hz = hz
        self.br = TransformBroadcaster()

        self.i = 0

    def process(self):
        self.i += 1
        if self.i % (self.clock_hz / self.hz) == 0:
            # base_link -> laser_link
            self.br.sendTransform(
                (0., 0., 0.),
                (0., 0., 0., 1.),
                rospy.Time.now(),
                'laser_link',
                'base_link'
            )

            pose_msg = self.simulation_manager.robot_manager.pose_msg
            if pose_msg is not None:
                x = pose_msg.pose.position.x
                y = pose_msg.pose.position.y
                _, _, yaw = euler_from_quaternion([
                    pose_msg.pose.orientation.x,
                    pose_msg.pose.orientation.y,
                    pose_msg.pose.orientation.z,
                    pose_msg.pose.orientation.w
                ])
                self.br.sendTransform(
                    (x, y, 0.),
                    quaternion_from_euler(0, 0, yaw),
                    rospy.Time.now(),
                    'base_link',
                    'map'
                )
