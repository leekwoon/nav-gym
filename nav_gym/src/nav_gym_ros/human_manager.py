import rospy
from tf import TransformListener
from pedsim_msgs.msg import TrackedPersons


class HumanManager(object):
    def __init__(self, simulation_manager, clock_hz, hz):
        self.simulation_manager = simulation_manager
        self.clock_hz = clock_hz
        self.hz = hz
        self.i = 0 

        self.humans_msg = None

        self.tf_listener = TransformListener()
        self.pub_humans = rospy.Publisher(
            '/humans', TrackedPersons, queue_size=10
        )

    def process(self):
        self.i += 1

        if self.i % (self.clock_hz / self.hz) == 0:
            if self.humans_msg is not None:
                self.pub_humans.publish(self.humans_msg)
         