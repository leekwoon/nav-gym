import rospy
from rosgraph_msgs.msg import Clock


class ClockManager(object):
    def __init__(self, clock_hz):
        self.clock_dt = 1. / clock_hz
        self.time = rospy.Time(0)
        self.pub_clock = rospy.Publisher(
            '/clock', Clock, queue_size=10
        )

    def process(self):
        self.time += rospy.Duration(self.clock_dt)
        clock = Clock()
        clock.clock = self.time
        self.pub_clock.publish(clock)
        
