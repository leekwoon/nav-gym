import rospy
from nav_msgs.msg import OccupancyGrid


class MapManager(object):
    def __init__(self, simulation_manager):
        self.simulation_manager = simulation_manager
        self.map_msg = None

        self.pub_map = rospy.Publisher(
            '/mcl3d/map/grid', OccupancyGrid, queue_size=1, latch=True
        )

    def process(self):
        self.pub_map.publish(self.map_msg)
       