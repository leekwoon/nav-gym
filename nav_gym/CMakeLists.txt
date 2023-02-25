cmake_minimum_required(VERSION 2.8.3)
project(nav_gym)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  pedsim_msgs
  nav_msgs
  std_msgs
  sensor_msgs
)

# Generate services in the 'srv' folder
add_service_files(
    FILES
    ResetMap.srv
    StrictUpdate.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
    DEPENDENCIES
    pedsim_msgs
    nav_msgs
    geometry_msgs
    sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS nav_msgs std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
