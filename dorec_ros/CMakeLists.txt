cmake_minimum_required(VERSION 2.8)

project(dorec_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  rospy_tutorials
  std_msgs
  geometry_msgs
  message_generation
  sensor_msgs
)

add_message_files(
  FILE
  ArmPose.msg
  CoefficientArray.msg
  Coefficient.msg
  KeypointArry.msg
  Keypoint.msg
  ManipPlan.msg
  SegmentationMap.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
