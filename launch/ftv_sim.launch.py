# File: ftv_sim.launch.py

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    ftv_pkg = get_package_share_directory('ftv')
    xarm_moveit_config_pkg = get_package_share_directory('xarm_moveit_config')
    xarm_description_pkg = get_package_share_directory('xarm_description')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Process the URDF file
    xacro_file = os.path.join(ftv_pkg, 'urdf', 'xarm7.urdf.xacro')
    robot_description_content = Command(['xacro ', xacro_file])

    robot_description = {'robot_description': robot_description_content}

    # Process the SRDF file
    srdf_xacro_file = os.path.join(xarm_moveit_config_pkg, 'srdf', 'xarm.srdf.xacro')
    robot_description_semantic_content = Command(['xacro ', srdf_xacro_file])

    robot_description_semantic = {'robot_description_semantic': robot_description_semantic_content}

    robot_state_publisher_node = Node(package='robot_state_publisher', executable='robot_state_publisher',
                                      name='robot_state_publisher', output='screen', parameters=[robot_description])

    xarm7_gazebo_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        [os.path.join(xarm_moveit_config_pkg, 'launch', 'xarm7_moveit_gazebo.launch.py')]),
        launch_arguments={'robot_model': 'xarm7', 'robot_description_file': xacro_file,
                          'use_sim_time': use_sim_time, }.items())

    # Launch our custom controller node
    ftv_ros_node = Node(package='ftv', executable='ftv_ros', name='ftv_ros', output='screen',
                        parameters=[robot_description, robot_description_semantic, {'use_sim_time': use_sim_time}],
                        arguments=['--ros-args', '--log-level', 'debug'], emulate_tty=True)

    # Delay the start of ftv_ros_node
    delayed_ftv_ros_node = TimerAction(period=10.0, actions=[ftv_ros_node])

    return LaunchDescription([DeclareLaunchArgument('use_sim_time', default_value='true',
                                                    description='Use simulation (Gazebo) clock if true'),
                              robot_state_publisher_node, xarm7_gazebo_launch, delayed_ftv_ros_node])
