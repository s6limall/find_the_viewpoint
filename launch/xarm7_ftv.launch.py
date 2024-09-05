# File: xarm7_with_perception.launch.py

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Include the XArm7 Gazebo launch file
    xarm7_gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('xarm_moveit_config'),
                'launch',
                'xarm7_moveit_gazebo.launch.py'
            ])
        ])
    )

    # Launch your RobotPerception node
    robot_perception_node = Node(
        package='ftv',  # Replace with your actual package name
        executable='ftv_ros',
        name='robot_perception',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Create and return launch description
    return LaunchDescription([
        xarm7_gazebo_launch,
        robot_perception_node
    ])
