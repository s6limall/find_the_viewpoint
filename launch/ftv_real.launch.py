import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch import LaunchDescription

def generate_launch_description():
    ftv_pkg = get_package_share_directory('ftv')
    xarm_moveit_config_pkg = get_package_share_directory('xarm_moveit_config')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')  # Real robot, so no simulation time
    robot_ip = LaunchConfiguration('robot_ip', default='192.168.1.120')  # TODO: Check the real robot IP
    hw_ns = LaunchConfiguration('hw_ns', default='xarm')

    # Process the URDF file for the real robot
    xacro_file = os.path.join(ftv_pkg, 'urdf', 'xarm7_with_object.urdf.xacro')
    robot_description_content = Command(['xacro ', xacro_file])
    robot_description = {'robot_description': robot_description_content}

    # Process the SRDF file
    # srdf_xacro_file = os.path.join(xarm_moveit_config_pkg, 'srdf', 'xarm.srdf.xacro')
    srdf_xacro_file = os.path.join(get_package_share_directory('ftv'), 'srdf', 'xarm.srdf.xacro')
    robot_description_semantic_content = Command(['xacro ', srdf_xacro_file])
    robot_description_semantic = {'robot_description_semantic': robot_description_semantic_content}

    # Include the real robot moveit launch file
    xarm7_moveit_realmove_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('xarm_moveit_config'), 'launch', '_robot_moveit_realmove.launch.py'])
        ),
        launch_arguments={
            'robot_ip': robot_ip,
            'dof': '7',
            'robot_type': 'xarm',
            'hw_ns': hw_ns,
            'no_gui_ctrl': 'false',
        }.items(),
    )

    # Start RViz with Move it for the real robot
    xarm7_rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ftv_pkg, 'launch', 'xarm7_moveit_rviz.launch.py')
        ),
        launch_arguments={
            'robot_model': 'xarm7',
            'robot_description_file': xacro_file,
            'use_sim_time': use_sim_time,
        }.items()
    )

    ftv_ros_node = Node(
        package='ftv', executable='ftv_ros', name='ftv_ros', output='screen',
        parameters=[robot_description, robot_description_semantic, {'use_sim_time': use_sim_time}],
        arguments=['--ros-args', '--log-level', 'debug'], emulate_tty=True
    )

    # Delay the start of ftv_ros_node to ensure hardware and MoveIt are initialized
    delayed_ftv_ros_node = TimerAction(period=10.0, actions=[ftv_ros_node])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation clock if true'),
        DeclareLaunchArgument('robot_ip', default_value='192.168.1.120', description='IP address of the real robot'),
        xarm7_moveit_realmove_launch,  # Launch real robot with Move it
        xarm7_rviz_launch,             # Launch RViz with Move it
        delayed_ftv_ros_node           # Delay custom controller node
    ])
