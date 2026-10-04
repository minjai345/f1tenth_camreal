"""Run from the repository root:  ros2 launch camsim_driver camsim_driver.launch.py [drive_enabled:=true]"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from camsim_driver.params import read_vehicle_yaml


def nodes(context):
    params = os.path.abspath(LaunchConfiguration('params_file').perform(context))
    if not os.path.isfile(params):
        raise RuntimeError(f'params_file not found: {params}. Run from the repository root '
                           '(cd ~/f1tenth_gym) or pass params_file:=/absolute/path/vehicle.yaml')
    read_vehicle_yaml(params)   # old layout, typos, 1 for 1.0
    enabled = LaunchConfiguration('drive_enabled').perform(context).strip().lower() in ('true', '1', 'yes')
    # Both nodes read the same /** section; only pure_pursuit_node gets the drive switch. While driving, its exit
    # (wheelbase_m not set, ...) ends the launch instead of leaving perception up with no controller.
    stop = dict(on_exit=Shutdown(reason='pure_pursuit_node exited')) if enabled else {}
    return [Node(package='camsim_driver', executable='waypoint_node', name='waypoint_node',
                 output='screen', parameters=[params]),
            Node(package='camsim_driver', executable='pure_pursuit_node', name='pure_pursuit_node',
                 output='screen', parameters=[params, {'drive_enabled': enabled}], **stop)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value='data/config/vehicle.yaml',
                              description='vehicle parameters (relative to the repository root)'),
        DeclareLaunchArgument('drive_enabled', default_value='false',
                              description='pure_pursuit_node publishes /drive; false = prediction and RViz only'),
        OpaqueFunction(function=nodes),
    ])
