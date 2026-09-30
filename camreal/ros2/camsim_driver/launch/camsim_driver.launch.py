"""Run from the repository root:  ros2 launch camsim_driver camsim_driver.launch.py [drive_enabled:=true]"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def driver(context):
    params = os.path.abspath(LaunchConfiguration('params_file').perform(context))
    if not os.path.isfile(params):
        raise RuntimeError(f'params_file not found: {params}. Run from the repository root '
                           '(cd ~/f1tenth_gym) or pass params_file:=/absolute/path/vehicle.yaml')
    enabled = LaunchConfiguration('drive_enabled').perform(context).strip().lower() in ('true', '1', 'yes')
    return [Node(package='camsim_driver', executable='camsim_driver_node', name='camsim_driver_node',
                 output='screen', parameters=[params, {'drive_enabled': enabled}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value='data/config/vehicle.yaml',
                              description='vehicle parameters (relative to the repository root)'),
        DeclareLaunchArgument('drive_enabled', default_value='false',
                              description='publish /drive; false = prediction and RViz only'),
        OpaqueFunction(function=driver),
    ])
