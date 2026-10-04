"""Both nodes' parameters and the vehicle.yaml check; no rclpy at import (the launch file and host tests use it)."""
import math
from pathlib import Path
import yaml

WAYPOINT_NODE = dict(camreal_config='data/camreal.yaml', waypoint_topic='/waypoint', path_topic='/predicted_path',
                     debug_image_topic='/camsim_driver/bev', debug_image_hz=5.0, device='cpu', cpu_threads=1,
                     input_timeout_s=0.25, future_tolerance_s=0.02, max_waypoint_m=3.0,
                     path_frame='rear_axle', image_qos_reliability='best_effort')
PURE_PURSUIT_NODE = dict(camreal_config='data/camreal.yaml', waypoint_topic='/waypoint', drive_topic='/drive',
                         control_hz=25.0, waypoint_timeout_s=0.25, future_tolerance_s=0.02, target_speed_mps=0.5,
                         wheelbase_m=0.0, steer_max_rad=0.30, max_waypoint_m=3.0, path_frame='rear_axle',
                         drive_enabled=False)
KNOWN = {**WAYPOINT_NODE, **PURE_PURSUIT_NODE, 'use_sim_time': False}
TEMPLATE, VEHICLE = 'camreal/ros2/camsim_driver/config/vehicle.yaml', 'vehicle.yaml(data/config/vehicle.yaml)'
MAX_SPEED_MPS = 2.0   # course limit: 5.0 typed for 0.5 must not reach the motor


def read_vehicle_yaml(path):
    """-> the /** ros__parameters. Each node silently ignores keys it does not declare, so typos stop here."""
    data = yaml.safe_load(Path(path).read_text())
    if isinstance(data, dict) and any(str(key).strip('/') == 'camsim_driver_node' for key in data):
        raise ValueError(f'이전 형식의 vehicle.yaml입니다: {path}. {TEMPLATE}을 data/config/로 다시 복사하고 '
                         '축간거리(wheelbase_m)를 채우세요.')
    section = data.get('/**') if isinstance(data, dict) and len(data) == 1 else None
    values = section.get('ros__parameters') if isinstance(section, dict) and len(section) == 1 else None
    if not isinstance(values, dict):
        raise ValueError(f'{path}: "/**:" 아래 "ros__parameters:" 한 섹션만 있어야 합니다. {TEMPLATE} 형식을 따르세요.')
    if 'drive_enabled' in values:   # the launch argument overrides it silently; ros2 run with the file would drive
        raise ValueError(f'{path}: drive_enabled는 vehicle.yaml에 두지 말고 launch 인자(drive_enabled:=true)나 '
                         'ros2 run의 -p drive_enabled:=true로 주세요. 그 줄을 지우세요.')
    unknown = sorted(set(values) - set(KNOWN))
    if unknown:
        raise ValueError(f'{path}: 모르는 파라미터 {unknown}. 노드는 모르는 이름을 무시하므로 {TEMPLATE}의 이름으로 고치세요.')
    wrong = sorted(key for key, value in values.items() if type(value) is not type(KNOWN[key]))
    if wrong:
        raise ValueError(f'{path}: 형식이 다른 값 {wrong}. 기본값과 같은 형식으로 적으세요(실수는 0.33, 1.0처럼 소수점 포함).')
    return values


def declare(node, defaults):
    """Startup-only (read-only) parameters; a wrong type such as 1 for 1.0 names the parameter and the fix."""
    from rcl_interfaces.msg import ParameterDescriptor
    from rclpy.exceptions import InvalidParameterTypeException
    values = {}
    for key, default in defaults.items():
        try:
            values[key] = node.declare_parameter(key, default, ParameterDescriptor(read_only=True)).value
        except InvalidParameterTypeException:
            raise ValueError(f'{key}의 형식이 기본값({default!r})과 다릅니다. '
                             '같은 형식으로 적으세요(실수는 1.0처럼 소수점 포함).') from None
    return values


def required(values, *keys):
    for key in keys:
        if not values[key]:
            raise ValueError(f'{key} 값이 비었습니다. {VEHICLE}에 적으세요.')


def positive(values, *keys):
    for key in keys:
        if not (math.isfinite(values[key]) and values[key] > 0):
            raise ValueError(f'{key}={values[key]}: 0보다 큰 유한한 값을 {VEHICLE}에 적으세요.')


def nonnegative(values, *keys):
    for key in keys:
        if not (math.isfinite(values[key]) and values[key] >= 0):
            raise ValueError(f'{key}={values[key]}: 0 이상의 유한한 값을 {VEHICLE}에 적으세요.')
