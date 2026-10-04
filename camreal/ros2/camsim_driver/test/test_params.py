"""vehicle.yaml rules the launch file applies; no ROS needed."""
from pathlib import Path
import pytest
import yaml
from camsim_driver.params import PURE_PURSUIT_NODE, WAYPOINT_NODE, read_vehicle_yaml

SHIPPED = Path(__file__).resolve().parents[1]/'config'/'vehicle.yaml'


def test_shipped_vehicle_yaml_holds_both_nodes_parameters():
    shipped = read_vehicle_yaml(SHIPPED)
    declared = {**WAYPOINT_NODE, **PURE_PURSUIT_NODE}
    del declared['drive_enabled']   # never from the file: the launch argument or -p only
    assert set(shipped) == set(declared)
    # rclpy refuses e.g. 25 for a parameter declared as 25.0.
    assert {key: type(value) for key, value in shipped.items()} == {key: type(v) for key, v in declared.items()}
    # Both nodes read one /** value, so a shared parameter has one default.
    assert all(WAYPOINT_NODE[key] == PURE_PURSUIT_NODE[key] for key in set(WAYPOINT_NODE) & set(PURE_PURSUIT_NODE))


@pytest.mark.parametrize('data,match', [
    ({'camsim_driver_node': {'ros__parameters': {'wheelbase_m': .3}}}, '이전 형식'),
    ({'/camsim_driver_node': {'ros__parameters': {'wheelbase_m': .3}}}, '이전 형식'),
    ({'waypoint_node': {'ros__parameters': {'wheelbase_m': .3}}}, 'ros__parameters'),
    ({'/**': {'wheelbase_m': .3}}, 'ros__parameters'),
    ({'/**': {'ros__parameters': None}}, 'ros__parameters'),
    (None, 'ros__parameters'),
    ({'/**': {'ros__parameters': {'wheelbase_m': .3, 'target_speed': .3, 'steer_max': .2}}},
     'steer_max.*target_speed'),
    ({'/**': {'ros__parameters': {'wheelbase_m': 1, 'control_hz': 25}}}, 'control_hz.*wheelbase_m.*1.0'),
    # The launch overrides it silently and `ros2 run ... --params-file` would drive.
    ({'/**': {'ros__parameters': {'wheelbase_m': .3, 'drive_enabled': True}}}, 'drive_enabled는 vehicle.yaml에 두지'),
    ({'/**': {'ros__parameters': {'wheelbase_m': .3, 'drive_enabled': False}}}, 'drive_enabled:=true'),
])
def test_vehicle_yaml_rejects_typos_and_other_layouts(tmp_path, data, match):
    path = tmp_path/'vehicle.yaml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=match):
        read_vehicle_yaml(path)


def test_vehicle_yaml_allows_use_sim_time(tmp_path):
    path = tmp_path/'vehicle.yaml'
    path.write_text(yaml.safe_dump({'/**': {'ros__parameters': {'wheelbase_m': .33, 'use_sim_time': True}}}))
    assert read_vehicle_yaml(path) == {'wheelbase_m': .33, 'use_sim_time': True}
