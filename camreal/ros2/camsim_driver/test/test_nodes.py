"""In-process ROS check: image -> waypoint_node -> /waypoint -> pure_pursuit_node -> /drive.

Test-only topic names and a random DDS domain keep it away from a live ackermann_mux.
"""
import importlib.util
from pathlib import Path
import random
import threading
import time
import numpy as np
import pytest
import yaml
pytest.importorskip('rclpy')
pytest.importorskip('sensor_msgs.msg')
pytest.importorskip('ackermann_msgs.msg')
pytest.importorskip('cv_bridge')
import rclpy
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from camsim import camera, config
from camsim.pure_pursuit import pure_pursuit
from camreal.tests.conftest import make_model_dir
from camsim_driver import pure_pursuit_node, waypoint_node
from camsim_driver.pure_pursuit_node import PurePursuitNode
from camsim_driver.waypoint_node import WaypointNode

WP, NORM = (1., .2), 2.   # the test model predicts WP for every image
IMAGE, WAYPOINT, DRIVE = '/test/image', '/test/waypoint', '/test/drive'


@pytest.fixture
def context():
    context = Context()
    rclpy.init(context=context, domain_id=random.randint(100, 200))
    yield context
    rclpy.try_shutdown(context=context)


def params(**values):
    return [Parameter(key, value=value) for key, value in values.items()]


def constant_head(net):
    import torch
    with torch.no_grad():
        net.head[-1].weight.zero_()
        net.head[-1].bias.copy_(torch.tensor(WP) / NORM)


def write_course(tmp_path, model, cfg, status='measured'):
    K = camera.intrinsics(cfg)
    calibration = tmp_path/'camera.yaml'
    calibration.write_text(yaml.safe_dump(dict(
        schema_version=1, calibration_status=status, distortion_model='plumb_bob', ground_frame='rear_axle',
        homography_space='undistorted_full_resolution', image_width=640, image_height=400,
        K=K.tolist(), new_K=K.tolist(), D=[.01, -.005, 0., 0., 0.], H_i2g=camera.build(cfg)[1].tolist())))
    course = tmp_path/'camreal.yaml'
    course.write_text(yaml.safe_dump(dict(
        model=str(model), calibration=str(calibration), image_topic=IMAGE,
        paths=dict(bags=str(tmp_path/'bags'), projects=str(tmp_path/'labeling'), datasets=str(tmp_path/'datasets')),
        sessions=dict(train=['run_train'], val=['run_val']), sampling=dict(interval_s=.5, max_frames=10))))
    return course


class Harness(Node):
    """Fake camera (~20 Hz, current stamps) that records /waypoint and /drive with receipt times."""

    def __init__(self, context):
        super().__init__('harness', context=context)
        self.lock = threading.Lock()
        self.streaming, self.stamps, self.waypoints, self.drives = True, set(), [], []
        image = np.random.default_rng(0).integers(0, 256, (400, 640, 3), np.uint8)
        self.image = CvBridge().cv2_to_imgmsg(image, encoding='bgr8')
        self.image_pub = self.create_publisher(Image, IMAGE, 1)
        self.create_subscription(PointStamped, WAYPOINT, lambda m: self.record(self.waypoints, m), 50)
        self.create_subscription(AckermannDriveStamped, DRIVE, lambda m: self.record(self.drives, m), 200)
        self.create_timer(.05, self.publish_image)

    def record(self, items, msg):
        with self.lock:
            items.append((time.monotonic(), msg))

    def publish_image(self):
        with self.lock:
            if self.streaming:
                self.image.header.stamp = self.get_clock().now().to_msg()
                self.stamps.add((self.image.header.stamp.sec, self.image.header.stamp.nanosec))
                self.image_pub.publish(self.image)

    def stream(self, on):
        with self.lock:
            self.streaming = on
        return time.monotonic()

    def since(self, items, start):
        with self.lock:
            return [(t, m) for t, m in items if t >= start]


def wait_for(condition, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(.02)
    return condition()


def spin(executor):
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return thread


def test_image_to_waypoint_to_drive_and_stop(tmp_path, context):
    model, cfg, _ = make_model_dir(tmp_path, constant_head, waypoints__norm_m=NORM)
    course = write_course(tmp_path, model, cfg)
    harness = Harness(context)
    perception = WaypointNode(context=context, parameter_overrides=params(
        camreal_config=str(course), device='cpu', waypoint_topic=WAYPOINT,
        path_topic='/test/predicted_path', debug_image_topic='/test/bev'))
    control = PurePursuitNode(context=context, parameter_overrides=params(
        camreal_config=str(course), drive_enabled=True, wheelbase_m=.3, waypoint_topic=WAYPOINT, drive_topic=DRIVE))
    driver, recorder = MultiThreadedExecutor(num_threads=2, context=context), SingleThreadedExecutor(context=context)
    driver.add_node(perception)
    driver.add_node(control)
    recorder.add_node(harness)
    threads = [spin(driver), spin(recorder)]
    alive = [perception, control, harness]
    moving = lambda start: [m for _, m in harness.since(harness.drives, start) if m.drive.speed > 0]
    try:
        start = time.monotonic()
        assert wait_for(lambda: time.monotonic() - start > 1.5 and len(moving(start)) >= 10, 20.)
        waypoints = harness.since(harness.waypoints, start)
        assert waypoints
        for _, msg in waypoints:
            assert msg.header.frame_id == 'rear_axle'
            assert (msg.header.stamp.sec, msg.header.stamp.nanosec) in harness.stamps
            np.testing.assert_allclose([msg.point.x, msg.point.y, msg.point.z], [*WP, 0.], atol=1e-6)
        for msg in moving(start):
            assert msg.header.frame_id == 'rear_axle' and msg.drive.speed == .5
            assert msg.drive.steering_angle == pytest.approx(pure_pursuit(WP, .3, .3), abs=1e-6)

        stopped = harness.stream(False)
        zero = lambda: [t for t, m in harness.since(harness.drives, stopped) if m.drive.speed == 0]
        assert wait_for(lambda: zero(), 2.)
        first_zero = min(zero())
        assert first_zero - stopped < .5
        assert wait_for(lambda: len(harness.since(harness.drives, first_zero)) >= 5, 2.)
        assert all(m.drive.speed == 0 and m.drive.steering_angle == 0
                   for _, m in harness.since(harness.drives, first_zero))

        resumed = harness.stream(True)
        assert wait_for(lambda: moving(resumed), 5.)

        driver.shutdown()
        threads[0].join(5.)
        perception.destroy_node()
        destroyed = time.monotonic()
        control.destroy_node()   # sends one last zero command
        alive.remove(perception)
        alive.remove(control)
        drives = lambda: harness.since(harness.drives, 0)
        assert wait_for(lambda: drives()[-1][0] >= destroyed and drives()[-1][1].drive.speed == 0, 2.)
        assert [m for t, m in drives() if t < destroyed][-1].drive.speed > 0
        count = len(drives())
        assert not wait_for(lambda: len(drives()) > count, .3)
    finally:
        driver.shutdown()
        recorder.shutdown()
        for thread in threads:
            thread.join(5.)
        for node in alive:
            node.destroy_node()


def test_drive_refuses_assumed_calibration(tmp_path, context):
    course = write_course(tmp_path, tmp_path/'model', config.load(), status='assumed')
    with pytest.raises(ValueError, match='ASSUMED'):
        PurePursuitNode(context=context, parameter_overrides=params(
            camreal_config=str(course), drive_enabled=True, wheelbase_m=.3))


@pytest.mark.parametrize('values', [dict(), dict(wheelbase_m=.3, control_hz=0.),
                                    dict(wheelbase_m=.3, waypoint_timeout_s=-1.), dict(wheelbase_m=.3, path_frame='')])
def test_control_startup_checks(context, values):
    with pytest.raises(ValueError):
        PurePursuitNode(context=context, parameter_overrides=params(**values))


def test_drive_disabled_reads_no_course_and_has_no_drive_publisher(context):
    node = PurePursuitNode(context=context, parameter_overrides=params(
        camreal_config='/nonexistent/camreal.yaml', wheelbase_m=.3))
    try:
        topics = [name for name, _ in node.get_publisher_names_and_types_by_node('pure_pursuit_node', '/')]
        assert '/drive' not in topics
    finally:
        node.destroy_node()


def test_vehicle_yaml_holds_both_nodes_parameters():
    data = yaml.safe_load((Path(__file__).resolve().parents[1]/'config'/'vehicle.yaml').read_text())
    shipped = data['/**']['ros__parameters']
    declared = {**waypoint_node.PARAMETERS, **pure_pursuit_node.PARAMETERS}
    assert set(data) == {'/**'} and set(shipped) == set(declared)
    # rclpy refuses e.g. 25 for a parameter declared as 25.0.
    assert {key: type(value) for key, value in shipped.items()} == {key: type(v) for key, v in declared.items()}


def test_launch_rejects_old_vehicle_yaml(tmp_path):
    pytest.importorskip('launch_ros')
    from launch import LaunchContext
    path = Path(__file__).resolve().parents[1]/'launch'/'camsim_driver.launch.py'
    spec = importlib.util.spec_from_file_location('camsim_driver_launch', path)
    launch_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch_file)
    vehicle = tmp_path/'vehicle.yaml'
    launch_context = LaunchContext()
    launch_context.launch_configurations.update(params_file=str(vehicle), drive_enabled='true')
    vehicle.write_text(yaml.safe_dump({'/**': {'ros__parameters': {'wheelbase_m': .3}}}))
    nodes = launch_file.nodes(launch_context)
    assert [n.node_executable for n in nodes] == ['waypoint_node', 'pure_pursuit_node']
    vehicle.write_text(yaml.safe_dump({'camsim_driver_node': {'ros__parameters': {'wheelbase_m': .3}}}))
    with pytest.raises(RuntimeError, match='이전 형식'):
        launch_file.nodes(launch_context)
