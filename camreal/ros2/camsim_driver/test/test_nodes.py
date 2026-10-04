"""In-process ROS check: image -> waypoint_node -> /waypoint -> pure_pursuit_node -> /drive; the real entry point
in a subprocess for startup errors and stop signals.

Test-only topic names and a random DDS domain keep it away from a live ackermann_mux.
"""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
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
from std_msgs.msg import String
from camsim import camera, config
from camsim.handoff import sha256_file
from camsim.pure_pursuit import pure_pursuit
from camreal.tests.conftest import make_model_dir, tape_lane
from camsim_driver.messages import CALIBRATION_TOPIC, LATCHED, make_waypoint
from camsim_driver.params import PURE_PURSUIT_NODE, WAYPOINT_NODE
from camsim_driver import waypoint_node
from camsim_driver.pure_pursuit_node import PurePursuitNode
from camsim_driver.waypoint_node import WaypointNode

WP, NORM = (1., .2), 2.   # the test model predicts WP for every image
IMAGE, WAYPOINT, DRIVE = '/test/image', '/test/waypoint', '/test/drive'
RNG = np.random.default_rng(0)
TRACK = np.clip(tape_lane(config.load()) + RNG.normal(0, 2, (400, 640, 3)), 0, 255).astype(np.uint8)
YY, XX = np.mgrid[:400, :640] - np.array([200, 320])[:, None, None]
SHADING = .7 + .3 / (1 + (XX ** 2 + YY ** 2) / 320 ** 2)[..., None] ** 2   # 30 % cos^4 lens shading
COVERED = {   # auto exposure brightens a covered lens: noise and smooth shading, no edges
    'black frame': np.zeros_like(TRACK),
    'covered, auto gain': np.clip(RNG.normal(20, 10, TRACK.shape), 0, 255).astype(np.uint8),
    'palm, auto exposure': np.clip(np.array([70, 95, 140]) * SHADING + RNG.normal(0, 2, TRACK.shape), 0, 255
                                   ).astype(np.uint8)}
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def context():
    context = Context()
    rclpy.init(context=context, domain_id=random.randint(30, 101))   # Linux-safe ports, never the car's 0
    yield context
    rclpy.try_shutdown(context=context)


def params(**values):
    return [Parameter(key, value=value) for key, value in values.items()]


def constant_head(net, wp=WP):
    import torch
    with torch.no_grad():
        net.head[-1].weight.zero_()
        net.head[-1].bias.copy_(torch.tensor(wp) / NORM)


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


def stamp_seconds(stamp):
    return stamp[0] + stamp[1] * 1e-9


class Harness(Node):
    """Fake camera (image) or fake waypoint_node (waypoint), ~20 Hz, current stamps; records with receipt times."""

    def __init__(self, context, image=None, waypoint=None):
        super().__init__('harness', context=context)
        self.lock = threading.Lock()
        self.streaming, self.stamps, self.waypoints, self.drives, self.calibrations = True, set(), [], [], []
        self.image = None if image is None else CvBridge().cv2_to_imgmsg(image, encoding='bgr8')
        self.waypoint, self.calibration_pub = waypoint, None
        if image is not None:
            self.image_pub = self.create_publisher(Image, IMAGE, 1)
            self.create_subscription(PointStamped, WAYPOINT, lambda m: self.record(self.waypoints, m), 50)
            self.create_subscription(String, CALIBRATION_TOPIC, lambda m: self.record(self.calibrations, m),
                                     LATCHED)
        if waypoint is not None:
            self.waypoint_pub = self.create_publisher(PointStamped, WAYPOINT, 1)
        self.create_subscription(AckermannDriveStamped, DRIVE, lambda m: self.record(self.drives, m), 200)
        self.create_timer(.05, self.publish)

    def record(self, items, msg):
        with self.lock:
            items.append((time.monotonic(), msg))

    def publish(self):
        with self.lock:
            if not self.streaming:
                return
            stamp = self.get_clock().now().to_msg()
            self.stamps.add((stamp.sec, stamp.nanosec))
            if self.image is not None:
                self.image.header.stamp = stamp
                self.image_pub.publish(self.image)
            if self.waypoint is not None:
                self.waypoint_pub.publish(make_waypoint(stamp, 'rear_axle', self.waypoint))

    def publish_calibration(self, text):
        if self.calibration_pub is None:
            self.calibration_pub = self.create_publisher(String, CALIBRATION_TOPIC, LATCHED)
        self.calibration_pub.publish(String(data=text))

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


@contextmanager
def spinning(context, nodes, harness):
    driver, recorder = MultiThreadedExecutor(num_threads=2, context=context), SingleThreadedExecutor(context=context)
    for node in nodes:
        driver.add_node(node)
    recorder.add_node(harness)
    threads = [spin(driver), spin(recorder)]
    try:
        yield
    finally:
        driver.shutdown()
        recorder.shutdown()
        for thread in threads:
            thread.join(5.)
        for node in (*nodes, harness):
            node.destroy_node()


def declared(node):
    return set(node.get_parameters_by_prefix('')) - {'use_sim_time'}


def test_image_to_waypoint_to_drive_and_stop(tmp_path, context):
    model, cfg, _ = make_model_dir(tmp_path, constant_head, waypoints__norm_m=NORM)
    course = write_course(tmp_path, model, cfg)
    harness = Harness(context, image=TRACK)
    perception = WaypointNode(context=context, parameter_overrides=params(
        camreal_config=str(course), device='cpu', waypoint_topic=WAYPOINT,
        path_topic='/test/predicted_path', debug_image_topic='/test/bev'))
    control = PurePursuitNode(context=context, parameter_overrides=params(
        camreal_config=str(course), drive_enabled=True, wheelbase_m=.3, waypoint_topic=WAYPOINT, drive_topic=DRIVE))
    # vehicle.yaml is checked against params.py; the nodes declare exactly those.
    assert declared(perception) == set(WAYPOINT_NODE) and declared(control) == set(PURE_PURSUIT_NODE)
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
        assert [m.data for _, m in harness.calibrations] == [f'measured {sha256_file(tmp_path/"camera.yaml")}']
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
        zero = lambda: [(t, m) for t, m in harness.since(harness.drives, stopped) if m.drive.speed == 0]
        assert wait_for(lambda: zero(), 2.)
        first_zero, msg = zero()[0]
        # Speed 0 within waypoint_timeout_s + two control periods of the last image's capture time.
        last_image = stamp_seconds(max(harness.stamps))
        assert stamp_seconds((msg.header.stamp.sec, msg.header.stamp.nanosec)) - last_image <= .25 + 2 / 25
        assert wait_for(lambda: len(harness.since(harness.drives, first_zero)) >= 5, 2.)
        assert all(m.drive.speed == 0 and m.drive.steering_angle == 0
                   for _, m in harness.since(harness.drives, first_zero))

        resumed = harness.stream(True)
        assert wait_for(lambda: moving(resumed), 5.)

        # A second /waypoint publisher (another waypoint_node, a bag) holds (0, 0) until it is gone.
        extra = harness.create_publisher(PointStamped, WAYPOINT, 1)
        assert wait_for(lambda: control.follower.reason == f'2 publishers on {WAYPOINT}', 5.)
        held = time.monotonic()
        assert wait_for(lambda: len(harness.since(harness.drives, held)) >= 5, 2.)
        assert not moving(held)
        harness.destroy_publisher(extra)
        single = time.monotonic()
        assert wait_for(lambda: moving(single), 5.)

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


@pytest.mark.parametrize('case', [*COVERED, 'point behind'])
def test_no_waypoint_and_only_zero_drive(tmp_path, context, case):
    wp, image, reason = ((WP, COVERED[case], 'no edges in view') if case in COVERED
                         else ((-1., .2), TRACK, 'invalid waypoint output'))
    model, cfg, _ = make_model_dir(tmp_path, lambda net: constant_head(net, wp), waypoints__norm_m=NORM)
    course = write_course(tmp_path, model, cfg)
    harness = Harness(context, image=image)
    perception = WaypointNode(context=context, parameter_overrides=params(
        camreal_config=str(course), device='cpu', waypoint_topic=WAYPOINT,
        path_topic='/test/predicted_path', debug_image_topic='/test/bev'))
    control = PurePursuitNode(context=context, parameter_overrides=params(
        camreal_config=str(course), drive_enabled=True, wheelbase_m=.3, waypoint_topic=WAYPOINT, drive_topic=DRIVE))
    with spinning(context, [perception, control], harness):
        assert wait_for(lambda: perception.mailbox.reason.startswith(reason), 20.)
        start = time.monotonic()
        assert wait_for(lambda: len(harness.since(harness.drives, start)) >= 25, 5.)
        assert not harness.waypoints
        assert all(m.drive.speed == 0 and m.drive.steering_angle == 0 for _, m in harness.since(harness.drives, 0))


def test_drive_waits_for_the_calibration_waypoint_node_loaded(tmp_path, context):
    course = write_course(tmp_path, tmp_path/'model', config.load())
    sha = sha256_file(tmp_path/'camera.yaml')
    harness = Harness(context, waypoint=WP)
    control = PurePursuitNode(context=context, parameter_overrides=params(
        camreal_config=str(course), drive_enabled=True, wheelbase_m=.3, waypoint_topic=WAYPOINT, drive_topic=DRIVE))
    moving = lambda start: [m for _, m in harness.since(harness.drives, start) if m.drive.speed > 0]
    with spinning(context, [control], harness):
        start = time.monotonic()
        assert wait_for(lambda: len(harness.since(harness.drives, start)) >= 10, 5.)
        assert not moving(start) and control.follower.reason == 'waiting for waypoint_node calibration'
        # e.g. waypoint_node started before `calibrate` rewrote the file, or with another camreal_config
        for text, reason in ((f'measured {"0" * 64}', 'differs'), (f'assumed {sha}', 'ASSUMED')):
            harness.publish_calibration(text)
            assert wait_for(lambda: reason in control.follower.reason, 5.)
            start = time.monotonic()
            assert wait_for(lambda: len(harness.since(harness.drives, start)) >= 5, 2.) and not moving(start)
        matched = time.monotonic()
        harness.publish_calibration(f'measured {sha}')
        assert wait_for(lambda: moving(matched), 5.)


def test_calibration_saved_while_waypoint_node_reads_it_is_refused(tmp_path, context, monkeypatch):
    """The announced SHA-256 must describe the H_i2g in use, or a later pure_pursuit_node drives on a stale one."""
    model, cfg, _ = make_model_dir(tmp_path, constant_head, waypoints__norm_m=NORM)
    course = write_course(tmp_path, model, cfg)
    load = waypoint_node.CameraPreprocessor

    def calibrate_saves_meanwhile(path, *args):
        preprocessor = load(path, *args)
        Path(path).write_text(Path(path).read_text() + '# saved by calibrate\n')
        return preprocessor
    monkeypatch.setattr(waypoint_node, 'CameraPreprocessor', calibrate_saves_meanwhile)
    with pytest.raises(ValueError, match='읽는 동안 바뀌었습니다'):
        WaypointNode(context=context, parameter_overrides=params(camreal_config=str(course), device='cpu'))


def test_final_zero_waits_until_subscribers_have_it(tmp_path, context):
    """Published is not delivered: under load DDS dropped the last sample of a process that exited right after."""
    course = write_course(tmp_path, tmp_path/'model', config.load())
    node = PurePursuitNode(context=context, parameter_overrides=params(
        camreal_config=str(course), drive_enabled=True, wheelbase_m=.3, drive_topic=DRIVE))
    calls, publish = [], node.drive_pub.publish
    node.drive_pub.publish = lambda msg: calls.append(('publish', msg.drive.speed)) or publish(msg)
    node.drive_pub.wait_for_all_acked = lambda timeout: calls.append(('acked', timeout.nanoseconds)) or True
    node.destroy_node()
    assert calls == [('publish', 0.), ('acked', 500_000_000)]


def test_drive_refuses_assumed_calibration(tmp_path, context):
    course = write_course(tmp_path, tmp_path/'model', config.load(), status='assumed')
    with pytest.raises(ValueError, match=r'가정 캘리브레이션\(calibration_status: assumed\)은 미리보기 전용'):
        PurePursuitNode(context=context, parameter_overrides=params(
            camreal_config=str(course), drive_enabled=True, wheelbase_m=.3))


@pytest.mark.parametrize('values,match', [
    (dict(), 'wheelbase_m=0.0.*vehicle.yaml'), (dict(wheelbase_m=1), 'wheelbase_m.*1.0'),
    (dict(wheelbase_m=.3, steer_max_rad=0.), 'steer_max_rad'), (dict(wheelbase_m=.3, control_hz=0.), 'control_hz'),
    (dict(wheelbase_m=.3, waypoint_timeout_s=-1.), 'waypoint_timeout_s=-1.0: 0보다 큰'),
    (dict(wheelbase_m=.3, path_frame=''), 'path_frame 값이 비었습니다'),
    (dict(wheelbase_m=.3, future_tolerance_s=-.1), 'future_tolerance_s=-0.1: 0 이상'),
    (dict(wheelbase_m=.3, target_speed_mps=-.1), 'target_speed_mps=-0.1: 0 이상'),
    (dict(wheelbase_m=.3, target_speed_mps=5.), r'target_speed_mps=5.0: .*2.0 m/s.*vehicle.yaml')])   # 0.5 typo
def test_control_startup_errors_name_the_parameter(context, values, match):
    with pytest.raises(ValueError, match=match):
        PurePursuitNode(context=context, parameter_overrides=params(**values))


@pytest.mark.parametrize('values,match', [
    (dict(waypoint_topic=''), 'waypoint_topic 값이 비었습니다'),
    (dict(image_qos_reliability='fast'), 'image_qos_reliability=fast: best_effort 또는 reliable'),
    (dict(input_timeout_s=0.), 'input_timeout_s=0.0: 0보다 큰'), (dict(future_tolerance_s=-1.), 'future_tolerance_s')])
def test_perception_startup_errors_name_the_parameter(context, values, match):
    with pytest.raises(ValueError, match=match):
        WaypointNode(context=context, parameter_overrides=params(**values))


def test_drive_disabled_reads_no_course_and_has_no_drive_publisher(context):
    node = PurePursuitNode(context=context, parameter_overrides=params(
        camreal_config='/nonexistent/camreal.yaml', wheelbase_m=.3))
    try:
        topics = [name for name, _ in node.get_publisher_names_and_types_by_node('pure_pursuit_node', '/')]
        assert '/drive' not in topics
        topics = [name for name, _ in node.get_subscriber_names_and_types_by_node('pure_pursuit_node', '/')]
        assert CALIBRATION_TOPIC not in topics
    finally:
        node.destroy_node()


def test_launch_checks_vehicle_yaml_and_gives_drive_only_to_the_controller(tmp_path):
    pytest.importorskip('launch_ros')
    from launch import LaunchContext
    from launch.actions import Shutdown
    from launch_ros.utilities import evaluate_parameters
    path = Path(__file__).resolve().parents[1]/'launch'/'camsim_driver.launch.py'
    spec = importlib.util.spec_from_file_location('camsim_driver_launch', path)
    launch_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch_file)
    vehicle = tmp_path/'vehicle.yaml'
    vehicle.write_text(yaml.safe_dump({'/**': {'ros__parameters': {'wheelbase_m': .3}}}))
    launch_context = LaunchContext()
    for enabled in (True, False):
        launch_context.launch_configurations.update(params_file=str(vehicle), drive_enabled=str(enabled).lower())
        perception, control = launch_file.nodes(launch_context)
        assert (perception.node_executable, control.node_executable) == ('waypoint_node', 'pure_pursuit_node')
        assert evaluate_parameters(launch_context, perception._Node__parameters) == (vehicle,)
        assert evaluate_parameters(launch_context, control._Node__parameters) == (vehicle,
                                                                                  {'drive_enabled': enabled})
        # While driving, a controller that exits (wheelbase_m not set, ...) ends the launch.
        assert isinstance(control._ExecuteLocal__on_exit, Shutdown) == enabled
        assert perception._ExecuteLocal__on_exit is None
    for data, match in (({'camsim_driver_node': {'ros__parameters': {'wheelbase_m': .3}}}, '이전 형식'),
                        ({'/**': {'ros__parameters': {'wheelbase_m': .3, 'target_speed': .3}}}, 'target_speed'),
                        ({'/**': {'ros__parameters': {'wheelbase_m': .3, 'drive_enabled': True}}}, 'drive_enabled')):
        vehicle.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match=match):
            launch_file.nodes(launch_context)


CONTROLLER = [sys.executable, '-c', 'from camsim_driver.pure_pursuit_node import main; main()']


def test_startup_setting_error_prints_the_fix_without_a_traceback(context):
    result = subprocess.run(CONTROLLER + ['--ros-args', '-p', 'wheelbase_m:=0.0'], cwd=ROOT, capture_output=True,
                            text=True, timeout=60, env=dict(os.environ, ROS_DOMAIN_ID=str(context.get_domain_id())))
    output = result.stdout + result.stderr
    assert result.returncode == 1 and 'Traceback' not in output
    assert 'pure_pursuit_node를 시작하지 못했습니다(/drive 없음): wheelbase_m=0.0: 축간거리를 실측해' in output


# A parent that dies without signalling its children, like ros2 launch after SIGTERM or SIGKILL.
ORPHANING_PARENT = [sys.executable, '-c', 'import subprocess, sys, time; p = subprocess.Popen(sys.argv[1:], '
                    'stdout=sys.stderr); print(p.pid, flush=True); time.sleep(60)']


@pytest.mark.parametrize('how', ['SIGINT twice', 'SIGINT until it exits', 'SIGHUP', 'parent killed'])
def test_stop_signal_or_dead_parent_sends_the_final_zero(tmp_path, context, how):
    """The real entry point (spin.run), started the way ros2 launch and ros2 run start it."""
    course = write_course(tmp_path, tmp_path/'model', config.load())
    harness = Harness(context, waypoint=WP)
    harness.publish_calibration(f'measured {sha256_file(tmp_path/"camera.yaml")}')
    recorder = SingleThreadedExecutor(context=context)
    recorder.add_node(harness)
    thread = spin(recorder)
    orphaned = how == 'parent killed'
    command = (ORPHANING_PARENT if orphaned else []) + CONTROLLER + [
        '--ros-args', '-p', f'camreal_config:={course}', '-p', 'drive_enabled:=true', '-p', 'wheelbase_m:=0.3',
        '-p', f'waypoint_topic:={WAYPOINT}', '-p', f'drive_topic:={DRIVE}']
    log_path = tmp_path/'node.log'
    log = log_path.open('w')
    process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, ROS_DOMAIN_ID=str(context.get_domain_id())),
                               stdout=subprocess.PIPE if orphaned else log, stderr=log, text=True)
    node_pid = int(process.stdout.readline()) if orphaned else process.pid
    drives = lambda: harness.since(harness.drives, 0)
    try:
        assert wait_for(lambda: any(m.drive.speed > 0 for _, m in drives()), 30.), log_path.read_text()
        signalled = time.monotonic()
        if how == 'SIGINT twice':   # Ctrl+C reaches the node directly and again through ros2 launch
            process.send_signal(signal.SIGINT)
            process.send_signal(signal.SIGINT)
        elif how == 'SIGINT until it exits':   # a late one used to kill it by signal: launch logged a crash
            while process.poll() is None:
                process.send_signal(signal.SIGINT)
                time.sleep(.002)
        else:
            process.send_signal(signal.SIGKILL if orphaned else signal.SIGHUP)
        if not orphaned:
            assert process.wait(10.) == 0, log_path.read_text()
        assert wait_for(lambda: time.monotonic() - drives()[-1][0] > .5, 10.), log_path.read_text()
        received, last = drives()[-1]
        assert received > signalled, log_path.read_text()
        assert (last.drive.speed, last.drive.steering_angle) == (0., 0.), log_path.read_text()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(5.)
        if orphaned:
            try:
                os.kill(node_pid, signal.SIGKILL)   # still alive only if the dead parent's SIGTERM was lost
            except ProcessLookupError:
                pass
        recorder.shutdown()
        thread.join(5.)
        harness.destroy_node()
        log.close()
