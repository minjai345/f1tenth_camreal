"""Control node: /waypoint -> pure pursuit -> /drive on a steady timer; (0, 0) unless the waypoint is fresh."""
import hashlib
from pathlib import Path
import threading
import time
import numpy as np
import yaml
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped
from camreal.__main__ import load_course
from .params import PURE_PURSUIT_NODE as PARAMETERS, declare, positive
from .runtime import WaypointFollower, calibration_problem
from .messages import CALIBRATION_TOPIC, LATCHED, read_waypoint
from .spin import run


class PurePursuitNode(Node):
    def __init__(self, **kwargs):
        super().__init__('pure_pursuit_node', **kwargs)
        p = declare(self, PARAMETERS)   # startup-only: no unvalidated live geometry/enable changes
        if not (np.isfinite(p['wheelbase_m']) and p['wheelbase_m'] > 0):
            raise ValueError(f'wheelbase_m={p["wheelbase_m"]}: 축간거리를 실측해 vehicle.yaml(data/config/vehicle.yaml)의 '
                             'wheelbase_m에 m 단위로 적으세요(예: 0.33).')
        positive(p, 'control_hz', 'waypoint_timeout_s', 'steer_max_rad', 'max_waypoint_m')
        for key in ('camreal_config', 'waypoint_topic', 'drive_topic', 'path_frame'):
            if not p[key]:
                raise ValueError(f'required parameter is empty: {key}')
        self.follower = WaypointFollower(p['waypoint_timeout_s'], p['future_tolerance_s'], p['target_speed_mps'],
                                         p['wheelbase_m'], p['steer_max_rad'], p['max_waypoint_m'], p['path_frame'])
        self.frame_id, self.enabled, self.topic = p['path_frame'], p['drive_enabled'], p['waypoint_topic']
        inputs = MutuallyExclusiveCallbackGroup()
        self.calibration_block, status = None, 'not checked'
        if self.enabled:
            # Same rule as CameraPreprocessor.require_driving_calibration, on the students' calibration file ...
            path = load_course(p['camreal_config'])['calibration']
            data = Path(path).read_bytes()
            c = yaml.safe_load(data)
            if not isinstance(c, dict):
                raise ValueError(f'calibration is not a YAML mapping: {path}')
            status = c.get('calibration_status', 'unspecified')
            if status == 'assumed':
                raise ValueError(f'ASSUMED calibration is preview-only ({path}); '
                                 'replace with measured calibration before drive_enabled:=true')
            # ... and it has to be the file waypoint_node loaded; (0, 0) until waypoint_node says so.
            self.sha256 = hashlib.sha256(data).hexdigest()
            self.calibration_block = 'waiting for waypoint_node calibration'
            self.create_subscription(String, CALIBRATION_TOPIC, self.on_calibration, LATCHED, callback_group=inputs)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, p['drive_topic'], 1) if self.enabled else None
        self.drive_lock, self.stopped, self.last_reason = threading.Lock(), False, None
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(PointStamped, self.topic, self.on_waypoint, qos,
                                                      callback_group=inputs)
        # Steady timer continues to stop on stalled /clock, using monotonic receipt ages.
        self.timer = self.create_timer(1.0 / p['control_hz'], self.control,
                                       callback_group=MutuallyExclusiveCallbackGroup(),
                                       clock=Clock(clock_type=ClockType.STEADY_TIME))
        output = p['drive_topic'] if self.enabled else 'no drive output'
        self.get_logger().info(
            f'Ready; drive_enabled={self.enabled}, {self.topic} -> {output}, {p["target_speed_mps"]} m/s, '
            f'wheelbase {p["wheelbase_m"]} m, steer_max {p["steer_max_rad"]} rad, calibration={status}')

    def ros_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_calibration(self, msg):
        problem = calibration_problem(msg.data, self.sha256)
        if problem and problem != self.calibration_block:
            self.get_logger().error(f'{problem}: 두 노드를 같은 camreal_config로 함께 다시 시작하세요'
                                    '(calibrate 뒤에는 launch를 다시). 그동안 속도 0.')
        self.calibration_block = problem

    def on_waypoint(self, msg):
        stamp, frame_id, wp = read_waypoint(msg)
        with self.follower.lock:   # clocks read in the order the follower sees them (see control)
            self.follower.update(wp, stamp, frame_id, time.monotonic(), self.ros_seconds())

    def control(self):
        with self.drive_lock:
            if self.stopped:
                return
            # Interleaved publishers drop the waypoint at every older stamp: surge and stop several times a second.
            publishers = self.count_publishers(self.topic)
            shared = f'{publishers} publishers on {self.topic}' if publishers > 1 else None
            with self.follower.lock:
                speed, steering = self.follower.command(time.monotonic(), self.ros_seconds(),
                                                        self.calibration_block or shared)
                reason = self.follower.reason
            if self.drive_pub is not None:
                self.publish_drive(speed, steering)
        if publishers > 1:
            names = ', '.join(sorted(info.node_name for info in self.get_publishers_info_by_topic(self.topic)))
            self.get_logger().warning(f'{self.topic} publisher가 {publishers}개입니다({names}). 하나만 남기세요. '
                                      '그동안 속도 0.', throttle_duration_sec=2.0)
        if reason != self.last_reason and self.get_logger().info(f'Control: {reason}', throttle_duration_sec=1.0):
            self.last_reason = reason   # a change hidden by the throttle is logged once it expires

    def publish_drive(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), self.frame_id
        msg.drive.speed, msg.drive.steering_angle = float(speed), float(steering)
        self.drive_pub.publish(msg)

    def destroy_node(self):
        self.timer.cancel()
        with self.drive_lock:   # a control callback still in flight cannot publish after the final stop
            self.stopped = True
            if self.drive_pub is not None and self.context.ok():
                self.publish_drive(0.0, 0.0)
        return super().destroy_node()


def main(args=None):
    run(PurePursuitNode, lambda: MultiThreadedExecutor(num_threads=2),
        'pure_pursuit_node startup failed; drive output inactive', args)
