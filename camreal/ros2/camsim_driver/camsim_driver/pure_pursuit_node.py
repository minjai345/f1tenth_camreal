"""Control node: /waypoint -> pure pursuit -> /drive on a steady timer; (0, 0) unless the waypoint is fresh."""
from pathlib import Path
import threading
import time
import numpy as np
import yaml
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from camreal.__main__ import load_course
from .params import PURE_PURSUIT_NODE as PARAMETERS, MAX_SPEED_MPS, VEHICLE, declare, nonnegative, positive, required
from .runtime import WaypointFollower
from .messages import read_waypoint
from .spin import run


class PurePursuitNode(Node):
    def __init__(self, **kwargs):
        super().__init__('pure_pursuit_node', **kwargs)
        p = declare(self, PARAMETERS)   # startup-only: no unvalidated live geometry/enable changes
        if not (np.isfinite(p['wheelbase_m']) and p['wheelbase_m'] > 0):
            raise ValueError(f'wheelbase_m={p["wheelbase_m"]}: 축간거리를 실측해 {VEHICLE}의 '
                             'wheelbase_m에 m 단위로 적으세요(예: 0.33).')
        positive(p, 'control_hz', 'waypoint_timeout_s', 'steer_max_rad', 'max_waypoint_m')
        nonnegative(p, 'future_tolerance_s', 'target_speed_mps')
        if p['target_speed_mps'] > MAX_SPEED_MPS:
            raise ValueError(f'target_speed_mps={p["target_speed_mps"]}: 수업 상한 {MAX_SPEED_MPS} m/s를 넘습니다. '
                             f'{VEHICLE}의 값을 확인하세요(예: 0.5).')
        required(p, 'camreal_config', 'waypoint_topic', 'drive_topic', 'path_frame')
        self.follower = WaypointFollower(p['waypoint_timeout_s'], p['future_tolerance_s'], p['target_speed_mps'],
                                         p['wheelbase_m'], p['steer_max_rad'], p['max_waypoint_m'], p['path_frame'])
        self.frame_id, self.enabled, self.topic = p['path_frame'], p['drive_enabled'], p['waypoint_topic']
        status = '확인 안 함'
        if self.enabled:
            # Same rule as CameraPreprocessor.require_driving_calibration, on the students' calibration file.
            path = load_course(p['camreal_config'])['calibration']
            c = yaml.safe_load(Path(path).read_text())
            if not isinstance(c, dict):
                raise ValueError(f'캘리브레이션 파일 형식이 아닙니다(키: 값 YAML이어야 함): {path}')
            status = c.get('calibration_status', 'unspecified')
            if status == 'assumed':
                raise ValueError(f'가정 캘리브레이션(calibration_status: assumed)은 미리보기 전용입니다: {path}. '
                                 'python3 -m camreal calibrate로 실측한 뒤 drive_enabled:=true로 실행하세요.')
        self.drive_pub = self.create_publisher(AckermannDriveStamped, p['drive_topic'], 1) if self.enabled else None
        self.drive_lock, self.stopped, self.last_reason = threading.Lock(), False, None
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(PointStamped, self.topic, self.on_waypoint, qos,
                                                      callback_group=MutuallyExclusiveCallbackGroup())
        # Steady timer continues to stop on stalled /clock, using monotonic receipt ages.
        self.timer = self.create_timer(1.0 / p['control_hz'], self.control,
                                       callback_group=MutuallyExclusiveCallbackGroup(),
                                       clock=Clock(clock_type=ClockType.STEADY_TIME))
        output = p['drive_topic'] if self.enabled else '출력 없음(주행 끔)'
        self.get_logger().info(
            f'준비됨: drive_enabled={self.enabled}, {self.topic} -> {output}, {p["target_speed_mps"]} m/s, '
            f'축간거리 {p["wheelbase_m"]} m, 조향 한계 {p["steer_max_rad"]} rad, 캘리브레이션 {status}')

    def ros_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

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
                speed, steering = self.follower.command(time.monotonic(), self.ros_seconds(), shared)
                reason = self.follower.reason
            if self.drive_pub is not None:
                self.publish_drive(speed, steering)
        if publishers > 1:
            names = ', '.join(sorted(info.node_name for info in self.get_publishers_info_by_topic(self.topic)))
            self.get_logger().warning(f'{self.topic} publisher가 {publishers}개입니다({names}). 하나만 남기세요. '
                                      '같은 ROS_DOMAIN_ID로 같은 네트워크에 있는 다른 차일 수도 있습니다. 그동안 속도 0.',
                                      throttle_duration_sec=2.0)
        if reason != self.last_reason and self.get_logger().info(f'제어: {reason}', throttle_duration_sec=1.0):
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
                # The process exits next; under load DDS dropped the sample unless the mux acknowledged it.
                self.drive_pub.wait_for_all_acked(Duration(seconds=0.5))
        return super().destroy_node()


def main(args=None):
    run(PurePursuitNode, lambda: MultiThreadedExecutor(num_threads=2),
        'pure_pursuit_node를 시작하지 못했습니다(/drive 없음)', args)
