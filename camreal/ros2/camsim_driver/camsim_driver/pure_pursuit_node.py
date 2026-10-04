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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from camreal.__main__ import load_course
from .runtime import WaypointFollower
from .messages import read_waypoint
from .spin import run


PARAMETERS = dict(camreal_config='data/camreal.yaml', waypoint_topic='/waypoint', drive_topic='/drive',
                  control_hz=25.0, waypoint_timeout_s=0.25, future_tolerance_s=0.02, target_speed_mps=0.5,
                  wheelbase_m=0.0, steer_max_rad=0.30, max_waypoint_m=3.0, path_frame='rear_axle',
                  drive_enabled=False)


class PurePursuitNode(Node):
    def __init__(self, **kwargs):
        super().__init__('pure_pursuit_node', **kwargs)
        # Startup-only parameters prevent unvalidated live geometry/enable changes.
        p = {key: self.declare_parameter(key, value, ParameterDescriptor(read_only=True)).value
             for key, value in PARAMETERS.items()}
        for key in ('camreal_config', 'waypoint_topic', 'drive_topic', 'path_frame'):
            if not p[key]:
                raise ValueError(f'required parameter is empty: {key}')
        if not np.isfinite(p['control_hz']) or p['control_hz'] <= 0:
            raise ValueError('control_hz must be positive')
        self.follower = WaypointFollower(p['waypoint_timeout_s'], p['future_tolerance_s'], p['target_speed_mps'],
                                         p['wheelbase_m'], p['steer_max_rad'], p['max_waypoint_m'], p['path_frame'])
        self.frame_id, self.enabled = p['path_frame'], p['drive_enabled']
        calibration = 'not checked'
        if self.enabled:
            # Same rule as CameraPreprocessor.require_driving_calibration, on the students' calibration file.
            path = load_course(p['camreal_config'])['calibration']
            c = yaml.safe_load(Path(path).read_text())
            if not isinstance(c, dict):
                raise ValueError(f'calibration is not a YAML mapping: {path}')
            calibration = c.get('calibration_status', 'unspecified')
            if calibration == 'assumed':
                raise ValueError(f'ASSUMED calibration is preview-only ({path}); '
                                 'replace with measured calibration before drive_enabled:=true')
        self.drive_pub = self.create_publisher(AckermannDriveStamped, p['drive_topic'], 1) if self.enabled else None
        self.drive_lock, self.stopped, self.last_reason = threading.Lock(), False, None
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(PointStamped, p['waypoint_topic'], self.on_waypoint, qos,
                                                      callback_group=MutuallyExclusiveCallbackGroup())
        # Steady timer continues to stop on stalled /clock, using monotonic receipt ages.
        self.timer = self.create_timer(1.0 / p['control_hz'], self.control,
                                       callback_group=MutuallyExclusiveCallbackGroup(),
                                       clock=Clock(clock_type=ClockType.STEADY_TIME))
        output = p['drive_topic'] if self.enabled else 'no drive output'
        self.get_logger().info(
            f'Ready; drive_enabled={self.enabled}, {p["waypoint_topic"]} -> {output}, {p["target_speed_mps"]} m/s, '
            f'wheelbase {p["wheelbase_m"]} m, steer_max {p["steer_max_rad"]} rad, calibration={calibration}')

    def ros_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_waypoint(self, msg):
        stamp, frame_id, wp = read_waypoint(msg)
        self.follower.update(wp, stamp, frame_id, time.monotonic(), self.ros_seconds())

    def control(self):
        with self.drive_lock:
            if self.stopped:
                return
            speed, steering = self.follower.command(time.monotonic(), self.ros_seconds())
            if self.drive_pub is not None:
                self.publish_drive(speed, steering)
        if self.follower.reason != self.last_reason:
            self.last_reason = self.follower.reason
            self.get_logger().info(f'Control: {self.last_reason}')

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
