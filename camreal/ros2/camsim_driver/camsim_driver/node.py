"""One ROS node; short image callback, dedicated inference worker, independent timer."""
import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from camreal.__main__ import load_course
from camreal.checkpoint import load_model
from camreal.overlay import bev_view, title_bar
from .preprocessing import CameraPreprocessor
from .runtime import DriverState
from .messages import decode_bgr8, make_path


class CamsimDriverNode(Node):
    def __init__(self):
        super().__init__('camsim_driver_node')
        defaults = dict(camreal_config='data/camreal.yaml', path_topic='/predicted_path', drive_topic='/drive',
                        debug_image_topic='/camsim_driver/bev', debug_image_hz=5.0,
                        device='cpu', cpu_threads=1, control_hz=25.0, input_timeout_s=0.25,
                        path_timeout_s=0.25, future_tolerance_s=0.02, target_speed_mps=0.5,
                        wheelbase_m=0.0, steer_max_rad=0.30, max_waypoint_m=3.0,
                        path_frame='rear_axle', drive_enabled=False, image_qos_reliability='best_effort')
        # Startup-only parameters prevent unvalidated live geometry/enable changes.
        p = {key: self.declare_parameter(key, value, ParameterDescriptor(read_only=True)).value
             for key, value in defaults.items()}
        for key in ('camreal_config', 'path_topic', 'drive_topic', 'path_frame'):
            if not p[key]:
                raise ValueError(f'required parameter is empty: {key}')
        if p['image_qos_reliability'] not in ('best_effort', 'reliable'):
            raise ValueError('image_qos_reliability must be best_effort or reliable')
        if not np.isfinite(p['control_hz']) or p['control_hz'] <= 0 or p['cpu_threads'] < 1:
            raise ValueError('control_hz / cpu_threads must be positive')
        # Model, calibration and image topic come from the same file the students use.
        course = load_course(p['camreal_config'])
        self.predictor, self.cfg, mask = load_model(course['model'], p['device'], p['cpu_threads'])
        self.preprocessor = CameraPreprocessor(course['calibration'], self.cfg, mask, p['path_frame'])
        if p['drive_enabled']:
            self.preprocessor.require_driving_calibration()
        self.state = DriverState(p['input_timeout_s'], p['path_timeout_s'], p['future_tolerance_s'],
                                 p['target_speed_mps'], p['wheelbase_m'], p['steer_max_rad'], p['max_waypoint_m'])
        # Exercise the exact shape/device before creating any drive publisher.
        probe = self.predictor.predict(np.zeros((*mask.shape, 3), dtype=np.uint8))
        if probe.shape != (2,) or not np.isfinite(probe).all():
            raise ValueError('model startup probe failed')
        self.frame_id, self.enabled = p['path_frame'], p['drive_enabled']
        self.bridge = CvBridge()
        self.path_pub = self.create_publisher(Path, p['path_topic'], 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, p['drive_topic'], 1) if self.enabled else None
        self.debug_pub = (self.create_publisher(Image, p['debug_image_topic'], 1)
                          if p['debug_image_topic'] and p['debug_image_hz'] > 0 else None)
        self.debug_period, self.last_debug = 1.0 / max(p['debug_image_hz'], 1e-6), 0.0
        self.image_group = MutuallyExclusiveCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=(ReliabilityPolicy.BEST_EFFORT if p['image_qos_reliability'] == 'best_effort'
                                      else ReliabilityPolicy.RELIABLE), durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(Image, course['image_topic'], self.on_image, qos,
                                                      callback_group=self.image_group)
        # Steady timer continues to stop on stalled /clock, using monotonic receipt ages.
        self.timer = self.create_timer(1.0 / p['control_hz'], self.control,
                                      callback_group=self.control_group,
                                      clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self.infer, name='camsim_inference', daemon=True)
        self.worker.start()
        self.last_reason = None
        self.get_logger().info(
            f'Ready; drive_enabled={self.enabled}, model={course["model"]} ({self.predictor.provider}), '
            f'waypoint {self.cfg.waypoints.ahead_m} m ahead in {self.frame_id}, '
            f'calibration={self.preprocessor.calibration_status}, image={course["image_topic"]}')

    def ros_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_image(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.state.offer(msg, stamp, time.monotonic(), self.ros_seconds())

    def infer(self):
        while not self.stop_event.is_set():
            frame = self.state.take()
            if frame is None:
                return
            try:
                bev = self.preprocessor.bev(decode_bgr8(self.bridge, frame.message))
                wp = self.predictor.predict(bev)
                if self.stop_event.is_set():
                    return
                if self.state.complete(frame, wp, time.monotonic(), self.ros_seconds()):
                    self.path_pub.publish(make_path(frame.message.header.stamp, self.frame_id, wp))
                self.publish_debug(frame, bev, wp)
            except Exception as exc:
                self.state.fail(frame, f'inference failure: {exc}')
                self.get_logger().error(f'Inference stopped driving: {exc}', throttle_duration_sec=2.0)

    def publish_debug(self, frame, bev, wp):
        """What the model sees (long side at most 400 px) with its waypoint: watch it change with lighting."""
        now = time.monotonic()
        if self.debug_pub is None or now - self.last_debug < self.debug_period:
            return
        self.last_debug = now
        view = bev_view(bev, self.cfg, wp)
        scale = 400 / max(view.shape[:2])
        if scale < 1:   # 380 x 300 BEV stays full size; finer BEVs are shrunk
            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        title_bar(view, f'x {wp[0]:.2f} y {wp[1]:+.2f} m | {self.state.reason} | drive {"ON" if self.enabled else "off"}')
        msg = self.bridge.cv2_to_imgmsg(view, encoding='bgr8')
        msg.header.stamp, msg.header.frame_id = frame.message.header.stamp, self.frame_id
        self.debug_pub.publish(msg)

    def control(self):
        speed, steering = self.state.command(time.monotonic(), self.ros_seconds())
        if self.state.reason != self.last_reason:
            self.get_logger().info(f'Control: {self.state.reason}')
            self.last_reason = self.state.reason
        if self.drive_pub is not None:
            self.publish_drive(speed, steering)

    def publish_drive(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), self.frame_id
        msg.drive.speed, msg.drive.steering_angle = float(speed), float(steering)
        self.drive_pub.publish(msg)

    def destroy_node(self):
        self.timer.cancel()
        self.stop_event.set()
        self.state.close()
        if self.drive_pub is not None and rclpy.ok():
            self.publish_drive(0.0, 0.0)
        self.worker.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        node = CamsimDriverNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is None:
            print(f'camsim_driver_node startup failed; drive output inactive: {exc}', flush=True)
        raise
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
