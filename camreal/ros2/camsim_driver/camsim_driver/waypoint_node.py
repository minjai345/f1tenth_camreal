"""Perception node: short image callback, dedicated inference worker, /waypoint for pure_pursuit_node."""
import threading
import time
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from camreal.__main__ import load_course
from camreal.checkpoint import load_model
from camreal.overlay import bev_view, title_bar
from .params import WAYPOINT_NODE as PARAMETERS, VEHICLE, declare, nonnegative, positive, required
from .preprocessing import CameraPreprocessor
from .runtime import FrameMailbox
from .messages import decode_bgr8, make_path, make_waypoint, seconds
from .spin import run


class WaypointNode(Node):
    def __init__(self, **kwargs):
        super().__init__('waypoint_node', **kwargs)
        p = declare(self, PARAMETERS)   # startup-only: no unvalidated live changes
        required(p, 'camreal_config', 'waypoint_topic', 'path_topic', 'path_frame')
        for key, allowed in (('image_qos_reliability', ('best_effort', 'reliable')), ('device', ('cpu', 'cuda'))):
            if p[key] not in allowed:
                raise ValueError(f'{key}={p[key]}: {" 또는 ".join(allowed)}로 {VEHICLE}에 적으세요.')
        positive(p, 'cpu_threads', 'input_timeout_s', 'max_waypoint_m')
        nonnegative(p, 'future_tolerance_s')
        self.mailbox = FrameMailbox(p['input_timeout_s'], p['future_tolerance_s'], p['max_waypoint_m'])
        # Model, calibration and image topic come from the same file the students use.
        course = load_course(p['camreal_config'])
        self.predictor, self.cfg, mask = load_model(course['model'], p['device'], p['cpu_threads'])
        self.preprocessor = CameraPreprocessor(course['calibration'], self.cfg, mask, p['path_frame'])
        # Exercise the exact shape/device before publishing anything.
        probe = self.predictor.predict(np.zeros((*mask.shape, 3), dtype=np.uint8))
        if probe.shape != (2,) or not np.isfinite(probe).all():
            raise ValueError(f'모델이 시험 영상에 유한한 (x, y)를 내지 않습니다: {course["model"]}. '
                             'camsim 5장이 저장한 model.onnx와 checkpoint.json인지 확인하세요.')
        self.frame_id = p['path_frame']
        self.bridge = CvBridge()
        self.waypoint_pub = self.create_publisher(PointStamped, p['waypoint_topic'], 1)
        self.path_pub = self.create_publisher(Path, p['path_topic'], 1)
        self.debug_pub = (self.create_publisher(Image, p['debug_image_topic'], 1)
                          if p['debug_image_topic'] and p['debug_image_hz'] > 0 else None)
        self.debug_period, self.last_debug = 1.0 / max(p['debug_image_hz'], 1e-6), time.monotonic()
        self.last_image = self.last_debug
        if self.debug_pub is not None:   # the worker publishes no BEV for a dropped or failed frame
            self.debug_lock, self.stale_after = threading.Lock(), self.debug_period + p['input_timeout_s']
            self.image_timeout = p['input_timeout_s']
            self.view = self.shrink(bev_view(np.full((*mask.shape, 3), self.cfg.lane.color_floor, np.uint8), self.cfg))
            self.create_timer(self.debug_period, self.publish_stale, clock=Clock(clock_type=ClockType.STEADY_TIME))
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=(ReliabilityPolicy.BEST_EFFORT if p['image_qos_reliability'] == 'best_effort'
                                      else ReliabilityPolicy.RELIABLE), durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(Image, course['image_topic'], self.on_image, qos)
        self.stop_event, self.last_reason = threading.Event(), None
        self.worker = threading.Thread(target=self.infer, name='camsim_inference', daemon=True)
        self.worker.start()
        self.get_logger().info(
            f'준비됨: 모델 {course["model"]} ({self.predictor.provider}), '
            f'{self.cfg.waypoints.ahead_m} m 앞 waypoint ({self.frame_id}) -> {p["waypoint_topic"]}, '
            f'캘리브레이션 {self.preprocessor.calibration_status}, 영상 {course["image_topic"]}')

    def ros_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_image(self, msg):
        stamp, ros_now, self.last_image = seconds(msg.header.stamp), self.ros_seconds(), time.monotonic()
        if not self.mailbox.offer(msg, stamp, self.last_image, ros_now):
            reason = self.mailbox.reason
            hint = ' bag이나 시계를 되감았으면 노드를 다시 시작하세요.' if reason.startswith('non-increasing') else ''
            self.get_logger().warning(f'영상 버림: {reason} (영상 나이 {ros_now - stamp:.3f} s).{hint}',
                                      throttle_duration_sec=2.0)

    def infer(self):
        while not self.stop_event.is_set():
            frame = self.mailbox.take()
            if frame is None:
                return
            try:
                bev = self.preprocessor.bev(decode_bgr8(self.bridge, frame.message))
                wp = self.predictor.predict(bev)
                if self.stop_event.is_set():
                    return
                if self.mailbox.complete(frame, wp, time.monotonic(), self.ros_seconds()):
                    stamp = frame.message.header.stamp
                    self.waypoint_pub.publish(make_waypoint(stamp, self.frame_id, wp))
                    self.path_pub.publish(make_path(stamp, self.frame_id, wp))
                self.publish_debug(frame, bev, wp)
            except Exception as exc:   # the message (Korean) goes to the log, the BEV title stays ASCII
                self.mailbox.fail(frame, f'inference failure ({type(exc).__name__})')
                self.get_logger().error(f'추론 실패, /waypoint 없음: {exc}', throttle_duration_sec=2.0)
            reason, log = self.mailbox.reason, self.get_logger()
            if reason != self.last_reason and log.info(f'예측: {reason}', throttle_duration_sec=1.0):
                self.last_reason = reason   # a change hidden by the throttle is logged once it expires

    @staticmethod
    def shrink(view):
        scale = 400 / max(view.shape[:2])   # 380 x 300 BEV stays full size; finer BEVs are shrunk
        return cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else view

    def publish_debug(self, frame, bev, wp):
        """What the model sees (long side at most 400 px) with its waypoint: watch it change with lighting."""
        now = time.monotonic()
        if self.debug_pub is None or now - self.last_debug < self.debug_period:
            return
        view = self.shrink(bev_view(bev, self.cfg, wp))
        with self.debug_lock:
            self.last_debug, self.view = now, view.copy()
            self.send_debug(title_bar(view, f'x {wp[0]:.2f} y {wp[1]:+.2f} m | {self.mailbox.reason}'),
                            frame.message.header.stamp)

    def publish_stale(self):
        """No new BEV (camera stopped, frames dropped, inference failing): the last one in grey, how long and why."""
        with self.debug_lock:
            now = time.monotonic()
            if now - self.last_debug <= self.stale_after:
                return
            why = self.mailbox.reason
            if now - self.last_image > self.image_timeout:   # not stale_after: the last BEV may lag the last image
                why = 'no image'
            elif why == 'valid':   # images arrive, the worker has returned nothing since
                why = 'inference stalled'
            view = cv2.cvtColor(cv2.cvtColor(self.view, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            # The age first: a long reason runs off the 300 px BEV, the seconds students read must not.
            self.send_debug(title_bar(view, f'{now - self.last_debug:.1f} s | {why}'), self.get_clock().now().to_msg())

    def send_debug(self, view, stamp):
        msg = self.bridge.cv2_to_imgmsg(view, encoding='bgr8')
        msg.header.stamp, msg.header.frame_id = stamp, self.frame_id
        self.debug_pub.publish(msg)

    def destroy_node(self):
        self.stop_event.set()
        self.mailbox.close()
        self.worker.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    run(WaypointNode, SingleThreadedExecutor, 'waypoint_node를 시작하지 못했습니다(/waypoint 없음)', args)
