"""Perception node: short image callback, dedicated inference worker, /waypoint for pure_pursuit_node."""
import threading
import time
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
from camsim.handoff import sha256_file
from camreal.__main__ import load_course
from camreal.checkpoint import load_model
from camreal.overlay import bev_view, title_bar
from .params import WAYPOINT_NODE as PARAMETERS, declare, positive
from .preprocessing import CameraPreprocessor
from .runtime import FrameMailbox, image_problem
from .messages import CALIBRATION_TOPIC, LATCHED, decode_bgr8, make_path, make_waypoint, seconds
from .spin import run


class WaypointNode(Node):
    def __init__(self, **kwargs):
        super().__init__('waypoint_node', **kwargs)
        p = declare(self, PARAMETERS)   # startup-only: no unvalidated live changes
        for key in ('camreal_config', 'waypoint_topic', 'path_topic', 'path_frame'):
            if not p[key]:
                raise ValueError(f'required parameter is empty: {key}')
        if p['image_qos_reliability'] not in ('best_effort', 'reliable'):
            raise ValueError('image_qos_reliability must be best_effort or reliable')
        positive(p, 'cpu_threads', 'input_timeout_s', 'max_waypoint_m')
        self.mailbox = FrameMailbox(p['input_timeout_s'], p['future_tolerance_s'], p['max_waypoint_m'])
        # Model, calibration and image topic come from the same file the students use.
        course = load_course(p['camreal_config'])
        self.predictor, self.cfg, mask = load_model(course['model'], p['device'], p['cpu_threads'])
        self.preprocessor = CameraPreprocessor(course['calibration'], self.cfg, mask, p['path_frame'])
        calibration = f'{self.preprocessor.calibration_status} {sha256_file(course["calibration"])}'
        # Exercise the exact shape/device before publishing anything.
        probe = self.predictor.predict(np.zeros((*mask.shape, 3), dtype=np.uint8))
        if probe.shape != (2,) or not np.isfinite(probe).all():
            raise ValueError('model startup probe failed')
        self.frame_id = p['path_frame']
        self.bridge = CvBridge()
        self.waypoint_pub = self.create_publisher(PointStamped, p['waypoint_topic'], 1)
        self.path_pub = self.create_publisher(Path, p['path_topic'], 1)
        self.debug_pub = (self.create_publisher(Image, p['debug_image_topic'], 1)
                          if p['debug_image_topic'] and p['debug_image_hz'] > 0 else None)
        self.debug_period, self.last_debug = 1.0 / max(p['debug_image_hz'], 1e-6), 0.0
        # pure_pursuit_node drives only on the calibration file it checked itself.
        self.calibration_pub = self.create_publisher(String, CALIBRATION_TOPIC, LATCHED)
        self.calibration_pub.publish(String(data=calibration))
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=(ReliabilityPolicy.BEST_EFFORT if p['image_qos_reliability'] == 'best_effort'
                                      else ReliabilityPolicy.RELIABLE), durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(Image, course['image_topic'], self.on_image, qos)
        self.stop_event, self.last_reason = threading.Event(), None
        self.worker = threading.Thread(target=self.infer, name='camsim_inference', daemon=True)
        self.worker.start()
        self.get_logger().info(
            f'Ready; model={course["model"]} ({self.predictor.provider}), '
            f'waypoint {self.cfg.waypoints.ahead_m} m ahead in {self.frame_id} -> {p["waypoint_topic"]}, '
            f'calibration={self.preprocessor.calibration_status}, image={course["image_topic"]}')

    def ros_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_image(self, msg):
        stamp, ros_now = seconds(msg.header.stamp), self.ros_seconds()
        if not self.mailbox.offer(msg, stamp, time.monotonic(), ros_now):
            self.get_logger().warning(f'Image dropped: {self.mailbox.reason} (age {ros_now - stamp:.3f} s)',
                                      throttle_duration_sec=2.0)

    def infer(self):
        while not self.stop_event.is_set():
            frame = self.mailbox.take()
            if frame is None:
                return
            try:
                bev = self.preprocessor.bev(decode_bgr8(self.bridge, frame.message))
                # A covered lens still yields a waypoint, so the model never sees a dark or featureless view.
                problem = image_problem(bev, self.preprocessor.mask)
                wp = None if problem else self.predictor.predict(bev)
                if self.stop_event.is_set():
                    return
                if problem:
                    self.mailbox.fail(frame, problem)
                elif self.mailbox.complete(frame, wp, time.monotonic(), self.ros_seconds()):
                    stamp = frame.message.header.stamp
                    self.waypoint_pub.publish(make_waypoint(stamp, self.frame_id, wp))
                    self.path_pub.publish(make_path(stamp, self.frame_id, wp))
                self.publish_debug(frame, bev, wp)
            except Exception as exc:
                self.mailbox.fail(frame, f'inference failure: {exc}')
                self.get_logger().error(f'Inference failed, no /waypoint: {exc}', throttle_duration_sec=2.0)
            reason, log = self.mailbox.reason, self.get_logger()
            if reason != self.last_reason and log.info(f'Waypoint: {reason}', throttle_duration_sec=1.0):
                self.last_reason = reason   # a change hidden by the throttle is logged once it expires

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
        reason = self.mailbox.reason
        title_bar(view, reason if wp is None else f'x {wp[0]:.2f} y {wp[1]:+.2f} m | {reason}')
        msg = self.bridge.cv2_to_imgmsg(view, encoding='bgr8')
        msg.header.stamp, msg.header.frame_id = frame.message.header.stamp, self.frame_id
        self.debug_pub.publish(msg)

    def destroy_node(self):
        self.stop_event.set()
        self.mailbox.close()
        self.worker.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    run(WaypointNode, SingleThreadedExecutor, 'waypoint_node startup failed; no /waypoint', args)
