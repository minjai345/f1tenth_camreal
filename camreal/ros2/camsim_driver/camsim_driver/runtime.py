"""ROS-independent latest-frame mailbox and fail-closed control state."""
from dataclasses import dataclass
from threading import Condition
import numpy as np
from camsim.pure_pursuit import pure_pursuit


@dataclass(frozen=True)
class Frame:
    message: object
    stamp: float
    received: float
    epoch: int


class DriverState:
    """One predicted waypoint (x, y) per frame; pure pursuit steers straight at it."""

    def __init__(self, input_timeout, path_timeout, future_tolerance,
                 speed, wheelbase, steer_max, max_waypoint_m):
        values = (input_timeout, path_timeout, wheelbase, steer_max, max_waypoint_m)
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError('timeouts, geometry and waypoint bound must be positive and finite')
        if not np.isfinite(speed) or speed < 0 or not np.isfinite(future_tolerance) or future_tolerance < 0:
            raise ValueError('speed/future tolerance must be finite and nonnegative')
        self.input_timeout, self.path_timeout = input_timeout, path_timeout
        self.future_tolerance = future_tolerance
        self.speed, self.wheelbase = speed, wheelbase
        self.steer_max, self.max_waypoint_m = steer_max, max_waypoint_m
        self.condition = Condition()
        self.pending = self.latest = self.prediction = None
        self.epoch, self.last_stamp = 0, None
        self.closed = False
        self.reason = 'waiting for image/prediction'

    def _fresh(self, frame, now, ros_now, timeout):
        return (0 <= now - frame.received <= timeout and
                -self.future_tolerance <= ros_now - frame.stamp <= timeout)

    def _invalidate(self, reason):
        self.prediction = None
        self.epoch += 1
        self.reason = reason

    def offer(self, message, stamp, now, ros_now):
        with self.condition:
            if (not np.isfinite(stamp) or stamp <= 0 or
                    not -self.future_tolerance <= ros_now - stamp <= self.input_timeout or
                    (self.last_stamp is not None and stamp <= self.last_stamp)):
                self._invalidate('invalid, stale or non-increasing image timestamp')
                self.pending = None
                return False
            self.last_stamp = stamp
            self.latest = self.pending = Frame(message, stamp, now, self.epoch)
            self.condition.notify()
            return True

    def take(self):
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.pending is not None)
            if self.closed:
                return None
            frame, self.pending = self.pending, None
            return frame

    def fail(self, frame, reason):
        with self.condition:
            if frame.epoch == self.epoch:
                self._invalidate(reason)
                # Frames queued before a failed inference cannot revive the old epoch.
                self.pending = None

    def complete(self, frame, waypoint, now, ros_now):
        wp = np.asarray(waypoint, dtype=float)
        valid = bool(wp.shape == (2,) and np.isfinite(wp).all() and wp[0] > 0 and
                     np.hypot(*wp) <= self.max_waypoint_m)
        with self.condition:
            if frame.epoch != self.epoch:
                return False
            if not valid:
                self._invalidate('invalid waypoint output')
                self.pending = None
                return False
            if not self._fresh(frame, now, ros_now, min(self.path_timeout, self.input_timeout)):
                self._invalidate('inference result expired')
                self.pending = None
                return False
            wp = wp.copy()
            wp.flags.writeable = False
            self.prediction = (frame, wp)
            self.reason = 'valid'
            return True

    def command(self, now, ros_now):
        with self.condition:
            if self.latest is None or not self._fresh(self.latest, now, ros_now, self.input_timeout):
                self.reason = 'image timeout'
                return 0.0, 0.0
            if self.prediction is None:
                return 0.0, 0.0
            frame, wp = self.prediction
            if not self._fresh(frame, now, ros_now, min(self.path_timeout, self.input_timeout)):
                self.prediction = None  # never refresh the source timestamp on timer reuse
                self.reason = 'path timeout'
                return 0.0, 0.0
            steering = pure_pursuit(wp, self.wheelbase, self.steer_max)
            if not np.isfinite(steering):
                self._invalidate('nonfinite steering')
                return 0.0, 0.0
            return self.speed, steering

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
