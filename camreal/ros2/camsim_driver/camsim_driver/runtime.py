"""ROS-independent state: latest-frame mailbox (waypoint_node) and fail-closed follower (pure_pursuit_node)."""
from dataclasses import dataclass
from threading import Condition, Lock
import numpy as np
from camsim.pure_pursuit import pure_pursuit


@dataclass(frozen=True)
class Frame:
    message: object
    stamp: float
    received: float
    epoch: int


def valid_waypoint(wp, max_waypoint_m):
    wp = np.asarray(wp, dtype=float)
    return bool(wp.shape == (2,) and np.isfinite(wp).all() and wp[0] > 0 and np.hypot(*wp) <= max_waypoint_m)


class FrameMailbox:
    """Newest image only; a failed, invalid or expired result invalidates every frame queued before it."""

    def __init__(self, input_timeout, future_tolerance, max_waypoint_m):
        if any(not np.isfinite(v) or v <= 0 for v in (input_timeout, max_waypoint_m)):
            raise ValueError('input timeout and waypoint bound must be positive and finite')
        if not np.isfinite(future_tolerance) or future_tolerance < 0:
            raise ValueError('future tolerance must be finite and nonnegative')
        self.input_timeout, self.future_tolerance = input_timeout, future_tolerance
        self.max_waypoint_m = max_waypoint_m
        self.condition = Condition()
        self.pending, self.last_stamp = None, None
        self.epoch, self.closed = 0, False
        self.reason = 'waiting for image'

    def _invalidate(self, reason):
        self.epoch += 1
        self.pending = None
        self.reason = reason

    def offer(self, message, stamp, now, ros_now):
        with self.condition:
            if (not np.isfinite(stamp) or stamp <= 0 or
                    not -self.future_tolerance <= ros_now - stamp <= self.input_timeout or
                    (self.last_stamp is not None and stamp <= self.last_stamp)):
                self._invalidate('invalid, stale or non-increasing image timestamp')
                return False
            self.last_stamp = stamp
            self.pending = Frame(message, stamp, now, self.epoch)
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

    def complete(self, frame, waypoint, now, ros_now):
        """True = publish: same epoch, valid waypoint and the source image still fresh."""
        valid = valid_waypoint(waypoint, self.max_waypoint_m)
        with self.condition:
            if frame.epoch != self.epoch:
                return False
            if not valid:
                self._invalidate('invalid waypoint output')
                return False
            if not (0 <= now - frame.received <= self.input_timeout and
                    -self.future_tolerance <= ros_now - frame.stamp <= self.input_timeout):
                self._invalidate('inference result expired')
                return False
            self.reason = 'valid'
            return True

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class WaypointFollower:
    """Pure pursuit straight at the newest accepted waypoint while it is fresh; otherwise (0, 0)."""

    def __init__(self, timeout, future_tolerance, speed, wheelbase, steer_max, max_waypoint_m, frame_id):
        if any(not np.isfinite(v) or v <= 0 for v in (timeout, wheelbase, steer_max, max_waypoint_m)):
            raise ValueError('waypoint timeout, wheelbase, steer_max and max_waypoint_m must be positive and finite')
        if not np.isfinite(speed) or speed < 0 or not np.isfinite(future_tolerance) or future_tolerance < 0:
            raise ValueError('speed/future tolerance must be finite and nonnegative')
        if not frame_id:
            raise ValueError('frame_id is empty')
        self.timeout, self.future_tolerance = timeout, future_tolerance
        self.speed, self.wheelbase, self.steer_max = speed, wheelbase, steer_max
        self.max_waypoint_m, self.frame_id = max_waypoint_m, frame_id
        self.lock = Lock()
        self.held = self.last_stamp = None
        self.reason = 'waiting for waypoint'

    def _fresh(self, stamp, received, now, ros_now):
        return 0 <= now - received <= self.timeout and -self.future_tolerance <= ros_now - stamp <= self.timeout

    def update(self, wp, stamp, frame_id, now, ros_now):
        """False = rejected; a rejected message also drops the held waypoint."""
        with self.lock:
            if frame_id != self.frame_id:
                reason = f'waypoint frame_id {frame_id!r} is not {self.frame_id!r}'
            elif not np.isfinite(stamp) or stamp <= 0 or (self.last_stamp is not None and stamp <= self.last_stamp):
                reason = 'invalid or non-increasing waypoint stamp'
            elif not valid_waypoint(wp, self.max_waypoint_m):
                reason = 'invalid waypoint (non-finite, x <= 0 or beyond max_waypoint_m)'
            elif not -self.future_tolerance <= ros_now - stamp <= self.timeout:
                reason = 'stale or future waypoint stamp'
            else:
                wp = np.array(wp, dtype=float)
                wp.flags.writeable = False
                self.held, self.last_stamp, self.reason = (wp, stamp, now), stamp, 'valid'
                return True
            self.held, self.reason = None, reason
            return False

    def command(self, now, ros_now):
        with self.lock:
            if self.held is None:
                return 0.0, 0.0
            wp, stamp, received = self.held
            if not self._fresh(stamp, received, now, ros_now):
                self.held, self.reason = None, 'waypoint timeout'  # never revived by later clock readings
                return 0.0, 0.0
            return self.speed, pure_pursuit(wp, self.wheelbase, self.steer_max)
