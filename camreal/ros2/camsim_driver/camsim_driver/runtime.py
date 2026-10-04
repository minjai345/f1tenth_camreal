"""ROS-independent state: latest-frame mailbox (waypoint_node) and fail-closed follower (pure_pursuit_node)."""
from dataclasses import dataclass
from threading import Condition, RLock
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
    """Newest image only; a failed, invalid or expired result invalidates every frame queued before it.
    reason is drawn on the debug BEV with cv2.putText, so it stays ASCII."""

    def __init__(self, input_timeout, future_tolerance, max_waypoint_m):
        if any(not np.isfinite(v) or v <= 0 for v in (input_timeout, max_waypoint_m)):
            raise ValueError('input_timeout_s와 max_waypoint_m은 0보다 큰 유한한 값이어야 합니다.')
        if not np.isfinite(future_tolerance) or future_tolerance < 0:
            raise ValueError('future_tolerance_s는 0 이상의 유한한 값이어야 합니다.')
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
            if not np.isfinite(stamp) or stamp <= 0:
                reason = 'invalid image timestamp'
            elif ros_now - stamp > self.input_timeout:
                reason = 'stale image timestamp'
            elif ros_now - stamp < -self.future_tolerance:
                reason = 'future image timestamp'
            elif self.last_stamp is not None and stamp <= self.last_stamp:
                reason = 'non-increasing image timestamp'
            else:
                self.last_stamp = stamp
                self.pending = Frame(message, stamp, now, self.epoch)
                self.condition.notify()
                return True
            self._invalidate(reason)
            return False

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
            raise ValueError('waypoint_timeout_s, wheelbase_m, steer_max_rad, max_waypoint_m은 0보다 큰 유한한 값이어야 합니다.')
        if not np.isfinite(speed) or speed < 0 or not np.isfinite(future_tolerance) or future_tolerance < 0:
            raise ValueError('target_speed_mps와 future_tolerance_s는 0 이상의 유한한 값이어야 합니다.')
        if not frame_id:
            raise ValueError('path_frame 값이 비었습니다.')
        self.timeout, self.future_tolerance = timeout, future_tolerance
        self.speed, self.wheelbase, self.steer_max = speed, wheelbase, steer_max
        self.max_waypoint_m, self.frame_id = max_waypoint_m, frame_id
        self.lock = RLock()   # the node reads its clocks under it, so a receipt is never newer than "now"
        self.held = self.last_seen = None
        self.reason = 'waiting for waypoint'

    def _window(self, stamp, ros_now):
        return -self.future_tolerance <= ros_now - stamp <= self.timeout

    def update(self, wp, stamp, frame_id, now, ros_now):
        """False = rejected; a rejected message also drops the held waypoint. reason: why the output is zero,
        'valid' only from command()."""
        with self.lock:
            seen = self.last_seen
            if np.isfinite(stamp) and self._window(stamp, ros_now) and (seen is None or stamp > seen):
                self.last_seen = stamp   # orders later messages even if this one is rejected below
            if frame_id != self.frame_id:
                reason = f'waypoint frame_id {frame_id!r} is not {self.frame_id!r}'
            elif not np.isfinite(stamp) or stamp <= 0:
                reason = 'invalid waypoint stamp'
            elif ros_now - stamp > self.timeout:
                reason = 'stale waypoint stamp'
            elif ros_now - stamp < -self.future_tolerance:
                reason = 'future waypoint stamp'
            elif seen is not None and stamp <= seen:
                reason = 'non-increasing waypoint stamp (bag/시계를 되감았으면 두 노드를 다시 시작)'
            elif not valid_waypoint(wp, self.max_waypoint_m):
                reason = 'invalid waypoint (non-finite, x <= 0 or beyond max_waypoint_m)'
            else:
                wp = np.array(wp, dtype=float)
                wp.flags.writeable = False
                self.held = wp, stamp, now
                return True
            self.held, self.reason = None, reason
            return False

    def command(self, now, ros_now, block=None):
        """-> (speed, steering); block = a reason to hold (0, 0) anyway, such as two /waypoint publishers."""
        with self.lock:
            if block:
                self.held, self.reason = None, block
            if self.held is None:
                return 0.0, 0.0
            wp, stamp, received = self.held
            if not (0 <= now - received <= self.timeout and self._window(stamp, ros_now)):
                self.held, self.reason = None, 'waypoint timeout'  # never revived by later clock readings
                return 0.0, 0.0
            steering = pure_pursuit(wp, self.wheelbase, self.steer_max)
            if not np.isfinite(steering):
                self.held, self.reason = None, 'nonfinite steering'
                return 0.0, 0.0
            self.reason = 'valid'
            return self.speed, steering
