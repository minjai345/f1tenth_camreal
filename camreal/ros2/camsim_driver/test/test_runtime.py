"""ROS-free checks of the perception mailbox and the control-side waypoint follower."""
import threading
import time
import numpy as np
import pytest
from camsim.pure_pursuit import pure_pursuit
from camsim_driver import runtime
from camsim_driver.runtime import FrameMailbox, WaypointFollower, valid_waypoint

BAD = [[], [np.nan, 0], [1, np.inf], [-1, 0], [0, 0], [[1, 0], [2, 0]], [100, 0], [4.3, 4.3]]


def mailbox():
    return FrameMailbox(.25, .02, 6.)


def follower():
    return WaypointFollower(.2, .02, .5, .3, .4, 6., 'rear_axle')


def test_pursuit_and_timestamp_expiry():
    box, follow = mailbox(), follower()
    assert box.offer('image', 10., 1., 10.)
    frame = box.take()
    assert box.complete(frame, [1., 1.], 1.01, 10.01) and box.reason == 'valid'
    assert follow.update(np.array([1., 1.]), frame.stamp, 'rear_axle', 1.01, 10.01)
    speed, steer = follow.command(1.02, 10.02)
    # Pure pursuit straight at the waypoint: curvature 2y/L^2 = 1, steer = atan(wheelbase * 1).
    assert speed == .5 and steer == pytest.approx(np.arctan(.3))
    assert box.offer('new image', 10.19, 1.19, 10.19)  # a newer image without a result keeps nothing alive
    assert follow.command(1.21, 10.21) == (0., 0.) and follow.reason == 'waypoint timeout'
    assert frame.stamp == 10.


@pytest.mark.parametrize('wp', BAD)
def test_bad_predictions_stop(wp):
    assert not valid_waypoint(wp, 6.)
    box, follow = mailbox(), follower()
    box.offer(None, 10., 1., 10.)
    assert not box.complete(box.take(), wp, 1.01, 10.01)  # never published
    assert box.reason == 'invalid waypoint output'
    assert follow.update([1., 0.], 10., 'rear_axle', 1.01, 10.01)
    assert not follow.update(wp, 10.02, 'rear_axle', 1.02, 10.02)  # a bad message drops the held waypoint
    assert follow.command(1.03, 10.03) == (0., 0.)


def test_valid_waypoint_bounds():
    assert valid_waypoint([1, 0], 6.) and valid_waypoint(np.array([6., 0.]), 6.)
    assert not valid_waypoint([6.01, 0.], 6.)


def test_input_loss_and_clock_pause():
    box, follow = mailbox(), follower()
    box.offer(None, 10., 1., 10.)
    assert box.complete(box.take(), [1, 0], 1., 10.)
    assert follow.update([1., 0.], 10., 'rear_axle', 1., 10.)
    assert follow.command(1.3, 10.) == (0., 0.)  # monotonic timeout even if ROS clock stalls
    assert box.offer(None, 10.01, 1.31, 10.)
    assert not box.complete(box.take(), [1, 0], 1.6, 10.)  # a late result is not published either
    assert box.reason == 'inference result expired'


def test_latest_only_and_failure_epoch():
    box, follow = mailbox(), follower()
    for i in range(10):
        assert box.offer(i, 10. + i / 100, 1. + i / 100, 10. + i / 100)
    frame = box.take()
    assert frame.message == 9
    box.fail(frame, 'network failed')
    assert box.reason == 'network failed'
    assert not box.complete(frame, [1, 0], 1.1, 10.1)
    assert follow.command(1.1, 10.1) == (0., 0.)
    box.offer(11, 10.11, 1.11, 10.11)
    frame = box.take()
    assert frame.message == 11 and box.complete(frame, [1, 0], 1.12, 10.12)
    assert follow.update([1., 0.], frame.stamp, 'rear_axle', 1.12, 10.12)
    assert follow.command(1.13, 10.13)[0] == .5


@pytest.mark.parametrize('stamp', [0., 9., 11., np.nan, 10.])
def test_invalid_timestamp_invalidates_running_result(stamp):
    box = mailbox()
    box.offer(None, 10., 1., 10.)
    frame = box.take()
    assert not box.offer(None, stamp, 1.01, 10.01)
    assert not box.complete(frame, [1, 0], 1.02, 10.02)
    assert 'timestamp' in box.reason


def test_delayed_worker_blocks_neither_images_nor_control():
    box, follow = mailbox(), follower()
    box.offer(None, 10., 1., 10.)
    entered, release, result = threading.Event(), threading.Event(), []

    def worker():
        frame = box.take()
        entered.set()
        release.wait(2.)
        result.append(box.complete(frame, [1, 0], 1.5, 10.5))
    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(1.)
    start = time.monotonic()
    for i in range(100):
        assert box.offer(i, 10.4 + i * 1e-4, 1.4, 10.41)
        assert follow.command(1.4, 10.4) == (0., 0.)
    assert time.monotonic() - start < .2
    release.set()
    thread.join(2.)
    assert not thread.is_alive() and result == [False]


def test_close_releases_waiting_worker():
    box, result = mailbox(), []
    thread = threading.Thread(target=lambda: result.append(box.take()))
    thread.start()
    box.close()
    thread.join(2.)
    assert not thread.is_alive() and result == [None]


def test_follower_rejects_wrong_frame_id():
    follow = follower()
    assert follow.update([1., 0.], 10., 'rear_axle', 1., 10.)
    assert not follow.update([1., 0.], 10.01, 'base_link', 1.01, 10.01)
    assert 'frame_id' in follow.reason
    assert follow.command(1.02, 10.02) == (0., 0.)


@pytest.mark.parametrize('stamp', [10., 9.99, 0., -1., np.nan, np.inf])
def test_follower_rejects_non_increasing_or_invalid_stamp(stamp):
    follow = follower()
    assert follow.update([1., 0.], 10., 'rear_axle', 1., 10.)
    assert not follow.update([1., 0.], stamp, 'rear_axle', 1.01, 10.01)
    assert follow.command(1.02, 10.02) == (0., 0.)
    assert follow.update([1., 0.], 10.02, 'rear_axle', 1.02, 10.02)  # newer stamps recover
    assert follow.command(1.03, 10.03)[0] == .5


def test_follower_future_and_stale_stamps():
    follow = follower()
    assert not follow.update([1., 0.], 10.05, 'rear_axle', 1., 10.)  # 50 ms ahead of ROS time > 20 ms tolerance
    assert follow.update([1., 0.], 10.015, 'rear_axle', 1., 10.)
    assert follow.command(1.01, 10.) == (.5, 0.)
    assert follow.command(1.02, 9.) == (0., 0.)  # ROS clock jumped back: stamp now in the future
    assert follow.update([1., 0.], 10.1, 'rear_axle', 1.03, 10.1)
    assert not follow.update([1., 0.], 10.2, 'rear_axle', 1.04, 10.5)  # 0.3 s old on arrival
    assert follow.command(1.05, 10.5) == (0., 0.)


@pytest.mark.parametrize('wp', [[np.nan, 0.], [1., np.inf], [0., 0.], [-1., 0.], [6.1, 0.], [4.3, 4.3]])
def test_follower_rejects_invalid_waypoint(wp):
    follow = follower()
    assert follow.update([1., 0.], 10., 'rear_axle', 1., 10.)
    assert not follow.update(np.array(wp), 10.01, 'rear_axle', 1.01, 10.01)
    assert follow.command(1.02, 10.02) == (0., 0.)


def test_follower_timeout_after_silence():
    follow = follower()
    assert follow.command(1., 10.) == (0., 0.) and follow.reason == 'waiting for waypoint'
    assert follow.update([1., 0.], 10., 'rear_axle', 1., 10.)
    assert follow.command(1.19, 10.19)[0] == .5 and follow.reason == 'valid'
    assert follow.command(1.21, 10.21) == (0., 0.) and follow.reason == 'waypoint timeout'


def test_reason_describes_the_last_command_not_the_last_message():
    """Messages arrive between control ticks: an accepted one is no reason to read 'valid' while held at zero."""
    follow = follower()
    assert follow.update([1., 0.], 10., 'rear_axle', 1., 10.) and follow.reason == 'waiting for waypoint'
    assert follow.command(1.01, 10.01, '2 publishers on /waypoint') == (0., 0.)
    assert follow.update([1., 0.], 10.02, 'rear_axle', 1.02, 10.02)
    assert follow.reason == '2 publishers on /waypoint'
    assert follow.command(1.03, 10.03)[0] == .5 and follow.reason == 'valid'


@pytest.mark.parametrize('wp', [[1., 0.], [1., 1.], [.5, -.4], [.3, .5]])
def test_follower_steering_is_camsim_pure_pursuit(wp):
    follow = follower()
    assert follow.update(np.array(wp), 10., 'rear_axle', 1., 10.)
    assert follow.command(1.01, 10.01) == (.5, pure_pursuit(wp, .3, .4))


@pytest.mark.parametrize('change,name', [
    (dict(timeout=0.), 'waypoint_timeout_s'), (dict(wheelbase=0.), 'wheelbase_m'),
    (dict(steer_max=np.nan), 'steer_max'), (dict(max_waypoint_m=-1.), 'max_waypoint_m'),
    (dict(speed=-.1), 'target_speed_mps'), (dict(future_tolerance=-1.), 'future_tolerance_s'),
    (dict(frame_id=''), 'path_frame 값이 비었습니다')])
def test_follower_rejects_bad_configuration(change, name):
    args = dict(timeout=.2, future_tolerance=.02, speed=.5, wheelbase=.3, steer_max=.4,
                max_waypoint_m=6., frame_id='rear_axle')
    with pytest.raises(ValueError, match=name):
        WaypointFollower(**{**args, **change})


@pytest.mark.parametrize('args,name', [((0., .02, 6.), 'input_timeout_s'), ((np.inf, .02, 6.), 'input_timeout_s'),
                                       ((.25, -.01, 6.), 'future_tolerance_s'), ((.25, .02, 0.), 'max_waypoint_m')])
def test_mailbox_rejects_bad_configuration(args, name):
    with pytest.raises(ValueError, match=f'{name}.*유한한 값'):
        FrameMailbox(*args)


def test_follower_orders_by_the_newest_stamp_seen_even_if_rejected():
    follow = follower()
    assert follow.update([1., 0.], 10., 'rear_axle', 1., 10.)
    assert not follow.update([np.nan, 0.], 10.1, 'rear_axle', 1.1, 10.1)
    assert not follow.update([1., -.1], 10.05, 'rear_axle', 1.11, 10.11)   # late, older than the rejected one
    assert follow.reason.startswith('non-increasing') and follow.command(1.12, 10.12) == (0., 0.)
    assert not follow.update([1., 0.], 10.5, 'rear_axle', 1.12, 10.12)    # far future: rejected, blocks nothing
    assert follow.update([1., 0.], 10.12, 'rear_axle', 1.12, 10.12)


def test_follower_stops_on_nonfinite_steering(monkeypatch):
    follow = follower()
    assert follow.update([1., 0.], 10., 'rear_axle', 1., 10.)
    monkeypatch.setattr(runtime, 'pure_pursuit', lambda *_: float('nan'))
    assert follow.command(1.01, 10.01) == (0., 0.) and follow.reason == 'nonfinite steering'
    monkeypatch.undo()
    assert follow.command(1.02, 10.02) == (0., 0.)   # the waypoint was dropped


def test_follower_block_holds_zero_and_drops_the_waypoint():
    follow = follower()
    assert follow.update([1., 0.], 10., 'rear_axle', 1., 10.)
    assert follow.command(1.01, 10.01, '2 publishers on /waypoint') == (0., 0.)
    assert follow.reason == '2 publishers on /waypoint'
    assert follow.command(1.02, 10.02) == (0., 0.)   # never revived
    assert follow.update([1., 0.], 10.03, 'rear_axle', 1.03, 10.03)
    assert follow.command(1.04, 10.04)[0] == .5
    with follow.lock:   # re-entrant: the node reads its clocks while holding it
        assert follow.command(1.05, 10.05)[0] == .5


@pytest.mark.parametrize('stamp,reason', [(np.nan, 'invalid'), (0., 'invalid'), (9.5, 'stale'),
                                          (10.5, 'future'), (10., 'non-increasing')])
def test_stamp_rejections_name_their_cause(stamp, reason):
    box, follow = mailbox(), follower()
    assert box.offer(None, 10., 1., 10.)
    assert not box.offer(None, stamp, 1.01, 10.01)
    # waypoint_node draws its reason on the BEV with cv2.putText, which shows Hangul as '?'.
    assert box.reason.startswith(reason) and 'timestamp' in box.reason and box.reason.isascii()
    assert follow.update([1., 0.], 10., 'rear_axle', 1., 10.)
    assert not follow.update([1., 0.], stamp, 'rear_axle', 1.01, 10.01)
    assert follow.reason.startswith(reason) and 'stamp' in follow.reason
    if reason == 'non-increasing':   # the usual cause is a rewound bag or clock
        assert '다시 시작' in follow.reason

