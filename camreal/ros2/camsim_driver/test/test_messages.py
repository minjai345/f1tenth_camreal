"""Actual ROS message/cv_bridge checks, no DDS or vehicle required."""
import numpy as np
import pytest
pytest.importorskip('nav_msgs.msg')
pytest.importorskip('cv_bridge')
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from camsim_driver.messages import make_path, decode_bgr8


def test_path_is_rear_axle_to_waypoint_with_capture_stamp():
    stamp = Time(sec=42, nanosec=123456789)
    path = make_path(stamp, 'rear_axle', np.array([2., 1.]))
    assert path.header.stamp == stamp
    assert path.header.frame_id == 'rear_axle'
    assert len(path.poses) == 2
    start, end = (p.pose.position for p in path.poses)
    assert (start.x, start.y) == (0., 0.) and (end.x, end.y) == (2., 1.)
    for pose in path.poses:
        assert pose.header == path.header
        q = pose.pose.orientation
        assert q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w == pytest.approx(1.)
        assert 2 * np.arctan2(q.z, q.w) == pytest.approx(np.arctan2(1., 2.))


@pytest.mark.parametrize('encoding', ['bgr8', 'rgb8', 'mono8', 'bgra8', 'rgba8'])
def test_cv_bridge_encoding_to_bgr(encoding):
    bridge = CvBridge()
    bgr = np.full((8, 10, 3), [10, 20, 30], np.uint8)
    if encoding == 'mono8':
        raw = bgr[:, :, 0].copy()
        expected = np.repeat(raw[:, :, None], 3, axis=2)
    else:
        raw = bgr if encoding.startswith('bgr') else bgr[:, :, ::-1].copy()
        if 'a' in encoding:
            raw = np.concatenate([raw, np.full((8, 10, 1), 255, np.uint8)], axis=2)
        expected = bgr
    msg = bridge.cv2_to_imgmsg(raw, encoding=encoding)
    np.testing.assert_array_equal(decode_bgr8(bridge, msg), expected)


def test_16bit_rejected():
    bridge = CvBridge()
    msg = bridge.cv2_to_imgmsg(np.ones((8, 10), np.uint16), encoding='mono16')
    with pytest.raises(ValueError, match='unsupported image encoding'):
        decode_bgr8(bridge, msg)
