"""ROS image conversion and visualization messages; no control feedback topic."""
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def decode_bgr8(bridge, message):
    # >8-bit intensity scaling requires a training-specific policy; never guess it.
    encoding = message.encoding.lower()
    if encoding not in ('bgr8', 'rgb8', 'bgra8', 'rgba8', 'mono8',
                        'bayer_rggb8', 'bayer_bggr8', 'bayer_gbrg8', 'bayer_grbg8', 'yuv422'):
        raise ValueError(f'unsupported image encoding: {encoding}; configure camera to 8-bit')
    return bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')


def make_path(stamp, frame_id, wp):
    """Rear axle -> predicted waypoint (the pure pursuit target); both poses face the waypoint."""
    wp = np.asarray(wp, dtype=float).reshape(2)
    yaw = float(np.arctan2(wp[1], wp[0]))
    path = Path()
    path.header.stamp, path.header.frame_id = stamp, frame_id
    for x, y in ((0., 0.), wp):
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        pose.pose.orientation.z, pose.pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
        path.poses.append(pose)
    return path
