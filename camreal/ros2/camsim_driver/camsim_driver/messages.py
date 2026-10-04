"""ROS image conversion, the /waypoint hand-off between the two nodes, and visualization messages."""
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped, PoseStamped


def decode_bgr8(bridge, message):
    # >8-bit intensity scaling requires a training-specific policy; never guess it.
    encoding = message.encoding.lower()
    if encoding not in ('bgr8', 'rgb8', 'bgra8', 'rgba8', 'mono8',
                        'bayer_rggb8', 'bayer_bggr8', 'bayer_gbrg8', 'bayer_grbg8', 'yuv422'):
        raise ValueError(f'unsupported image encoding: {encoding}; configure camera to 8-bit')
    return bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def make_waypoint(stamp, frame_id, wp):
    """/waypoint: predicted pure pursuit target (x, y, z = 0) stamped with the source image capture time."""
    wp = np.asarray(wp, dtype=float).reshape(2)
    msg = PointStamped()
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    msg.point.x, msg.point.y = float(wp[0]), float(wp[1])
    return msg


def read_waypoint(msg):
    """-> (stamp in seconds, frame_id, waypoint (x, y) in metres)."""
    return seconds(msg.header.stamp), msg.header.frame_id, np.array([msg.point.x, msg.point.y])


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
