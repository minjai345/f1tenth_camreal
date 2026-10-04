"""Week-1 ost.yaml + floor marker clicks -> camreal schema-1 calibration (H_i2g).

Clicks are pixels of the full-resolution undistorted image made with CameraPreprocessor's maps,
so H_i2g lands in the homography_space the car uses. ROS is imported only by read_bag_frame.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import math
from pathlib import Path
import re
import cv2
import numpy as np
import yaml
from camsim import camera
from camsim.handoff import sha256_file
from camsim.render import bev_pixels
from camreal.checkpoint import training_mask
from camreal.overlay import TRUTH, bev_view
from camreal.preprocessing import CameraPreprocessor

MIN_FIT, MIN_LOO, MIN_SAVE = 4, 5, 6
WARN_CM, REJECT_CM = 5.0, 20.0
MAX_COND = 1e8   # camera homographies stay near 1e3-1e5; collinear clicks give 1e10 and more
GREY = (200, 200, 200)
HEADER = '# camreal 지면 캘리브레이션: python3 -m camreal calibrate가 만든 파일. 고칠 때는 다시 실행하세요.\n'


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    new_K: np.ndarray
    path: str
    sha256: str


def _read_yaml(path):
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        lines = sorted({m.line + 1 for m in (getattr(exc, 'context_mark', None), getattr(exc, 'problem_mark', None)) if m})
        where = f' ({"~".join(map(str, lines))}번째 줄)' if lines else ''
        raise ValueError(f'{path}: YAML 문법 오류입니다{where}. 들여쓰기, 콜론, 괄호를 확인하세요.') from exc


def _ost_matrix(data, key, rows, cols, path):
    m = data[key]
    try:
        values = np.asarray(m['data'], dtype=float) if (m['rows'], m['cols']) == (rows, cols) else None
    except (KeyError, TypeError, ValueError):
        values = None
    if values is None or values.shape != (rows * cols,) or not np.isfinite(values).all():
        raise ValueError(f'{path}: {key}는 rows: {rows}, cols: {cols}, data: 유한한 숫자 {rows * cols}개여야 합니다.')
    return values.reshape(rows, cols)


def _pinhole(m):
    return m[0, 0] > 0 and m[1, 1] > 0 and np.allclose(m[2], [0, 0, 1])


def read_ost(path):
    """Week-1 cameracalibrator ost.yaml (ROS camera_info YAML) -> Intrinsics.

    new_K = P[:, :3], the undistorted view week-1 image_proc showed; K when P is missing or unusable.
    """
    path = Path(path).expanduser().resolve()
    data = _read_yaml(path)
    keys = ('image_width', 'image_height', 'camera_matrix', 'distortion_model', 'distortion_coefficients')
    missing = [k for k in keys if not isinstance(data, dict) or k not in data]
    if missing:
        raise ValueError(f'{path}: {", ".join(missing)} 항목이 없습니다. 1주차 cameracalibrator가 저장한 ost.yaml을 지정하세요.')
    width, height = data['image_width'], data['image_height']
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError(f'{path}: image_width, image_height는 양의 정수여야 합니다.')
    if data['distortion_model'] != 'plumb_bob':
        raise ValueError(f"{path}: distortion_model이 {data['distortion_model']}입니다. plumb_bob(계수 5개)만 지원합니다. "
                         '1주차 방식(cameracalibrator 기본 핀홀 모델)으로 다시 캘리브레이션하세요.')
    K = _ost_matrix(data, 'camera_matrix', 3, 3, path)
    D = _ost_matrix(data, 'distortion_coefficients', 1, 5, path)[0]
    if not _pinhole(K):
        raise ValueError(f'{path}: camera_matrix가 카메라 행렬(fx, fy > 0, 마지막 행 [0, 0, 1])이 아닙니다.')
    new_K = K
    if 'projection_matrix' in data:
        try:
            P = _ost_matrix(data, 'projection_matrix', 3, 4, path)[:, :3]
            new_K = P if _pinhole(P) else K
        except ValueError:
            pass
    return Intrinsics(width, height, K, D, new_K.copy(), str(path), sha256_file(path))


def load_markers(path):
    """markers.yaml {markers: {id: [x, y]}} -> {id: (x, y)} in rear-axle metres, file order."""
    path = Path(path).expanduser()
    data = _read_yaml(path)
    if not isinstance(data, dict) or not isinstance(data.get('markers'), dict):
        raise ValueError(f'{path}: markers: 아래에 id: [x, y]를 적으세요 (camreal/config/markers.yaml 참고).')
    markers, owner = {}, {}
    for key, value in data['markers'].items():
        name = str(key) if type(key) in (str, int) else ''
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,16}', name) or name in markers:
            raise ValueError(f'{path}: 마커 id {key!r}: 영문/숫자/_/- 1~16자로 겹치지 않게 지으세요.')
        if not (isinstance(value, list) and len(value) == 2 and
                all(type(v) in (int, float) and math.isfinite(v) for v in value)):
            raise ValueError(f'{path}: {name}의 좌표는 유한한 숫자 두 개 [x, y]여야 합니다 (m).')
        x, y = float(value[0]), float(value[1])
        if x <= 0:
            raise ValueError(f'{path}: {name}의 x({x:g} m)는 0보다 커야 합니다 (후륜축보다 앞).')
        if (x, y) in owner:
            raise ValueError(f'{path}: {owner[(x, y)]}와 {name}의 좌표가 같습니다 ({x:g}, {y:g}).')
        owner[(x, y)], markers[name] = name, (x, y)
    if len(markers) < MIN_SAVE:
        raise ValueError(f'{path}: 마커가 {len(markers)}개입니다. {MIN_SAVE}개 이상 적으세요.')
    return markers


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'이미지를 읽을 수 없습니다: {path} (원본 해상도 PNG/JPG를 지정하세요)')
    return image


def read_bag_frame(bag_path, image_topic):
    """Middle image_topic message of a rosbag2 -> (BGR uint8, header stamp ns)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from camreal.labeling.bag import ENCODINGS
    bag = Path(bag_path).expanduser()
    if not (bag/'metadata.yaml').is_file():
        raise FileNotFoundError(f'rosbag이 없습니다: {bag} (metadata.yaml이 있는 ros2 bag record --output 폴더를 지정하세요)')
    meta = rosbag2_py.Info().read_metadata(str(bag), '')
    if meta.compression_mode not in ('', 'none'):
        raise ValueError('압축되지 않은 rosbag2를 사용하세요. 현재 compressed storage는 지원하지 않습니다.')
    topics = {t.topic_metadata.name: t for t in meta.topics_with_message_count}
    if image_topic not in topics:
        raise ValueError(f'{bag}에 {image_topic} 토픽이 없습니다. 기록된 토픽: {", ".join(sorted(topics)) or "없음"}')
    topic = topics[image_topic]
    if topic.topic_metadata.type != 'sensor_msgs/msg/Image':
        raise ValueError(f'{image_topic}: sensor_msgs/msg/Image 토픽이 아닙니다 ({topic.topic_metadata.type}).')
    if topic.message_count == 0:
        raise ValueError(f'{bag}의 {image_topic} 메시지가 0개입니다. 카메라를 켠 상태에서 다시 기록하세요.')
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=meta.storage_identifier),
                rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[image_topic]))
    for _ in range(topic.message_count // 2 + 1):
        if not reader.has_next():
            raise ValueError(f'{image_topic} 메시지가 metadata보다 적습니다. bag이 손상되었습니다.')
        data = reader.read_next()[1]
    message = deserialize_message(data, Image)
    if message.encoding.lower() not in ENCODINGS:
        raise ValueError(f'unsupported image encoding: {message.encoding}')
    stamp = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
    return CvBridge().imgmsg_to_cv2(message, desired_encoding='bgr8'), stamp


def check_resolution(frame, intr):
    h, w = frame.shape[:2]
    if (w, h) != (intr.width, intr.height):
        raise ValueError(
            f'영상 해상도 {w}x{h}가 ost.yaml({intr.path})의 {intr.width}x{intr.height}와 다릅니다. 다음 중 하나로 맞추세요: '
            f'1) 카메라 해상도를 캘리브레이션 때와 같은 {intr.width}x{intr.height}로 설정 '
            f'2) 1주차 방식(cameracalibrator)으로 {w}x{h}에서 다시 캘리브레이션 '
            '3) --ost로 기준 파일 지정 (--ost camreal/config/ost_reference_1920x1200.yaml)')


def undistort(frame, intr):
    """Full-resolution undistorted image: CameraPreprocessor's maps, black border."""
    check_resolution(frame, intr)
    mapx, mapy = cv2.initUndistortRectifyMap(intr.K, intr.D, None, intr.new_K, (intr.width, intr.height), cv2.CV_32FC1)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _pairs(points, markers):
    """Clicked ids in marker-file order, their (n, 2) pixels and ground metres."""
    if not isinstance(points, dict):
        raise ValueError('points는 {마커 id: [u, v]} 형식이어야 합니다.')
    unknown = [str(k) for k in points if k not in markers]
    if unknown:
        raise ValueError(f'마커 파일에 없는 id: {", ".join(unknown)}')
    ids, pixels = [k for k in markers if k in points], []
    for k in ids:
        try:
            p = [float(c) for c in points[k]] if isinstance(points[k], (list, tuple, np.ndarray)) else []
        except (TypeError, ValueError):
            p = []
        if len(p) != 2 or not all(math.isfinite(c) for c in p):
            raise ValueError(f'{k}: 픽셀 좌표는 유한한 [u, v]여야 합니다.')
        pixels.append(p)
    return ids, np.array(pixels, float).reshape(-1, 2), np.array([markers[k] for k in ids], float).reshape(-1, 2)


def _spread(ground):
    """Some 4 markers with no 3 on one line (1 mm^2): otherwise H is not determined, even by exact clicks."""
    def area(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return any(all(area(*t) > 1e-6 for t in combinations(q, 3)) for q in combinations(ground.tolist(), 4))


def _fit(img, ground):
    H = cv2.findHomography(img, ground, 0)[0] if len(img) >= MIN_FIT and _spread(ground) else None
    if H is None or H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.cond(H) > MAX_COND:
        raise ValueError('찍은 점으로 H를 계산할 수 없습니다. 한 줄에 몰리지 않게 여러 줄의 마커를 찍으세요.')
    return H / np.linalg.norm(H)


def fit_homography(points, markers):
    """H_i2g (undistorted px -> rear-axle ground m, Frobenius norm 1) from MIN_FIT or more clicks."""
    ids, img, ground = _pairs(points, markers)
    if len(ids) < MIN_FIT:
        raise ValueError(f'마커를 {MIN_FIT}개 이상 찍어야 H를 계산합니다 (지금 {len(ids)}개).')
    return _fit(img, ground)


def loo_errors(points, markers):
    """Leave-one-out ground error (m) per clicked marker, inf when the others cannot fit H; {} below MIN_LOO."""
    ids, img, ground = _pairs(points, markers)
    if len(ids) < MIN_LOO:
        return {}
    errors = {}
    for i, k in enumerate(ids):
        rest = np.arange(len(ids)) != i
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                e = float(np.hypot(*(camera.project(_fit(img[rest], ground[rest]), img[i]) - ground[i])))
        except ValueError:
            e = math.inf
        errors[k] = e if math.isfinite(e) else math.inf
    return errors


def estimate_pose(H_i2g, new_K):
    """Camera pose from inv(H_i2g) = s * new_K [r1 r2 t] in camsim.camera conventions.

    x_m, y_m, height_m: camera centre in the rear-axle frame (x forward, y left, z up).
    The mount B = R^T _R_VC (columns: optical axis, left, up) is Rz(yaw) Ry(pitch) Rx(roll):
    pitch + looks down, yaw + looks left, roll + turns the camera clockwise seen from behind
    (right side down, so the floor appears turned counter-clockwise in the image).
    s puts the floor below the horizon in front of the camera (|roll| < 90 deg); a mirrored
    marker frame (y sign flipped or x, y swapped) then gives height_m < 0.
    """
    H = np.asarray(H_i2g, dtype=float)
    M = np.linalg.inv(new_K) @ np.linalg.inv(H)
    r1, r2, t = math.copysign(1., H[2, 1]) / math.sqrt(np.linalg.norm(M[:, 0]) * np.linalg.norm(M[:, 1])) * M.T
    U, _, Vt = np.linalg.svd(np.column_stack((r1, r2, np.cross(r1, r2))))
    R = U @ Vt
    x, y, z = -R.T @ t
    B = R.T @ camera._R_VC
    return dict(x_m=float(x), y_m=float(y), height_m=float(z),
                pitch_deg=math.degrees(math.asin(min(1., max(-1., -B[2, 0])))),
                yaw_deg=math.degrees(math.atan2(B[1, 0], B[0, 0])),
                roll_deg=math.degrees(math.atan2(B[2, 1], B[2, 2])))


def evaluate(points, markers, intr):
    """What the UI shows for the clicks so far; the server recomputes it on save.

    savable: MIN_SAVE or more markers, every leave-one-out error <= REJECT_CM and the camera above
    the floor (a mirrored marker frame fits just as well but puts the camera below it).
    """
    ids, img, _ = _pairs(points, markers)
    if len(ids) and not ((img >= 0) & (img < [intr.width, intr.height])).all():
        raise ValueError(f'영상({intr.width}x{intr.height}) 밖의 픽셀이 있습니다.')
    n = len(ids)
    result = dict(n=n, H_i2g=None, errors_cm={}, rms_cm=None, pose=None, savable=False,
                  reason=f'마커를 {MIN_FIT}개 이상 찍으면 계산합니다 (지금 {n}개, 저장은 {MIN_SAVE}개부터).')
    if n < MIN_FIT:
        return result
    try:
        H = fit_homography(points, markers)
    except ValueError as exc:
        return dict(result, reason=str(exc))
    errors = {k: round(e * 100, 2) if math.isfinite(e) else None for k, e in loo_errors(points, markers).items()}
    known = [e for e in errors.values() if e is not None]
    rms = round(math.sqrt(sum(e * e for e in known) / n), 2) if len(known) == n else None
    pose = {k: round(v, 4) + 0. for k, v in estimate_pose(H, intr.new_K).items()}   # + 0. turns -0.0 into 0.0
    far = [k for k, e in errors.items() if e is not None and e > REJECT_CM]
    lost = [k for k, e in errors.items() if e is None]
    problems = []
    if n < MIN_SAVE:
        problems.append(f'저장하려면 마커를 {MIN_SAVE}개 이상 찍으세요 (지금 {n}개).')
    if far:
        problems.append(f'오차가 {REJECT_CM:g} cm를 넘는 마커: {", ".join(far)}. 다른 마커를 찍은 것으로 보입니다. 다시 찍거나 건너뛰세요.')
    if lost:
        problems.append(f'검증할 수 없는 마커: {", ".join(lost)} (빼면 나머지가 한 줄). 다른 줄의 마커를 더 찍으세요.')
    if pose['height_m'] <= 0:
        problems.append(f'추정 카메라 높이가 {pose["height_m"]:.2f} m(바닥 아래)입니다. 마커 좌표의 y 부호(왼쪽 +)와 [x, y] 순서를 확인하세요.')
    warn = ', '.join(f'{k} {e:.1f} cm' for k, e in errors.items() if e is not None and WARN_CM < e <= REJECT_CM)
    reason = ' '.join(problems) or f'저장 가능: 마커 {n}개, RMS {rms:.1f} cm.'
    if warn:
        reason += f' 경고(오차 {WARN_CM:g} cm 초과): {warn}'
    return dict(result, H_i2g=H.tolist(), errors_cm=errors, rms_cm=rms, pose=pose, savable=not problems, reason=reason)


def calibration_dict(intr, H_i2g, status='measured'):
    """camreal schema-1 calibration fields as plain lists (preview and saved file)."""
    return dict(schema_version=1, calibration_status=status, distortion_model='plumb_bob',
                image_width=intr.width, image_height=intr.height, K=intr.K.tolist(), D=intr.D.tolist(),
                new_K=intr.new_K.tolist(), homography_space='undistorted_full_resolution',
                H_i2g=np.asarray(H_i2g, dtype=float).tolist(), ground_frame='rear_axle')


def bev_preview(raw_frame, calibration, cfg, markers, points):
    """The car's BEV (CameraPreprocessor) with the ahead_m ring and a cross per marker (green: clicked)."""
    out = bev_view(CameraPreprocessor(calibration, cfg, training_mask(cfg), 'rear_axle').bev(raw_frame), cfg)
    for k, xy in markers.items():
        u, v = np.rint(bev_pixels(np.asarray(xy, dtype=float), cfg)).astype(int)
        cv2.drawMarker(out, (int(u), int(v)), TRUTH if k in points else GREY, cv2.MARKER_CROSS, 11, 2 if k in points else 1)
    return out


def build_calibration(intr, result, points, markers, source):
    """calibration_dict + provenance + per-marker check + pose, YAML-safe. result = evaluate(points, markers, intr).

    source: frame (bag session or image path), frame_stamp_ns (None for an image), markers (markers.yaml path).
    """
    if not result['savable']:
        raise ValueError(f'저장할 수 없습니다: {result["reason"]}')
    if set(result['errors_cm']) != set(points):
        raise ValueError('result가 이 points로 계산한 것이 아닙니다. evaluate를 다시 실행하세요.')
    stamp = source.get('frame_stamp_ns')
    data = calibration_dict(intr, result['H_i2g'])
    data['source'] = dict(ost=intr.path, ost_sha256=intr.sha256, frame=str(source['frame']),
                          frame_stamp_ns=None if stamp is None else int(stamp),
                          markers=str(Path(source['markers']).resolve()), markers_sha256=sha256_file(source['markers']),
                          created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    data['markers'] = [dict(id=k, ground_m=[float(c) for c in markers[k]], pixel=[round(float(c), 2) for c in points[k]],
                            error_cm=float(result['errors_cm'][k])) for k in markers if k in points]
    data['camera_pose_estimate'] = {k: float(v) for k, v in result['pose'].items()}
    return data


def save_calibration(data, out_path, preview_bgr):
    """Write YAML atomically (an old file goes to <parent>/old/<stem>-YYYYmmdd-HHMMSS<suffix>) and <stem>_bev.png."""
    out = Path(out_path)
    text = HEADER + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=None)
    ok, png = cv2.imencode('.png', preview_bgr)
    if not ok:
        raise ValueError('BEV 미리보기를 PNG로 만들 수 없습니다.')
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    backup = None
    if out.exists():
        old = out.parent/'old'
        old.mkdir(exist_ok=True)
        stamp, i = datetime.now().strftime('%Y%m%d-%H%M%S'), 0
        backup = old/f'{out.stem}-{stamp}{out.suffix}'
        while backup.exists():
            i += 1
            backup = old/f'{out.stem}-{stamp}-{i}{out.suffix}'
        out.replace(backup)
    temporary.replace(out)
    preview = out.with_name(out.stem + '_bev.png')
    temporary = preview.with_name(preview.name + '.tmp')
    temporary.write_bytes(png.tobytes())
    temporary.replace(preview)
    return out, backup
