"""Offline rosbag2 reader; never replays messages onto the vehicle graph."""
import hashlib
from pathlib import Path
import re
import shutil
import cv2
import numpy as np
import yaml
from camsim.handoff import sha256_file as sha256
from camreal.checkpoint import contract, load_config, model_files, read_manifest, training_mask
from camreal.preprocessing import CameraPreprocessor
from .core import MODEL_DIR, write_json

# Same explicit encoding policy as the live driver, without importing the ROS node.
ENCODINGS = ('bgr8', 'rgb8', 'bgra8', 'rgba8', 'mono8',
             'bayer_rggb8', 'bayer_bggr8', 'bayer_gbrg8', 'bayer_grbg8', 'yuv422')


def bag_digest(root):
    """Content identity detects reuse even when a bag directory is renamed/copied."""
    meta = yaml.safe_load((root/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    paths = meta['relative_file_paths']
    h = hashlib.sha256()
    for name in paths:
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('invalid bag storage path')
        h.update(sha256(path).encode('ascii'))
    return h.hexdigest(), meta


def mask_digest(mask):
    return hashlib.sha256(np.packbits(mask).tobytes() + repr(mask.shape).encode()).hexdigest()


def existing_project(output, bag_path, image_topic, model_dir, calibration_path,
                     session_id, interval_s, max_frames):
    """Reuse only a complete, intact extraction of the same inputs; never write labels."""
    from .core import Project
    output = Path(output)
    if not output.exists():
        return None
    try:
        project = Project(output)
        meta = project.meta
        digest, _ = bag_digest(Path(bag_path).resolve())
        if (meta['session_id'] != session_id or meta['source']['bag_sha256'] != digest
                or meta['source']['image_topic'] != image_topic
                or meta['sampling'] != dict(interval_s=interval_s, max_frames=max_frames)):
            raise ValueError('bag·토픽·세션·추출 설정이 기존 결과와 다릅니다')
        snapshots = {f'{MODEL_DIR}/{p.name}': sha256(p) for p in model_files(model_dir)}
        snapshots['calibration.yaml'] = sha256(calibration_path)
        if snapshots != meta['snapshot_sha256']:
            raise ValueError('모델 또는 캘리브레이션이 기존 결과와 다릅니다')
        if not project.frames:
            raise ValueError('추출된 프레임이 없습니다')
        for frame in project.frames.values():
            for folder in ('raw', 'bev'):
                if sha256(output/folder/(frame['id']+'.png')) != frame[folder+'_sha256']:
                    raise ValueError('추출 이미지가 변경되었습니다')
        return meta
    except (ValueError, OSError, KeyError) as exc:
        raise ValueError(f'기존 프로젝트를 재사용할 수 없습니다: {exc}. '
                         f'기존 라벨은 그대로 두었습니다. {output}을 다른 이름으로 옮긴 뒤 '
                         'prepare를 다시 실행하세요.') from exc


def extract_bag(bag_path, image_topic, model_dir, calibration_path, output,
                session_id, interval_s=.5, max_frames=1000):
    # ROS imports stay lazy: labeling, export and training need no ROS installation.
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', session_id):
        raise ValueError('session_id: 영문/숫자/_/- 1~80자만 사용하세요.')
    if not np.isfinite(interval_s) or interval_s < 0 or max_frames <= 0:
        raise ValueError('interval must be >= 0; max_frames must be > 0')
    bag_path, output = Path(bag_path).resolve(), Path(output)
    digest, metadata = bag_digest(bag_path)
    if metadata.get('compression_mode') not in (None, '', 'none'):
        raise ValueError('압축되지 않은 rosbag2를 사용하세요. 현재 compressed storage는 지원하지 않습니다.')
    manifest, cfg = read_manifest(model_dir), load_config(model_dir)
    mask = training_mask(cfg)
    pre = CameraPreprocessor(calibration_path, cfg, mask, 'rear_axle')
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=metadata['storage_identifier']),
                rosbag2_py.ConverterOptions('', ''))
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topics.get(image_topic) != 'sensor_msgs/msg/Image':
        raise ValueError(f'{image_topic}: sensor_msgs/msg/Image 토픽이 아닙니다. 토픽: {topics}')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[image_topic]))
    output.mkdir(parents=True, exist_ok=False)
    (output/MODEL_DIR).mkdir()
    for path in model_files(model_dir):
        shutil.copy2(path, output/MODEL_DIR/path.name)
    shutil.copy2(calibration_path, output/'calibration.yaml')
    for directory in ('raw', 'bev', 'annotations'):
        (output/directory).mkdir()
    snapshots = {str(p.relative_to(output)): sha256(p) for p in sorted((output/MODEL_DIR).iterdir())}
    snapshots['calibration.yaml'] = sha256(output/'calibration.yaml')
    project = dict(schema_version=2, complete=False, session_id=session_id, contract=contract(cfg),
        model=dict(sha256=manifest['sha256'], git_commit=manifest.get('git_commit'),
                   created_utc=manifest.get('created_utc')),
        calibration_status=pre.calibration_status, training_mask_sha256=mask_digest(mask),
        source=dict(bag_path=str(bag_path), bag_sha256=digest, image_topic=image_topic,
                    storage_id=metadata['storage_identifier']), snapshot_sha256=snapshots,
        sampling=dict(interval_s=interval_s, max_frames=max_frames), frames=[])
    write_json(output/'project.json', project)
    bridge = CvBridge()
    last_stamp = last_selected = None
    sequence = 0
    while reader.has_next() and len(project['frames']) < max_frames:
        _, serialized, recorded_ns = reader.read_next()
        message = deserialize_message(serialized, Image)
        stamp = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
        sequence += 1
        if stamp <= 0 or (last_stamp is not None and stamp <= last_stamp):
            raise ValueError(f'image timestamp is zero/duplicate/backward at message {sequence}; split or repair bag')
        last_stamp = stamp
        if last_selected is not None and stamp-last_selected < round(interval_s*1e9):
            continue
        if message.encoding.lower() not in ENCODINGS:
            raise ValueError(f'unsupported image encoding: {message.encoding}')
        raw = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        bev = pre.bev(raw)
        frame_id = f'{len(project["frames"]):06d}'
        for folder, image in [('raw',raw), ('bev',bev)]:
            if not cv2.imwrite(str(output/folder/f'{frame_id}.png'), image):
                raise OSError('image write failed')
        project['frames'].append(dict(id=frame_id, header_stamp_ns=str(stamp), bag_stamp_ns=str(recorded_ns),
            source_sequence=sequence, camera_frame=message.header.frame_id, encoding=message.encoding,
            raw_sha256=sha256(output/'raw'/f'{frame_id}.png'), bev_sha256=sha256(output/'bev'/f'{frame_id}.png')))
        last_selected = stamp
    if not project['frames']:
        raise ValueError('no frames extracted')
    project['complete'] = True
    write_json(output/'project.json', project)
    return project
