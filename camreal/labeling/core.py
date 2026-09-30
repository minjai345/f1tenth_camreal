"""Versioned student labels (one waypoint on the ahead_m ring) and camsim-format export."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
import shutil
import numpy as np
from camsim.handoff import sha256_file as sha256
from camreal.checkpoint import contract, load_config

MODEL_DIR = 'model'   # project snapshot of the camsim hand-off folder
# Same as camsim.dataset.LABEL_HEADER (checked in tests); importing it would pull torch into the editor.
LABEL_HEADER = ['file', 'x', 'y', 'theta', 'wp_x', 'wp_y']


class ConflictError(ValueError):
    pass


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def ring_point(point, ahead_m, bev):
    """Put a click on the ahead_m circle around the rear axle: the real-car label definition.

    Sim GT is ahead_m of arc length along the centre line. On the car the lane centre on
    this circle is used instead; the two differ by a few cm, mostly along the lane.
    """
    p = np.asarray(point, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError('waypoint는 유한한 (x, y) 미터 좌표여야 합니다.')
    r = float(np.hypot(*p))
    if r < 1e-6:
        raise ValueError('후륜축에서 떨어진 곳을 클릭하세요.')
    wp = p * (float(ahead_m) / r)
    if not (bev['x_range_m'][0] <= wp[0] <= bev['x_range_m'][1] and bev['y_range_m'][0] <= wp[1] <= bev['y_range_m'][1]):
        raise ValueError('waypoint가 BEV 영상 범위 밖입니다. 차선 중앙이 안 보이면 이 프레임은 제외하세요.')
    return wp


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.meta = json.loads((self.root/'project.json').read_text())
        if self.meta.get('schema_version') != 2 or not self.meta.get('complete'):
            raise ValueError('추출이 끝나지 않았거나 이전 버전(waypoint 6개) 프로젝트입니다.')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', self.meta['session_id']):
            raise ValueError('invalid session_id')
        for relative, digest in self.meta['snapshot_sha256'].items():
            path = (self.root/relative).resolve()
            if not path.is_relative_to(self.root) or sha256(path) != digest:
                raise ValueError(f'프로젝트 snapshot 변경: {relative}')
        self.cfg = load_config(self.root/MODEL_DIR)
        if self.meta['contract'] != contract(self.cfg):
            raise ValueError('프로젝트의 모델 설정이 변경되었습니다.')
        self.ahead_m = float(self.cfg.waypoints.ahead_m)
        self.frames = {f['id']: f for f in self.meta['frames']}
        if len(self.frames) != len(self.meta['frames']) or any(not re.fullmatch(r'\d{6}', f) for f in self.frames):
            raise ValueError('invalid or duplicate frame IDs')

    def annotation(self, frame_id):
        if frame_id not in self.frames:
            raise KeyError(frame_id)
        path = self.root/'annotations'/f'{frame_id}.json'
        if path.exists():
            return json.loads(path.read_text())
        return dict(frame_id=frame_id, revision=0, status='unlabeled', waypoint_m=None, note='')

    def save(self, frame_id, payload):
        current = self.annotation(frame_id)
        if payload.get('revision') != current['revision']:
            raise ConflictError('다른 탭에서 수정되었습니다. 프레임을 다시 불러오세요.')
        status = payload.get('status')
        if status not in ('accepted', 'rejected'):
            raise ValueError('invalid annotation status')
        wp = None
        if status == 'accepted':
            if payload.get('waypoint_m') is None:
                raise ValueError(f'승인하려면 {self.ahead_m:g} m 원 위에 waypoint를 찍으세요.')
            wp = ring_point(payload['waypoint_m'], self.ahead_m, self.meta['contract']['bev'])
        record = dict(frame_id=frame_id, revision=current['revision'] + 1, status=status,
                      note=str(payload.get('note', ''))[:2000],
                      updated_utc=datetime.now(timezone.utc).isoformat(), ahead_m=self.ahead_m,
                      waypoint_m=None if wp is None else wp.tolist())
        history = self.root/'annotation_history'/frame_id
        history.mkdir(parents=True, exist_ok=True)
        (self.root/'annotations').mkdir(exist_ok=True)
        # Archive every saved revision, including rejections.
        write_json(history/f'{record["revision"]:06d}.json', record)
        write_json(self.root/'annotations'/f'{frame_id}.json', record)
        return record

    def counts(self):
        statuses = [self.annotation(f)['status'] for f in self.frames]
        return {s: statuses.count(s) for s in ('accepted', 'rejected', 'unlabeled')}


def export_dataset(train_projects, val_projects, output):
    """Each split becomes a camsim DiskDataset folder (images/ + labels.csv).

    Splits are whole recording sessions; adjacent frames of one drive never cross splits.
    """
    groups = {'train': [Project(p) for p in train_projects], 'val': [Project(p) for p in val_projects]}
    if not groups['train'] or not groups['val']:
        raise ValueError('train과 val에는 서로 다른 기록 세션이 최소 하나씩 필요합니다.')
    projects = groups['train'] + groups['val']
    base = projects[0]
    sessions, bags = set(), set()
    for p in projects:
        session, bag_id = p.meta['session_id'], p.meta['source']['bag_sha256']
        if session in sessions or bag_id in bags:
            raise ValueError('동일 session 또는 rosbag을 중복/교차 split에 사용할 수 없습니다.')
        if p.meta['contract'] != base.meta['contract']:
            raise ValueError(f'{session}: 다른 BEV/waypoint 규격의 모델로 준비된 세션입니다.')
        if p.meta['training_mask_sha256'] != base.meta['training_mask_sha256']:
            raise ValueError(f'{session}: 학습 카메라 가시 영역이 다른 모델로 준비된 세션입니다.')
        sessions.add(session)
        bags.add(bag_id)
    bev = base.meta['contract']['bev']
    accepted = {}
    for split, items in groups.items():
        rows = []
        for p in items:
            for frame_id, frame in p.frames.items():
                label = p.annotation(frame_id)
                if label['status'] != 'accepted':
                    continue
                wp = ring_point(label['waypoint_m'], p.ahead_m, bev)
                image = p.root/'bev'/f'{frame_id}.png'
                if sha256(image) != frame['bev_sha256']:
                    raise ValueError(f'BEV image changed: {image}')
                rows.append((p, frame_id, wp, image))
        if not rows:
            raise ValueError(f'{split} 세션에 승인된 라벨이 없습니다.')
        accepted[split] = rows
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=2, format='camsim DiskDataset: images/*.png + labels.csv',
                    contract=base.meta['contract'], split_unit='recording_session',
                    world_pose_available=False, counts={}, sources=[])
    for split, rows in accepted.items():
        (output/split/'images').mkdir(parents=True)
        with (output/split/'labels.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(LABEL_HEADER)
            for p, frame_id, wp, image in rows:
                name = f'{p.meta["session_id"]}_{frame_id}.png'
                shutil.copy2(image, output/split/'images'/name)
                # No world pose on the car: explicit NaN, never an invented odometry pose.
                writer.writerow([name, 'nan', 'nan', 'nan', f'{wp[0]:.4f}', f'{wp[1]:.4f}'])
        manifest['counts'][split] = len(rows)
        manifest['sources'].extend(dict(split=split, session_id=p.meta['session_id'], source=p.meta['source'],
                                        model=p.meta['model'], calibration_status=p.meta['calibration_status'],
                                        labels=p.counts()) for p in groups[split])
    manifest['label_sha256'] = {s: sha256(output/s/'labels.csv') for s in accepted}
    write_json(output/'dataset.json', manifest)
    return manifest
