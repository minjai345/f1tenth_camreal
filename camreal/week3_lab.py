"""Week-3 notebook helpers: read a rosbag2 without ROS, make BEV frames, label them by clicking in Colab,
and write a week-2 style dataset (images/*.png + labels.csv) that camsim's DiskDataset reads.

notebooks/week3_bag_label_train.ipynb uses this. Bags are read with the pure-Python `rosbags` package,
so the same code runs on Colab, a laptop or the car.
"""
import base64
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from camreal.checkpoint import training_mask
from camreal.labeling.core import LABEL_HEADER, ring_point
from camreal.preprocessing import CameraPreprocessor

IMAGE_TOPIC = '/flir_camera/image_raw'
MIN_ACCEPTED = 5          # the notebook keeps 20 % of the dataset for validation: at least one frame
SAMPLE = Path(__file__).resolve().parent/'sample'/'week3'
LABELER_JS = Path(__file__).resolve().parent/'week3_labeler.js'
# Same conversions cv_bridge uses (OpenCV names Bayer patterns one row down).
BAYER = {'bayer_rggb8': cv2.COLOR_BayerBG2BGR, 'bayer_bggr8': cv2.COLOR_BayerRG2BGR,
         'bayer_gbrg8': cv2.COLOR_BayerGR2BGR, 'bayer_grbg8': cv2.COLOR_BayerGB2BGR}
COLOUR = {'bgr8': (3, None), 'rgb8': (3, cv2.COLOR_RGB2BGR), 'bgra8': (4, cv2.COLOR_BGRA2BGR),
          'rgba8': (4, cv2.COLOR_RGBA2BGR), 'mono8': (1, cv2.COLOR_GRAY2BGR)}


def to_bgr(msg):
    """sensor_msgs/Image (as deserialized by rosbags) -> uint8 BGR image."""
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    data = np.asarray(msg.data, np.uint8)[:h * step].reshape(h, step)
    if msg.encoding in BAYER:
        return cv2.cvtColor(np.ascontiguousarray(data[:, :w]), BAYER[msg.encoding])
    if msg.encoding in COLOUR:
        channels, code = COLOUR[msg.encoding]
        image = np.ascontiguousarray(data[:, :w * channels]).reshape(h, w, channels) if channels > 1 else data[:, :w]
        return image.copy() if code is None else cv2.cvtColor(np.ascontiguousarray(image), code)
    raise ValueError(f'{msg.encoding}: 지원하지 않는 영상 인코딩입니다 (bayer_*8, bgr8, rgb8, bgra8, rgba8, mono8).')


def _reader(bag):
    """rosbags Reader for a bag folder, or for one .db3/.mcap file of it (Colab's file panel uploads files, not folders)."""
    from rosbags.rosbag2 import Reader
    path = Path(bag)
    if not path.exists():
        raise ValueError(f'{path}: 없는 경로입니다. Colab 파일 창에 올린 파일은 /content/ 아래에 있습니다 (예: /content/my_run_0.db3).')
    if path.is_dir() and not (path/'metadata.yaml').is_file():
        raise ValueError(f'{path}: metadata.yaml이 없는 폴더입니다. bag 폴더나 그 안의 .db3 파일 하나를 지정하세요.')
    return Reader(path)


def bag_summary(bag):
    """Files, length and topics of a bag (folder or one .db3 file), the same facts `ros2 bag info` prints."""
    path = Path(bag)
    with _reader(path) as reader:
        return dict(files=sorted(p.name for p in path.iterdir()) if path.is_dir() else [path.name],
                    duration_s=reader.duration / 1e9,
                    topics=[dict(topic=c.topic, type=c.msgtype, count=c.msgcount) for c in reader.connections])


def sampler(every_s):
    """keep(stamp_ns) -> True for the first stamp at or after each every_s step.

    A frame may come a little early (jitter): up to 10 % of the step, but never more than half the camera's
    frame interval seen so far. So a 40 Hz recording keeps the 0.5 s, 1.0 s, ... images and a bag already
    recorded at 2 Hz keeps every image.
    """
    step = int(round(every_s * 1e9))
    if step <= 0:
        raise ValueError('every_s는 0보다 커야 합니다.')
    due = previous = None
    interval = step

    def keep(stamp):
        nonlocal due, previous, interval
        if previous is not None:
            interval = min(interval, stamp - previous)
        previous = stamp
        slack = min(step // 10, interval // 2)
        if due is not None and stamp < due - slack:
            return False
        due = stamp + step if due is None else due + step
        while due - slack <= stamp:               # skip steps a gap in the recording already passed
            due += step
        return True
    return keep


def read_frames(bag, every_s=.5, topic=IMAGE_TOPIC):
    """One image per every_s seconds of camera time (header stamp), as [{'stamp_ns', 'bgr'}].

    An image whose stamp is not later than the one before (a repeat, or the clock jumping back) is skipped and counted.
    """
    from rosbags.typesys import Stores, get_typestore
    store, keep = get_typestore(Stores.ROS2_HUMBLE), sampler(every_s)
    frames, last, skipped = [], None, 0
    with _reader(bag) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise ValueError(f'{topic} 토픽이 bag에 없습니다. 있는 토픽: {sorted({c.topic for c in reader.connections})}')
        for connection, _, raw in reader.messages(connections=connections):
            msg = store.deserialize_cdr(raw, connection.msgtype)
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            if last is not None and stamp <= last:
                skipped += 1
                continue
            last = stamp
            if keep(stamp):
                frames.append(dict(stamp_ns=stamp, bgr=to_bgr(msg)))
    if skipped:
        print(f'영상 시각(header stamp)이 앞 영상과 같거나 거꾸로 간 메시지 {skipped}개는 건너뜀.')
    if not frames:
        raise ValueError(f'{topic}에 영상 메시지가 없습니다.')
    return frames


def make_bevs(frames, calibration, cfg):
    """The model's BEV input for each frame, made exactly as on the car (camreal CameraPreprocessor)."""
    pre = CameraPreprocessor(calibration, cfg, training_mask(cfg), 'rear_axle')
    return [pre.bev(f['bgr']) for f in frames]


def png_url(image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('PNG 인코딩 실패')
    return 'data:image/png;base64,' + base64.b64encode(data.tobytes()).decode()


class Labeler:
    """Click labels for the BEVs of read_frames' frames: one point on the ahead_m circle, or 'rejected'.

    Saved to JSON on every change with the frames' stamps: running the cell again keeps the clicks,
    another bag (or another every_s) starts fresh.
    """

    def __init__(self, frames, bevs, cfg, path):
        if len(frames) != len(bevs):
            raise ValueError(f'프레임 {len(frames)}장과 BEV {len(bevs)}장의 수가 다릅니다. BEV 셀부터 다시 실행하세요.')
        self.bevs, self.path = bevs, Path(path)
        self.stamps = [int(f['stamp_ns']) for f in frames]
        self.ahead_m = float(cfg.waypoints.ahead_m)
        self.bev = dict(x_range_m=list(cfg.bev.x_range_m), y_range_m=list(cfg.bev.y_range_m),
                        resolution_m=float(cfg.bev.resolution_m))
        self.labels = [dict(status='unlabeled', waypoint_m=None) for _ in bevs]
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if isinstance(saved, dict) and saved.get('stamps') == self.stamps:
                self.labels = saved['labels']

    def counts(self):
        statuses = [label['status'] for label in self.labels]
        return {s: statuses.count(s) for s in ('accepted', 'rejected', 'unlabeled')}

    def frame(self, index):
        index = self._index(index)
        return dict(index=index, n=len(self.bevs), image=png_url(self.bevs[index]), label=self.labels[index],
                    counts=self.counts(), statuses=[label['status'] for label in self.labels])

    def save(self, index, status, x=None, y=None):
        index = self._index(index)
        if status not in ('accepted', 'rejected'):
            raise ValueError("status는 'accepted' 또는 'rejected'여야 합니다.")
        point = ring_point([x, y], self.ahead_m, self.bev).tolist() if status == 'accepted' else None
        self.labels[index] = dict(status=status, waypoint_m=point)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(stamps=self.stamps, labels=self.labels), indent=1))
        return dict(label=self.labels[index], counts=self.counts(), statuses=[label['status'] for label in self.labels])

    def _index(self, index):
        index = int(index)
        if not 0 <= index < len(self.bevs):
            raise ValueError(f'프레임 번호는 0~{len(self.bevs) - 1}입니다.')
        return index

    def html(self, prefix='camreal.week3.'):
        config = dict(n=len(self.bevs), ahead_m=self.ahead_m, bev=self.bev, scale=2, prefix=prefix)
        return f"""<div id="w3"><style>
#w3{{font:15px system-ui,sans-serif}} #w3 .bar{{display:flex;gap:8px;align-items:center;margin:6px 0;flex-wrap:wrap}}
#w3 button{{font:inherit;padding:6px 12px;border:1px solid #8a97ad;border-radius:5px;background:#eef2f8;cursor:pointer}}
#w3 .accept{{background:#d4f0e2}} #w3 .reject{{background:#f6dcdc}} #w3 canvas{{border:1px solid #8a97ad;cursor:crosshair;touch-action:none}}
#w3 #msg{{color:#9a5a00;min-height:22px}}</style>
<div class="bar"><button id="prev">← 이전</button><b id="pos"></b><button id="next">다음 →</button>
<button id="todo">다음 미작업</button><span id="counts"></span></div>
<canvas id="cv"></canvas>
<div class="bar"><button id="accept" class="accept">승인 (Enter)</button><button id="reject" class="reject">제외 (X)</button>
<span>이 프레임: <b id="state"></b></span></div><div id="msg"></div></div>
<script>const CONFIG={json.dumps(config)};
async function call(name,args){{const r=await google.colab.kernel.invokeFunction(CONFIG.prefix+name,args,{{}});
const d=r.data['application/json'];if(d.error)throw Error(d.error);return d}}</script>
<script>{LABELER_JS.read_text()}</script>"""

    def show(self, prefix='camreal.week3.'):
        """Colab only: register the page's two callbacks and draw the labeler in this cell's output."""
        from IPython.display import HTML, display
        try:
            from google.colab import output
        except ImportError:
            print('클릭 라벨링 화면은 Colab에서만 뜹니다 (google.colab 없음). Colab에서 이 노트북을 여세요.')
            return
        output.register_callback(prefix + 'frame', _answer(self.frame))
        output.register_callback(prefix + 'save', _answer(self.save))
        display(HTML(self.html(prefix)))


def _answer(method):
    """method as a Colab callback: its result as JSON, or {'error': message} the page shows for any exception."""
    from IPython.display import JSON

    def callback(*args):
        try:
            return JSON(method(*args))
        except ValueError as exc:                  # our own checks, written for the student
            return JSON(dict(error=str(exc)))
        except Exception as exc:                   # anything else: the page cannot show a traceback, so name it
            return JSON(dict(error=f'{type(exc).__name__}: {exc}'))
    return callback


def write_dataset(bevs, labels, out_dir, prefix='frame'):
    """Accepted frames -> out_dir/images/*.png + out_dir/labels.csv, the folder camsim's DiskDataset reads.

    No car pose on a real recording, so x, y, theta are NaN (as camreal export writes them); training uses wp_x, wp_y.
    """
    if len(labels) != len(bevs):
        raise ValueError(f'BEV {len(bevs)}장과 라벨 {len(labels)}개의 수가 다릅니다. 라벨링 셀부터 다시 실행하세요.')
    accepted = [i for i, label in enumerate(labels) if label['status'] == 'accepted']
    if len(accepted) < MIN_ACCEPTED:
        raise ValueError(f'승인한 프레임이 {len(accepted)}장입니다. 20%를 검증용으로 떼고 학습하려면 '
                         f'{MIN_ACCEPTED}장 이상 점을 찍고 승인(Enter)하세요.')
    out = Path(out_dir)
    if out.exists():
        if any(p.name not in ('images', 'labels.csv') for p in out.iterdir()):
            raise ValueError(f'{out}에 데이터셋이 아닌 파일이 있어 지우지 않았습니다. 다른 폴더 이름을 쓰세요.')
        shutil.rmtree(out)
    (out/'images').mkdir(parents=True)
    rows = []
    for i in accepted:
        name, (x, y) = f'{prefix}_{i:04d}.png', labels[i]['waypoint_m']
        cv2.imwrite(str(out/'images'/name), bevs[i])
        rows.append([name, 'nan', 'nan', 'nan', f'{x:.4f}', f'{y:.4f}'])
    with (out/'labels.csv').open('w', newline='') as stream:
        csv.writer(stream, lineterminator='\n').writerows([LABEL_HEADER, *rows])
    return len(rows)
