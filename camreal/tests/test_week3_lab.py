"""camreal.week3_lab on the sample bag (28 images at 2 Hz) and on synthetic messages."""
import csv
import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from camsim import config

from camreal import week3_lab as lab
from camreal.checkpoint import training_mask

BAG = lab.SAMPLE/'run_train2_part1_2hz'
DB3 = BAG/'run_train2_part1_2hz_0.db3'
PAGE = Path(__file__).with_name('week3_labeler.js')
SCENARIOS = re.findall(r'^  async (\w+)\(\)\{$', PAGE.read_text(), re.M)
MS = 1_000_000
ACCEPTED, REJECTED = dict(status='accepted', waypoint_m=[1.0, 0.0]), dict(status='rejected', waypoint_m=None)


@pytest.fixture(scope='module')
def frames():
    pytest.importorskip('rosbags')
    return lab.read_frames(BAG)


@pytest.fixture(scope='module')
def bevs(frames):
    return lab.make_bevs(frames, lab.SAMPLE/'car.yaml', config.load())


def write_bag(path, stamps_ns):
    """A rosbag2 of 8x6 mono8 images, image i filled with value i, header stamps as given, received in this order."""
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore
    store = get_typestore(Stores.ROS2_HUMBLE)
    image, header, time = (store.types[name] for name in
                           ('sensor_msgs/msg/Image', 'std_msgs/msg/Header', 'builtin_interfaces/msg/Time'))
    with Writer(path, version=9) as writer:
        connection = writer.add_connection(lab.IMAGE_TOPIC, 'sensor_msgs/msg/Image', typestore=store)
        for i, stamp in enumerate(stamps_ns):
            msg = image(header=header(stamp=time(sec=stamp // 10**9, nanosec=stamp % 10**9), frame_id='cam'),
                        height=6, width=8, encoding='mono8', is_bigendian=0, step=8, data=np.full(48, i, np.uint8))
            writer.write(connection, stamps_ns[0] + i * 25 * MS, store.serialize_cdr(msg, 'sensor_msgs/msg/Image'))
    return path


def test_sample_folder_has_the_bag_and_both_calibrations():
    assert (BAG/'metadata.yaml').is_file() and (lab.SAMPLE/'car.yaml').is_file() and (lab.SAMPLE/'ost.yaml').is_file()
    assert all(p.stat().st_size < 100 * 2**20 for p in lab.SAMPLE.rglob('*') if p.is_file())   # GitHub's file limit


def test_bag_summary_matches_ros2_bag_info():
    pytest.importorskip('rosbags')
    info = lab.bag_summary(BAG)
    counts = {t['topic']: (t['type'], t['count']) for t in info['topics']}
    assert counts == {'/flir_camera/image_raw': ('sensor_msgs/msg/Image', 28),
                      '/flir_camera/camera_info': ('sensor_msgs/msg/CameraInfo', 30)}
    assert 14 < info['duration_s'] < 15
    assert info['files'] == ['metadata.yaml', 'run_train2_part1_2hz_0.db3']


def test_one_db3_file_reads_like_its_bag_folder():      # Colab's file panel uploads files, not folders
    pytest.importorskip('rosbags')
    info = lab.bag_summary(DB3)
    assert info['files'] == ['run_train2_part1_2hz_0.db3']
    assert {t['topic']: t['count'] for t in info['topics']} == {'/flir_camera/image_raw': 28, '/flir_camera/camera_info': 30}
    assert len(lab.read_frames(DB3)) == 28


def test_wrong_bag_paths_are_value_errors(tmp_path):
    pytest.importorskip('rosbags')
    with pytest.raises(ValueError, match='/content/'):
        lab.bag_summary(tmp_path/'my_run_0.db3')                # not uploaded yet, or a path relative to the repo
    (tmp_path/'my_run').mkdir()
    (tmp_path/'my_run'/'my_run_0.db3').write_bytes(b'')
    with pytest.raises(ValueError, match='metadata.yaml'):
        lab.read_frames(tmp_path/'my_run')                       # a folder holding only the .db3


def test_every_frame_of_the_2hz_bag_is_kept(frames):
    stamps = [f['stamp_ns'] for f in frames]
    assert len(frames) == 28 and frames[0]['bgr'].shape == (1200, 1920, 3) and frames[0]['bgr'].dtype == np.uint8
    assert all(b > a for a, b in zip(stamps, stamps[1:]))


def test_frames_whose_stamp_does_not_move_forward_are_skipped(tmp_path, capsys):
    pytest.importorskip('rosbags')
    t = 1_791_268_815 * 10**9
    bag = write_bag(tmp_path/'bag', [t, t + 500 * MS, t + 500 * MS, t + 300 * MS, t + 1000 * MS])   # a repeat, a jump back
    frames = lab.read_frames(bag, every_s=.5)
    assert [f['stamp_ns'] - t for f in frames] == [0, 500 * MS, 1000 * MS]
    assert [int(f['bgr'][0, 0, 0]) for f in frames] == [0, 1, 4]
    assert '2개' in capsys.readouterr().out                     # the student sees that two images were dropped


def test_sampler_keeps_one_image_per_step():
    def kept(stamps, every_s=.5):
        keep = lab.sampler(every_s)
        return [i for i, s in enumerate(stamps) if keep(s)]
    forty_hz = [i * 25 * MS for i in range(80)]                          # 2 s at 40 Hz
    assert kept(forty_hz) == [0, 20, 40, 60]
    two_hz = [i * 500 * MS + (3 * MS if i % 2 else -2 * MS) for i in range(6)]   # 2 Hz with jitter
    assert kept(two_hz) == list(range(6))
    with_gap = [0, 500 * MS, 2600 * MS, 2650 * MS, 3100 * MS]           # 2.1 s gap, then on again
    assert kept(with_gap) == [0, 1, 2, 4]
    with pytest.raises(ValueError):
        lab.sampler(0)


def test_to_bgr_matches_cv_bridge_conversions():
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (4, 6, 3), np.uint8)
    msg = lambda enc, data, step: SimpleNamespace(height=4, width=6, step=step, encoding=enc, data=data.ravel())
    assert np.array_equal(lab.to_bgr(msg('bgr8', bgr, 18)), bgr)
    assert np.array_equal(lab.to_bgr(msg('rgb8', bgr[..., ::-1].copy(), 18)), bgr)
    mono = bgr[..., 0].copy()
    assert np.array_equal(lab.to_bgr(msg('mono8', mono, 6)), cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR))
    bayer = rng.integers(0, 256, (4, 6), np.uint8)
    assert np.array_equal(lab.to_bgr(msg('bayer_rggb8', bayer, 6)), cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR))
    padded = np.zeros((4, 8), np.uint8)
    padded[:, :6] = bayer                                             # rows padded to step 8
    assert np.array_equal(lab.to_bgr(msg('bayer_rggb8', padded, 8)), cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR))
    with pytest.raises(ValueError, match='yuv422'):
        lab.to_bgr(msg('yuv422', bayer, 6))


def test_bevs_are_the_models_input_size(bevs):
    cfg = config.load()
    assert len(bevs) == 28 and bevs[0].shape == (380, 300, 3) and bevs[0].dtype == np.uint8
    hidden = ~training_mask(cfg)                                         # what the training camera never saw
    assert hidden.any() and (bevs[0][hidden] == np.array(cfg.lane.color_floor)).all()


def test_labels_snap_to_the_circle(frames, bevs, tmp_path):
    labeler = lab.Labeler(frames[:3], bevs[:3], config.load(), tmp_path/'labels.json')
    assert labeler.save(0, 'accepted', 1.0, 0.0)['label'] == dict(status='accepted', waypoint_m=[1.0, 0.0])
    labeler.save(1, 'rejected')
    second = labeler.save(2, 'accepted', .5, .5)
    assert np.allclose(second['label']['waypoint_m'], [2**-.5, 2**-.5])
    assert second['counts'] == dict(accepted=2, rejected=1, unlabeled=0)


def test_saved_labels_come_back_only_for_the_same_frames(frames, bevs, tmp_path):
    cfg, path = config.load(), tmp_path/'labels.json'
    labeler = lab.Labeler(frames[:3], bevs[:3], cfg, path)
    labeler.save(0, 'accepted', 1.0, 0.0)
    labeler.save(1, 'rejected')
    again = lab.Labeler(frames[:3], bevs[:3], cfg, path)               # the cell is run again
    assert again.labels == [ACCEPTED, REJECTED, dict(status='unlabeled', waypoint_m=None)]
    other_bag = [dict(f, stamp_ns=f['stamp_ns'] + 1) for f in frames[:3]]   # same number of frames, other images
    assert lab.Labeler(other_bag, bevs[:3], cfg, path).counts() == dict(accepted=0, rejected=0, unlabeled=3)
    with pytest.raises(ValueError):
        lab.Labeler(frames[:2], bevs[:3], cfg, path)                   # BEVs of other frames


def test_labeler_refuses_bad_clicks(frames, bevs, tmp_path):
    labeler = lab.Labeler(frames[:2], bevs[:2], config.load(), tmp_path/'labels.json')
    for args in [(5, 'accepted', 1, 0), (0, 'maybe'), (0, 'accepted', 0, 0), (0, 'accepted', -1, 0)]:
        with pytest.raises(ValueError):
            labeler.save(*args)
    assert not (tmp_path/'labels.json').exists()


def test_frame_payload_for_the_page(frames, bevs, tmp_path):
    page = lab.Labeler(frames[:2], bevs[:2], config.load(), tmp_path/'labels.json').frame(1)
    assert page['index'] == 1 and page['n'] == 2 and page['statuses'] == ['unlabeled', 'unlabeled']
    assert page['image'].startswith('data:image/png;base64,') and page['label']['status'] == 'unlabeled'


def test_show_registers_the_callbacks_the_page_calls(frames, bevs, tmp_path, monkeypatch):
    pytest.importorskip('IPython')
    registered, shown = {}, []
    colab = types.ModuleType('google.colab')
    colab.output = SimpleNamespace(register_callback=registered.__setitem__)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    monkeypatch.setattr('IPython.display.display', shown.append)
    labeler = lab.Labeler(frames[:2], bevs[:2], config.load(), tmp_path/'labels.json')
    labeler.show('t.')
    assert sorted(registered) == ['t.frame', 't.save'] and len(shown) == 1   # the page calls prefix + frame/save
    assert registered['t.frame'](1).data['index'] == 1
    assert registered['t.save'](0, 'accepted', 1.0, 0.0).data['label'] == ACCEPTED
    assert registered['t.save'](0, 'maybe').data == {'error': "status는 'accepted' 또는 'rejected'여야 합니다."}
    labeler.bevs[1] = np.zeros((0, 0, 3), np.uint8)                   # an error nobody planned for (cv2.error here)
    assert list(registered['t.frame'](1).data) == ['error']           # reaches the page as a message, not a hang


def test_dataset_is_what_week2_reads(frames, bevs, tmp_path):
    torch = pytest.importorskip('torch')
    from camsim import dataset
    cfg = config.load()
    labeler = lab.Labeler(frames[:6], bevs[:6], cfg, tmp_path/'labels.json')
    for i in (0, 2, 4, 5):
        labeler.save(i, 'accepted', 1.0, 0.0)
    labeler.save(1, 'rejected')
    labeler.save(3, 'accepted', .6, .8)
    out = tmp_path/'week3_real'
    assert lab.write_dataset(bevs[:6], labeler.labels, out, 'run') == 5
    rows = list(csv.reader((out/'labels.csv').open()))
    assert rows[:4] == [['file', 'x', 'y', 'theta', 'wp_x', 'wp_y'], ['run_0000.png', 'nan', 'nan', 'nan', '1.0000', '0.0000'],
                        ['run_0002.png', 'nan', 'nan', 'nan', '1.0000', '0.0000'],
                        ['run_0003.png', 'nan', 'nan', 'nan', '0.6000', '0.8000']]
    data = dataset.DiskDataset(str(out), cfg, 'all')
    image, target = data[2]
    assert len(data) == 5 and tuple(image.shape) == (3, 380, 300) and isinstance(target, torch.Tensor)
    assert np.allclose(target.numpy() * cfg.waypoints.norm_m, [.6, .8], atol=1e-4)
    assert lab.write_dataset(bevs[:6], labeler.labels, out, 'run') == 5      # running the cell again rebuilds it


@pytest.mark.parametrize('accepted', [0, 2])
def test_too_few_accepted_labels_leave_nothing_to_validate_on(bevs, tmp_path, accepted):
    labels = [ACCEPTED] * accepted + [REJECTED] * (8 - accepted)
    with pytest.raises(ValueError):
        lab.write_dataset(bevs[:8], labels, tmp_path/'week3_real')
    assert not (tmp_path/'week3_real').exists()


def test_five_accepted_labels_give_the_notebook_a_validation_frame(bevs, tmp_path):
    pytest.importorskip('torch')
    from camsim import dataset
    out = tmp_path/'week3_real'
    assert lab.write_dataset(bevs[:8], [ACCEPTED] * 5 + [REJECTED] * 3, out) == 5
    assert len(dataset.DiskDataset(str(out), config.load(), 'val', val_frac=0.2)) >= 1    # the notebook's 20 % split


def test_dataset_refuses_labels_of_other_frames(bevs, tmp_path):
    out = tmp_path/'week3_real'
    assert lab.write_dataset(bevs[:6], [ACCEPTED] * 6, out) == 6
    with pytest.raises(ValueError):
        lab.write_dataset(bevs[:7], [ACCEPTED] * 6, out)                 # labels left from a shorter run
    assert len(list((out/'images').iterdir())) == 6                      # the earlier dataset is still there


def test_dataset_keeps_foreign_folders(bevs, tmp_path):
    mine = tmp_path/'mine'
    mine.mkdir()
    (mine/'notes.txt').write_text('keep')
    with pytest.raises(ValueError, match='지우지 않았습니다'):
        lab.write_dataset(bevs[:6], [ACCEPTED] * 6, mine)
    assert (mine/'notes.txt').read_text() == 'keep'


def test_page_html_carries_the_config_and_the_script(frames, bevs, tmp_path):
    html = lab.Labeler(frames[:2], bevs[:2], config.load(), tmp_path/'l.json').html('x.')
    config_json = json.loads(re.search(r'const CONFIG=(\{.*?\});', html).group(1))
    assert config_json == dict(n=2, ahead_m=1.0, bev=dict(x_range_m=[.2, 4.], y_range_m=[-1.5, 1.5], resolution_m=.01),
                               scale=2, prefix='x.')
    assert lab.LABELER_JS.read_text() in html and 'id="cv"' in html


def test_every_page_scenario_is_found():
    assert len(SCENARIOS) == 9


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
@pytest.mark.parametrize('scenario', SCENARIOS)
def test_labeler_page(scenario):
    run = subprocess.run(['node', str(PAGE), scenario], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
