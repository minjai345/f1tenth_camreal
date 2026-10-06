"""camreal.week3_lab on the sample bag (28 images at 2 Hz) and on synthetic messages."""
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from camsim import config

from camreal import week3_lab as lab
from camreal.checkpoint import training_mask

BAG = lab.SAMPLE/'run_train2_part1_2hz'
PAGE = Path(__file__).with_name('week3_labeler.js')
SCENARIOS = re.findall(r'^  async (\w+)\(\)\{$', PAGE.read_text(), re.M)


@pytest.fixture(scope='module')
def bevs():
    pytest.importorskip('rosbags')
    return lab.make_bevs(lab.read_frames(BAG), lab.SAMPLE/'car.yaml', config.load())


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


def test_every_frame_of_the_2hz_bag_is_kept():
    pytest.importorskip('rosbags')
    frames = lab.read_frames(BAG)
    stamps = [f['stamp_ns'] for f in frames]
    assert len(frames) == 28 and frames[0]['bgr'].shape == (1200, 1920, 3) and frames[0]['bgr'].dtype == np.uint8
    assert all(b > a for a, b in zip(stamps, stamps[1:]))


def test_sampler_keeps_one_image_per_step():
    def kept(stamps, every_s=.5):
        keep = lab.sampler(every_s)
        return [i for i, s in enumerate(stamps) if keep(s)]
    ms = 1_000_000
    forty_hz = [i * 25 * ms for i in range(80)]                          # 2 s at 40 Hz
    assert kept(forty_hz) == [0, 20, 40, 60]
    two_hz = [i * 500 * ms + (3 * ms if i % 2 else -2 * ms) for i in range(6)]   # 2 Hz with jitter
    assert kept(two_hz) == list(range(6))
    with_gap = [0, 500 * ms, 2600 * ms, 2650 * ms, 3100 * ms]           # 2.1 s gap, then on again
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


def test_labels_snap_to_the_circle_and_survive_a_rerun(bevs, tmp_path):
    cfg, path = config.load(), tmp_path/'labels.json'
    labeler = lab.Labeler(bevs[:3], cfg, path)
    assert labeler.save(0, 'accepted', 1.0, 0.0)['label'] == dict(status='accepted', waypoint_m=[1.0, 0.0])
    labeler.save(1, 'rejected')
    second = labeler.save(2, 'accepted', .5, .5)
    assert np.allclose(second['label']['waypoint_m'], [2**-.5, 2**-.5])
    assert second['counts'] == dict(accepted=2, rejected=1, unlabeled=0)
    assert lab.Labeler(bevs[:3], cfg, path).labels == labeler.labels      # the cell can be run again
    assert lab.Labeler(bevs[:4], cfg, path).counts()['unlabeled'] == 4   # other frames: start fresh


def test_labeler_refuses_bad_clicks(bevs, tmp_path):
    labeler = lab.Labeler(bevs[:2], config.load(), tmp_path/'labels.json')
    for args in [(5, 'accepted', 1, 0), (0, 'maybe'), (0, 'accepted', 0, 0), (0, 'accepted', -1, 0)]:
        with pytest.raises(ValueError):
            labeler.save(*args)
    assert not (tmp_path/'labels.json').exists()


def test_frame_payload_for_the_page(bevs, tmp_path):
    page = lab.Labeler(bevs[:2], config.load(), tmp_path/'labels.json').frame(1)
    assert page['index'] == 1 and page['n'] == 2 and page['statuses'] == ['unlabeled', 'unlabeled']
    assert page['image'].startswith('data:image/png;base64,') and page['label']['status'] == 'unlabeled'


def test_dataset_is_what_week2_reads(bevs, tmp_path):
    torch = pytest.importorskip('torch')
    from camsim import dataset
    cfg = config.load()
    labeler = lab.Labeler(bevs[:4], cfg, tmp_path/'labels.json')
    labeler.save(0, 'accepted', 1.0, 0.0)
    labeler.save(1, 'rejected')
    labeler.save(3, 'accepted', .6, .8)
    out = tmp_path/'week3_real'
    assert lab.write_dataset(bevs[:4], labeler.labels, out, 'run') == 2
    rows = list(csv.reader((out/'labels.csv').open()))
    assert rows == [['file', 'x', 'y', 'theta', 'wp_x', 'wp_y'], ['run_0000.png', 'nan', 'nan', 'nan', '1.0000', '0.0000'],
                    ['run_0003.png', 'nan', 'nan', 'nan', '0.6000', '0.8000']]
    data = dataset.DiskDataset(str(out), cfg, 'all')
    image, target = data[1]
    assert len(data) == 2 and tuple(image.shape) == (3, 380, 300) and isinstance(target, torch.Tensor)
    assert np.allclose(target.numpy() * cfg.waypoints.norm_m, [.6, .8], atol=1e-4)
    assert lab.write_dataset(bevs[:4], labeler.labels, out, 'run') == 2      # running the cell again rebuilds it


def test_dataset_needs_an_accepted_label_and_keeps_foreign_folders(bevs, tmp_path):
    labels = [dict(status='rejected', waypoint_m=None)] * 2
    with pytest.raises(ValueError, match='승인한 라벨이 없습니다'):
        lab.write_dataset(bevs[:2], labels, tmp_path/'empty')
    assert not (tmp_path/'empty').exists()
    mine = tmp_path/'mine'
    mine.mkdir()
    (mine/'notes.txt').write_text('keep')
    with pytest.raises(ValueError, match='지우지 않았습니다'):
        lab.write_dataset(bevs[:2], [dict(status='accepted', waypoint_m=[1., 0.])] * 2, mine)
    assert (mine/'notes.txt').read_text() == 'keep'


def test_page_html_carries_the_config_and_the_script(bevs, tmp_path):
    html = lab.Labeler(bevs[:2], config.load(), tmp_path/'l.json').html('x.')
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
