import dataclasses
from datetime import datetime
import hashlib
import json
import math
import shutil
import subprocess
import sys
import cv2
import numpy as np
import pytest
import yaml
from camsim import camera, config
from camsim.render import bev_pixels
from camreal.calibration import core
from camreal.checkpoint import training_mask
from camreal.preprocessing import CameraPreprocessor
from conftest import OST, TEMPLATE, assumed, clicks

K_LINE = '  data: [983.23425, 0., 909.40739, 0., 974.13698, 573.79454, 0., 0., 1.]'
P_LINE = '  data: [784.36676, 0., 950.00529, 0., 0., 849.48163, 577.98068, 0., 0., 0., 1., 0.]'
# camera_calibration's lryaml really writes matrices as multi-line flow lists.
WRITTEN = OST.replace(K_LINE, '  data: [983.23425,   0.     , 909.40739,\n           0.     , 974.13698, 573.79454,\n'
                              '           0.     ,   0.     ,   1.     ]').replace(
    P_LINE, '  data: [784.36676,   0.     , 950.00529,   0.     ,\n           0.     , 849.48163, 577.98068,   0.     ,\n'
            '           0.     ,   0.     ,   1.     ,   0.     ]')


def write(tmp_path, text, name='ost.yaml'):
    path = tmp_path/name
    path.write_text(text)
    return path


@pytest.mark.parametrize('text', [OST, WRITTEN])
def test_read_ost_slide84(tmp_path, text):
    path = write(tmp_path, text)
    intr = core.read_ost(path)
    assert (intr.width, intr.height) == (1480, 1080)
    np.testing.assert_allclose(intr.K, [[983.23425, 0, 909.40739], [0, 974.13698, 573.79454], [0, 0, 1]])
    np.testing.assert_allclose(intr.D, [-0.364955, 0.124123, -0.003946, 0.000371, 0])
    np.testing.assert_allclose(intr.new_K, [[784.36676, 0, 950.00529], [0, 849.48163, 577.98068], [0, 0, 1]])
    assert intr.path == str(path.resolve()) and intr.sha256 == hashlib.sha256(text.encode()).hexdigest()
    with pytest.raises(FileNotFoundError):
        core.read_ost(tmp_path/'missing.yaml')


@pytest.mark.parametrize('p', [None, '[.nan, 0., 950., 0., 0., 849., 578., 0., 0., 0., 1., 0.]',
                               '[0., 0., 950., 0., 0., 849., 578., 0., 0., 0., 1., 0.]',
                               '[784., 0., 950., 0., 0., 849., 578., 0., 0., 0., 2., 0.]',
                               '[784., 0., 950., 0., 849., 578., 0., 0., 1.]'])
def test_new_K_falls_back_to_K_without_a_usable_P(tmp_path, p):
    text = OST.split('projection_matrix:')[0] if p is None else OST.replace(P_LINE, '  data: ' + p)
    intr = core.read_ost(write(tmp_path, text))
    np.testing.assert_array_equal(intr.new_K, intr.K)


@pytest.mark.parametrize('old,new,match', [
    ('distortion_model: plumb_bob', 'distortion_model: equidistant', 'plumb_bob'),
    ('distortion_model: plumb_bob', 'distortion_model: rational_polynomial', 'plumb_bob'),
    ('cols: 5\n  data: [-0.364955, 0.124123, -0.003946, 0.000371, 0.000000]',
     'cols: 4\n  data: [-0.364955, 0.124123, -0.003946, 0.000371]', 'distortion_coefficients'),
    ('0.000371, 0.000000]', '0.000371]', 'distortion_coefficients'),
    (K_LINE, '  data: [983.23425, 0., 909.40739, 0., 974.13698, 573.79454]', 'camera_matrix'),
    ('rows: 3\n  cols: 3\n  data: [983', 'rows: 3\n  cols: 4\n  data: [983', 'camera_matrix'),
    ('data: [983.23425,', 'data: [0.,', 'camera_matrix'),
    ('image_width: 1480\n', '', 'image_width'),
    ('image_height: 1080', 'image_height: 0', 'image_height'),
])
def test_read_ost_rejects_bad_files(tmp_path, old, new, match):
    assert old in OST
    with pytest.raises(ValueError, match=match):
        core.read_ost(write(tmp_path, OST.replace(old, new)))


def test_marker_template(markers):
    assert list(markers) == [r + c for r in 'ABCD' for c in '123']
    assert markers['A1'] == (.6, .4) and markers['B2'] == (1., 0.) and markers['D3'] == (2., -.4)


GOOD = {'M0': [.5, .4], 'M1': [.5, 0.], 'M2': [.5, -.4], 'M3': [.8, .4], 'M4': [.8, 0.], 'M5': [.8, -.4]}


@pytest.mark.parametrize('edit,match', [
    (lambda m: m.pop('M5'), '6'),
    (lambda m: m.update({'M 1': [3., 0.]}), 'id'),
    (lambda m: m.update({'x' * 17: [3., 0.]}), 'id'),
    (lambda m: m.update(M0=[float('nan'), 0.]), 'M0'),
    (lambda m: m.update(M0=[1., 2., 3.]), 'M0'),
    (lambda m: m.update(M0='1, 2'), 'M0'),
    (lambda m: m.update(M0=[0., .1]), 'M0'),
    (lambda m: m.update(M0=list(m['M4'])), 'M4'),
    (lambda m: m.update({k: [.5 + .1 * i, 0.] for i, k in enumerate(m)}), '한 줄'),   # no H from any clicks
    (lambda m: m.update({k: [.5 + .1 * i, 0.] for i, k in enumerate(list(m)[:5])}), '한 줄'),
])
def test_load_markers_validation(tmp_path, edit, match):
    m = {k: list(v) for k, v in GOOD.items()}
    assert core.load_markers(write(tmp_path, yaml.safe_dump({'markers': m}), 'good.yaml'))[0]['M4'] == (.8, 0.)
    edit(m)
    with pytest.raises(ValueError, match=match):
        core.load_markers(write(tmp_path, yaml.safe_dump({'markers': m}), 'markers.yaml'))


def test_load_markers_refuses_a_key_written_twice(tmp_path):
    text = TEMPLATE.read_text()
    path = write(tmp_path, text, 'markers.yaml')
    markers, sha256 = core.load_markers(path)
    assert markers['B2'] == (1., 0.) and sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    line = text.splitlines().index('  B2: [1.00, 0.00]') + 1
    path.write_text(text.replace('  B3:', '  B2: [1.5, 0.0]\n  B3:'))   # a pasted line; plain YAML keeps the last B2
    with pytest.raises(ValueError, match=f'B2.*{line}.*{line + 1}'):
        core.load_markers(path)


def test_yaml_syntax_errors_are_value_errors(tmp_path):
    with pytest.raises(ValueError, match='2~3번째 줄'):   # hand-edited markers.yaml: no traceback for the CLI
        core.load_markers(write(tmp_path, 'markers:\n  A1: [0.6, 0.4\n  A2: [0.6, 0.0]\n', 'markers.yaml'))
    with pytest.raises(ValueError, match='YAML'):
        core.read_ost(write(tmp_path, OST.replace('  rows: 3', '  rows: [3', 1)))


def test_read_image(tmp_path):
    image = np.random.default_rng(1).integers(0, 256, (12, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path/'frame.png'), image)
    np.testing.assert_array_equal(core.read_image(tmp_path/'frame.png'), image)
    with pytest.raises(ValueError):
        core.read_image(write(tmp_path, 'not an image', 'broken.png'))


def test_ost_kind_names_the_week1_file_and_the_reference(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    assert core.ost_kind('~/camera_calibration/ost.yaml') == core.ost_kind(tmp_path/'camera_calibration'/'ost.yaml') == '1주차 학생 파일'
    assert core.ost_kind(TEMPLATE.parent/'ost_reference_1920x1200.yaml') == '기준 파일'
    assert core.ost_kind(tmp_path/'ost.yaml') == '--ost로 지정한 파일'


def test_resolution_mismatch_names_both_sizes_and_remedies(ost, tmp_path, monkeypatch):
    intr = core.read_ost(ost)
    frame = np.zeros((1080, 1440, 3), np.uint8)
    monkeypatch.setattr(core, 'REFERENCE', tmp_path/'missing.yaml')   # section 7 not done: no reference file to offer
    with pytest.raises(ValueError) as error:
        core.check_resolution(frame, intr)
    for text in ('1440x1080', '1480x1080', '카메라 해상도', 'cameracalibrator'):
        assert text in str(error.value)
    assert '3)' not in str(error.value) and '--ost' not in str(error.value)
    reference = write(tmp_path, OST.replace('1480', '1920').replace('1080', '1200'), 'ost_reference_1920x1200.yaml')
    monkeypatch.setattr(core, 'REFERENCE', reference)
    with pytest.raises(ValueError, match=r'3\) 카메라 해상도를 1920x1200으로 바꾸고 --ost camreal/config/ost_reference_1920x1200\.yaml'):
        core.check_resolution(frame, intr)
    with pytest.raises(ValueError, match=r'3\) --ost camreal/config/ost_reference_1920x1200\.yaml'):
        core.check_resolution(np.zeros((1200, 1920, 3), np.uint8), intr)
    with pytest.raises(ValueError) as error:   # the reference already in use: remedy 1 is all it can offer
        core.check_resolution(frame, core.read_ost(reference))
    assert '1920x1200' in str(error.value) and '3)' not in str(error.value)
    with pytest.raises(ValueError, match='1440x1080'):
        core.undistort(frame, intr)
    core.check_resolution(np.zeros((1080, 1480, 3), np.uint8), intr)


def test_synthetic_round_trip_noise_and_wrong_clicks(ost, markers):
    cfg = assumed()
    H_g2i, H_i2g = camera.build(cfg)
    points = clicks(H_g2i, markers)
    assert len(points) == 12
    H = core.fit_homography(points, markers)
    assert np.linalg.norm(H) == pytest.approx(1.)
    ground = np.array([[.5, .3], [1.2, -.5], [2.5, .1], [3., 0.]])
    np.testing.assert_allclose(camera.project(H, camera.project(H_g2i, ground)), ground, atol=1e-4)
    # Pixels come from the camsim pinhole, so the pose needs its K as new_K.
    intr = dataclasses.replace(core.read_ost(ost), new_K=camera.intrinsics(cfg))
    result = core.evaluate(points, markers, intr)
    json.dumps(result, allow_nan=False)   # the server sends it as is
    assert result['n'] == 12 and result['savable'] and '저장 가능' in result['reason']
    assert max(result['errors_cm'].values()) < .05 and result['rms_cm'] < .05
    assert result['pose'] == pytest.approx(dict(x_m=.1, y_m=0., height_m=.2, pitch_deg=10., yaw_deg=0., roll_deg=0.), abs=1e-3)
    np.testing.assert_allclose(camera.project(np.array(result['H_i2g']), points['C2']), markers['C2'], atol=1e-4)
    # 1 px click noise: the 0.6 m and 1.0 m rows stay under 2 cm.
    rng = np.random.default_rng(0)
    noisy = {k: (np.array(p) + rng.normal(0, 1, 2)).tolist() for k, p in points.items()}
    errors = core.evaluate(noisy, markers, intr)['errors_cm']
    assert max(errors[k] for k in markers if markers[k][0] <= 1.) < 2.
    rest = {k: p for k, p in noisy.items() if k != 'B2'}   # leave-one-out: B2 judged by an H fitted without it
    expected = np.hypot(*(camera.project(core.fit_homography(rest, markers), noisy['B2']) - markers['B2']))
    assert core.loo_errors(noisy, markers)['B2'] == pytest.approx(expected) and expected > .001
    # The neighbour's cross clicked instead: both markers are far off and saving is refused.
    swapped = dict(points, B2=points['C2'], C2=points['B2'])
    result = core.evaluate(swapped, markers, intr)
    assert result['errors_cm']['B2'] > core.REJECT_CM and result['errors_cm']['C2'] > core.REJECT_CM
    assert not result['savable'] and 'B2' in result['reason'] and 'C2' in result['reason']


def test_fit_needs_enough_valid_points(ost, markers):
    points = clicks(camera.build(assumed())[0], markers)
    intr = dataclasses.replace(core.read_ost(ost), new_K=camera.intrinsics(assumed()))
    few = {k: points[k] for k in ('A1', 'A3', 'D1')}
    result = core.evaluate(few, markers, intr)
    assert result['H_i2g'] is None and result['pose'] is None and not result['savable'] and str(core.MIN_FIT) in result['reason']
    four = dict(few, D3=points['D3'])
    assert core.loo_errors(four, markers) == {}
    result = core.evaluate(four, markers, intr)
    assert result['H_i2g'] is not None and result['errors_cm'] == {} and result['rms_cm'] is None and not result['savable']
    five = dict(four, B2=points['B2'])
    result = core.evaluate(five, markers, intr)
    assert len(result['errors_cm']) == 5 and not result['savable'] and str(core.MIN_SAVE) in result['reason']
    with pytest.raises(ValueError, match='Z9'):
        core.fit_homography(dict(four, Z9=[1., 2.]), markers)
    with pytest.raises(ValueError, match='A1'):
        core.fit_homography(dict(four, A1=[float('nan'), 2.]), markers)
    with pytest.raises(ValueError, match='A1'):
        core.fit_homography(dict(four, A1=[1., 2., 3.]), markers)
    with pytest.raises(ValueError):
        core.fit_homography(few, markers)
    for line in (('A2', 'B2', 'C2', 'D2'), ('A1', 'A2', 'A3', 'B1')):   # no 4 markers with 3 off one line
        with pytest.raises(ValueError, match='한 줄'):
            core.fit_homography({k: points[k] for k in line}, markers)
    with pytest.raises(ValueError, match='한 줄'):   # 3 clicks on one pixel row for markers that are not: singular H
        core.fit_homography(dict(zip(('A1', 'A3', 'D1', 'D3'), ([100., 500.], [200., 500.], [300., 500.], [250., 700.]))), markers)
    # Without B1 or C1 the rest has 3 markers on row A: those two cannot be checked.
    result = core.evaluate({k: points[k] for k in ('A1', 'A2', 'A3', 'B1', 'C1')}, markers, intr)
    assert result['errors_cm']['B1'] is None and result['errors_cm']['C1'] is None and result['rms_cm'] is None
    assert result['errors_cm']['A1'] < .05 and 'B1, C1' in result['reason']
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError):
        core.evaluate(dict(five, C2=[-5., 10.]), markers, intr)


def test_one_misclick_is_named_even_when_it_bends_every_other_fit(ost, markers):
    cfg = assumed(pitch=12.)
    points = clicks(camera.build(cfg)[0], markers)
    intr = dataclasses.replace(core.read_ost(ost), new_K=camera.intrinsics(cfg))
    six = {k: points[k] for k in ('A1', 'A3', 'B2', 'C1', 'C3', 'D2')}
    result = core.evaluate(dict(six, A3=points['B2']), markers, intr)   # A3 clicked on B2's cross
    errors = result['errors_cm']
    assert max(errors, key=errors.get) != 'A3' and errors['A1'] > core.REJECT_CM   # leave-one-out alone blames A1
    assert result['suspect'] == 'A3' and not result['savable'] and 'A3 마커' in result['reason'] and 'A1' not in result['reason']
    assert 'A3·B2' in result['reason']   # one cross clicked for two markers
    json.dumps(result, allow_nan=False)
    result = core.evaluate(dict(points, C3=points['D3']), markers, intr)   # 12 clicks: the good D3 is flagged too
    assert result['errors_cm']['D3'] > core.REJECT_CM and result['suspect'] == 'C3' and 'D3 마커' not in result['reason']
    result = core.evaluate(dict(points, A1=points['A2'], D3=points['D2']), markers, intr)   # two misclicks: no single culprit
    assert result['suspect'] is None and not result['savable'] and '하나 이상' in result['reason']
    assert 'A1·A2' in result['reason'] and 'D2·D3' in result['reason']
    assert core.evaluate(points, markers, intr)['suspect'] is None


@pytest.mark.parametrize('pitch', [0., 10.])
def test_pose_round_trips_camsim_build(pitch):
    cfg = assumed(pitch=pitch)
    expected = dict(x_m=.1, y_m=0., height_m=.2, pitch_deg=pitch, yaw_deg=0., roll_deg=0.)
    for scale in (1., -3.):   # H is defined up to scale and sign
        assert core.estimate_pose(scale * camera.build(cfg)[1], camera.intrinsics(cfg)) == pytest.approx(expected, abs=1e-6)


def turned(cfg, yaw, roll, dy):
    """H_g2i of the camsim camera moved dy left, turned by yaw about its vertical, then rolled about its optical axis."""
    K, a, b = camera.intrinsics(cfg), np.radians(yaw), np.radians(roll)
    c = np.array([cfg.camera.offset_x_m, 0.])
    A = np.eye(3)
    A[:2, :2] = [[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]   # the floor as the turned camera sees it
    A[:2, 2] = c - A[:2, :2] @ (c + [0., dy])
    Rc = np.array([[np.cos(b), -np.sin(b), 0.], [np.sin(b), np.cos(b), 0.], [0., 0., 1.]])   # x_c -> y_c
    return K @ Rc.T @ np.linalg.inv(K) @ camera.build(cfg)[0] @ A


def test_pose_yaw_and_roll_conventions():
    cfg = assumed()
    K = camera.intrinsics(cfg)
    # yaw +10: looks left, so the floor point 10 deg left of the camera is on the centre column.
    ray = np.array([.1, 0.]) + 1.5 * np.array([np.cos(np.radians(10.)), np.sin(np.radians(10.))])
    assert camera.project(turned(cfg, 10., 0., 0.), ray)[0] == pytest.approx(K[0, 2])
    # roll +5: camera turned clockwise seen from behind (right side down); a left floor point sits lower.
    left, right = camera.project(turned(cfg, 0., 5., 0.), np.array([[1.5, .3], [1.5, -.3]]))
    assert left[1] > right[1]
    pose = core.estimate_pose(np.linalg.inv(turned(cfg, 8., -3., .05)), K)
    assert pose == pytest.approx(dict(x_m=.1, y_m=.05, height_m=.2, pitch_deg=10., yaw_deg=8., roll_deg=-3.), abs=1e-6)
    assert core.estimate_pose(np.linalg.inv(turned(cfg, 0., 5., 0.)), K)['roll_deg'] == pytest.approx(5.)


def test_mirrored_marker_frame_is_refused(ost, markers):
    cfg = assumed()
    points = clicks(camera.build(cfg)[0], markers)
    intr = dataclasses.replace(core.read_ost(ost), new_K=camera.intrinsics(cfg))
    mirrored = {k: (x, -y) for k, (x, y) in markers.items()}   # y measured to the right
    result = core.evaluate(points, mirrored, intr)
    assert max(result['errors_cm'].values()) < .05   # the fit alone cannot tell
    assert result['pose']['height_m'] < 0 and not result['savable'] and '높이' in result['reason']


def test_saved_calibration_drives_camera_preprocessor(ost, markers, tmp_path, monkeypatch):
    intr = core.read_ost(ost)
    cfg = assumed()
    # Undistorted pixels of the slide-84 camera (new_K = P) mounted like the camsim camera.
    H_g2i = intr.new_K @ camera.extrinsics(cfg, 10.)[:, [0, 1, 3]]
    points = clicks(H_g2i, markers)
    assert core.MIN_SAVE <= len(points) < 12 and 'A3' not in points
    result = core.evaluate(points, markers, intr)
    assert result['savable'], result['reason']
    raw = np.random.default_rng(3).integers(0, 256, (1080, 1480, 3), dtype=np.uint8)
    bev_cfg = config.load()
    preview = core.bev_preview(raw, core.calibration_dict(intr, result['H_i2g']), bev_cfg, markers, points)
    for name, color in (('B2', (0, 200, 0)), ('A3', (200, 200, 200))):   # clicked green, skipped grey
        u, v = np.rint(bev_pixels(np.array(markers[name]), bev_cfg)).astype(int)
        assert tuple(int(c) for c in preview[v, u]) == color
    path = tmp_path/'markers.yaml'
    shutil.copy(TEMPLATE, path)
    source = dict(frame='calib', frame_stamp_ns=10100000000, markers=str(path), markers_sha256=core.load_markers(path)[1])
    data = core.build_calibration(intr, result, points, markers, source)
    assert yaml.safe_load(yaml.safe_dump(data)) == data
    assert '-0.0' not in [str(v) for v in data['camera_pose_estimate'].values()]   # no "y_m: -0.0" in the file
    assert data['source']['ost_sha256'] == intr.sha256 and data['source']['markers_sha256'] == hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    assert [m['id'] for m in data['markers']] == list(points) and data['markers'][0]['ground_m'] == [.6, .4]
    # hfov_deg: the FOV camsim needs, with the height and pitch, to retrain for this camera (spec 3.3).
    assert data['camera_pose_estimate'] == pytest.approx(dict(x_m=.1, y_m=0., height_m=.2, pitch_deg=10., yaw_deg=0., roll_deg=0.,
                                                              hfov_deg=math.degrees(2 * math.atan(740 / 784.36676))), abs=1e-3)
    path.write_text(TEMPLATE.read_text().replace('B2: [1.00, 0.00]', 'B2: [1.02, 0.01]'))   # re-measured after the start
    with pytest.raises(ValueError, match='다시 실행'):
        core.build_calibration(intr, result, points, markers, source)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 8, 9, 30, 0, tzinfo=tz)
    monkeypatch.setattr(core, 'datetime', Clock)
    out = tmp_path/'calibration'/'car.yaml'
    saved, backup = core.save_calibration(data, out, preview)
    assert saved == out and backup is None and out.read_text().startswith('# ')
    np.testing.assert_array_equal(cv2.imread(str(tmp_path/'calibration'/'car_bev.png')), preview)
    pre = CameraPreprocessor(out, bev_cfg, training_mask(bev_cfg), 'rear_axle')
    pre.require_driving_calibration()
    assert pre.calibration_status == 'measured'
    np.testing.assert_allclose(camera.project(pre.H, points['B2']), markers['B2'], atol=1e-3)
    # Same undistortion maps as the car, so clicks are in H_i2g's homography_space.
    expected = cv2.remap(raw, pre.mapx, pre.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    np.testing.assert_array_equal(core.undistort(raw, intr), expected)
    first = out.read_text()
    saved, backup = core.save_calibration(dict(data, calibration_status='measured'), out, preview)
    assert backup == tmp_path/'calibration'/'old'/'car-20261008-093000.yaml' and backup.read_text() == first
    saved, backup = core.save_calibration(data, out, preview)
    assert backup.name == 'car-20261008-093000-1.yaml' and len(list(backup.parent.iterdir())) == 2
    with pytest.raises(ValueError):
        core.build_calibration(intr, dict(result, savable=False), points, markers, dict(source, frame='x', frame_stamp_ns=None))


def test_a_failed_save_keeps_the_old_calibration(tmp_path):
    out, preview = tmp_path/'calibration'/'car.yaml', np.zeros((4, 4, 3), np.uint8)
    core.save_calibration({'H_i2g': 1}, out, preview)
    first = out.read_text()
    bev = out.with_name('car_bev.png')
    bev.unlink()
    bev.mkdir()   # the preview cannot be written (a folder here; a full disk on the car)
    with pytest.raises(OSError, match='저장하지 못했습니다'):
        core.save_calibration({'H_i2g': 2}, out, preview)
    assert out.read_text() == first and sorted(p.name for p in out.parent.iterdir()) == ['car.yaml', 'car_bev.png', 'old']
    assert len(list((out.parent/'old').iterdir())) <= 1   # at most a copy; car.yaml itself never left
    bev.rmdir()
    shutil.rmtree(out.parent/'old')
    (out.parent/'old').write_text('')   # no backup folder possible
    with pytest.raises(OSError, match='저장하지 못했습니다'):
        core.save_calibration({'H_i2g': 3}, out, preview)
    assert out.read_text() == first and sorted(p.name for p in out.parent.iterdir()) == ['car.yaml', 'old']


def test_read_bag_frame_returns_middle_message(tmp_path):
    rosbag = pytest.importorskip('rosbag2_py')
    from rclpy.serialization import serialize_message
    from cv_bridge import CvBridge
    bag, bridge = tmp_path/'calib', CvBridge()
    writer = rosbag.SequentialWriter()
    writer.open(rosbag.StorageOptions(uri=str(bag), storage_id='sqlite3'), rosbag.ConverterOptions('', ''))
    for name, kind in [('/flir_camera/image_raw', 'Image'), ('/flir_camera/camera_info', 'CameraInfo'),
                       ('/empty', 'Image'), ('/depth', 'Image')]:
        writer.create_topic(rosbag.TopicMetadata(name=name, type='sensor_msgs/msg/' + kind, serialization_format='cdr'))
    frames = [np.full((6, 8, 3), (10 * i, 20, 30), np.uint8) for i in range(3)]
    for i, bgr in enumerate(frames):
        msg = bridge.cv2_to_imgmsg(bgr[:, :, ::-1].copy(), encoding='rgb8')
        msg.header.stamp.sec, msg.header.stamp.nanosec = 10, i * 100000000
        writer.write('/flir_camera/image_raw', serialize_message(msg), 10000000000 + i * 100000000)
    writer.write('/depth', serialize_message(bridge.cv2_to_imgmsg(np.ones((6, 8), np.uint16), encoding='mono16')), 10000000000)
    del writer
    bgr, stamp = core.read_bag_frame(bag, '/flir_camera/image_raw')
    np.testing.assert_array_equal(bgr, frames[1])
    assert stamp == 10100000000
    with pytest.raises(ValueError, match='/missing'):
        core.read_bag_frame(bag, '/missing')
    with pytest.raises(ValueError, match='Image'):
        core.read_bag_frame(bag, '/flir_camera/camera_info')
    with pytest.raises(ValueError, match='0개'):
        core.read_bag_frame(bag, '/empty')
    with pytest.raises(ValueError, match='encoding'):
        core.read_bag_frame(bag, '/depth')
    with pytest.raises(FileNotFoundError):
        core.read_bag_frame(tmp_path/'nope', '/flir_camera/image_raw')


def test_core_import_needs_no_ros_or_torch():
    subprocess.run([sys.executable, '-c', 'import sys, camreal.calibration.core\n'
                    'assert "torch" not in sys.modules and "rclpy" not in sys.modules'], check=True)
