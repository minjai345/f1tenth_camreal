import threading
import time
import cv2
import numpy as np
import pytest
import torch
import yaml
from camsim import config, handoff, model, camera, render
from camsim.dataset import to_tensor
from camreal.checkpoint import load_model, training_mask
from camsim_driver.preprocessing import CameraPreprocessor
from camsim_driver.runtime import DriverState


def state():
    return DriverState(.25, .2, .02, .5, .3, .4, 6.)


def test_pursuit_and_timestamp_expiry():
    s = state()
    s.offer('image', 10., 1., 10.)
    frame = s.take()
    assert s.complete(frame, [1., 1.], 1.01, 10.01)
    speed, steer = s.command(1.02, 10.02)
    # Pure pursuit straight at the waypoint: curvature 2y/L^2 = 1, steer = atan(wheelbase * 1).
    assert speed == .5 and steer == pytest.approx(np.arctan(.3))
    s.offer('new image', 10.19, 1.19, 10.19)  # no new inference: must expire old path
    assert s.command(1.21, 10.21) == (0., 0.)
    assert frame.stamp == 10.


@pytest.mark.parametrize('wp', [[], [np.nan, 0], [1, np.inf], [-1, 0], [0, 0], [[1, 0], [2, 0]], [100, 0]])
def test_bad_predictions_stop(wp):
    s = state()
    s.offer(None, 10., 1., 10.)
    assert not s.complete(s.take(), wp, 1.01, 10.01)
    assert s.command(1.02, 10.02) == (0., 0.)


def test_input_loss_and_clock_pause():
    s = state()
    s.offer(None, 10., 1., 10.)
    assert s.complete(s.take(), [1, 0], 1., 10.)
    assert s.command(1.3, 10.) == (0., 0.)  # monotonic timeout even if ROS clock stalls


def test_latest_only_and_failure_epoch():
    s = state()
    for i in range(10):
        assert s.offer(i, 10. + i / 100, 1. + i / 100, 10. + i / 100)
    frame = s.take()
    assert frame.message == 9
    s.fail(frame, 'network failed')
    assert s.command(1.1, 10.1) == (0., 0.)
    assert not s.complete(frame, [1, 0], 1.1, 10.1)
    s.offer(11, 10.11, 1.11, 10.11)
    assert s.complete(s.take(), [1, 0], 1.12, 10.12)
    assert s.command(1.13, 10.13)[0] == .5


@pytest.mark.parametrize('stamp', [0., 9., 11., np.nan, 10.])
def test_invalid_timestamp_invalidates_running_result(stamp):
    s = state()
    s.offer(None, 10., 1., 10.)
    frame = s.take()
    assert not s.offer(None, stamp, 1.01, 10.01)
    assert not s.complete(frame, [1, 0], 1.02, 10.02)
    assert s.command(1.02, 10.02) == (0., 0.)


def test_delayed_worker_does_not_block_control():
    s = state()
    s.offer(None, 10., 1., 10.)
    entered, release = threading.Event(), threading.Event()
    def worker():
        frame = s.take()
        entered.set()
        release.wait(2.)
        assert not s.complete(frame, [1, 0], 1.5, 10.5)
    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(1.)
    start = time.monotonic()
    for _ in range(100):
        assert s.command(1.4, 10.4) == (0., 0.)
    assert time.monotonic() - start < .2
    release.set()
    thread.join(2.)
    assert not thread.is_alive()


@pytest.fixture
def model_folder(tmp_path):
    """camsim notebook ch.5 output: model.onnx + checkpoint.json."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = config.load()
    cfg.bev.resolution_m = .05
    cfg.waypoints.norm_m = 2.5
    cfg.model.arch = 'small'
    net = model.WaypointNet(cfg).eval()
    onnx = model.export_onnx(net, cfg, tmp_path/'model.onnx')
    handoff.export_checkpoint(onnx, tmp_path/'model', cfg, 'test')
    return tmp_path/'model', cfg, net


def test_handoff_model_and_predictor_equivalence(model_folder):
    root, cfg, net = model_folder
    predictor, restored, mask = load_model(root)
    assert restored.waypoints.norm_m == 2.5
    bev = np.random.default_rng(4).integers(0, 256, (*render.bev_size(cfg), 3), dtype=np.uint8)
    # The ONNX graph on the car gives what camsim's PyTorch Predictor gives in Colab.
    np.testing.assert_allclose(predictor.predict(bev), model.Predictor(net, cfg).predict(bev), atol=1e-4)
    tensor = to_tensor(np.array([[[10, 20, 30]]], np.uint8))
    np.testing.assert_allclose(tensor[:, 0, 0], np.array([10, 20, 30]) / 255)
    with torch.no_grad():
        expected = net(to_tensor(bev)[None])[0].numpy() * 2.5
    np.testing.assert_allclose(predictor.predict(bev), expected, atol=1e-4)


def calibration(tmp_path, cfg):
    H = camera.build(cfg)[1]
    K = camera.intrinsics(cfg)
    data = dict(schema_version=1, distortion_model='plumb_bob', ground_frame='rear_axle',
                homography_space='undistorted_full_resolution', image_width=640, image_height=400,
                K=K.tolist(), new_K=K.tolist(), D=[.01, -.005, 0., 0., 0.], H_i2g=H.tolist())
    path = tmp_path/'camera.yaml'
    path.write_text(yaml.safe_dump(data))
    return path, data


def test_preprocessing_matches_explicit_predictor_pipeline(model_folder, tmp_path):
    root, cfg, net = model_folder
    path, c = calibration(tmp_path, cfg)
    mask = training_mask(cfg)
    pre = CameraPreprocessor(path, cfg, mask, 'rear_axle')
    raw = np.random.default_rng(6).integers(0, 256, (400, 640, 3), dtype=np.uint8)
    undistorted = cv2.remap(raw, pre.mapx, pre.mapy, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(cfg.lane.color_floor))
    expected = render.ipm_bev(undistorted, np.array(c['H_i2g']), cfg)
    expected[~pre.mask] = cfg.lane.color_floor
    np.testing.assert_array_equal(pre.bev(raw), expected)
    predictor = model.Predictor(net, cfg)
    np.testing.assert_allclose(predictor.predict(pre.bev(raw)), predictor.predict(expected))
    # camsim's predict_camera is IPM only; the car adds undistortion and the training mask.
    np.testing.assert_allclose(predictor.predict_camera(undistorted, pre.H),
                               predictor.predict(render.ipm_bev(undistorted, pre.H, cfg)))
    assert np.all(pre.bev(raw)[~mask] == cfg.lane.color_floor)
    with pytest.raises(ValueError, match='resolution'):
        pre.bev(raw[::2, ::2])
    with pytest.raises(ValueError, match='ground_frame'):
        CameraPreprocessor(path, cfg, mask, 'base_link')


@pytest.mark.parametrize('field,value', [('H_i2g', [[0, 0, 0]] * 3),
                                        ('D', [float('nan')] * 5),
                                        ('homography_space', 'raw_distorted')])
def test_bad_calibration_rejected(model_folder, tmp_path, field, value):
    _, cfg, _ = model_folder
    path, data = calibration(tmp_path, cfg)
    data[field] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        CameraPreprocessor(path, cfg, training_mask(cfg), 'rear_axle')


def test_missing_calibration_rejected(model_folder, tmp_path):
    _, cfg, _ = model_folder
    with pytest.raises(FileNotFoundError):
        CameraPreprocessor(tmp_path/'missing.yaml', cfg, training_mask(cfg), 'rear_axle')
