import cv2
import numpy as np
import pytest
import torch
import yaml
from camsim import config, handoff, model, camera, render
from camsim.dataset import to_tensor
from camreal.checkpoint import load_model, training_mask
from camsim_driver.preprocessing import CameraPreprocessor


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
    # No resize/crop: the error names both sizes and the three fixes.
    with pytest.raises(ValueError, match=r'영상 해상도\(320x200\)와 캘리브레이션 해상도\(640x400\)가 다릅니다.* '
                                         r'1\) 카메라 해상도를 640x400에 맞추기 '
                                         r'2\) 320x200에서 1주차 방식으로 다시 캘리브레이션.* '
                                         r'3\) 카메라 해상도를 1920x1200에 맞추고 .*'
                                         r'--ost camreal/config/ost_reference_1920x1200\.yaml'):
        pre.bev(raw[::2, ::2])
    with pytest.raises(ValueError, match=r'ground_frame\(rear_axle\)과 path_frame\(base_link\)이 같아야'):
        CameraPreprocessor(path, cfg, mask, 'base_link')


@pytest.mark.parametrize('field,value,match', [
    ('H_i2g', [[0, 0, 0]] * 3, 'H_i2g는 유한하고 역행렬이 있는 3x3'), ('D', [float('nan')] * 5, r'D\(왜곡 계수\)'),
    ('homography_space', 'raw_distorted', '왜곡 보정된 원본 해상도'), ('image_width', 0, 'image_width'),
    ('distortion_model', 'equidistant', r'plumb_bob인 캘리브레이션 파일이어야 합니다\(지금 1, equidistant\)')],
    ids=['H_i2g', 'D', 'homography_space', 'image_width', 'distortion_model'])
def test_bad_calibration_rejected(model_folder, tmp_path, field, value, match):
    _, cfg, _ = model_folder
    path, data = calibration(tmp_path, cfg)
    data[field] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=match):
        CameraPreprocessor(path, cfg, training_mask(cfg), 'rear_axle')


def test_missing_calibration_rejected(model_folder, tmp_path):
    _, cfg, _ = model_folder
    with pytest.raises(FileNotFoundError):
        CameraPreprocessor(tmp_path/'missing.yaml', cfg, training_mask(cfg), 'rear_axle')
