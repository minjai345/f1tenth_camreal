import cv2
import numpy as np
import pytest
from camreal.overlay import project_waypoints, to_distorted_pixels, draw_trajectory


def test_ground_projection_known_points():
    H = np.array([[0., -100., 320.], [0., 0., 100.], [1., 0., 0.]])
    np.testing.assert_allclose(project_waypoints([[1., 0.], [2., 1.]], H),
                               [[320., 100.], [110., 50.]])
    assert np.isnan(project_waypoints([[0., 0.]], H)).all()


def test_raw_projection_roundtrip_with_distortion():
    K = np.array([[500., 0., 320.], [0., 500., 200.], [0., 0., 1.]])
    new_K = K.copy()
    new_K[0, 0] = 470.
    D = np.array([.03, -.01, .001, -.002, .001])
    undistorted = np.array([[100., 280.], [320., 200.], [560., 340.]])
    raw = to_distorted_pixels(undistorted, K, D, new_K)
    back = cv2.undistortPoints(raw[:, None, :], K, D, P=new_K).reshape(-1, 2)
    np.testing.assert_allclose(back, undistorted, atol=1e-5)


def test_overlay_does_not_mutate_input_or_bridge_invalid_points():
    image = np.zeros((100, 100, 3), np.uint8)
    uv = np.array([[10., 50.], [np.nan, np.nan], [90., 50.]])
    result = draw_trajectory(image, uv, 'Test')
    assert not image.any()
    assert result[50, 10].any() and result[50, 90].any()
    assert not result[50, 50].any()


def test_bev_view_keeps_off_image_prediction_visible():
    from camsim import config
    from camreal.overlay import bev_view, PRED
    cfg = config.load()
    cfg.bev.resolution_m = .05                       # 76 x 60 BEV
    out = bev_view(np.zeros((76, 60, 3), np.uint8), cfg, prediction=[1., -2.5])   # beyond the -1.5 m edge
    assert (out == PRED).all(axis=-1)[:, -12:].any()


@pytest.mark.parametrize('wp', [[], [[float('nan'), 0]], [[1., 2., 3.]]])
def test_invalid_prediction_rejected(wp):
    with pytest.raises(ValueError):
        project_waypoints(wp, np.eye(3))


def test_offline_image_to_html_and_waypoint(tmp_path):
    import json
    import torch
    import yaml
    from camsim import camera, config, model
    from camreal.tools.infer_images import run
    from conftest import make_model_dir
    def constant(net):   # waypoint (1.0, 0.25) m for any image, norm_m = 1
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
            net.head[-1].bias.copy_(torch.tensor([1., .25]))
    root, cfg, _ = make_model_dir(tmp_path, constant)
    K = camera.intrinsics(cfg)
    calibration = dict(schema_version=1, distortion_model='plumb_bob', ground_frame='rear_axle',
        homography_space='undistorted_full_resolution', image_width=640, image_height=400,
        K=K.tolist(), new_K=K.tolist(), D=[0.] * 5, H_i2g=camera.build(cfg)[1].tolist())
    (tmp_path/'camera.yaml').write_text(yaml.safe_dump(calibration))
    cv2.imwrite(str(tmp_path/'frame.png'), np.full((400, 640, 3), 128, np.uint8))
    run(root, tmp_path/'camera.yaml', tmp_path/'frame.png', tmp_path/'results')
    records = json.loads((tmp_path/'results'/'predictions.json').read_text())
    np.testing.assert_allclose(records[0]['waypoint_m'], [1., .25], atol=1e-6)
    assert records[0]['waypoint_frame'] == 'rear_axle'
    assert (tmp_path/'results'/'index.html').is_file()
    assert len(list((tmp_path/'results').glob('*.png'))) == 4
