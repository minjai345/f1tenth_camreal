from pathlib import Path
import numpy as np
import pytest
from camsim import camera, config
from camreal.calibration import core

TEMPLATE = Path(__file__).resolve().parents[1]/'config'/'markers.yaml'
# Week-1 slide 84 ost.yaml (cameracalibrator, 1480x1080) until the 1920x1200 reference file exists.
OST = """image_width: 1480
image_height: 1080
camera_name: narrow_stereo
camera_matrix:
  rows: 3
  cols: 3
  data: [983.23425, 0., 909.40739, 0., 974.13698, 573.79454, 0., 0., 1.]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [-0.364955, 0.124123, -0.003946, 0.000371, 0.000000]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1., 0., 0., 0., 1., 0., 0., 0., 1.]
projection_matrix:
  rows: 3
  cols: 4
  data: [784.36676, 0., 950.00529, 0., 0., 849.48163, 577.98068, 0., 0., 0., 1., 0.]
"""


def make_model_dir(root, edit_net=None, **overrides):
    """A model folder exactly as camsim notebook ch.5 writes it (model.onnx + checkpoint.json).

    edit_net(net) may change the weights before export. Returns (model_dir, cfg, net).
    """
    import torch
    from camsim import config, handoff, model
    torch.set_num_threads(1)
    cfg = config.load()
    cfg.bev.resolution_m = .05   # 76 x 60 BEV keeps tests fast
    cfg.model.arch = 'small'     # no torchvision download; the car only sees the ONNX graph anyway
    for key, value in overrides.items():
        section, name = key.split('__')
        setattr(getattr(cfg, section), name, value)
    torch.manual_seed(0)
    net = model.WaypointNet(cfg).eval()
    if edit_net is not None:
        edit_net(net)
    (root/'weights').mkdir(parents=True, exist_ok=True)
    onnx = model.export_onnx(net, cfg, root/'weights'/'model.onnx')
    handoff.export_checkpoint(onnx, root/'model', cfg, 'test-commit')
    return root/'model', cfg, net


def tape_lane(cfg):
    """A straight lane of 5 cm yellow tape 0.8 m apart on the floor, as the camera of cfg sees it (BGR uint8)."""
    from camsim import camera, render
    x = np.arange(.3, 4., .05)
    quads = np.array([[[a, y - .025], [a + .05, y - .025], [a + .05, y + .025], [a, y + .025]]
                      for y in (.4, -.4) for a in x])
    return render.render((0., 0., 0.), quads, None, camera.build(cfg)[0], cfg)


@pytest.fixture
def model_dir(tmp_path):
    return make_model_dir(tmp_path)


def assumed(pitch=10., offset=.1, width=1480, height=1080, hfov=90.):
    """camsim's assumed pinhole camera at the ost.yaml resolution."""
    cfg = config.load()
    cfg.camera.image_width, cfg.camera.image_height, cfg.camera.hfov_deg = width, height, hfov
    cfg.camera.height_m, cfg.camera.pitch_deg, cfg.camera.offset_x_m = .2, pitch, offset
    return cfg


def clicks(H_g2i, markers, size=(1480, 1080)):
    """Marker pixels a perfect clicker would give; markers outside the image are skipped."""
    points = {k: camera.project(H_g2i, np.array(v)).tolist() for k, v in markers.items()}
    return {k: p for k, p in points.items() if 0 <= p[0] < size[0] and 0 <= p[1] < size[1]}


@pytest.fixture
def ost(tmp_path):
    path = tmp_path/'ost.yaml'
    path.write_text(OST)
    return path


@pytest.fixture
def markers():
    return core.load_markers(TEMPLATE)[0]
