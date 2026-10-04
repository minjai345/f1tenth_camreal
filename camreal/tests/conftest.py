import numpy as np
import pytest


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
