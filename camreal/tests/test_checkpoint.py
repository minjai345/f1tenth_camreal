import json
import subprocess
import sys
import numpy as np
import pytest
from camsim import camera, config, model, render
from camreal.checkpoint import load_config, load_model, training_mask
from conftest import make_model_dir


def test_handoff_folder_round_trip(model_dir):
    root, cfg, net = model_dir
    predictor, restored, mask = load_model(root)
    assert restored.waypoints.ahead_m == cfg.waypoints.ahead_m and restored.bev == cfg.bev
    bev = np.random.default_rng(4).integers(0, 256, (*render.bev_size(cfg), 3), dtype=np.uint8)
    out = predictor.predict(bev)
    assert out.shape == (2,) and predictor.provider == 'CPUExecutionProvider'
    np.testing.assert_allclose(out, model.Predictor(net, cfg).predict(bev), atol=1e-4)
    np.testing.assert_array_equal(mask, render.bev_visibility_mask(camera.build(cfg)[0], cfg))


def test_modified_or_missing_files_rejected(model_dir):
    root, _, _ = model_dir
    with (root/'model.onnx').open('ab') as stream:
        stream.write(b'\n')
    with pytest.raises(ValueError, match='SHA-256'):
        load_config(root)
    (root/'model.onnx').unlink()
    with pytest.raises(FileNotFoundError, match='model.onnx'):
        load_config(root)
    (root/'checkpoint.json').unlink()
    with pytest.raises(FileNotFoundError, match='checkpoint.json'):
        load_config(root)


def test_old_model_pt_handoff_rejected(model_dir):
    root, _, _ = model_dir
    info = json.loads((root/'checkpoint.json').read_text())
    info['file'] = 'model.pt'   # notebook ch.5 before the ONNX hand-off
    (root/'checkpoint.json').write_text(json.dumps(info))
    with pytest.raises(ValueError, match='model.onnx'):
        load_config(root)


def test_bev_size_mismatch_rejected(model_dir):
    root, _, _ = model_dir
    info = json.loads((root/'checkpoint.json').read_text())
    info['config']['bev']['resolution_m'] = .1   # config says 38 x 30, the graph takes 76 x 60
    (root/'checkpoint.json').write_text(json.dumps(info))
    with pytest.raises(ValueError, match='BEV'):
        load_model(root)


def test_unknown_device_rejected(model_dir):
    with pytest.raises(ValueError, match='device'):
        load_model(model_dir[0], device='gpu')


def test_colab_h_i2g_path_resolved_next_to_model(tmp_path):
    root, _, _ = make_model_dir(tmp_path)
    info = json.loads((root/'checkpoint.json').read_text())
    info['config']['camera']['h_i2g_file'] = '/content/drive/MyDrive/calib/H_i2g.npy'
    (root/'checkpoint.json').write_text(json.dumps(info))
    with pytest.raises(FileNotFoundError, match='h_i2g_file'):
        load_config(root)
    np.save(root/'H_i2g.npy', camera.build(config.load())[1])
    cfg = load_config(root)
    assert cfg.camera.h_i2g_file == str((root/'H_i2g.npy').resolve())
    assert training_mask(cfg).any()


def test_loading_a_model_needs_neither_torch_nor_the_simulator(model_dir):
    subprocess.run([sys.executable, '-c',
        'import sys; from camreal.checkpoint import load_model; '
        f'load_model({str(model_dir[0])!r})[0].predict(__import__("numpy").zeros((76, 60, 3), "uint8")); '
        'assert not any(m in sys.modules for m in ("torch", "gym", "f110_gym", "pyglet")), '
        '[m for m in ("torch", "gym", "f110_gym", "pyglet") if m in sys.modules]'], check=True)
