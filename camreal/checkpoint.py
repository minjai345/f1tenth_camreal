"""Read the camsim hand-off (notebook ch.5: model.onnx + checkpoint.json) on the car.

The training config always comes from checkpoint.json, never from guesses. The ONNX file is
checked against the manifest SHA-256 and its input shape against the config's BEV size.
Inference runs in onnxruntime (CUDA on the Jetson), so the car needs neither PyTorch nor a
separately built TensorRT engine.
"""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import yaml
from camsim import camera, config, render
from camsim.handoff import sha256_file

WEIGHTS, MANIFEST = 'model.onnx', 'checkpoint.json'


def read_manifest(model_dir):
    model_dir = Path(model_dir)
    manifest = model_dir/MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f'{manifest} 없음: camsim 5장이 드라이브에 저장한 model.onnx와 checkpoint.json을 '
                                f'같은 폴더({model_dir})에 두세요.')
    info = json.loads(manifest.read_text())
    if not isinstance(info, dict) or not isinstance(info.get('config'), dict):
        raise ValueError(f'{manifest}: camsim.handoff가 만든 checkpoint.json이 아닙니다.')
    if info.get('file') != WEIGHTS:
        raise ValueError(f'{manifest}는 {info.get("file")}의 기록입니다. 최신 camsim 노트북 5장을 다시 실행해 '
                         'model.onnx와 checkpoint.json을 받으세요.')
    if not (model_dir/WEIGHTS).is_file():
        raise FileNotFoundError(f'{model_dir/WEIGHTS} 없음: checkpoint.json과 같은 폴더에 두세요.')
    if sha256_file(model_dir/WEIGHTS) != info.get('sha256'):
        raise ValueError(f'{model_dir/WEIGHTS}의 SHA-256이 checkpoint.json과 다릅니다. '
                         '같은 학습 결과의 두 파일을 다시 받으세요.')
    return info


def load_config(model_dir):
    """checkpoint.json config -> camsim Config, validated exactly like camsim/config.yaml."""
    model_dir = Path(model_dir)
    data = copy.deepcopy(read_manifest(model_dir)['config'])
    h_file = (data.get('camera') or {}).get('h_i2g_file')
    if h_file and not Path(h_file).is_file():
        # The manifest keeps the Colab path; the same .npy has to travel next to model.onnx.
        local = model_dir/Path(h_file).name
        if not local.is_file():
            raise FileNotFoundError(f'checkpoint.json의 camera.h_i2g_file({h_file})이 없습니다. '
                                    f'같은 .npy 파일을 {model_dir}에 두세요.')
        data['camera']['h_i2g_file'] = str(local.resolve())
    fd, path = tempfile.mkstemp(suffix='.yaml')
    try:
        with os.fdopen(fd, 'w') as stream:
            yaml.safe_dump(data, stream)
        return config.load(path)
    finally:
        os.unlink(path)


def model_files(model_dir):
    """Files that define a model folder (extra Drive files such as model.pt or videos are ignored)."""
    model_dir = Path(model_dir)
    return [model_dir/WEIGHTS, model_dir/MANIFEST] + sorted(model_dir.glob('*.npy'))


def training_mask(cfg):
    """BEV pixels the training camera could see; the car's IPM output is cut to the same area."""
    return render.bev_visibility_mask(camera.build(cfg)[0], cfg)


def contract(cfg):
    """What a BEV frame and its label mean. Sessions export together only if this matches."""
    return dict(tensor='BGR_uint8_CHW_float32_div255', output='rear_axle_x_forward_y_left_m',
                waypoints=asdict(cfg.waypoints), bev=asdict(cfg.bev), bev_hw=list(render.bev_size(cfg)),
                floor_bgr=[int(c) for c in cfg.lane.color_floor], ipm_interpolation='nearest')


class OnnxPredictor:
    """Same interface as camsim.model.Predictor: predict(bev_bgr) -> waypoint (x, y) in metres."""

    def __init__(self, path, cfg, device='cpu', cpu_threads=0):
        import onnxruntime as ort
        if device not in ('cpu', 'cuda'):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('onnxruntime에 CUDA가 없습니다 (onnxruntime-gpu 필요). device: cpu로 바꾸거나 설치하세요.')
            providers.insert(0, 'CUDAExecutionProvider')
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = int(cpu_threads)
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        self.provider = self.session.get_providers()[0]
        if device == 'cuda' and self.provider != 'CUDAExecutionProvider':
            raise RuntimeError(f'CUDA로 모델을 열지 못했습니다 ({self.provider}).')
        inp, out = self.session.get_inputs()[0], self.session.get_outputs()[0]
        if list(inp.shape) != [1, 3, *render.bev_size(cfg)] or list(out.shape) != [1, 2]:
            raise ValueError(f'model.onnx 입력 {inp.shape}·출력 {out.shape}이 checkpoint.json의 '
                             f'BEV {render.bev_size(cfg)}와 맞지 않습니다.')
        self.input_name, self.cfg, self.norm_m = inp.name, cfg, float(cfg.waypoints.norm_m)

    def predict(self, bev_bgr):
        x = np.ascontiguousarray(bev_bgr.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
        return self.session.run(None, {self.input_name: x})[0][0].astype(np.float64) * self.norm_m


def load_model(model_dir, device='cpu', cpu_threads=0):
    """-> (OnnxPredictor, cfg, training_mask). predict(bev) returns one (x, y) waypoint in metres."""
    cfg = load_config(model_dir)
    return OnnxPredictor(Path(model_dir)/WEIGHTS, cfg, device, cpu_threads), cfg, training_mask(cfg)
