"""Offline real camera image -> calibrated waypoint overlay; no ROS/Gym required.

    python -m camreal.tools.infer_images --input data/labeling/run_train/raw --out out/run_train_sim

Model and calibration default to data/camreal.yaml (override with --model / --calibration).
"""
import argparse
import html
import json
from pathlib import Path
import cv2
from camreal.checkpoint import load_model
from camreal.preprocessing import CameraPreprocessor
from camreal.overlay import predict_overlays


def run(model_dir, calibration, source, output, device='cpu', label='Camera image predictions'):
    source, output = Path(source), Path(output)
    predictor, cfg, mask = load_model(model_dir, device)
    pre = CameraPreprocessor(calibration, cfg, mask, 'rear_axle')
    if pre.calibration_status == 'assumed':
        label += ' — ASSUMED calibration / geometry not validated'
        print('ASSUMED calibration: visual pipeline check only; metre coordinates are not validated.', flush=True)
    candidates = sorted(source.iterdir()) if source.is_dir() else [source]
    files = [p for p in candidates if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')]
    if not files:
        raise ValueError(f'no image files: {source}')
    output.mkdir(parents=True, exist_ok=False)
    records, cards = [], []
    for i, path in enumerate(files):
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f'cannot read {path}')
        # Do not silently scale 16-bit data or assume a Bayer pattern in a saved file.
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif raw.ndim == 3 and raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        wp, images = predict_overlays(raw, pre, predictor)
        names = {}
        for kind, image in images.items():
            name = f'{i:04d}_{kind}.png'
            if not cv2.imwrite(str(output/name), image):
                raise OSError(f'failed to write {output/name}')
            names[kind] = name
        records.append(dict(source=str(path.resolve()), waypoint_frame='rear_axle',
                            waypoint_m=wp.tolist(), images=names))
        cards.append(f'<section><h2>{html.escape(path.name)} · x {wp[0]:.2f} m, y {wp[1]:+.2f} m</h2><div class="row">' +
                     ''.join(f'<figure><img src="{names[kind]}"><figcaption>{kind}</figcaption></figure>'
                             for kind in ('raw_overlay', 'undistorted_overlay', 'bev_overlay')) + '</div></section>')
    (output/'predictions.json').write_text(json.dumps(records, indent=2))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Waypoint predictions</title>'
        '<style>body{font:16px sans-serif;background:#151923;color:#eee;margin:24px}'
        '.row{display:flex;gap:12px;align-items:flex-start}figure{margin:0;flex:1;min-width:0}'
        'img{width:100%}figcaption{padding:8px}section{margin-bottom:32px}</style>'
        f'<h1>{html.escape(label)}</h1><p>Magenta: predicted waypoint ({cfg.waypoints.ahead_m} m ahead, '
        'teal ring in BEV). No driving commands are published. Overlays are not an accuracy measurement.</p>' + ''.join(cards))
    print(f'Saved {len(records)} predictions: {output / "index.html"}', flush=True)
    return records


def main():
    from camreal.__main__ import load_course
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='image or directory of decoded camera images')
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--config', default='data/camreal.yaml')
    parser.add_argument('--model', help='model folder (model.onnx + checkpoint.json); default: config model')
    parser.add_argument('--calibration', help='default: config calibration')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    course = load_course(args.config) if not (args.model and args.calibration) else {}
    run(args.model or course['model'], args.calibration or course['calibration'], args.input, args.out, args.device)


if __name__ == '__main__':
    main()
