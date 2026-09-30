"""Waypoint error of a model on an exported real dataset, per recording session.

    python -m camreal.evaluate --model data/models/my_model --dataset data/datasets/week3_real --out out/my_model_eval

`python -m camreal export` runs this for the configured model (usually the sim model) as the baseline.
"""
import argparse
import html
import json
from pathlib import Path
import cv2
import numpy as np
from camreal.checkpoint import contract, load_model
from camreal.overlay import bev_view


def summary(errs_m):
    e = np.asarray(errs_m, float)
    return dict(n=int(len(e)), mean_m=round(float(e.mean()), 4), median_m=round(float(np.median(e)), 4),
                max_m=round(float(e.max()), 4))


def evaluate(model_dir, dataset_dir, output=None, device='cpu', n_preview=24):
    from camsim.dataset import DiskDataset
    dataset_dir = Path(dataset_dir)
    info = json.loads((dataset_dir/'dataset.json').read_text())
    predictor, cfg, _ = load_model(model_dir, device)
    if info['contract'] != contract(cfg):
        raise ValueError('모델의 BEV/waypoint 규격(waypoint 거리 등)이 데이터셋과 다릅니다.')
    result, val_rows = dict(model=str(model_dir), dataset=str(dataset_dir), splits={}), []
    for split in ('train', 'val'):
        ds = DiskDataset(str(dataset_dir/split), cfg, 'all')
        sessions = {}
        for i in range(len(ds)):
            truth, name = ds.wps[ds.idx[i]], ds.files[ds.idx[i]]
            pred = predictor.predict(ds.load_image(i))
            err = float(np.hypot(*(pred - truth)))
            sessions.setdefault(name.rsplit('_', 1)[0], []).append(err)   # file = <session>_<frame>.png
            if split == 'val':
                val_rows.append((err, i, pred, truth))
        errs = [e for v in sessions.values() for e in v]
        result['splits'][split] = dict(summary(errs), sessions={s: summary(v) for s, v in sessions.items()})
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        ds = DiskDataset(str(dataset_dir/'val'), cfg, 'all')
        cards = []
        # Worst first: that is where the model breaks on real images.
        for rank, (err, i, pred, truth) in enumerate(sorted(val_rows, key=lambda r: -r[0])[:n_preview]):
            name = ds.files[ds.idx[i]]
            image = bev_view(ds.load_image(i), cfg, pred, truth, title=f'{name} | error {err * 100:.1f} cm')
            cv2.imwrite(str(output/f'{rank:03d}.png'), image)
            cards.append(f'<figure><img src="{rank:03d}.png"><figcaption>{html.escape(name)} · '
                         f'{err * 100:.1f} cm</figcaption></figure>')
        (output/'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Waypoint error</title>'
            '<style>body{font:16px sans-serif;background:#151923;color:#eee;margin:24px}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}'
            'figure{margin:0}img{width:100%}pre{background:#1d2635;padding:12px}</style>'
            f'<h1>{html.escape(str(model_dir))} on real validation frames</h1>'
            f'<p>Green: label · magenta: prediction · teal: {cfg.waypoints.ahead_m} m ring. Worst frames first.</p>'
            f'<pre>{html.escape(report(result))}</pre><div class="grid">' + ''.join(cards) + '</div>')
    return result


def report(result):
    lines = ['waypoint 오차 (평균 / 최대, 장 수)']
    for split, value in result['splits'].items():
        for session, s in value['sessions'].items():
            lines.append(f'  {split:5s} {session:20s} {s["mean_m"] * 100:6.1f} cm / {s["max_m"] * 100:6.1f} cm ({s["n"]}장)')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='model folder (model.onnx + checkpoint.json)')
    parser.add_argument('--dataset', required=True, help='folder made by python -m camreal export')
    parser.add_argument('--out', help='new folder for metrics.json and a worst-first preview')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    result = evaluate(args.model, args.dataset, args.out, args.device)
    print(report(result))
    if args.out:
        print(f'미리보기: {Path(args.out)/"index.html"}')


if __name__ == '__main__':
    main()
