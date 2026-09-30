"""Generate an explicitly ASSUMED pinhole calibration for pipeline previews only."""
import argparse
from pathlib import Path
import numpy as np
import yaml
from camsim import camera, config


def make_assumed(width=1920, height=1200, hfov_deg=90., height_m=.20, pitch_deg=0., offset_x_m=0.):
    if width <= 0 or height <= 0 or not np.isfinite([hfov_deg,height_m,pitch_deg,offset_x_m]).all():
        raise ValueError('image dimensions and camera assumptions must be finite and valid')
    if not 0 < hfov_deg < 180 or height_m <= 0 or not -89 < pitch_deg < 89:
        raise ValueError('invalid FOV, camera height or pitch')
    cfg = config.load()
    cfg.camera.image_width, cfg.camera.image_height = width, height
    cfg.camera.hfov_deg, cfg.camera.height_m = hfov_deg, height_m
    cfg.camera.pitch_deg, cfg.camera.offset_x_m = pitch_deg, offset_x_m
    cfg.camera.h_i2g_file = None
    K = camera.intrinsics(cfg)
    return dict(schema_version=1, calibration_status='assumed',
        assumptions=dict(hfov_deg=hfov_deg, camera_height_m=height_m, pitch_deg=pitch_deg,
                         offset_x_m=offset_x_m, distortion='zero, NOT measured',
                         purpose='pipeline preview only; replace with Week 1 calibration'),
        distortion_model='plumb_bob', image_width=width, image_height=height,
        K=K.tolist(), D=[0.] * 5, new_K=K.tolist(),
        homography_space='undistorted_full_resolution', H_i2g=camera.build(cfg)[1].tolist(),
        ground_frame='rear_axle')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--width',type=int,default=1920);p.add_argument('--height',type=int,default=1200)
    p.add_argument('--hfov-deg',type=float,default=90.)
    p.add_argument('--camera-height-m',type=float,default=.20)
    p.add_argument('--pitch-deg',type=float,default=0.)
    p.add_argument('--offset-x-m',type=float,default=0.)
    a=p.parse_args()
    c=make_assumed(a.width,a.height,a.hfov_deg,a.camera_height_m,a.pitch_deg,a.offset_x_m)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:
        stream.write('# ASSUMED calibration: preview only, NOT measured. Drive output is blocked.\n')
        yaml.safe_dump(c,stream,sort_keys=False)
    print(f'Assumed calibration saved: {out}; fx={c["K"][0][0]:.3f}px; drive output blocked')


if __name__=='__main__':main()
