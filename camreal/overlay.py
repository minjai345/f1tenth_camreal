"""Waypoint overlays on calibrated raw/undistorted camera images and BEV."""
import cv2
import numpy as np
from camsim.render import bev_pixels

PRED, TRUTH, RING = (255, 0, 255), (0, 200, 0), (207, 229, 83)   # BGR: magenta, green, teal


def project_waypoints(waypoints, H_g2i):
    wp = np.asarray(waypoints, dtype=float)
    if wp.ndim != 2 or wp.shape[1] != 2 or not len(wp) or not np.isfinite(wp).all():
        raise ValueError('prediction must be a nonempty finite (K,2) array')
    hom = np.column_stack((wp, np.ones(len(wp)))) @ np.asarray(H_g2i).T
    valid = np.abs(hom[:, 2]) > 1e-9
    uv = np.full((len(wp), 2), np.nan)
    uv[valid] = hom[valid, :2] / hom[valid, 2:3]
    return uv


def to_distorted_pixels(uv, K, D, new_K):
    """Invert undistorted projection via camera rays, then apply original distortion."""
    uv = np.asarray(uv, dtype=float)
    result = np.full_like(uv, np.nan)
    valid = np.isfinite(uv).all(axis=1)
    if valid.any():
        rays = np.column_stack((uv[valid], np.ones(valid.sum()))) @ np.linalg.inv(new_K).T
        pixels, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, D)
        result[valid] = pixels.reshape(-1, 2)
    return result


def title_bar(image, title):
    w = image.shape[1]
    cv2.rectangle(image, (0, 0), (w, 30), (25, 25, 25), -1)
    cv2.putText(image, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def draw_trajectory(image, uv, title, color=PRED):
    out = image.copy()
    h, w = out.shape[:2]
    uv = np.asarray(uv)
    valid = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) &
             (uv[:, 1] >= 0) & (uv[:, 1] < h))
    # Never connect through invalid/out-of-view projections at the horizon.
    for i in range(1, len(uv)):
        if valid[i - 1] and valid[i]:
            cv2.line(out, tuple(np.rint(uv[i - 1]).astype(int)),
                     tuple(np.rint(uv[i]).astype(int)), color, 3, cv2.LINE_AA)
    for i in np.flatnonzero(valid):
        cv2.circle(out, tuple(np.rint(uv[i]).astype(int)), 8, color, -1, cv2.LINE_AA)
    return title_bar(out, title)


def bev_view(bev, cfg, prediction=None, truth=None, title=None):
    """BEV with the ahead_m ring around the rear axle, GT in green and prediction in magenta."""
    out = bev.copy()
    a = np.linspace(-np.pi / 2, np.pi / 2, 91)
    ring = cfg.waypoints.ahead_m * np.column_stack((np.cos(a), np.sin(a)))
    cv2.polylines(out, [np.rint(bev_pixels(ring, cfg)).astype(np.int32).reshape(-1, 1, 2)],
                  False, RING, 1, cv2.LINE_AA)
    h, w = out.shape[:2]
    for wp, color in ((truth, TRUTH), (prediction, PRED)):
        if wp is None or not np.isfinite(wp).all():
            continue
        u, v = bev_pixels(np.asarray(wp, float).reshape(2), cfg)
        center = (int(round(np.clip(u, 9, w - 10))), int(round(np.clip(v, 9, h - 10))))
        # Off-image points stay visible: hollow circle on the border in their direction.
        inside = 0 <= u < w and 0 <= v < h
        cv2.circle(out, center, 7, color, -1 if inside else 2, cv2.LINE_AA)
    return title_bar(out, title) if title else out


def predict_overlays(raw_bgr, preprocessor, predictor):
    bev = preprocessor.bev(raw_bgr)
    wp = predictor.predict(bev)
    uv = project_waypoints(wp.reshape(1, 2), np.linalg.inv(preprocessor.H))
    raw_uv = to_distorted_pixels(uv, preprocessor.K, preprocessor.D, preprocessor.new_K)
    images = {
        'raw_overlay': draw_trajectory(raw_bgr, raw_uv, 'RAW CAMERA | magenta: prediction'),
        'undistorted_overlay': draw_trajectory(preprocessor.undistort(raw_bgr), uv,
                                               'UNDISTORTED | magenta: prediction'),
        'bev_overlay': bev_view(bev, preprocessor.cfg, wp, title='BEV | forward up, left left'),
        'bev_input': bev,
    }
    return wp, images
