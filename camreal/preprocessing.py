"""Explicit camera pixel contract; no implicit resize or crop."""
import cv2
import numpy as np
import yaml
from pathlib import Path
from camsim.render import ipm_bev, bev_size


class CameraPreprocessor:
    def __init__(self, calibration, cfg, training_mask, frame_id):
        # A path, or a calibration already loaded as a dict (calibrate's preview before saving).
        c = calibration if isinstance(calibration, dict) else yaml.safe_load(Path(calibration).read_text())
        self.calibration_status = c.get('calibration_status', 'unspecified')
        if c['schema_version'] != 1 or c['distortion_model'] != 'plumb_bob':
            raise ValueError('requires schema 1, OpenCV plumb_bob calibration')
        if c['ground_frame'] != frame_id:
            raise ValueError('calibration ground_frame must equal path_frame')
        if c['homography_space'] != 'undistorted_full_resolution':
            raise ValueError('H_i2g must refer to full-resolution undistorted pixels')
        self.width, self.height = int(c['image_width']), int(c['image_height'])
        if self.width <= 0 or self.height <= 0:
            raise ValueError('invalid calibration image dimensions')
        self.K = self._matrix(c['K'], 'K')
        self.new_K = self._matrix(c['new_K'], 'new_K')
        self.H = self._matrix(c['H_i2g'], 'H_i2g')
        self.D = np.asarray(c['D'], dtype=float)
        if self.D.ndim != 1 or len(self.D) not in (4, 5, 8, 12, 14) or not np.isfinite(self.D).all():
            raise ValueError('invalid OpenCV distortion coefficients')
        for matrix in (self.K, self.new_K):
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
                raise ValueError('invalid camera intrinsics')
        self.cfg = cfg
        self.training_mask = np.asarray(training_mask, dtype=bool)
        if self.training_mask.shape != bev_size(cfg):
            raise ValueError('training mask / BEV size mismatch')
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.new_K, (self.width, self.height), cv2.CV_32FC1)
        valid = ((self.mapx >= 0) & (self.mapx < self.width - 1) &
                 (self.mapy >= 0) & (self.mapy < self.height - 1)).astype(np.uint8)
        # ipm_bev uses the training floor border; explicit zero border for validity.
        from camsim.render import ground_to_bev_matrix
        h, w = bev_size(cfg)
        coverage = cv2.warpPerspective(valid, ground_to_bev_matrix(cfg) @ self.H,
                                      (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        self.mask = self.training_mask & (coverage != 0)
        if not self.mask.any():
            raise ValueError('calibration has no visible overlap with training BEV')

    def require_driving_calibration(self):
        if self.calibration_status == 'assumed':
            raise ValueError('ASSUMED calibration is preview-only; replace with measured calibration before enabling drive')

    @staticmethod
    def _matrix(value, name):
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
            raise ValueError(f'{name} must be a finite, nonsingular 3x3 matrix')
        return matrix

    def undistort(self, bgr):
        if bgr.shape != (self.height, self.width, 3) or bgr.dtype != np.uint8:
            raise ValueError('image must be uint8 BGR at calibration reference resolution; no resize/crop allowed')
        return cv2.remap(bgr, self.mapx, self.mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(self.cfg.lane.color_floor))

    def bev(self, bgr):
        bev = ipm_bev(self.undistort(bgr), self.H, self.cfg)
        bev[~self.mask] = self.cfg.lane.color_floor
        return bev
