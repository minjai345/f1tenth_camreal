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
            raise ValueError('schema_version 1, distortion_model plumb_bob인 캘리브레이션 파일이어야 합니다'
                             f'(지금 {c["schema_version"]}, {c["distortion_model"]}).')
        if c['ground_frame'] != frame_id:
            raise ValueError(f'캘리브레이션의 ground_frame({c["ground_frame"]})과 path_frame({frame_id})이 같아야 합니다.')
        if c['homography_space'] != 'undistorted_full_resolution':
            raise ValueError('H_i2g는 왜곡 보정된 원본 해상도 영상 기준이어야 합니다'
                             '(homography_space: undistorted_full_resolution).')
        self.width, self.height = int(c['image_width']), int(c['image_height'])
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f'캘리브레이션의 image_width, image_height가 잘못되었습니다({self.width}x{self.height}).')
        self.K = self._matrix(c['K'], 'K')
        self.new_K = self._matrix(c['new_K'], 'new_K')
        self.H = self._matrix(c['H_i2g'], 'H_i2g')
        self.D = np.asarray(c['D'], dtype=float)
        if self.D.ndim != 1 or len(self.D) not in (4, 5, 8, 12, 14) or not np.isfinite(self.D).all():
            raise ValueError('D(왜곡 계수)는 유한한 값 4, 5, 8, 12, 14개 중 하나여야 합니다.')
        for matrix in (self.K, self.new_K):
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
                raise ValueError('K, new_K는 fx, fy가 0보다 크고 마지막 행이 [0, 0, 1]이어야 합니다.')
        self.cfg = cfg
        self.training_mask = np.asarray(training_mask, dtype=bool)
        if self.training_mask.shape != bev_size(cfg):
            raise ValueError('학습 가시 영역(training mask)과 BEV 크기가 다릅니다.')
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
            raise ValueError('캘리브레이션(H_i2g)으로 보이는 바닥이 학습 BEV와 겹치지 않습니다. 캘리브레이션을 확인하세요.')

    def require_driving_calibration(self):
        if self.calibration_status == 'assumed':
            raise ValueError('가정 캘리브레이션(calibration_status: assumed)은 미리보기 전용입니다. '
                             'python3 -m camreal calibrate로 실측한 뒤 주행하세요.')

    @staticmethod
    def _matrix(value, name):
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
            raise ValueError(f'{name}는 유한하고 역행렬이 있는 3x3 행렬이어야 합니다.')
        return matrix

    def undistort(self, bgr):
        h, w = bgr.shape[:2]
        if (w, h) != (self.width, self.height):   # no resize/crop: the fix is a setting
            fixes = [f'카메라 해상도를 {self.width}x{self.height}에 맞추기',
                     f'{w}x{h}에서 1주차 방식으로 다시 캘리브레이션한 뒤 python3 -m camreal calibrate 다시 실행',
                     ('' if (w, h) == (1920, 1200) else '카메라 해상도를 1920x1200에 맞추고 ') +
                     'python3 -m camreal calibrate에 --ost camreal/config/ost_reference_1920x1200.yaml을 붙여 다시 실행']
            raise ValueError(f'영상 해상도({w}x{h})와 캘리브레이션 해상도({self.width}x{self.height})가 다릅니다. '
                             '다음 중 하나로 맞추세요: ' + ' '.join(f'{i}) {fix}' for i, fix in enumerate(fixes, 1)))
        if bgr.shape[2:] != (3,) or bgr.dtype != np.uint8:
            raise ValueError('uint8 BGR(3채널) 영상이어야 합니다.')
        return cv2.remap(bgr, self.mapx, self.mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(self.cfg.lane.color_floor))

    def bev(self, bgr):
        bev = ipm_bev(self.undistort(bgr), self.H, self.cfg)
        bev[~self.mask] = self.cfg.lane.color_floor
        return bev
