"""Single-user localhost marker clicker, stdlib only (same security model as the label editor).

The page sends clicks only; every number it shows and everything /api/save writes is computed here.
"""
import base64
from dataclasses import asdict, dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys
import threading
from urllib.parse import urlparse
import cv2
import numpy as np
from . import core

WEB = Path(__file__).parent/'web'


@dataclass
class CalibrationSession:
    intr: core.Intrinsics
    raw: np.ndarray           # BGR frame as recorded; the BEV preview undistorts it like the car
    undistorted: np.ndarray   # core.undistort(raw): the image the clicks are pixels of
    markers: dict             # core.load_markers: {id: (x, y)} rear-axle metres
    cfg: object               # camsim Config of the course model: BEV size, training mask, ahead_m
    out_path: Path            # course calibration file (data/calibration/car.yaml)
    source: dict              # core.build_calibration source: frame, frame_stamp_ns, markers, markers_sha256


class Server(ThreadingHTTPServer):
    """A daemon thread per request: an idle socket a browser preconnected cannot stall the page's next request."""
    block_on_close = False   # nor Ctrl+C

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # a fetch the browser cancelled is not an error
            super().handle_error(request, client_address)


def png(image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('PNG로 인코딩할 수 없습니다.')
    return data.tobytes()


def parse_points(value, session):
    """{"points": {id: [u, v]}} -> {id: [u, v]}: marker-file ids, two numbers inside the undistorted image."""
    if not isinstance(value, dict) or set(value) != {'points'} or not isinstance(value['points'], dict):
        raise ValueError('요청 형식은 {"points": {마커 id: [u, v]}}입니다.')
    w, h = session.intr.width, session.intr.height
    points = {}
    for k, p in value['points'].items():
        if k not in session.markers:
            raise ValueError(f'마커 파일에 없는 id입니다: {k}')
        if not (isinstance(p, list) and len(p) == 2 and all(type(c) in (int, float) for c in p)):
            raise ValueError(f'{k}: 픽셀 좌표는 숫자 두 개 [u, v]여야 합니다.')
        if not (0 <= p[0] < w and 0 <= p[1] < h):   # also false for NaN and inf
            raise ValueError(f'{k}: 픽셀 좌표 {p}가 영상({w}x{h}) 밖이거나 유한하지 않습니다.')
        points[k] = [float(c) for c in p]
    return points


def fit(session, points):
    """core.evaluate + the car's BEV as a PNG data URL -> (JSON-safe result, preview BGR or None)."""
    result = dict(core.evaluate(points, session.markers, session.intr), bev=None)
    if result['H_i2g'] is None:
        return result, None
    try:
        calibration = core.calibration_dict(session.intr, result['H_i2g'])
        preview = core.bev_preview(session.raw, calibration, session.cfg, session.markers, points)
    except ValueError as exc:   # CameraPreprocessor refuses this H, so the car would refuse the saved file too
        problem = f'BEV 미리보기를 만들 수 없습니다: 이 H로는 학습 BEV 영역이 영상에 보이지 않습니다 ({exc}). 찍은 점을 확인하세요.'
        return dict(result, savable=False, reason=problem if result['savable'] else f'{problem} {result["reason"]}'), None
    return dict(result, bev='data:image/png;base64,' + base64.b64encode(png(preview)).decode('ascii')), preview


def make_server(session, port=0):
    """Server on 127.0.0.1 for one session; POSTs need the per-run token /api/session hands to the page."""
    token, image, intr, saving = secrets.token_urlsafe(32), png(session.undistorted), session.intr, threading.Lock()
    info = dict(width=intr.width, height=intr.height, markers=[dict(id=k, x=x, y=y) for k, (x, y) in session.markers.items()],
                ost=intr.path, ost_kind=core.ost_kind(intr.path), frame=str(session.source['frame']), out=str(session.out_path),
                image_sha256=hashlib.sha256(image).hexdigest(), hfov_deg=core.hfov_deg(intr), bev=asdict(session.cfg.bev),
                warn_cm=core.WARN_CM, reject_cm=core.REJECT_CM, min_fit=core.MIN_FIT, min_loo=core.MIN_LOO,
                min_save=core.MIN_SAVE, ahead_m=float(session.cfg.waypoints.ahead_m), token=token)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value, content_type='application/json'):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if content_type == 'application/json' else value
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def not_local(self):
            return self.reply(403, {'error':f'http://127.0.0.1:{self.server.server_port} 또는 localhost 주소로만 열 수 있습니다.'})

        def do_GET(self):
            if not self.local_request():
                return self.not_local()
            path = urlparse(self.path).path
            try:
                if path in ('/', '/app.js'):
                    file = WEB/('index.html' if path == '/' else 'app.js')
                    return self.reply(200, file.read_bytes(), 'text/html; charset=utf-8' if path == '/' else 'text/javascript; charset=utf-8')
                if path == '/api/session':
                    return self.reply(200, info)
                if path == '/image/undistorted.png':
                    return self.reply(200, image, 'image/png')
                self.reply(404, {'error':'없는 주소입니다.'})
            except OSError as exc:
                self.reply(400, {'error':f'파일을 읽을 수 없습니다: {exc.filename} ({exc.strerror})'})

        def do_POST(self):
            if not self.local_request():
                return self.not_local()
            if self.headers.get('X-Calibration-Token') != token:
                return self.reply(403, {'error':'토큰이 맞지 않습니다. 페이지를 새로 고치세요 (calibrate를 다시 실행하면 토큰이 바뀝니다).'})
            path = urlparse(self.path).path
            if path not in ('/api/fit', '/api/save'):
                return self.reply(404, {'error':'없는 주소입니다.'})
            try:
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    size = 0
                if not 0 < size <= 131072 or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('JSON 본문이 필요합니다 (Content-Type: application/json, 128 KiB 이하).')
                try:
                    value = json.loads(self.rfile.read(size))
                except (ValueError, RecursionError):   # not JSON, not UTF-8, or nested too deep
                    raise ValueError('본문이 올바른 JSON이 아닙니다 (UTF-8 JSON 객체를 보내세요).') from None
                points = parse_points(value, session)
                core.check_markers(session.source)   # an edited markers.yaml: restart, never fit or save stale markers
                result, preview = fit(session, points)
                if path == '/api/fit':
                    return self.reply(200, result)
                with saving:   # one save at a time: backup names and temporaries are per file
                    data = core.build_calibration(intr, result, points, session.markers, session.source)   # refuses unless savable
                    saved, backup = core.save_calibration(data, session.out_path, preview)
                self.reply(200, dict(saved=str(saved), backup=None if backup is None else str(backup), rms_cm=result['rms_cm']))
            except (KeyError, ValueError, TypeError, OSError) as exc:
                self.reply(400, {'error':str(exc)})

        def log_message(self, *_):
            pass

    return Server(('127.0.0.1', port), Handler)


def serve(session, port=8765):
    try:
        server = make_server(session, port)
    except OSError as exc:
        raise OSError(f'포트 {port}를 열 수 없습니다 ({exc.strerror}). 실행 중인 calibrate/label을 끄거나 --port로 다른 포트를 지정하세요.') from exc
    print(f'지면 캘리브레이션: http://127.0.0.1:{server.server_port} (브라우저에서 열기)', flush=True)
    print(f'저장 위치: {session.out_path} (기존 파일은 {session.out_path.parent/"old"}/에 백업) · 종료: Ctrl+C', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
