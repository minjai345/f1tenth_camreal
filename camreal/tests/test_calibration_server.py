import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import select
import shutil
import socket
import subprocess
import sys
import threading
from urllib.request import Request, urlopen
import cv2
import numpy as np
import pytest
import yaml
from camsim import camera, config, render
from camreal.__main__ import main
from camreal.calibration import core, server as calibration_server
from camreal.calibration.server import CalibrationSession, make_server
from camreal.checkpoint import training_mask
from camreal.overlay import to_distorted_pixels
from camreal.preprocessing import CameraPreprocessor
from conftest import TEMPLATE, assumed, clicks, make_model_dir

POSE = dict(x_m=.1, y_m=0., height_m=.2, pitch_deg=10., yaw_deg=0., roll_deg=0.)


def mounted(intr):
    """Undistorted-image H_g2i of the slide-84 lens (new_K = P) mounted like camsim's camera: 0.20 m, pitch 10, 0.1 m ahead."""
    return intr.new_K @ camera.extrinsics(assumed(), 10.)[:, [0, 1, 3]]


def tape_frame(intr, H_g2i, markers):
    """Grey floor with a white 5 cm tape cross (16 cm arms) on every marker, as the distorting lens records it."""
    raw = np.full((intr.height, intr.width, 3), 128, np.uint8)
    corners = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.], [-1., -1.]])
    t = np.linspace(0., 1., 16, endpoint=False)[:, None]
    square = np.vstack([a + t * (b - a) for a, b in zip(corners[:-1], corners[1:])])   # densified: the lens bends straight tape
    for xy in markers.values():
        for half in ([.08, .025], [.025, .08]):
            uv = to_distorted_pixels(camera.project(H_g2i, np.asarray(xy) + square * half), intr.K, intr.D, intr.new_K)
            cv2.fillPoly(raw, [np.rint(uv * 16).astype(np.int32)], (255, 255, 255), cv2.LINE_AA, 4)
    return raw


@pytest.fixture
def session(ost, tmp_path):
    intr = core.read_ost(ost)
    path = tmp_path/'markers.yaml'
    shutil.copy(TEMPLATE, path)   # a copy a test may edit during the session
    markers, sha256 = core.load_markers(path)
    raw = tape_frame(intr, mounted(intr), markers)
    return CalibrationSession(intr, raw, core.undistort(raw, intr), markers, config.load(), tmp_path/'calibration'/'car.yaml',
                              dict(frame=str(tmp_path/'frame.png'), frame_stamp_ns=None, markers=str(path), markers_sha256=sha256))


@pytest.fixture
def points(session):
    return clicks(mounted(session.intr), session.markers)   # exact clicks; A3 falls outside the P view


@pytest.fixture
def server(session):
    srv = make_server(session)
    thread = threading.Thread(target=srv.serve_forever, kwargs=dict(poll_interval=.02), daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(5)


def request(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection('127.0.0.1', srv.server_port, timeout=10)
    try:
        conn.request(method, path, body, headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def info(srv):
    return json.loads(request(srv, 'GET', '/api/session')[2])


def post(srv, path, value):
    headers = {'Content-Type': 'application/json', 'X-Calibration-Token': info(srv)['token']}
    status, _, body = request(srv, 'POST', path, json.dumps(value).encode(), headers)
    return status, json.loads(body)


def test_synthetic_frame_needs_the_undistortion(session, points):
    assert len(points) == 11 and 'A3' not in points
    for k, (u, v) in points.items():
        assert session.undistorted[round(v), round(u)].min() == 255, k   # each exact click is on its tape cross
    assert session.raw[round(points['D1'][1]), round(points['D1'][0])].max() == 128   # but not in the raw frame


def test_session_page_and_image(server, session):
    s = info(server)
    assert (s['width'], s['height'], s['ahead_m']) == (1480, 1080, 1.)
    assert s['markers'][0] == dict(id='A1', x=.6, y=.4) and [m['id'] for m in s['markers']] == list(session.markers)
    assert (s['ost'], s['ost_kind'], s['frame'], s['out']) == (session.intr.path, '--ost로 지정한 파일', session.source['frame'],
                                                               str(session.out_path))
    assert (s['warn_cm'], s['reject_cm'], s['min_fit'], s['min_loo'], s['min_save']) == (5., 20., 4, 5, 6)
    assert s['hfov_deg'] == pytest.approx(math.degrees(2 * math.atan(740 / 784.36676)), abs=.05) and len(s['token']) > 30
    assert s['bev'] == dict(x_range_m=[.2, 4.], y_range_m=[-1.5, 1.5], resolution_m=.01)   # the course model's BEV
    status, headers, body = request(server, 'GET', '/image/undistorted.png')
    assert status == 200 and headers['Content-Type'] == 'image/png'
    assert s['image_sha256'] == hashlib.sha256(body).hexdigest()   # the page keeps clicks per image under this key
    assert headers['Cache-Control'] == 'no-store' and headers['X-Content-Type-Options'] == 'nosniff'
    np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR), session.undistorted)
    for path, text in (('/', '지면 캘리브레이션'), ('/app.js', '/api/fit')):
        status, _, body = request(server, 'GET', path)
        assert status == 200 and text in body.decode()
    assert request(server, 'GET', '/api/nothing')[0] == 404


def test_token_and_local_host_required(server):
    body, port = json.dumps({'points': {}}).encode(), server.server_port
    for headers in ({'Content-Type': 'application/json'}, {'Content-Type': 'application/json', 'X-Calibration-Token': 'guess'}):
        status, _, reply = request(server, 'POST', '/api/fit', body, headers)
        assert status == 403 and '토큰' in json.loads(reply)['error']
    good = {'Content-Type': 'application/json', 'X-Calibration-Token': info(server)['token']}
    assert request(server, 'POST', '/api/fit', body, good)[0] == 200
    for host in (f'evil.example:{port}', '127.0.0.1:1'):   # DNS rebinding, another port
        for status, _, reply in (request(server, 'GET', '/api/session', headers={'Host': host}),
                                 request(server, 'POST', '/api/fit', body, dict(good, Host=host))):
            assert status == 403 and f'127.0.0.1:{port}' in json.loads(reply)['error'] and '토큰' not in json.loads(reply)['error']
    assert request(server, 'GET', '/api/session', headers={'Host': f'localhost:{port}'})[0] == 200


@pytest.mark.parametrize('value,words', [
    ([], ['"points"']), ({'pts': {}}, ['"points"']), ({'points': []}, ['"points"']),
    ({'points': {}, 'savable': True}, ['"points"']),   # clicks only: the server computes everything else
    ({'points': {'Z9': [1., 2.]}}, ['Z9']), ({'points': {'A1': [1., 2., 3.]}}, ['A1']), ({'points': {'A1': ['1', 2.]}}, ['A1']),
    ({'points': {'A1': [True, 2.]}}, ['A1']), ({'points': {'A1': [float('nan'), 2.]}}, ['A1', '1480x1080']),
    ({'points': {'A1': [1480, 2.]}}, ['A1', '1480x1080']), ({'points': {'A1': [10., -.5]}}, ['A1', '1480x1080']),
])
def test_bad_points_are_400(server, value, words):
    status, reply = post(server, '/api/fit', value)
    assert status == 400 and all(w in reply['error'] for w in words)


def test_body_must_be_small_json(server):
    token = info(server)['token']
    for body, headers in ((b'{"points": {', {}), (b'{"points": {}}', {'Content-Type': 'text/plain'}),
                          (b'{"points": {}}', {'Content-Length': '200000'}), (b'{"points": {}}', {'Content-Length': 'abc'}),
                          ('{"points": {"A1": "\u00ff"}}'.encode('latin-1'), {}),   # not UTF-8
                          (b'[' * 100000, {})):   # nested too deep for json (RecursionError)
        status, _, reply = request(server, 'POST', '/api/fit', body,
                                   {'Content-Type': 'application/json', 'X-Calibration-Token': token, **headers})
        error = json.loads(reply)['error']
        assert status == 400 and 'JSON' in error and re.search('[가-힣]', error), error   # Korean, not Python's words
        assert 'invalid literal' not in error and 'codec' not in error


def test_a_stalled_connection_does_not_block_the_page(server):
    port, token = server.server_port, info(server)['token']
    half = socket.create_connection(('127.0.0.1', port))   # a body that never fully arrives
    half.sendall(f'POST /api/fit HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nX-Calibration-Token: {token}\r\n'
                 f'Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{{"points"'.encode())
    idle = socket.create_connection(('127.0.0.1', port))   # Chrome keeps preconnected sockets like this one open
    try:
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
        conn.request('GET', '/api/session')
        assert conn.getresponse().status == 200
        conn.close()
        assert post(server, '/api/fit', {'points': {}})[0] == 200
    finally:
        half.close()
        idle.close()


def test_a_cancelled_fetch_is_not_an_error(server, capsys):
    try:
        raise BrokenPipeError(32, 'Broken pipe')   # what writing to a closed browser socket raises
    except BrokenPipeError:
        server.handle_error(None, ('127.0.0.1', 1))
    assert capsys.readouterr().err == ''
    try:
        raise RuntimeError('a real bug')
    except RuntimeError:
        server.handle_error(None, ('127.0.0.1', 1))
    assert 'a real bug' in capsys.readouterr().err


def test_fit_grows_with_the_clicks(server, session, points):
    three = {k: points[k] for k in ('A1', 'C3', 'D1')}
    status, r = post(server, '/api/fit', {'points': three})
    assert status == 200 and r['n'] == 3 and r['errors_cm'] == {} and r['bev'] is None and not r['savable'] and '4개' in r['reason']
    five = dict(three, B2=points['B2'], D3=points['D3'])
    status, r = post(server, '/api/fit', {'points': five})
    assert r['n'] == 5 and set(r['errors_cm']) == set(five) and not r['savable'] and '6개' in r['reason']
    assert r['bev'].startswith('data:image/png;base64,')
    status, r = post(server, '/api/fit', {'points': dict(five, A2=points['A2'])})
    assert r['n'] == 6 and r['savable'] and max(r['errors_cm'].values()) < .05 and r['rms_cm'] < .05
    assert r['pose'] == pytest.approx(POSE, abs=1e-3)
    bev = cv2.imdecode(np.frombuffer(base64.b64decode(r['bev'].split(',', 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert bev.shape[:2] == render.bev_size(session.cfg)
    for name, color in (('B2', (0, 200, 0)), ('C2', (200, 200, 200))):   # clicked green, not clicked grey
        u, v = np.rint(render.bev_pixels(np.array(session.markers[name]), session.cfg)).astype(int)
        assert tuple(int(c) for c in bev[v, u]) == color


def test_preview_the_car_would_reject_blocks_saving(server, session, points, monkeypatch):
    def no_overlap(*_):
        raise ValueError('calibration has no visible overlap with training BEV')   # CameraPreprocessor's words
    monkeypatch.setattr(core, 'bev_preview', no_overlap)
    status, r = post(server, '/api/fit', {'points': points})
    assert status == 200 and r['bev'] is None and not r['savable'] and 'BEV' in r['reason']
    status, r = post(server, '/api/save', {'points': points})
    assert status == 400 and 'BEV' in r['error'] and not session.out_path.parent.exists()


def test_markers_edited_after_the_start_are_refused(server, session, points):
    assert post(server, '/api/fit', {'points': points})[0] == 200
    path = Path(session.source['markers'])
    path.write_text(path.read_text().replace('B2: [1.00, 0.00]', 'B2: [1.02, 0.01]'))   # re-measured mid-session
    for api in ('/api/fit', '/api/save'):
        status, r = post(server, api, {'points': points})
        assert status == 400 and 'markers.yaml' in r['error'] and '다시 실행' in r['error']
    assert not session.out_path.parent.exists()


def test_saves_at_the_same_time_keep_every_backup(server, session, points):
    with ThreadPoolExecutor(6) as pool:
        replies = list(pool.map(lambda _: post(server, '/api/save', {'points': points}), range(6)))
    assert [status for status, _ in replies] == [200] * 6
    assert sorted(r['backup'] is None for _, r in replies) == [False] * 5 + [True]
    assert len(list((session.out_path.parent/'old').iterdir())) == 5 and not list(session.out_path.parent.glob('*.tmp'))


def test_save_recomputes_writes_and_backs_up(server, session, points):
    status, r = post(server, '/api/save', {'points': dict(list(points.items())[:5])})
    assert status == 400 and '6개' in r['error'] and not session.out_path.parent.exists()
    status, r = post(server, '/api/save', {'points': points})
    assert status == 200 and r == dict(saved=str(session.out_path), backup=None, rms_cm=r['rms_cm']) and r['rms_cm'] < .05
    first = session.out_path.read_text()
    data = yaml.safe_load(first)
    assert data['source']['frame'] == session.source['frame'] and data['source']['frame_stamp_ns'] is None
    assert [m['id'] for m in data['markers']] == list(points)
    assert data['camera_pose_estimate'] == pytest.approx(dict(POSE, hfov_deg=info(server)['hfov_deg']), abs=1e-3)
    assert data['source']['markers_sha256'] == session.source['markers_sha256']
    assert cv2.imread(str(session.out_path.with_name('car_bev.png'))).shape[:2] == render.bev_size(session.cfg)
    pre = CameraPreprocessor(session.out_path, session.cfg, training_mask(session.cfg), 'rear_axle')
    pre.require_driving_calibration()
    np.testing.assert_allclose(camera.project(pre.H, points['B2']), session.markers['B2'], atol=1e-3)
    status, r = post(server, '/api/save', {'points': points})
    assert status == 200 and Path(r['backup']).parent == session.out_path.parent/'old' and Path(r['backup']).read_text() == first


def course(tmp_path, model):
    c = yaml.safe_load(Path('camreal/config/course.yaml').read_text())
    c.update(model=str(model), calibration=str(tmp_path/'out'/'car.yaml'))
    c['paths']['bags'] = str(tmp_path)
    (tmp_path/'course.yaml').write_text(yaml.safe_dump(c))
    return str(tmp_path/'course.yaml')


@contextmanager
def calibrate_command(*args):
    """python -m camreal calibrate ... --port 0 in the background -> (what it printed, base URL)."""
    with subprocess.Popen([sys.executable, '-u', '-m', 'camreal', 'calibrate', *args, '--port', '0'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        try:
            out = ''
            while 'http://' not in out:
                ready, _, _ = select.select([proc.stdout], [], [], 30)
                line = proc.stdout.readline() if ready else ''
                if not line:
                    proc.kill()
                    raise AssertionError(out + proc.stderr.read())
                out += line
            yield out, re.search(r'http://127\.0\.0\.1:\d+', out).group()
        finally:
            proc.terminate()
            proc.wait(timeout=10)


def save_through(base, points):
    with urlopen(base + '/api/session', timeout=10) as response:
        token = json.load(response)['token']
    save = Request(base + '/api/save', json.dumps({'points': points}).encode(),
                   {'Content-Type': 'application/json', 'X-Calibration-Token': token})
    with urlopen(save, timeout=30) as response:
        return json.load(response)


@pytest.mark.parametrize('trained', [True, False])
def test_calibrate_command_serves_an_image(session, points, tmp_path, trained):
    model, cfg = make_model_dir(tmp_path)[:2] if trained else (tmp_path/'no_model', config.load())
    cv2.imwrite(str(tmp_path/'frame.png'), session.raw)
    args = ('--image', os.path.relpath(tmp_path/'frame.png'), '--ost', session.intr.path, '--markers', str(TEMPLATE))
    with calibrate_command(*args, '--config', course(tmp_path, model)) as (out, base):
        assert '--ost로 지정한 파일' in out and '1480x1080' in out and ('주행 모델과 같음' if trained else 'camsim 기본 설정') in out
        assert save_through(base, points)['saved'] == str(tmp_path/'out'/'car.yaml')
    assert yaml.safe_load((tmp_path/'out'/'car.yaml').read_text())['source']['frame'] == str((tmp_path/'frame.png').resolve())
    # BEV size of the course model (0.05 m/px in the test model), camsim's default without one.
    assert cv2.imread(str(tmp_path/'out'/'car_bev.png')).shape[:2] == render.bev_size(cfg)


def calibrate_here(monkeypatch, *args):
    """main(['calibrate', ...]) in this process, the server swapped out -> the CalibrationSession it would serve."""
    started = []
    monkeypatch.setattr(calibration_server, 'serve', lambda session, port: started.append(session))
    main(['calibrate', *args])
    return started[0]


def test_calibrate_expands_home_and_checks_the_image(session, tmp_path, monkeypatch, capfd):
    home = tmp_path/'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    cv2.imwrite(str(home/'frame.png'), session.raw)
    shutil.copy(TEMPLATE, home/'markers.yaml')
    options = ('--ost', session.intr.path, '--config', course(tmp_path, tmp_path/'no_model'))
    started = calibrate_here(monkeypatch, '--image', '~/frame.png', '--markers', '~/markers.yaml', *options)
    assert started.source['frame'] == str((home/'frame.png').resolve()) and started.markers == session.markers
    assert started.source['markers'] == str((home/'markers.yaml').resolve())
    assert started.source['markers_sha256'] == hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    capfd.readouterr()
    with pytest.raises(SystemExit) as error:
        calibrate_here(monkeypatch, '--image', '~/missing.png', '--markers', '~/markers.yaml', *options)
    err = capfd.readouterr().err
    assert error.value.code == 2 and '이미지 파일이 없습니다' in err and str((home/'missing.png').resolve()) in err
    assert 'findDecoder' not in err   # OpenCV never tried


def test_missing_ost_offers_the_reference_only_when_there_is_one(tmp_path, monkeypatch, capsys):
    args = ('--image', 'frame.png', '--ost', str(tmp_path/'ost.yaml'), '--config', course(tmp_path, tmp_path/'no_model'))
    monkeypatch.setattr(core, 'REFERENCE', tmp_path/'missing_reference.yaml')   # before section 7
    with pytest.raises(SystemExit):
        calibrate_here(monkeypatch, *args)
    err = capsys.readouterr().err
    assert 'ost.yaml이 없습니다' in err and 'ost_reference' not in err
    monkeypatch.setattr(core, 'REFERENCE', TEMPLATE)   # any existing file stands in for the reference
    with pytest.raises(SystemExit):
        calibrate_here(monkeypatch, *args)
    assert '--ost camreal/config/ost_reference_1920x1200.yaml' in capsys.readouterr().err


def test_calibrate_command_reads_the_middle_bag_frame(session, points, tmp_path):
    rosbag = pytest.importorskip('rosbag2_py')
    from rclpy.serialization import serialize_message
    from cv_bridge import CvBridge
    writer, bridge = rosbag.SequentialWriter(), CvBridge()
    writer.open(rosbag.StorageOptions(uri=str(tmp_path/'calib'), storage_id='sqlite3'), rosbag.ConverterOptions('', ''))
    writer.create_topic(rosbag.TopicMetadata(name='/flir_camera/image_raw', type='sensor_msgs/msg/Image', serialization_format='cdr'))
    for i in range(3):
        msg = bridge.cv2_to_imgmsg(session.raw, encoding='bgr8')
        msg.header.stamp.sec, msg.header.stamp.nanosec = 10, i * 100000000
        writer.write('/flir_camera/image_raw', serialize_message(msg), 10000000000 + i * 100000000)
    del writer
    args = ('calib', '--ost', session.intr.path, '--markers', str(TEMPLATE))
    with calibrate_command(*args, '--config', course(tmp_path, tmp_path/'no_model')) as (out, base):
        assert '가운데 메시지' in out and '10100000000' in out and 'camsim 기본 설정' in out
        save_through(base, points)
    source = yaml.safe_load((tmp_path/'out'/'car.yaml').read_text())['source']
    assert source['frame'] == str(tmp_path/'calib') and source['frame_stamp_ns'] == 10100000000
