"""Single-user localhost label editor, stdlib only; revisions prevent lost updates."""
import fcntl
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import urlparse
from .core import Project, ConflictError


def serve(project_path, port=8765):
    project = Project(project_path)
    lock = (project.root/'.labeler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    token = secrets.token_urlsafe(32)
    web = Path(__file__).parent/'web'

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

        def do_GET(self):
            if not self.local_request():
                return self.reply(403, {'error':'localhost only'})
            path = urlparse(self.path).path
            try:
                if path in ('/', '/app.js'):
                    file = web/('index.html' if path == '/' else 'app.js')
                    return self.reply(200, file.read_bytes(), 'text/html; charset=utf-8' if path == '/' else 'text/javascript; charset=utf-8')
                if path == '/api/project':
                    return self.reply(200, dict(session_id=project.meta['session_id'], frames=[dict(f, status=project.annotation(f['id'])['status']) for f in project.meta['frames']],
                        bev=project.meta['contract']['bev'], ahead_m=project.ahead_m,
                        path_line=project.cfg.waypoints.line, calibration_status=project.meta['calibration_status'],
                        token=token))
                parts = path.strip('/').split('/')
                if len(parts) == 3 and parts[:2] == ['api','frame']:
                    return self.reply(200, project.annotation(parts[2]))
                if len(parts) == 3 and parts[0] == 'images' and parts[1] in ('bev','raw') and parts[2] in project.frames:
                    return self.reply(200, (project.root/parts[1]/f'{parts[2]}.png').read_bytes(), 'image/png')
                self.reply(404, {'error':'not found'})
            except (KeyError, ValueError, OSError) as exc:
                self.reply(400, {'error':str(exc)})

        def do_POST(self):
            if not self.local_request() or self.headers.get('X-Label-Token') != token:
                return self.reply(403, {'error':'invalid local editor token'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 131072 or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('JSON body required (max 128 KiB)')
                value = json.loads(self.rfile.read(size))
                if not isinstance(value, dict):
                    raise ValueError('JSON object required')
                parts = urlparse(self.path).path.strip('/').split('/')
                if len(parts) == 3 and parts[:2] == ['api','annotation']:
                    return self.reply(200, project.save(parts[2], value))
                self.reply(404, {'error':'not found'})
            except ConflictError as exc:
                self.reply(409, {'error':str(exc)})
            except (KeyError, ValueError, TypeError, OSError) as exc:
                self.reply(400, {'error':str(exc)})

        def log_message(self, *_):
            pass

    server = HTTPServer(('127.0.0.1', port), Handler)
    print(f'Label editor: http://127.0.0.1:{server.server_port} | {project.meta["session_id"]}', flush=True)
    print(f'라벨 저장 위치: {project.root/"annotations"} · 종료: Ctrl+C', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        lock.close()
