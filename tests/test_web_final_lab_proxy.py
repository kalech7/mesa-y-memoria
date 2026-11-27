"""Ensure saved bookmarks share a browser origin with their archived v5 chats."""
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

from web_demo.__main__ import FINAL_GET, FINAL_POST, Handler, MAX_REQUEST, STATIC


@contextmanager
def running(handler):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, args=(0.02,), daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join()


def fetch(server, method, path, payload=None, content_type="application/json"):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        headers = {"Content-Type": content_type} if payload is not None else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read()
    finally:
        connection.close()


class Upstream(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(self.path.encode())

    def do_POST(self):
        # A failed model request must preserve its status, body and media type.
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(422)
        self.send_header("Content-Type", self.headers["Content-Type"])
        self.end_headers()
        self.wfile.write(body)


def test_allowlisted_final_routes_and_existing_archive_keep_same_origin():
    with running(Handler) as legacy, running(Upstream) as final:
        legacy.final_lab_port = final.server_port
        legacy.lab = SimpleNamespace(profiles=SimpleNamespace(catalog=lambda: {"legacy": True}))
        for path in FINAL_GET:
            request = path + "?v=2"
            assert fetch(legacy, "GET", request) == (200, "text/plain", request.encode())
        for path in FINAL_POST:
            content_type = "application/json; charset=utf-8"
            body = '{"text":"¿Qué recuerdo?"}'.encode()
            assert fetch(legacy, "POST", path, body, content_type) == (422, content_type, body)
            assert fetch(legacy, "POST", path, b"x" * (FINAL_POST[path] + 1))[0] == 413
            assert fetch(legacy, "POST", path, b"{}", "text/plain")[0] == 415
        long_message = b'{"message":"' + b"x" * MAX_REQUEST + b'"}'
        assert fetch(legacy, "POST", "/api/food/chat", long_message)[2] == long_message
        assert fetch(legacy, "GET", "/archivo")[2] == (STATIC / "index.html").read_bytes()
        assert fetch(legacy, "GET", "/api/profiles/catalog")[2] == b'{"legacy": true}'
        assert fetch(legacy, "GET", "/static/theme.js")[0] == 200
        assert fetch(legacy, "GET", "/not-final")[0] == 404


def test_default_v5_entry_and_unavailable_final_service():
    with running(Handler) as legacy:
        assert fetch(legacy, "GET", "/")[2] == (STATIC / "index.html").read_bytes()
        assert fetch(legacy, "GET", "/archivo")[0] == 404
        assert fetch(legacy, "GET", "/chat")[0] == 404
        with running(Upstream) as final:
            legacy.final_lab_port = final.server_port
        status, kind, body = fetch(legacy, "GET", "/chat")
        assert status == 502 and kind.startswith("application/json") and b"v6" in body
        assert fetch(legacy, "GET", "/archivo")[0] == 200
