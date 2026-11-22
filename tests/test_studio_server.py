"""Regression tests for StudioHandler CORS Origin and Host validation (#830)."""

import http.client
import socketserver
import threading
import pytest

from demo.studio_server import StudioHandler


@pytest.fixture(scope="module")
def studio_addr():
    class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True

    httpd = Server(("127.0.0.1", 0), StudioHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address
    httpd.shutdown()
    httpd.server_close()


def _request_acao(addr, origin, host):
    conn = http.client.HTTPConnection(addr[0], addr[1], timeout=5)
    try:
        conn.request("GET", "/api/model_info", headers={"Origin": origin, "Host": host})
        resp = conn.getresponse()
        resp.read()
        return resp.status, resp.getheader("Access-Control-Allow-Origin")
    finally:
        conn.close()


def test_cors_rejects_dns_rebinding_and_spoofed_origins(studio_addr):
    local = f"127.0.0.1:{studio_addr[1]}"
    for origin, host in [
        ("http://evil.example:8080", "evil.example:8080"),
        ("http://localhost:8080", "evil.example:8080"),
        ("http://evil.example@localhost:8080", local),
        ("file://localhost", local),
        ("http://localhost:badport", local),
        ("http://localhost:8080/api", local),
        ("http://localhost.evil.example:8080", local),
        ("http://evil-localhost:8080", local),
    ]:
        status, acao = _request_acao(studio_addr, origin, host)
        assert status == 200 and acao is None, f"unexpected ACAO={acao!r} for {origin!r}"


def test_cors_allows_strict_loopback_origins(studio_addr):
    for origin, host in [
        ("http://localhost:8080", f"127.0.0.1:{studio_addr[1]}"),
        ("https://127.0.0.1:8443", "localhost:8080"),
        ("http://[::1]:8080", "[::1]:8080"),
    ]:
        status, acao = _request_acao(studio_addr, origin, host)
        assert status == 200 and acao == origin
