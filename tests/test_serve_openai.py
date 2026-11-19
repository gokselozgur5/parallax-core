import http.client, socket, subprocess, sys, time

def _check(port):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    c.request("GET", "/v1/models")
    return c.getresponse().status

def test_idle_connection_does_not_block_other_clients():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    proc = subprocess.Popen([sys.executable, "-c", f"from scripts.serve_openai import run_server; run_server(port={port})"])
    idle = None
    try:
        for _ in range(50):
            try:
                if _check(port) == 200: break
            except OSError: time.sleep(0.05)
        idle = socket.create_connection(("127.0.0.1", port))
        assert _check(port) == 200
    finally:
        if idle: idle.close()
        proc.kill()
        proc.wait()
