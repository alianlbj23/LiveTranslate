"""Child process for test_abandoned_streams_do_not_deadlock_the_pool: exits 0 when
200 abandoned streams + one full translation complete, hangs (and is killed by the
parent's timeout) if closing an abandoned stream deadlocks the connection pool."""
import gc
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, sys.argv[1])
from translator import Translator  # noqa: E402

SENTENCE = "今天天氣很好我們一起去公園散步然後吃午餐"


class SSE(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for ch in SENTENCE:
                d = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                     "choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}]}
                self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(0.01)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass  # client hung up on an abandoned stream


class Quiet(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass


srv = Quiet(("127.0.0.1", 0), SSE)
threading.Thread(target=srv.serve_forever, daemon=True).start()
tr = Translator(api_base=f"http://127.0.0.1:{srv.server_address[1]}/v1", api_key="x", model="m",
                target_language="zh", streaming=True, timeout=10)
pool = tr._client._client._transport._pool
original = pool._assign_requests_to_connections


def assign_with_gc():
    gc.collect()  # what an allocation-triggered collection does at this point
    return original()


pool._assign_requests_to_connections = assign_with_gc
gc.disable()  # only the collection inside the pool lock may finalize streams
for _ in range(20):
    for _partial in tr.with_target_language("zh").translate_iter("テスト", "ja"):
        break  # stale live preview
out = None
for out in tr.translate_iter("こんにちは", "ja"):
    pass
tr._client.close()
srv.shutdown()
print("OUT=" + out)
sys.exit(0 if out == SENTENCE else 1)
