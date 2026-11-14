"""Static server for the demo with cross-origin isolation headers (COOP/COEP) so ONNX Runtime Web can use
SharedArrayBuffer and run WASM inference multi-threaded. Usage: python3 demo/web/serve.py [port]"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class H(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def log_message(self, *a):
        pass


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
ThreadingHTTPServer(("", port), partial(H, directory=str(Path(__file__).parent))).serve_forever()
