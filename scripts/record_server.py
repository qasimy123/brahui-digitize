"""Local recording tool for the native speaker: shows one English sentence, records her Brahui, saves 16 kHz WAV.

  uv run python scripts/record_server.py --speaker spk_f01 [--sentences data/recording/sentences.jsonl] [--port 8765]
Then open http://localhost:8765 in Chrome. Keys: Space = start/stop recording, Enter = accept & next, R = redo, S = skip, ← = back.

Writes data/recording/<speaker>/<rec_id>.wav (16 kHz mono PCM16, via ffmpeg) and appends to
data/recording/<speaker>/manifest.jsonl (rec_id, en, brh_ref, duration, timestamp). Resumes from the manifest.
Everything stays on this machine; nothing is uploaded anywhere.
"""
from __future__ import annotations

import argparse, json, subprocess, sys, tempfile, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "scripts" / "record" / "index.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True); ap.add_argument("--sentences", default="data/recording/sentences.jsonl")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    sents = [json.loads(l) for l in open(ROOT / a.sentences, encoding="utf-8")]
    out = ROOT / "data" / "recording" / a.speaker; out.mkdir(parents=True, exist_ok=True)
    man = out / "manifest.jsonl"

    def done_ids():
        if not man.exists():
            return {}
        d = {}
        for l in open(man, encoding="utf-8"):
            r = json.loads(l); d[r["rec_id"]] = r
        return d

    class H(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                b = HTML.read_bytes()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            elif u.path == "/api/state":
                d = done_ids()
                self._json({"speaker": a.speaker, "sentences": sents, "done": {k: {"duration": v["duration"], "skipped": v.get("skipped", False)} for k, v in d.items()}})
            elif u.path.startswith("/audio/"):
                p = out / Path(u.path).name
                if p.exists() and p.suffix == ".wav":
                    b = p.read_bytes(); self.send_response(200); self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
                else:
                    self.send_response(404); self.end_headers()
            else:
                self.send_response(404); self.end_headers()

        def do_POST(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n)
            rec_id = q.get("rec_id", [""])[0]
            sent = next((s for s in sents if s["rec_id"] == rec_id), None)
            if sent is None:
                return self._json({"error": "unknown rec_id"}, 400)
            if u.path == "/api/skip":
                row = {"rec_id": rec_id, "speaker": a.speaker, "en": sent["en"], "brh_ref": sent.get("brh_ref"), "skipped": True,
                       "duration": 0.0, "ts": time.time()}
                with open(man, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                return self._json({"ok": True})
            if u.path != "/api/upload":
                return self._json({"error": "bad path"}, 404)
            wav = out / f"{rec_id}.wav"
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as t:
                t.write(body); tmp = t.name
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                               capture_output=True, text=True)
            Path(tmp).unlink(missing_ok=True)
            if r.returncode != 0:
                return self._json({"error": "ffmpeg: " + r.stderr[-300:]}, 500)
            info = sf.info(wav); dur = info.frames / info.samplerate
            data, _ = sf.read(wav, dtype="float32"); peak = float(abs(data).max()) if len(data) else 0.0
            row = {"rec_id": rec_id, "speaker": a.speaker, "en": sent["en"], "brh_ref": sent.get("brh_ref"), "source": sent.get("source"),
                   "audio": str(wav.relative_to(ROOT)), "duration": round(dur, 3), "peak": round(peak, 3), "ts": time.time()}
            with open(man, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._json({"ok": True, "duration": dur, "peak": peak})

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"recording for speaker={a.speaker}: {len(sents)} sentences, {len(done_ids())} done. Open http://localhost:{a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
