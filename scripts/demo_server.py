"""Live demo: speak Brahui into the mic -> fine-tuned ASR (runs/ctc1b_brh_s2/final, on MPS) -> NLLB student
(runs/nllb_r4_1p3b/final, offline) -> English text -> English speech (macOS `say`).

  uv run python scripts/demo_server.py [--port 8766] [--mt nllb:runs/nllb_r4_1p3b/final | opus | none]
      [--asr ctc:runs/ctc1b_brh_s2/final] [--tts say|none] [--host 0.0.0.0 --https]
Then open http://localhost:8766 in Chrome. Space = start/stop recording. With --mt nllb:... everything runs on this
machine, no network. For phones on the same Wi-Fi use --host 0.0.0.0 --https (browsers only allow the microphone on
HTTPS or localhost; a self-signed certificate is generated in data/demo/ and must be accepted once).

Every take is kept: data/demo/<session>/<n>.wav (16 kHz mono) and data/demo/<session>/takes.jsonl with the
transcript, translation and an optional 1-5 rating of the English, so the session doubles as eval data.
"""
from __future__ import annotations

import argparse, json, subprocess, sys, tempfile, threading, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brahui.text import normalize, romanize  # noqa: E402
from asr_eval import make_backend, device_for, load_audio  # noqa: E402
from demo_cascade import translate  # noqa: E402


class NLLB:
    """Offline brh->en with the fine-tuned NLLB student (fp16 on MPS/CUDA so the 1.3B fits beside the ASR model)."""
    def __init__(self, path, device):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from train_nllb import BRH, ENG
        self.src, self.tgt, self.device = BRH, ENG, device
        self.tok = AutoTokenizer.from_pretrained(path)
        dt = torch.float16 if str(device) in ("mps", "cuda") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=dt).to(device).eval()
        self.name = f"nllb:{Path(path).parent.name}"

    def __call__(self, brh: str) -> str:
        import re, torch
        sents = [x.strip() for x in re.split(r"(?<=[۔؟!])\s+", brh) if x.strip()] or [brh]
        self.tok.src_lang = self.src
        with torch.no_grad():
            enc = self.tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            gen = self.model.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(self.tgt), num_beams=4, max_new_tokens=200)
        return " ".join(t.strip() for t in self.tok.batch_decode(gen, skip_special_tokens=True))


def speak(text: str, wav: Path, voice: str | None = None) -> bool:
    """English TTS with macOS `say` -> 22 kHz mono wav. Returns False if unavailable."""
    aiff = wav.with_suffix(".aiff")
    cmd = ["say", "-o", str(aiff)] + (["-v", voice] if voice else []) + [text]
    try:
        if subprocess.run(cmd, capture_output=True, timeout=60).returncode != 0:
            return False
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", str(wav)], capture_output=True)
        aiff.unlink(missing_ok=True)
        return r.returncode == 0
    except Exception:
        return False

HTML = ROOT / "scripts" / "demo" / "index.html"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766); ap.add_argument("--mt", default="nllb:runs/nllb_r4_1p3b/final")
    ap.add_argument("--asr", default="ctc:runs/ctc1b_brh_s2/final"); ap.add_argument("--device")
    ap.add_argument("--tts", default="say", help="say (macOS voice) or none"); ap.add_argument("--voice", default=None)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--https", action="store_true")
    ap.add_argument("--session", default=time.strftime("%Y%m%d_%H%M"))
    a = ap.parse_args()
    kind, _, mid = a.asr.partition(":")
    dev = device_for(a.device)
    asr = make_backend(f"{kind}:{ROOT / mid}" if (ROOT / mid).exists() else a.asr, dev)
    nllb = NLLB(ROOT / a.mt[5:], dev) if a.mt.startswith("nllb:") else None
    mt_name = nllb.name if nllb else a.mt
    lock = threading.Lock()
    out = ROOT / "data" / "demo" / a.session; out.mkdir(parents=True, exist_ok=True)
    log = out / "takes.jsonl"
    takes = []
    if log.exists():
        for l in open(log, encoding="utf-8"):
            r = json.loads(l)
            if "rate" in r:
                for t in takes:
                    if t["id"] == r["rate"]:
                        t.update({k: r[k] for k in ("score", "meant") if k in r})
            else:
                takes.append(r)

    def append(row):
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
                self._json({"session": a.session, "mt": mt_name, "asr": asr.name, "tts": a.tts, "takes": [{**t, "lat": romanize(t["brh"])} for t in takes]})
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
            if u.path == "/api/rate":
                d = json.loads(body); append({"rate": d["id"], **{k: d[k] for k in ("score", "meant") if k in d}})
                for t in takes:
                    if t["id"] == d["id"]:
                        t.update({k: d[k] for k in ("score", "meant") if k in d})
                return self._json({"ok": True})
            if u.path != "/api/transcribe":
                return self._json({"error": "bad path"}, 404)
            tid = len([t for t in takes]) + 1
            wav = out / f"{tid:04d}.wav"
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as t:
                t.write(body); tmp = t.name
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                               capture_output=True, text=True)
            Path(tmp).unlink(missing_ok=True)
            if r.returncode != 0:
                return self._json({"error": "ffmpeg: " + r.stderr[-300:]}, 500)
            info = sf.info(wav); dur = info.frames / info.samplerate
            t0 = time.time()
            with lock:
                brh = normalize(asr.transcribe(load_audio(wav)))
            t_asr = time.time() - t0
            en, t_mt, t_tts, err, en_audio = "", 0.0, 0.0, None, None
            if a.mt != "none" and brh.strip():
                t0 = time.time()
                try:
                    with lock:
                        en = nllb(brh) if nllb else translate(brh, a.mt)
                except Exception as e:  # keep the demo alive if the model or CLI hiccups
                    err = f"translation failed: {e}"[:200]
                t_mt = time.time() - t0
            if a.tts == "say" and en.strip():
                t0 = time.time(); ew = out / f"{tid:04d}_en.wav"
                if speak(en, ew, a.voice):
                    en_audio = ew.name
                t_tts = time.time() - t0
            row = {"id": tid, "audio": str(wav.relative_to(ROOT)), "duration": round(dur, 2), "brh": brh, "en": en, "en_audio": en_audio,
                   "t_asr": round(t_asr, 2), "t_mt": round(t_mt, 2), "t_tts": round(t_tts, 2), "ts": time.time()}
            takes.append(row); append(row)
            self._json({**row, "lat": romanize(brh), "error": err})

    srv = ThreadingHTTPServer((a.host, a.port), H)
    scheme = "http"
    if a.https:
        import ssl
        cert = ROOT / "data" / "demo" / "cert.pem"
        if not cert.exists():
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(cert), "-out", str(cert),
                            "-days", "365", "-subj", "/CN=brahui-demo"], check=True, capture_output=True)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(str(cert))
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True); scheme = "https"
    ip = ""
    if a.host == "0.0.0.0":
        import socket
        try:
            s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s_.connect(("8.8.8.8", 80)); ip = s_.getsockname()[0]; s_.close()
        except Exception:
            pass
    print(f"demo ready: asr={asr.name} mt={mt_name} tts={a.tts} session=data/demo/{a.session}  ->  open {scheme}://localhost:{a.port}"
          + (f"  (phones on this Wi-Fi: {scheme}://{ip}:{a.port})" if ip else ""), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
