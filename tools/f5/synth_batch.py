"""Teacher synthesis for the small-model data set: speak a script (jsonl with id, vowelled) in one reference voice.
  python tools/f5/synth_batch.py --ckpt <model.safetensors> --vocab <vocab.txt> --script data/tts/distill/script.jsonl
      --ref data/tts/distill/ref --out data/tts/distill/synth [--shard 0/4] [--limit 0] [--speed 0.75] [--nfe 32]
Writes <out>/wavs/<id>.wav (24 kHz). Resumable; run one process per shard (several fit on one GPU). Each sentence gets its
own seed so the set does not share one noise pattern."""
import argparse, json, time, zlib
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--vocab", required=True); ap.add_argument("--script", required=True); ap.add_argument("--ref", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--speed", type=float, default=0.75); ap.add_argument("--nfe", type=int, default=32)
a = ap.parse_args(); k, n = map(int, a.shard.split("/"))
rows = [json.loads(l) for l in open(a.script, encoding="utf-8")]
if a.limit: rows = rows[:a.limit]
rows = rows[k::n]; out = Path(a.out) / "wavs"; out.mkdir(parents=True, exist_ok=True)
ref = Path(a.ref); rw, rt = [l.rstrip("\n").split("|", 1) for l in open(ref / "metadata.csv", encoding="utf-8")][1]
from f5_tts.api import F5TTS  # noqa: E402
import soundfile as sf  # noqa: E402
tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=a.ckpt, vocab_file=a.vocab); quiet = lambda *x, **k: None  # noqa: E731
t0 = time.time(); sec = 0.0; done = 0
for i, r in enumerate(rows):
    dst = out / f"{r['id']}.wav"
    if not dst.exists():
        tmp = out / f".{r['id']}.tmp.wav"
        try:
            tts.infer(ref_file=str(ref / rw), ref_text=rt, gen_text=r["vowelled"], file_wave=str(tmp), nfe_step=a.nfe, speed=a.speed, seed=zlib.crc32(r["id"].encode()) % 2**31, show_info=quiet, progress=None)
            sec += sf.info(str(tmp)).duration; tmp.rename(dst); done += 1
        except Exception as e: print("FAILED", r["id"], repr(e)[:200], flush=True)
    if (i + 1) % 50 == 0 or i + 1 == len(rows):
        el = time.time() - t0; print(f"shard {a.shard}: {i + 1}/{len(rows)}  new {done}  audio {sec / 60:.1f} min in {el / 60:.1f} min  (x{sec / max(el, 1):.1f} real time)", flush=True)
print(f"SHARD {a.shard} DONE", flush=True)
