"""Pod side of a streamed bulk transcription: transcribe files as the Mac's rsync delivers them.

  python scripts/gpu/transcribe_stream.py --list pending.tsv [--inbox data/media/flac_up] [--batch 16]

pending.tsv has one "<source>\t<id>" per line; audio arrives as <inbox>/<source>/<id>.flac (rsync renames into place, so
an existing file is complete). Same model, chunking and decoding as scripts/transcribe_media.py; transcripts go to
data/media/transcripts/<source>/. Exits when every listed file is done.
"""
import argparse, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from transcribe_media import CTC, transcribe_file, MEDIA  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--list", required=True); ap.add_argument("--inbox", default=str(MEDIA / "flac_up")); ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()
todo = [l.split("\t") for l in Path(a.list).read_text().split("\n") if l.strip()]
ctc = CTC(str(ROOT / "runs/ctc1b_brh_s3/final"), torch.device("cuda")); t0 = time.time(); audio_s = 0.0; n = 0
print(f"{len(todo)} files to transcribe, model loaded on {torch.cuda.get_device_name(0)}", flush=True)
while todo:
    ready = [(s, i) for s, i in todo if (Path(a.inbox) / s / f"{i}.flac").exists()]
    if not ready: time.sleep(2); continue
    for s, i in ready:
        t = time.time(); out = MEDIA / "transcripts" / s
        if not (out / f"{i}.jsonl").exists():
            st = transcribe_file(ctc, {"id": i, "source": s, "path": str(Path(a.inbox) / s / f"{i}.flac")}, out, a.batch, 20.0)
            audio_s += st["total_s"]; n += 1
            print(f"{n} {s}/{i}: {st['total_s'] / 60:.1f} min, conf {st['conf']:.3f}, {st['total_s'] / (time.time() - t):.0f}x RT | total {audio_s / 3600:.2f} h in {(time.time() - t0) / 60:.1f} min", flush=True)
        todo.remove([s, i])
print(f"ALL DONE: {audio_s / 3600:.2f} h in {(time.time() - t0) / 60:.1f} min", flush=True)
