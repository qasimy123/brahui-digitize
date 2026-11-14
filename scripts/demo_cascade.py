"""Non-realtime cascade demo: Brahui audio -> fine-tuned Brahui ASR (runs/ctc1b_brh_s2/final) -> Claude -> English (optionally spoken via macOS `say`).

Usage:
  uv run python scripts/demo_cascade.py path/to/clip.mp3 [more clips...] [--speak] [--model opus]
  uv run python scripts/demo_cascade.py --sample 3          # three random sealed CV test clips, with references shown
"""
from __future__ import annotations

import argparse, json, random, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from brahui.text import normalize  # noqa: E402

SYSTEM = (ROOT / "evals" / "mt" / "prompts" / "system_v0.md").read_text(encoding="utf-8")
SCHEMA = json.dumps({"type": "object", "properties": {"english": {"type": "string"}}, "required": ["english"]})


def translate(brahui: str, model: str) -> str:
    shots = [json.loads(l) for l in open(ROOT / "data" / "manifests" / "bouquet_sent_dev.jsonl", encoding="utf-8")][:10]
    msg = "## Reference translations (Brahui -> English)\n\n" + "".join(f"Brahui: {s['src']}\nEnglish: {s['tgt']}\n\n" for s in shots)
    msg += f"## Translate this Brahui ASR transcript into natural English (it may contain spelling errors from speech recognition):\n\n{brahui}"
    out = subprocess.run(["claude", "-p", "--model", model, "--system-prompt", SYSTEM, "--tools", "", "--restricted",
                          "--no-session-persistence", "--output-format", "json", "--json-schema", SCHEMA],
                         input=msg, capture_output=True, text=True, timeout=300)
    d = json.loads(out.stdout)
    return (d.get("structured_output") or json.loads(d["result"]))["english"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*"); ap.add_argument("--sample", type=int); ap.add_argument("--speak", action="store_true")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--asr", default="ctc:runs/ctc1b_brh_s2/final", help="asr_eval.py backend spec; e.g. whisper:data/raw/models/brahvi_whisper_small")
    a = ap.parse_args()
    from asr_eval import make_backend, device_for, load_audio
    refs = {}
    clips = [Path(c) for c in a.clips]
    if a.sample:
        rows = [json.loads(l) for l in open(ROOT / "data" / "eval" / "cv_test.jsonl", encoding="utf-8")]
        for r in random.Random(7).sample(rows, a.sample):
            clips.append(ROOT / r["audio"]); refs[str(ROOT / r["audio"])] = r["text"]
    kind, _, mid = a.asr.partition(":")
    asr = make_backend(f"{kind}:{ROOT / mid}" if (ROOT / mid).exists() else a.asr, device_for("cpu"))
    for c in clips:
        hyp = normalize(asr.transcribe(load_audio(c)))
        en = translate(hyp, a.model)
        print(f"\n[{c.name}]")
        if str(c) in refs:
            print(f"  reference : {refs[str(c)]}")
        print(f"  ASR       : {hyp}")
        print(f"  English   : {en}")
        if a.speak:
            subprocess.run(["say", en])


if __name__ == "__main__":
    main()
