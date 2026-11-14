"""Blind A/B listening page, old voice vs the voice retrained on vowelled text (scripts/gpu/f5_vow.sh):
evals/tts/listen_v3/{old,new}/wavs/LNN.wav for the 24 complete sentences of data/tts/listen_v3/set.jsonl (human English).
  python3 scripts/make_blind_review_v3.py  ->  evals/tts/listen_v3/blind_review_v3.html (self-contained, audio as mp3)
A/B order is shuffled per sentence (fixed seed); the key is only written into the exported results file."""
import base64, json, random, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "evals/tts/listen_v3"
def mp3(p): return base64.b64encode(subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-b:a", "64k", "-f", "mp3", "-"], capture_output=True, check=True).stdout).decode()
rng = random.Random(20261113); items = []; key = {}
for l in open(ROOT / "data/tts/listen_v3/set.jsonl", encoding="utf-8"):
    r = json.loads(l); n = r["id"]; a_is_new = rng.random() < 0.5; key[n] = {"A": "new" if a_is_new else "old", "B": "old" if a_is_new else "new"}
    new, old = mp3(D / "new/wavs" / f"{n}.wav"), mp3(D / "old/wavs" / f"{n}.wav")
    items.append({"n": n, "brh": r["text"], "rom": r["vowelled"], "en": r["en"], "A": new if a_is_new else old, "B": old if a_is_new else new})
KEY = base64.b64encode(json.dumps(key).encode()).decode()
html = (Path(__file__).parent / "blind_review_template.html").read_text(encoding="utf-8").replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False)).replace("__KEY__", KEY)
html = html.replace('"brahui_blind_review"', '"brahui_blind_review_v3"').replace("SK = \"", "SK = \"v3_")
(D / "blind_review_v3.html").write_text(html, encoding="utf-8"); print(D / "blind_review_v3.html", round(len(html) / 1e6, 1), "MB", sum(v["A"] == "new" for v in key.values()), "of", len(key), "have new as A")
