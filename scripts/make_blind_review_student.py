"""Blind A/B page: the small browser voice (runs/tts_student/best) against its teacher (F5, same voice) on held-out sentences.
  python3 scripts/make_blind_review_student.py [--n 20]  ->  evals/tts/student/blind_student_vs_teacher.html
Small model clips: evals/tts/student/best_full/wavs; teacher clips: data/tts/distill/synth/flac. Key hidden in the export."""
import argparse, base64, json, random, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); a = ap.parse_args()
def mp3(p): return base64.b64encode(subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-ar", "16000", "-b:a", "64k", "-f", "mp3", "-"], capture_output=True, check=True).stdout).decode()
script = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/tts/distill/script.jsonl", encoding="utf-8")}; en = {}
for f in ("retranslated_brh_en", "synthetic_fineweb_brh_en_filtered", "synthetic_bt_everyday", "gold_train_nodev"):
    for l in open(ROOT / "data/parallel" / f"{f}.jsonl", encoding="utf-8"): r = json.loads(l); en[r["id"]] = r.get("en", "")
dev = [json.loads(l) for l in open(ROOT / "data/tts/student/dev.jsonl", encoding="utf-8")]
dev = [r for r in dev if 6 <= script[r["id"]]["n_words"] <= 15 and en.get(script[r["id"]]["orig"]) and (ROOT / "evals/tts/student/best_full/wavs" / f"{r['id']}.wav").exists()]
rng = random.Random(20261114); rng.shuffle(dev); items = []; key = {}
for k, r in enumerate(dev[:a.n]):
    n = f"S{k + 1:02d}"; a_new = rng.random() < 0.5; key[n] = {"A": "new" if a_new else "old", "B": "old" if a_new else "new", "id": r["id"]}   # new = small model, old = teacher (both resampled to 16 kHz)
    new, old = mp3(ROOT / "evals/tts/student/best_full/wavs" / f"{r['id']}.wav"), mp3(ROOT / "data/tts/distill/synth/flac" / f"{r['id']}.flac")
    items.append({"n": n, "brh": r["text"], "rom": r["vowelled"], "en": en[script[r["id"]]["orig"]], "A": new if a_new else old, "B": old if a_new else new})
html = (Path(__file__).parent / "blind_review_template.html").read_text(encoding="utf-8").replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False)).replace("__KEY__", base64.b64encode(json.dumps(key).encode()).decode())
html = html.replace('"brahui_blind_review"', '"brahui_blind_student"').replace('SK = "', 'SK = "student_').replace("24 sentences, each spoken by two versions", f"{len(items)} sentences, each spoken by two versions")
out = ROOT / "evals/tts/student/blind_student_vs_teacher.html"; out.write_text(html, encoding="utf-8"); print(out, len(items), "sentences", round(len(html) / 1e6, 1), "MB;", sum(v["A"] == "new" for v in key.values()), "have the small model as A")
