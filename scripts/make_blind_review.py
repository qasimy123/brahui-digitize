"""Blind A/B listening page for the F5 old-vs-new comparison: evals/tts/listen/set30/{NN.wav (old), NN_new.wav (new)}.
  python3 scripts/make_blind_review.py   ->  evals/tts/listen/set30/blind_review.html (self-contained, audio embedded as mp3)
A/B order is shuffled per sentence (fixed seed); the key is only written into the exported results file."""
import base64, json, random, subprocess
from pathlib import Path
D = Path(__file__).resolve().parent.parent / "evals/tts/listen/set30"
EN = {  # Claude's own translations (2026-11-11), sentence fragments from running text; "?" = unsure
 "01": "So then, what is the cure for it?",
 "02": "…if it isn't read in [that], he won't understand it well.",
 "03": "He was a friend of Ahmad Naeem.",
 "04": "Or he befriends and bonds with a person only to serve his own interest.",
 "05": "…so much does he sympathise with people.",
 "06": "? The wicked have always dishonoured such people in society.",
 "07": "He had a great deal of work.",
 "08": "? If we are to know anyone, let it be Krishan Chander; for us, Yasin Bismil…",
 "09": "…tell me, what kind of madman are you?",
 "10": "The first to write parody in [Brahui] history is the honest Jabbar Yar.",
 "11": "? …the scenes of joking and mockery…",
 "12": "? …he asked after him; the news was good.",
 "13": "…kept pleading: in God's name, at least say something.",
 "14": "…that CNG rickshaws be shut down.",
 "15": "? …passion in a work of art by…",
 "16": "He tries to make people laugh.",
 "17": "Here too there used to be a gathering all the time.",
 "18": "? …'s heart sank… he said",
 "19": "You will surely be surprised that even the printing press was on loan.",
 "20": "Ahmad Naeem would say 'bravo, bravo' to him.",
 "21": "He was sitting in that same room of his.",
 "22": "But I remember PTV Bolan.",
 "23": "What closeness could there be with Jabbar Yar or Abdul Khaliq Ababaki?",
 "24": "? …your voice is coming through… oh uncle",
}
def mp3(p):
    return base64.b64encode(subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-b:a", "64k", "-f", "mp3", "-"], capture_output=True, check=True).stdout).decode()
rng = random.Random(20261111); items = []; key = {}
for l in open(D / "sentences.tsv", encoding="utf-8"):
    n, brh, rom = l.rstrip("\n").split("\t"); a_is_new = rng.random() < 0.5; key[n] = {"A": "new" if a_is_new else "old", "B": "old" if a_is_new else "new"}
    new, old = mp3(D / f"{n}_new.wav"), mp3(D / f"{n}.wav")
    items.append({"n": n, "brh": brh, "rom": rom, "en": EN[n], "A": new if a_is_new else old, "B": old if a_is_new else new})
KEY = base64.b64encode(json.dumps(key).encode()).decode()
html = (Path(__file__).parent / "blind_review_template.html").read_text(encoding="utf-8").replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False)).replace("__KEY__", KEY)
(D / "blind_review.html").write_text(html, encoding="utf-8"); print(D / "blind_review.html", round(len(html) / 1e6, 1), "MB", sum(v["A"] == "new" for v in key.values()), "of", len(key), "have new as A")
