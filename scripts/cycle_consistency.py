"""Cycle-consistency pilot: can ASR -> TTS re-speak -> acoustic comparison flag ASR errors without a label?

  uv run python scripts/cycle_consistency.py asr [--n 400]           # s3 ASR on Common Voice test, pick 30 wrong + 30 right clips
  tools/f5/.venv/bin/python scripts/cycle_consistency.py synth       # F5 re-speaks the ASR text and the human label in the clip's own voice
  uv run python scripts/cycle_consistency.py compare                 # mHuBERT-147 features + DTW cost vs the original; AUC, paired win rate

The comparator (utter-project/mHuBERT-147) is not our ASR, so its mistakes are not the ASR's mistakes. Human labels are
used only to grade the detector. Outputs under evals/cycle/.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "evals/cycle"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["asr", "synth", "compare"]); ap.add_argument("--n", type=int, default=400); a = ap.parse_args()

if a.stage == "asr":
    import torch, librosa, jiwer
    from brahui.text import normalize, for_scoring, romanize
    from transcribe_media import CTC
    asr = CTC(str(ROOT / "runs/ctc1b_brh_s3/final"), torch.device("mps"))
    rows = [json.loads(l) for l in open(ROOT / "data/manifests/cv_test.jsonl")]; rows = [r for r in rows if 4 <= r["duration"] <= 10]
    random.Random(0).shuffle(rows); res = []
    for r in rows[:a.n]:
        y, _ = librosa.load(ROOT / r["audio"], sr=16000); t = asr.transcribe([y])[0]
        ref, hyp = normalize(r.get("text_norm") or r["text"]), t["text"]
        res.append({"id": r["id"], "audio": r["audio"], "ref": ref, "hyp": hyp, "conf": t["conf"], "cer": jiwer.cer(for_scoring(ref) or " ", for_scoring(hyp) or " "),
                    "ref_rom": romanize(ref, speak=True), "hyp_rom": romanize(hyp, speak=True)})
    bad = sorted([x for x in res if 0.08 <= x["cer"] <= 0.5], key=lambda x: -x["cer"])[:30]; good = [x for x in res if x["cer"] == 0][:30]
    print(len(res), "clips, mean CER", round(float(np.mean([x["cer"] for x in res])), 4), "| picked", len(bad), "wrong", len(good), "right")
    with open(OUT / "pick.jsonl", "w", encoding="utf-8") as f:
        for x in bad + good: f.write(json.dumps(x, ensure_ascii=False) + "\n")

elif a.stage == "synth":
    import librosa, soundfile as sf
    from f5_tts.api import F5TTS
    tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=str(ROOT / "runs/f5_multi_roman/final/model_final.safetensors"), vocab_file=str(ROOT / "runs/f5_multi_roman/final/vocab.txt"))
    for i, l in enumerate(open(OUT / "pick.jsonl", encoding="utf-8")):
        x = json.loads(l); ref = OUT / "orig" / f"{x['id']}.wav"; ref.parent.mkdir(parents=True, exist_ok=True)
        if not ref.exists():
            y, _ = librosa.load(ROOT / x["audio"], sr=24000); sf.write(ref, y, 24000)
        for k in ("hyp", "ref"):
            if k == "ref" and x["cer"] == 0: continue
            dst = OUT / k / f"{x['id']}.wav"; dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():  # the clip is its own voice prompt; its prompt text is the ASR output (what we would have without labels)
                tts.infer(ref_file=str(ref), ref_text=x["hyp_rom"], gen_text=x[k + "_rom"], file_wave=str(dst), nfe_step=16, speed=1.0, seed=0, progress=None)
        print(i + 1, flush=True)

else:
    import torch, librosa
    from transformers import AutoFeatureExtractor, HubertModel
    fe = AutoFeatureExtractor.from_pretrained("utter-project/mHuBERT-147"); m = HubertModel.from_pretrained("utter-project/mHuBERT-147").eval()
    def feats(p):
        y, _ = librosa.load(p, sr=16000); y, _ = librosa.effects.trim(y, top_db=35)
        with torch.no_grad(): h = m(**fe(y, sampling_rate=16000, return_tensors="pt"), output_hidden_states=True).hidden_states[9][0]
        return torch.nn.functional.normalize(h, dim=-1).numpy()
    def dtw(A, B):
        C = 1 - A @ B.T; _, wp = librosa.sequence.dtw(C=C); c = C[wp[:, 0], wp[:, 1]]
        w = np.convolve(c, np.ones(10) / 10, mode="valid") if len(c) >= 10 else c   # worst 200 ms window: a local error should not be averaged away
        return float(c.mean()), float(w.max())
    res = []
    for l in open(OUT / "pick.jsonl", encoding="utf-8"):
        x = json.loads(l); o = feats(OUT / "orig" / f"{x['id']}.wav"); r = {"id": x["id"], "cer": x["cer"], "conf": x["conf"]}
        r["hyp_mean"], r["hyp_max"] = dtw(o, feats(OUT / "hyp" / f"{x['id']}.wav"))
        if x["cer"] > 0: r["ref_mean"], r["ref_max"] = dtw(o, feats(OUT / "ref" / f"{x['id']}.wav"))
        res.append(r)
    json.dump(res, open(OUT / "compare.json", "w"), indent=1)
    def auc(pos, neg): return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
    bad, good = [r for r in res if r["cer"] > 0], [r for r in res if r["cer"] == 0]
    for k in ("hyp_mean", "hyp_max"): print(f"AUC wrong-vs-right transcripts by {k}: {auc([r[k] for r in bad], [r[k] for r in good]):.3f}")
    print(f"AUC by ASR confidence (baseline): {auc([-r['conf'] for r in bad], [-r['conf'] for r in good]):.3f}")
    for k in ("mean", "max"): print(f"paired: label re-speak closer to the original than ASR re-speak ({k}): {np.mean([r['ref_' + k] < r['hyp_' + k] for r in bad]):.2f} of {len(bad)}")
    z = lambda v: (np.array(v) - np.mean(v)) / (np.std(v) + 1e-9)  # noqa: E731
    comb = z([r["hyp_mean"] for r in res]) - z([r["conf"] for r in res]); lab = np.array([r["cer"] > 0 for r in res])
    print(f"AUC confidence + cycle combined: {auc(comb[lab], comb[~lab]):.3f}")
