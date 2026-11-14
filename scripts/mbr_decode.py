"""MBR decoding with chrF utility and reverse-model (round-trip) reranking for en->brh.

For each English sentence: candidates = beam-4 output + N sampled outputs (epsilon sampling). Score each candidate by
  score = (1-lam) * MBR(c)   [mean chrF++ of c against the other candidates]
        +  lam   * RT(c)     [chrF++ of brh->en(c) against the English source, through the same model]
and pick the best. No training; decided on dev, sealed only to confirm. Outputs go to evals/mt/nllb/<tag>/ in the
same format as train_nllb.py so evals/mt/compare_nllb.py can compare against the beam baseline.

Usage: uv run python scripts/mbr_decode.py --model runs/nllb_r2_bt/final --tag mbr_r2 --n 16 [--lam 0.3] [--sealed]
           [--sets dev|smol|bouquet,...] [--device mps] [--limit N]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from sacrebleu.metrics import CHRF
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brahui.text import normalize  # noqa: E402
from train_nllb import read_jsonl, translate, score, write_rows, BRH, ENG, EVAL, DEV  # noqa: E402
chrf = CHRF(word_order=2)


@torch.no_grad()
def sample(model, tok, texts, n, bs=8, eps=0.02, max_new=200):
    tok.src_lang = ENG; out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(BRH), do_sample=True, epsilon_cutoff=eps,
                             num_return_sequences=n, max_new_tokens=max_new, top_k=0)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        out += [dec[j * n:(j + 1) * n] for j in range(len(dec) // n)]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--lam", type=float, default=0.3); ap.add_argument("--sealed", action="store_true")
    ap.add_argument("--sets", default=None); ap.add_argument("--device", default=None); ap.add_argument("--limit", type=int); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForSeq2SeqLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    sets = {"dev": (read_jsonl(DEV), "en", "brh"), "smol": (read_jsonl(EVAL / "smol_heldout.jsonl"), "src", "tgt"),
            "bouquet": (read_jsonl(EVAL / "bouquet_test.jsonl"), "tgt", "src")}
    names = a.sets.split(",") if a.sets else (["dev", "smol", "bouquet"] if a.sealed else ["dev"])
    outdir = ROOT / "evals/mt/nllb" / a.tag; outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    for name in names:
        rows, ek, bk = sets[name]; rows = rows[:a.limit] if a.limit else rows
        en, ref = [r[ek] for r in rows], [r[bk] for r in rows]; t0 = time.time()
        beam = translate(model, tok, en, ENG, BRH, bs=16)
        samp = sample(model, tok, en, a.n, bs=a.bs)
        cands = [[b] + list(dict.fromkeys(s)) for b, s in zip(beam, samp)]
        # reverse-model scores for every candidate, batched
        flat = [c for cs in cands for c in cs]; back = translate(model, tok, flat, BRH, ENG, bs=32, beams=1, max_new=200)
        k = 0; picks_mbr, picks_rt, picks_mix = [], [], []
        for e, cs in zip(en, cands):
            bt = back[k:k + len(cs)]; k += len(cs)
            rt = [chrf.sentence_score(b, [e]).score for b in bt]
            nc = [normalize(c) for c in cs]
            mbr = [sum(chrf.sentence_score(nc[i], [nc[j]]).score for j in range(len(cs)) if j != i) / max(1, len(cs) - 1) for i in range(len(cs))]
            mix = [(1 - a.lam) * m + a.lam * r for m, r in zip(mbr, rt)]
            picks_mbr.append(cs[max(range(len(cs)), key=lambda i: mbr[i])]); picks_rt.append(cs[max(range(len(cs)), key=lambda i: rt[i])])
            picks_mix.append(cs[max(range(len(cs)), key=lambda i: mix[i])])
        res = {"beam": score(beam, ref, True), "mbr": score(picks_mbr, ref, True), "rt": score(picks_rt, ref, True), f"mix_lam{a.lam}": score(picks_mix, ref, True),
               "n_cands_mean": round(sum(len(c) for c in cands) / len(cands), 1), "sec": round(time.time() - t0)}
        print(name, json.dumps(res), flush=True)
        for kind, picks in [("beam", beam), ("mbr", picks_mbr), ("rt", picks_rt), ("mix", picks_mix)]:
            write_rows(outdir, f"en2brh_{name}_{kind}", rows, picks, ek, bk)
        with open(outdir / f"cands_{name}.jsonl", "w", encoding="utf-8") as f:
            for r, cs in zip(rows, cands):
                f.write(json.dumps({"id": r["id"], "cands": cs}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
