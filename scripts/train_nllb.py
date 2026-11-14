"""Fine-tune NLLB-200 on Brahui<->English: gold pairs (data/parallel/gold_train.jsonl minus the fixed dev split in
data/dev/mt_dev.jsonl) + every synthetic corpus (data/parallel/synthetic_*.jsonl: real Brahui, Claude English).
Both directions in one model. Brahui is tagged with NLLB's `urd_Arab` code (NLLB has no Brahui; the closest
script-sharing code is reused and overwritten by training). Rows listed in data/parallel/eval_overlap.jsonl
(near-duplicates of sealed test sentences, found by scripts/dedup_against_eval.py) are never trained on.

Instruments (all per-sentence outputs go to evals/mt/nllb/<tag>/*.jsonl; evals/mt/compare_nllb.py does the CIs):
  dev (every epoch and at the end): en2brh_dev, brh2en_dev, roundtrip_dev (en->brh->en vs the English source)
  sealed test (before/after):      en2brh_smol (smol_heldout, n=200), en2brh_bouquet (bouquet_test used in reverse,
                                   n=854), brh2en_bouquet (n=854), roundtrip_smol, roundtrip_bouquet

Usage:
  uv run python scripts/train_nllb.py --out runs/nllb_brh [--base facebook/nllb-200-distilled-600M] [--epochs 3]
      [--lr 1e-4] [--bs 16] [--grad-accum 2] [--max-len 256] [--seed 0] [--synthetic-weight 1.0] [--no-synthetic]
      [--init <dir>] [--smoke]
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402

PAR, EVAL, DEV = ROOT / "data" / "parallel", ROOT / "data" / "eval", ROOT / "data" / "dev" / "mt_dev.jsonl"
BRH, ENG = "urd_Arab", "eng_Latn"
ACTIVE_DIRS = {"en2brh", "brh2en"}  # narrowed by --dirs (single-direction students)
chrf, bleu = CHRF(word_order=2), BLEU(effective_order=True)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def excluded_ids():
    p = PAR / "eval_overlap.jsonl"
    return {r["id"] for r in read_jsonl(p)} if p.exists() else set()


def load_pairs(a):
    dev_ids = {r["id"] for r in read_jsonl(DEV)}
    if a.pairs:  # distillation: one explicit corpus of {brh, en[, en_ref, dirs, real]} rows replaces the built-in mix
        rows = [r for r in read_jsonl(ROOT / a.pairs) if r.get("brh") and r.get("en") and r["id"] not in dev_ids]
        if a.smoke:
            rows = rows[:64]
        pairs, n_ref = [], 0
        for r in rows:
            tgt = r.get("en_ref") or r["en"]; n_ref += bool(r.get("en_ref"))
            for d in set(r.get("dirs", ["en2brh", "brh2en"])) & ACTIVE_DIRS:
                pairs.append({"src": r["brh"] if d == "brh2en" else tgt, "tgt": tgt if d == "brh2en" else r["brh"], "dir": d, "w": 1})
        print(f"  {a.pairs}: {len(rows)} rows ({n_ref} with human refs) -> {len(pairs)} examples in dirs {sorted(ACTIVE_DIRS)}")
        return pairs, n_ref, len(rows) - n_ref, len(dev_ids), 0
    bad = excluded_ids()
    gold = [r for r in read_jsonl(PAR / "gold_train.jsonl") if r["id"] not in dev_ids and r["id"] not in bad]
    syn = []
    if not a.no_synthetic:
        skip = set(a.skip_corpus.split(",")) if a.skip_corpus else set()
        for f in sorted(PAR.glob("synthetic_*.jsonl")):  # speech-transcript corpus + FineWeb-2 corpus (+ any later ones)
            if f.name in skip:
                print(f"  {f.name}: skipped"); continue
            rows = [r for r in read_jsonl(f) if r["id"] not in bad]; syn += rows; print(f"  {f.name}: {len(rows)} pairs")
    syn = [r for r in syn if r.get("en") and r.get("brh")]
    if a.smoke:
        gold, syn = gold[:64], syn[:64]
    pairs = []
    for r in gold + syn:
        w = 1 if r in gold else a.synthetic_weight
        dirs = r.get("dirs", ["en2brh", "brh2en"])  # back-translated rows (synthetic Brahui, human English) carry dirs=["brh2en"]
        if "en2brh" in dirs and "en2brh" in ACTIVE_DIRS:
            pairs.append({"src": r["en"], "tgt": r["brh"], "dir": "en2brh", "w": w})
        if "brh2en" in dirs and "brh2en" in ACTIVE_DIRS:
            pairs.append({"src": r["brh"], "tgt": r["en"], "dir": "brh2en", "w": w})
    return pairs, len(gold), len(syn), len(dev_ids), len(bad)


class Collator:
    """Pads and adds decoder_input_ids = labels shifted right with NLLB's decoder start (eos), since the Trainer pops
    `labels` before the forward pass when label smoothing is on."""
    def __init__(self, tok, model):
        self.pad, self.start = tok.pad_token_id, model.config.decoder_start_token_id

    def __call__(self, feats):
        L = max(len(f["input_ids"]) for f in feats); T = max(len(f["labels"]) for f in feats)
        ids = torch.full((len(feats), L), self.pad); am = torch.zeros((len(feats), L), dtype=torch.long)
        lab = torch.full((len(feats), T), -100); dec = torch.full((len(feats), T), self.pad)
        for i, f in enumerate(feats):
            n, m = len(f["input_ids"]), len(f["labels"])
            ids[i, :n] = torch.tensor(f["input_ids"]); am[i, :n] = 1
            lab[i, :m] = torch.tensor(f["labels"]); dec[i, 0] = self.start; dec[i, 1:m] = torch.tensor(f["labels"][:-1])
        return {"input_ids": ids, "attention_mask": am, "labels": lab, "decoder_input_ids": dec}


def tokenize(tok, pairs, max_len, seed):
    from datasets import Dataset
    feats = []
    for d in ["en2brh", "brh2en"]:
        rows = [p for p in pairs if p["dir"] == d]
        if not rows:
            continue
        tok.src_lang, tok.tgt_lang = (ENG, BRH) if d == "en2brh" else (BRH, ENG)
        enc = tok([p["src"] for p in rows], text_target=[p["tgt"] for p in rows], max_length=max_len, truncation=True)
        for i in range(len(rows)):
            feats.append({"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": enc["labels"][i]})
    random.Random(seed + 1).shuffle(feats)
    return Dataset.from_list(feats)


@torch.no_grad()
def translate(model, tok, texts, src, tgt, bs=16, beams=4, max_new=256):
    was_training = model.training
    model.eval(); tok.src_lang = src
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), num_beams=beams, max_new_tokens=max_new)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    if was_training:
        model.train()
    return out


def score(hyps, refs, brh_side):
    if brh_side:
        hyps, refs = [normalize(h) for h in hyps], [normalize(r) for r in refs]
    return {"chrfpp": round(chrf.corpus_score(hyps, [refs]).score, 2), "bleu": round(bleu.corpus_score(hyps, [refs]).score, 2)}


def write_rows(outdir, name, rows, hyps, sk, tk):
    with open(outdir / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r, h in zip(rows, hyps):
            f.write(json.dumps({"id": r["id"], "src": r[sk], "ref": r[tk], "hyp": h}, ensure_ascii=False) + "\n")


def run_sets(model, tok, tag, sets, limit=None, roundtrip=True):
    """sets: list of (name, rows, en_key, brh_key). Scores en2brh, brh2en and the round trip for each."""
    res = {}
    outdir = ROOT / "evals" / "mt" / "nllb" / tag; outdir.mkdir(parents=True, exist_ok=True)
    for name, rows, ek, bk, dirs in sets:
        dirs = [d for d in dirs if d in ACTIVE_DIRS]
        rows = rows[:limit] if limit else rows
        en, brh = [r[ek] for r in rows], [r[bk] for r in rows]
        t0 = time.time()
        if "en2brh" in dirs:
            h = translate(model, tok, en, ENG, BRH)
            res[f"en2brh_{name}"] = {"n": len(rows), **score(h, brh, True)}
            write_rows(outdir, f"en2brh_{name}", rows, h, ek, bk)
            if roundtrip:
                rt = translate(model, tok, h, BRH, ENG)
                res[f"roundtrip_{name}"] = {"n": len(rows), **score(rt, en, False)}
                with open(outdir / f"roundtrip_{name}.jsonl", "w", encoding="utf-8") as f:
                    for r, b, e in zip(rows, h, rt):
                        f.write(json.dumps({"id": r["id"], "src": r[ek], "brh": b, "hyp": e}, ensure_ascii=False) + "\n")
        if "brh2en" in dirs:
            h = translate(model, tok, brh, BRH, ENG)
            res[f"brh2en_{name}"] = {"n": len(rows), **score(h, en, False)}
            write_rows(outdir, f"brh2en_{name}", rows, h, bk, ek)
        res[f"sec_{name}"] = round(time.time() - t0)
    print(tag, json.dumps(res), flush=True)
    return res


def evaluate(model, tok, tag, limit=None):
    sets = [("smol", read_jsonl(EVAL / "smol_heldout.jsonl"), "src", "tgt", ("en2brh",)),
            ("bouquet", read_jsonl(EVAL / "bouquet_test.jsonl"), "tgt", "src", ("en2brh", "brh2en"))]
    return run_sets(model, tok, tag, sets, limit)


def evaluate_dev(model, tok, tag, limit=None):
    return run_sets(model, tok, tag, [("dev", read_jsonl(DEV), "en", "brh", ("en2brh", "brh2en"))], limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--base", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--init", default=None, help="start from these weights instead of --base (tokenizer still from --base)")
    ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256); ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-weight", type=float, default=1.0); ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--skip-corpus", default="", help="comma-separated synthetic_*.jsonl file names to leave out (ablations)")
    ap.add_argument("--label-smoothing", type=float, default=0.1); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-before", action="store_true"); ap.add_argument("--eval-limit", type=int)
    ap.add_argument("--no-dev-curve", action="store_true", help="skip the per-epoch dev eval")
    ap.add_argument("--allow-mps", action="store_true", help="permit a small run on the Mac (e.g. gold-only adaptation)")
    ap.add_argument("--skip-after", action="store_true", help="skip the sealed-set eval (dev only; run it later if dev is kept)")
    ap.add_argument("--pairs", default=None, help="jsonl of {id,brh,en[,en_ref,dirs]} rows replacing the built-in corpora (distillation)")
    ap.add_argument("--dirs", default="en2brh,brh2en", help="directions to train/evaluate, e.g. brh2en for a one-way student")
    ap.add_argument("--save-steps", type=int, default=0, help="checkpoint every N steps instead of every epoch")
    ap.add_argument("--brh-tag", default="urd_Arab", help="language code for Brahui; a new code (e.g. brh_Arab) is added "
                    "to the tokenizer with its embedding copied from --brh-tag-init (evals/interp: Brahui collapses onto Urdu)")
    ap.add_argument("--brh-tag-init", default="urd_Arab", help="existing language code whose embedding seeds a new --brh-tag")
    a = ap.parse_args()
    global BRH
    BRH = a.brh_tag
    ACTIVE_DIRS.intersection_update(a.dirs.split(","))
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
    dev = device()
    if dev != "cuda" and not a.smoke and not a.allow_mps:
        sys.exit("no CUDA; refusing a full run (use --smoke, or --allow-mps for a small gold-only adaptation)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    # --init dirs carry their own tokenizer (which may already hold a new Brahui code); otherwise the base one
    tok = AutoTokenizer.from_pretrained(a.init if a.init and (Path(a.init) / "tokenizer.json").exists() else a.base)
    # fp32 master weights; bf16 autocast comes from TrainingArguments(bf16=True). Loading the model in bf16 and
    # training it directly stalls: run 1 (2026-10-18) plateaued at loss 4.55 from epoch 1 because updates rounded away.
    model = AutoModelForSeq2SeqLM.from_pretrained(a.init or a.base, dtype=torch.float32).to(dev)
    if BRH not in tok.get_vocab():
        tok.add_tokens([BRH], special_tokens=True)
        new_id, seed_id = tok.convert_tokens_to_ids(BRH), tok.convert_tokens_to_ids(a.brh_tag_init)
        if new_id >= model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tok))
        with torch.no_grad():  # shared embedding, tied to lm_head: one copy seeds both
            model.get_input_embeddings().weight[new_id] = model.get_input_embeddings().weight[seed_id]
        print(f"added language code {BRH} (id {new_id}), embedding seeded from {a.brh_tag_init} (id {seed_id})", flush=True)
    pairs, n_gold, n_syn, n_dev, n_bad = load_pairs(a)
    print(f"device={dev} base={a.base} init={a.init} seed={a.seed} gold={n_gold} synthetic={n_syn} dev={n_dev} "
          f"excluded={n_bad} training examples={len(pairs)}", flush=True)
    lim = 8 if a.smoke else a.eval_limit
    before = {} if a.skip_before else evaluate(model, tok, f"{out.name}_before", lim)
    (out / "eval_before.json").write_text(json.dumps(before, indent=2))
    ds = tokenize(tok, pairs, a.max_len, a.seed)
    args = Seq2SeqTrainingArguments(
        output_dir=str(out), per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, num_train_epochs=a.epochs, warmup_steps=a.warmup, lr_scheduler_type="linear", weight_decay=0.01,
        label_smoothing_factor=a.label_smoothing, bf16=(dev == "cuda"), logging_steps=50, save_total_limit=1,
        save_strategy="steps" if a.save_steps else "epoch", save_steps=a.save_steps or 500,
        save_only_model=True, report_to=[], dataloader_num_workers=2 if dev == "cuda" else 0, seed=a.seed, data_seed=a.seed,
        max_steps=10 if a.smoke else -1, use_cpu=(dev == "cpu"))

    curve = out / "dev_curve.jsonl"

    class DevCurve(TrainerCallback):
        def on_epoch_end(self, args_, state, control, model=None, **kw):
            r = evaluate_dev(model, tok, f"{out.name}_dev_ep{int(round(state.epoch))}", lim)
            with open(curve, "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": state.epoch, "step": state.global_step, **r}) + "\n")

    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds, data_collator=Collator(tok, model), processing_class=tok,
                             callbacks=[] if a.no_dev_curve else [DevCurve()])
    trainer.train()
    model.save_pretrained(out / "final"); tok.save_pretrained(out / "final")
    dev_final = evaluate_dev(model, tok, f"{out.name}_dev", lim)
    after = {} if a.skip_after else evaluate(model, tok, f"{out.name}_after", lim)
    (out / "eval_after.json").write_text(json.dumps({**after, "dev": dev_final}, indent=2))
    print("before", before, "\nafter ", after, "\ndev   ", dev_final)


if __name__ == "__main__":
    main()
