"""Print markdown tables of every ASR run and MT variant found under evals/. Recomputes from per-row results."""
from __future__ import annotations

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def asr_table():
    import jiwer
    print("## ASR baselines (sealed test sets)\n")
    print("| model | set | clips | hours | CER | CER 95% CI | WER | empty | RTF |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in sorted((ROOT / "evals" / "asr" / "runs").glob("*")):
        rp = d / "results.jsonl"
        if not rp.exists():
            continue
        rows = [json.loads(l) for l in open(rp, encoding="utf-8")]
        if not rows:
            continue
        refs = [r["ref_norm"] or " " for r in rows]; hyps = [r["hyp_norm"] or " " for r in rows]
        cer, wer = jiwer.cer(refs, hyps), jiwer.wer(refs, hyps)
        sp = d / "summary.json"; s = json.load(open(sp)) if sp.exists() else {}
        ci = s.get("cer_ci95"); ci_s = f"{ci[0]*100:.1f}–{ci[1]*100:.1f}" if ci else "–"
        hours = sum(r["duration"] or 0 for r in rows) / 3600
        rtf = sum(r["latency_s"] for r in rows) / max(1e-9, sum(r["duration"] or 0 for r in rows))
        model = rows[0]["model"]
        if model.endswith(":final"):  # fine-tuned runs all save to <run>/final; name them by the run tag instead
            model = {"ctc_final": "ctc:ctc1b_brh (stage 1)", "ctc_s2": "ctc:ctc1b_brh_s2 (stage 2)", "ctc_s3": "ctc:ctc1b_brh_s3 (stage 3, normalized BS)"}.get(d.name.split("__")[0], d.name)
        print(f"| {model} | {rows[0]['set']} | {len(rows)} | {hours:.2f} | {cer*100:.1f} | {ci_s} | {wer*100:.1f} | "
              f"{sum(1 for h in hyps if h.strip()=='')} | {rtf:.2f} |")
    print()


def mt_table():
    from sacrebleu.metrics import BLEU, CHRF
    from brahui.text import normalize
    chrf, bleu = CHRF(word_order=2), BLEU(effective_order=True)
    print("## MT variants (chrF++ corpus / BLEU corpus)\n")
    print("| variant | model | en→brh train | en→brh test | brh→en train | brh→en test | headline (mean test chrF++) | errors |")
    print("|---|---|---|---|---|---|---|---|")
    for d in sorted((ROOT / "evals" / "mt" / "hillclimb").glob("*/")):
        rp = d / "results.jsonl"
        if not rp.exists():
            continue
        rows = [json.loads(l) for l in open(rp, encoding="utf-8")]
        sp = d / "summary.json"; s = json.load(open(sp)) if sp.exists() else {}
        cells = {}; errs = sum(1 for r in rows if r["meta"].get("error"))
        for dr in ["en2brh", "brh2en"]:
            for split in ["train", "test"]:
                rs = [r for r in rows if r["tags"][0] == dr and r["tags"][1] == split]
                if not rs:
                    cells[f"{dr}/{split}"] = "–"; continue
                if dr == "en2brh":
                    h = [normalize(r["meta"]["hyp"]) for r in rs]; f = [normalize(r["meta"]["ref"]) for r in rs]
                else:
                    h = [r["meta"]["hyp"] for r in rs]; f = [r["meta"]["ref"] for r in rs]
                c = chrf.corpus_score(h, [f]).score; b = bleu.corpus_score(h, [f]).score
                cells[f"{dr}/{split}"] = f"{c:.1f} / {b:.1f}"; cells[f"{dr}/{split}/c"] = c
        head = (cells.get("en2brh/test/c", 0) + cells.get("brh2en/test/c", 0)) / 2
        print(f"| {d.name} | {s.get('model','?')} | {cells['en2brh/train']} | {cells['en2brh/test']} | {cells['brh2en/train']} | "
              f"{cells['brh2en/test']} | {head:.2f} | {errs} |")
    print()


if __name__ == "__main__":
    asr_table(); mt_table()
