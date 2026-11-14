"""Map BrahuiSpeech-70H spelling conventions onto the Common Voice / Omnilingual ones, word by word.

BrahuiSpeech writes many high-frequency words differently from the other two corpora (ای/اے for ئے, اس for ئس,
آ for ء, و for ؤ, ...). Stage 2 showed that these conventions cost about 1 CER point on the sealed sets. This
script learns a word-level map from the training manifests and writes a normalized BrahuiSpeech manifest, so
all 80 h can be trained with one orthography.

Method: group words from both corpora by a lenient key (variants of alef/hamza, ye, and the Arabic-only letters
collapsed). For each key where CV+Omni have a clearly dominant surface form (>= --min-share of the key's mass,
>= --min-count occurrences), every other BrahuiSpeech surface form under that key is mapped to it. Keys that
CV/Omni never use are left alone, so BrahuiSpeech-only vocabulary is untouched. Word-level, not character-level,
so a letter is never changed where the target corpora themselves keep it.

Outputs: data/manifests/bs_norm_map.json (the map + stats) and data/manifests/brahuispeech_train_norm.jsonl.
"""
from __future__ import annotations

import argparse, collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brahui.text import for_scoring  # noqa: E402

MAN = ROOT / "data" / "manifests"
LENIENT = str.maketrans({"ں": "ن", "آ": "ا", "ء": "ا", "ئ": "ا", "ؤ": "و", "ے": "ی", "ح": "ہ", "ص": "س", "ض": "ز",
                         "ظ": "ز", "ط": "ت", "ث": "س", "ھ": "ہ"})


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def word_counts(rows):
    c = collections.Counter()
    for r in rows:
        c.update(for_scoring(r["text"]).split())
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-share", type=float, default=0.5, help="target form must hold this share of the key's CV+Omni mass")
    ap.add_argument("--min-count", type=int, default=3, help="target form must occur this often in CV+Omni")
    a = ap.parse_args()

    target_rows = read_jsonl(MAN / "cv_train.jsonl") + read_jsonl(MAN / "cv_dev.jsonl") + read_jsonl(MAN / "omni_train.jsonl")
    bs_rows = read_jsonl(MAN / "brahuispeech_train.jsonl")
    tgt, bs = word_counts(target_rows), word_counts(bs_rows)

    by_key = collections.defaultdict(collections.Counter)
    for w, c in tgt.items():
        by_key[w.translate(LENIENT)][w] += c

    mapping, skipped_ambiguous = {}, []
    for w, c in bs.items():
        forms = by_key.get(w.translate(LENIENT))
        if not forms:
            continue
        dom, dc = forms.most_common(1)[0]
        share = dc / sum(forms.values())
        if dom == w:
            continue
        if dc >= a.min_count and share >= a.min_share:
            mapping[w] = dom
        else:
            skipped_ambiguous.append((w, dict(forms.most_common(3))))

    changed_tokens = sum(bs[w] for w in mapping)
    total_tokens = sum(bs.values())
    out_rows, changed_clips = [], 0
    for r in bs_rows:
        words = r["text"].split()
        new = [mapping.get(for_scoring(w), w) if for_scoring(w) in mapping else w for w in words]
        if new != words:
            changed_clips += 1
        r = dict(r); r["text"] = " ".join(new); r["text_norm"] = for_scoring(r["text"]); r["bs_norm"] = True
        out_rows.append(r)

    (MAN / "brahuispeech_train_norm.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    top = sorted(mapping.items(), key=lambda kv: -bs[kv[0]])
    stats = {"entries": len(mapping), "bs_tokens": total_tokens, "changed_tokens": changed_tokens,
             "changed_token_share": changed_tokens / total_tokens, "changed_clips": changed_clips, "clips": len(bs_rows),
             "skipped_ambiguous": len(skipped_ambiguous), "min_share": a.min_share, "min_count": a.min_count,
             "top_30": [{"from": w, "to": t, "bs_count": bs[w]} for w, t in top[:30]]}
    json.dump({"map": mapping, "stats": stats, "skipped_ambiguous_examples": skipped_ambiguous[:50]},
              open(MAN / "bs_norm_map.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(mapping)} map entries; {changed_tokens}/{total_tokens} BrahuiSpeech tokens change "
          f"({100*changed_tokens/total_tokens:.1f}%), {changed_clips}/{len(bs_rows)} clips touched; "
          f"{len(skipped_ambiguous)} BS forms left alone as ambiguous")
    print("top mappings:", ", ".join(f"{w}→{t} ({bs[w]})" for w, t in top[:15]))
    print("wrote", MAN / "brahuispeech_train_norm.jsonl", "and", MAN / "bs_norm_map.json")


if __name__ == "__main__":
    main()
