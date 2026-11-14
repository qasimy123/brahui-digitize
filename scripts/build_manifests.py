"""Build unified manifests from data/raw and seal the evaluation sets.

Outputs
  data/manifests/<source>_<split>.jsonl   one row per utterance / sentence pair
  data/audio/<source>/*.wav               audio extracted from parquet (16 kHz mono)
  data/eval/*.jsonl + SHA256SUMS          sealed test sets - never train on these

Audio row:    {id, source, split, audio, duration, text, text_norm, speaker, extra}
Parallel row: {id, source, split, src_lang, tgt_lang, src, tgt, domain}

Usage: uv run python scripts/build_manifests.py [--no-extract] [--only cv omni brahuispeech smol bouquet]
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, random, sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brahui.text import normalize  # noqa: E402

RAW = ROOT / "data" / "raw"
MAN = ROOT / "data" / "manifests"
AUD = ROOT / "data" / "audio"
EVAL = ROOT / "data" / "eval"
SEED = 20261006


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):6d}  {path.relative_to(ROOT)}")


def read_jsonl(path: Path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------- Common Voice
def build_cv():
    base = RAW / "mdc" / "common_voice_27_brh" / "cv-corpus-27.0-2026-09-11" / "brh"
    durations = {}
    with open(base / "clip_durations.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            durations[r["clip"]] = float(r["duration[ms]"]) / 1000.0
    out = {}
    for split in ["train", "dev", "test"]:
        rows = []
        with open(base / f"{split}.tsv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                p = r["path"]
                rows.append({
                    "id": f"cv_{Path(p).stem}", "source": "common_voice_27", "split": split,
                    "audio": str((base / "clips" / p).relative_to(ROOT)),
                    "duration": durations.get(p), "text": r["sentence"], "text_norm": normalize(r["sentence"]),
                    "speaker": r["client_id"][:16], "extra": {"sentence_id": r["sentence_id"]},
                })
        write_jsonl(MAN / f"cv_{split}.jsonl", rows)
        out[split] = rows
    return out


# ---------------------------------------------------------------- parquet audio sources
def extract_parquet_audio(parquets, out_dir: Path, id_fn, text_fn, meta_fn, source, split, extract=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pf in parquets:
        t = pq.read_table(pf)
        cols = t.column_names
        for i in tqdm(range(t.num_rows), desc=f"{source}/{split}:{Path(pf).name}", leave=False):
            r = {c: t.column(c)[i].as_py() for c in cols if c != "audio"}
            uid = id_fn(r)
            wav = out_dir / f"{uid}.wav"
            dur = None
            if extract and not wav.exists():
                a = t.column("audio")[i].as_py()
                data, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
                data = data.mean(axis=1)
                if sr != 16000:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000); sr = 16000
                sf.write(wav, data, sr, subtype="PCM_16")
                dur = len(data) / sr
            if dur is None and wav.exists():
                info = sf.info(wav); dur = info.frames / info.samplerate
            text = text_fn(r)
            rows.append({"id": uid, "source": source, "split": split, "audio": str(wav.relative_to(ROOT)),
                         "duration": dur, "text": text, "text_norm": normalize(text), **meta_fn(r)})
    return rows


def build_omni(extract=True):
    base = RAW / "omnilingual_asr_brh" / "data" / "brh_Arab"
    out = {}
    for split in ["train", "dev", "test"]:
        files = sorted(base.glob(f"{split}-*.parquet"))
        rows = extract_parquet_audio(
            files, AUD / "omnilingual",
            id_fn=lambda r: f"omni_{r['speaker_id']}_{r['prompt_id']}_{r['segment_id']}",
            text_fn=lambda r: r["raw_text"],
            meta_fn=lambda r: {"speaker": r["speaker_id"], "extra": {"prompt": r["prompt"], "orig_duration": r["duration"]}},
            source="omnilingual_asr", split=split, extract=extract)
        write_jsonl(MAN / f"omni_{split}.jsonl", rows)
        out[split] = rows
    return out


def build_brahuispeech(extract=True):
    base = RAW / "brahuispeech_70h_v2" / "data"
    files = sorted(base.glob("train-*.parquet"))
    rows = extract_parquet_audio(
        files, AUD / "brahuispeech",
        id_fn=lambda r: f"bs_{r['utterance_id']}",
        text_fn=lambda r: r["transcription"] or r["text"],
        meta_fn=lambda r: {"speaker": None, "extra": {"review_stage": r["review_stage"], "orig_duration": r["duration_seconds"]}},
        source="brahuispeech_70h_v2", split="train", extract=extract)
    write_jsonl(MAN / "brahuispeech_train.jsonl", rows)
    return {"train": rows}


# ---------------------------------------------------------------- parallel text
def build_smol():
    base = RAW / "smol_brh"
    sent = read_jsonl(base / "smolsent" / "en_brh.jsonl")
    rows = [{"id": f"smolsent_{r['id']}", "source": "smol_sent", "split": None, "src_lang": "eng_Latn", "tgt_lang": "brh_Arab",
             "src": r["src"], "tgt": r["trg"], "tgt_norm": normalize(r["trg"]), "domain": "general"} for r in sent]
    rng = random.Random(SEED); rng.shuffle(rows)
    heldout, train = rows[:200], rows[200:]
    for r in heldout: r["split"] = "test"
    for r in train: r["split"] = "train"
    write_jsonl(MAN / "smol_sent_train.jsonl", sorted(train, key=lambda r: r["id"]))
    write_jsonl(MAN / "smol_sent_test.jsonl", sorted(heldout, key=lambda r: r["id"]))
    docs = read_jsonl(base / "smoldoc" / "en_brh.jsonl")
    drow = []
    for d in docs:
        for j, (s, t) in enumerate(zip(d["srcs"], d["trgs"])):
            drow.append({"id": f"smoldoc_{d['id']}_{j}", "source": "smol_doc", "split": "train", "src_lang": "eng_Latn",
                         "tgt_lang": "brh_Arab", "src": s, "tgt": t, "tgt_norm": normalize(t), "domain": d["id"].split("__")[0],
                         "doc_id": d["id"]})
    write_jsonl(MAN / "smol_doc_train.jsonl", drow)
    return {"train": train, "test": heldout, "doc": drow}


def build_bouquet():
    base = RAW / "bouquet_brh" / "benchmark_data" / "sentence_level"
    out = {}
    for split in ["dev", "test"]:
        t = pq.read_table(base / split / "brh_Arab-eng_Latn.parquet").to_pylist()
        rows = [{"id": f"bouquet_{r['uniq_id']}", "source": "bouquet", "split": split, "src_lang": "brh_Arab", "tgt_lang": "eng_Latn",
                 "src": r["src_text"], "src_norm": normalize(r["src_text"]), "tgt": r["tgt_text"], "domain": r["domain"],
                 "register": r["register"], "par_id": r["par_id"]} for r in t]
        write_jsonl(MAN / f"bouquet_sent_{split}.jsonl", rows)
        out[split] = rows
    return out


# ---------------------------------------------------------------- seal
def seal(sets: dict[str, list]):
    EVAL.mkdir(parents=True, exist_ok=True)
    sums = []
    for name, rows in sets.items():
        p = EVAL / f"{name}.jsonl"
        write_jsonl(p, rows)
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (EVAL / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (EVAL / "README.md").write_text(
        "# Sealed evaluation sets\n\nNever train, tune prompts, or select models on these. "
        "Hill-climbing reads only the *train* splits under data/manifests/.\n\n"
        "| file | what |\n|---|---|\n"
        "| cv_test.jsonl | Common Voice 27.0 Brahui official test split (CC0) |\n"
        "| omni_test.jsonl | Omnilingual ASR corpus brh_Arab test split (CC-BY-4.0) |\n"
        "| smol_heldout.jsonl | 200 SmolSent en->brh pairs, seeded random hold-out (CC-BY-4.0) |\n"
        "| bouquet_test.jsonl | BOUQuET brh_Arab->eng_Latn sentence-level test (CC-BY-4.0) |\n\n"
        "Verify with `shasum -a 256 -c SHA256SUMS` from this directory.\n")
    print("sealed", list(sets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    want = lambda k: a.only is None or k in a.only
    sets = {}
    if want("cv"):
        cv = build_cv(); sets["cv_test"] = cv["test"]
    if want("omni"):
        om = build_omni(extract=not a.no_extract); sets["omni_test"] = om["test"]
    if want("smol"):
        sm = build_smol(); sets["smol_heldout"] = sm["test"]
    if want("bouquet"):
        bq = build_bouquet(); sets["bouquet_test"] = bq["test"]
    if want("brahuispeech"):
        build_brahuispeech(extract=not a.no_extract)
    if sets and a.only is None:
        seal(sets)
    elif sets:
        # partial rebuild: refresh only the named sealed files but keep the others
        existing = {p.stem: read_jsonl(p) for p in EVAL.glob("*.jsonl")} if EVAL.exists() else {}
        existing.update(sets); seal(existing)


if __name__ == "__main__":
    main()
