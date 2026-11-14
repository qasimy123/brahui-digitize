"""Download untranscribed Brahui media audio (YouTube) as 16 kHz mono WAV for pseudo-labeling.

Usage:
  uv run python scripts/download_media.py [--sources data/media/sources.json] [--only NAME] [--dry-run]

Reads data/media/sources.json (per-source list of video ids or a channel URL + limit), downloads best audio with
yt-dlp, converts to 16 kHz mono PCM WAV under data/media/audio/<source>/<id>.wav and writes
data/media/meta.jsonl (id, source, title, channel, duration, upload_date, url). Resumable: yt-dlp's download
archive skips what is already there. Audio stays local (data/media/audio is gitignored); fair dealing for research,
not for redistribution.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "data" / "media"


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def targets(src: dict) -> list[str]:
    if src["kind"] == "video":
        return [f"https://www.youtube.com/watch?v={i}" for i in src["ids"]]
    return [src["url"]]


def download(src: dict, dry: bool) -> None:
    out = MEDIA / "audio" / src["name"]
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["yt-dlp", "--no-update", "--ignore-errors", "--no-overwrites",
           "--download-archive", str(MEDIA / "download_archive.txt"),
           "-f", "bestaudio[format_note*=original]/bestaudio/best", "-x", "--audio-format", "wav",
           "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000 -sample_fmt s16",
           "--write-info-json", "--no-write-playlist-metafiles",
           "-o", str(out / "%(id)s.%(ext)s")]
    if src["kind"] == "channel":
        cmd += ["--playlist-end", str(src.get("limit", 20))]
    if src.get("reject_title"):  # e.g. the same channel's Urdu bulletins
        cmd += ["--reject-title", src["reject_title"]]
    if src.get("keep_orig"):  # keep the full-rate original next to the 16 kHz wav (music separation / enhancement models want 44.1-48 kHz)
        cmd += ["-k"]
    if src.get("limit", 0) > 100 or len(src.get("ids", [])) > 20:
        cmd += ["--sleep-requests", "1", "--sleep-interval", "1", "--max-sleep-interval", "4"]
    if os.environ.get("YTDLP_COOKIES_BROWSER"):  # only with the user's OK: ties the downloads to their YouTube account
        cmd += ["--cookies-from-browser", os.environ["YTDLP_COOKIES_BROWSER"]]
    if dry:
        cmd += ["--skip-download", "--no-write-info-json", "--print", "%(id)s|%(duration)s|%(title)s"]
    cmd += targets(src)
    run(cmd)


def build_meta() -> None:
    rows = []
    for info in sorted((MEDIA / "audio").glob("*/*.info.json")):
        wav = info.with_name(info.name.replace(".info.json", ".wav"))
        if not wav.exists():
            continue
        d = json.loads(info.read_text())
        rows.append({"id": d["id"], "source": info.parent.name, "title": d.get("title"), "channel": d.get("channel") or d.get("uploader"),
                     "duration": d.get("duration"), "upload_date": d.get("upload_date"), "url": d.get("webpage_url"), "path": str(wav.relative_to(ROOT))})
    with open(MEDIA / "meta.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum((r["duration"] or 0) for r in rows) / 3600
    by = {}
    for r in rows:
        by.setdefault(r["source"], [0, 0.0]); by[r["source"]][0] += 1; by[r["source"]][1] += (r["duration"] or 0) / 3600
    for k, (n, h) in sorted(by.items()):
        print(f"  {k:32s} {n:4d} files {h:6.2f} h")
    print(f"meta.jsonl: {len(rows)} files, {hours:.2f} h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(MEDIA / "sources.json"))
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    srcs = json.loads(Path(a.sources).read_text())["sources"]
    for s in srcs:
        if a.only and s["name"] != a.only:
            continue
        download(s, a.dry_run)
    if not a.dry_run:
        build_meta()


if __name__ == "__main__":
    main()
