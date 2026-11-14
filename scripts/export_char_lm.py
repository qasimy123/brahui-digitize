"""Export the char KN LM (scripts/build_char_lm.py counts) as a compact backoff table for the browser decoder (demo/web/ctc_lm.js)
and for scripts/ctc_lm_decode.py --lm-bin. Interpolated KN probabilities are stored per n-gram; per-context backoff weights
log(lambda) make lookups exact: P(c|h) = p[h+c] if present else lambda[h] * P(c|h[1:]).
Layout (little-endian): magic 'BLM1', u8 order, u32 charset_bytes, UTF-16LE charset, u32 n_entries, then entries sorted by key:
u32 key (FNV-1a over UTF-16 code units of the n-gram string), f32 logp, f32 logbo (backoff of this n-gram as a context; 0 if none).
Orders >= --prune-order keep only n-grams with count >= --min-count.
  uv run python scripts/export_char_lm.py --lm data/lm/brh_char6 --min-count 2 --out demo/web/models/brh_char6.blm"""
import argparse, json, math, struct, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from ctc_lm_decode import CharKN  # noqa: E402

def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for u in s.encode("utf-16-le"):
        h ^= u; h = (h * 0x01000193) & 0xFFFFFFFF
    return h

ap = argparse.ArgumentParser(); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--min-count", type=int, default=2); ap.add_argument("--prune-order", type=int, default=3); a = ap.parse_args()
lm = CharKN(ROOT / (a.lm + ".json"))
keep = {g for g, c in lm.cnt.items() if len(g) < a.prune_order or c >= a.min_count}
# a context must be present whenever any of its extensions is, so the backoff weight can be found
ctxs = {g[:-1] for g in keep if len(g) > 1}; keep |= ctxs
def logbo(h):
    if len(h) == lm.N - 1: tot, types = lm.ctx_tot.get(h, 0), lm.ctx_types.get(h, 0)
    else: tot, types = lm.cont_tot.get(h, 0), lm.cont_types.get(h, 0)
    return math.log(lm.D * types / tot) if tot else 0.0
entries = {}
for g in keep:
    h, c = g[:-1], g[-1]; p = lm._p(h, c)
    entries[fnv1a(g)] = (math.log(max(p, 1e-12)), logbo(g) if g in ctxs else 0.0)
keys = np.array(sorted(entries), dtype=np.uint32); lp = np.array([entries[k][0] for k in keys], dtype=np.float32); bo = np.array([entries[k][1] for k in keys], dtype=np.float32)
charset = "".join(lm.chars)
with open(a.out, "wb") as f:
    cb = charset.encode("utf-16-le"); f.write(b"BLM1"); f.write(struct.pack("<BI", lm.N, len(cb))); f.write(cb); f.write(struct.pack("<I", len(keys)))
    f.write(keys.tobytes()); f.write(lp.tobytes()); f.write(bo.tobytes())
print(f"{len(lm.cnt)} n-grams -> {len(keys)} kept (min-count {a.min_count} from order {a.prune_order}); {Path(a.out).stat().st_size/1e6:.1f} MB -> {a.out}")
