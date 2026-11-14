"""CTC prefix beam search with a character n-gram LM (Kneser-Ney over counts from scripts/build_char_lm.py), scored on dumped
log-probs (scripts/ctc_dump_logits.py). Greedy baseline is reported alongside. Tune alpha/beta on cv_dev, report cv_test.
  uv run python scripts/ctc_lm_decode.py --logits data/eval/logits/ctc300m_cv_dev --lm data/lm/brh_char6 --alpha 0.5 --beta 1.0
  uv run python scripts/ctc_lm_decode.py --logits ... --lm ... --grid  # alpha x beta sweep on --limit clips"""
import argparse, collections, functools, json, math, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brahui.text import for_scoring  # noqa: E402
import jiwer  # noqa: E402

NEG = -1e30
def lse(a, b):
    if a == NEG: return b
    if b == NEG: return a
    m = max(a, b); return m + math.log(math.exp(a - m) + math.exp(b - m))

class CharKN:
    """Interpolated Kneser-Ney over raw counts. P(c|h) = max(c(hc)-D,0)/c(h.) + D*N1+(h.)/c(h.) * P(c|h[1:]); lower orders use
    continuation counts (number of distinct left extensions). Unigram backs off to uniform over the charset."""
    def __init__(self, path):
        d = json.load(open(path, encoding="utf-8")); self.N, self.D, self.bos, self.eos = d["order"], d["D"], d["bos"], d["eos"]
        self.cnt = d["ngrams"]; self.chars = d["chars"]; self.V = len(self.chars)
        # continuation counts for orders < N: cont[g] = |{x : x+g seen}|, derived from (order+1)-grams
        self.cont = collections.Counter(); self.ctx_tot = collections.Counter(); self.ctx_types = collections.Counter()
        self.cont_tot = collections.Counter(); self.cont_types = collections.Counter()
        for g, c in self.cnt.items():
            if len(g) == self.N:
                self.ctx_tot[g[:-1]] += c; self.ctx_types[g[:-1]] += 1
            if len(g) >= 2:
                self.cont[g[1:]] += 1
        for g, c in self.cont.items():
            self.cont_tot[g[:-1]] += c; self.cont_types[g[:-1]] += 1
    @functools.lru_cache(maxsize=1_000_000)
    def logp(self, h, c):
        """h: history string (len <= N-1), c: single char."""
        h = h[-(self.N - 1):] if self.N > 1 else ""
        return math.log(max(self._p(h, c), 1e-12))
    def _p(self, h, c):
        if len(h) == self.N - 1:
            tot = self.ctx_tot.get(h, 0)
            if tot == 0: return self._p(h[1:], c)
            num = max(self.cnt.get(h + c, 0) - self.D, 0.0); lam = self.D * self.ctx_types[h] / tot
            return num / tot + lam * self._p(h[1:], c)
        if h == "":
            tot = self.cont_tot.get("", 0)
            if tot == 0: return 1.0 / self.V
            num = max(self.cont.get(c, 0) - self.D, 0.0); lam = self.D * self.cont_types[""] / tot
            return num / tot + lam / self.V
        tot = self.cont_tot.get(h, 0)
        if tot == 0: return self._p(h[1:], c)
        num = max(self.cont.get(h + c, 0) - self.D, 0.0); lam = self.D * self.cont_types[h] / tot
        return num / tot + lam * self._p(h[1:], c)

class BackoffLM:
    """Reads the .blm table written by scripts/export_char_lm.py; mirrors demo/web/ctc_lm.js exactly."""
    def __init__(self, path):
        import struct
        b = open(path, "rb").read(); assert b[:4] == b"BLM1"; self.N, ncb = struct.unpack_from("<BI", b, 4); o = 9
        self.chars = b[o:o + ncb].decode("utf-16-le"); o += ncb; n, = struct.unpack_from("<I", b, o); o += 4
        self.keys = np.frombuffer(b, dtype="<u4", count=n, offset=o); o += 4 * n
        self.lp = np.frombuffer(b, dtype="<f4", count=n, offset=o); o += 4 * n; self.bo = np.frombuffer(b, dtype="<f4", count=n, offset=o)
        self.bos, self.eos, self.V = "\x02", "\x03", len(self.chars); self.cache = {}
        self.index = {int(k): i for i, k in enumerate(self.keys.tolist())}; self.lp = self.lp.tolist(); self.bo = self.bo.tolist()
    @staticmethod
    def h32(s):
        h = 0x811C9DC5
        for u in s.encode("utf-16-le"): h ^= u; h = (h * 0x01000193) & 0xFFFFFFFF
        return h
    def find(self, g):
        return self.index.get(self.h32(g), -1)
    def logp(self, h, c):
        h = h[-(self.N - 1):]; k = (h, c)
        if k in self.cache: return self.cache[k]
        bo = 0.0
        while True:
            i = self.find(h + c)
            if i >= 0: r = bo + float(self.lp[i]); break
            if h == "": r = bo + math.log(1.0 / self.V); break
            j = self.find(h); bo += float(self.bo[j]) if j >= 0 else 0.0; h = h[1:]
        self.cache[k] = r; return r

def prefix_beam(lp, id2ch, blank, lm, alpha, beta, beam=16, cand=8, prune=-12.0):
    """lp: [T,V] log-probs. Returns best string. Score = log P_ctc + alpha*log P_lm + beta*len(chars)."""
    T, V = lp.shape; bos = lm.bos * (lm.N - 1) if lm else ""
    beams = {"": (0.0, NEG, 0.0)}  # prefix -> (p_blank, p_nonblank, lm_score)
    for t in range(T):
        row = lp[t]; top = np.argpartition(-row, cand)[:cand]; top = [int(i) for i in top if row[i] > prune]
        if blank not in top: top.append(blank)
        nb = {}
        def add(pfx, pb, pnb, lmsc):
            ob, onb, _ = nb.get(pfx, (NEG, NEG, lmsc)); nb[pfx] = (lse(ob, pb), lse(onb, pnb), lmsc)
        for pfx, (pb, pnb, lmsc) in beams.items():
            tot = lse(pb, pnb)
            for i in top:
                p = float(row[i])
                if i == blank: add(pfx, tot + p, NEG, lmsc); continue
                ch = id2ch.get(i)
                if ch is None: continue
                last = pfx[-1] if pfx else None
                if ch == last:  # repeat: stay (non-blank path) or extend only via blank path
                    add(pfx, NEG, pnb + p, lmsc)
                    if pb > NEG:
                        add(pfx + ch, NEG, pb + p, lmsc + (alpha * lm.logp(bos + pfx, ch) if lm else 0.0))
                else:
                    add(pfx + ch, NEG, tot + p, lmsc + (alpha * lm.logp(bos + pfx, ch) if lm else 0.0))
        items = sorted(nb.items(), key=lambda kv: -(lse(kv[1][0], kv[1][1]) + kv[1][2] + beta * len(kv[0])))[:beam]
        beams = dict(items)
    def final(kv):
        pfx, (pb, pnb, lmsc) = kv; eos = alpha * lm.logp(bos + pfx, lm.eos) if lm else 0.0
        return lse(pb, pnb) + lmsc + eos + beta * len(pfx)
    return max(beams.items(), key=final)[0]

def greedy(lp, id2ch, blank):
    ids = lp.argmax(-1); out = []; prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != blank and i in id2ch: out.append(id2ch[i])
        prev = i
    return "".join(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--logits", required=True); ap.add_argument("--lm"); ap.add_argument("--lm-bin", help=".blm table from scripts/export_char_lm.py (browser semantics)"); ap.add_argument("--vocab", default="export/ctc300m_student/vocab.json")
    ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--beta", type=float, default=1.0); ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--limit", type=int); ap.add_argument("--grid", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args()
    vocab = json.load(open(ROOT / a.vocab, encoding="utf-8")); id2ch = {i: t for t, i in vocab.items() if len(t) == 1}; blank = vocab["<pad>"] if "<pad>" in vocab else 0
    blank = 0  # Wav2Vec2 CTC blank = pad_token_id = 0 for these checkpoints
    z = np.load(a.logits + ".npz"); refs = [json.loads(l) for l in open(a.logits + ".refs.jsonl", encoding="utf-8")][: a.limit or None]
    lm = BackoffLM(ROOT / a.lm_bin) if a.lm_bin else (CharKN(ROOT / (a.lm + ".json")) if a.lm else None)
    def run(alpha, beta):
        hyps, gs = [], []; t0 = time.time()
        for r in refs:
            lp = z[r["id"]].astype(np.float32); gs.append(for_scoring(greedy(lp, id2ch, blank)))
            hyps.append(for_scoring(prefix_beam(lp, id2ch, blank, lm, alpha, beta, beam=a.beam)))
        R = [for_scoring(r["text"]) for r in refs]
        return jiwer.cer(R, gs), jiwer.cer(R, hyps), time.time() - t0, hyps
    if a.grid:
        for alpha in (0.3, 0.5, 0.7, 1.0):
            for beta in (0.0, 0.5, 1.0, 2.0):
                g, c, dt, _ = run(alpha, beta); print(f"alpha {alpha} beta {beta}: greedy CER {g:.4f}  beam+LM CER {c:.4f}  ({dt:.0f}s)", flush=True)
        return
    g, c, dt, hyps = run(a.alpha, a.beta); print(f"n={len(refs)} greedy CER {g:.4f}  beam+LM CER {c:.4f} (alpha {a.alpha}, beta {a.beta}, beam {a.beam}, {dt:.0f}s)")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r, h in zip(refs, hyps): f.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": h}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
