"""Vowel-aware acoustic model: a CTC model that writes Brahui speech in vowelled Latin letters, trained only on words whose
spelling is trusted, then used to test every other spelling against the recordings.

Training target per utterance: the trusted words, spelled out. The frames of every other word (located beforehand by
scripts/vowel_align.py) are overwritten with "blank, probability 1" before the loss, so that audio gives no gradient and an
uncertain spelling never becomes a label. A run of hidden words leaves a word delimiter on each side ("w1 | | w3").
(First attempt, 2026-11-12: a wildcard token with log-prob 0 in the target. Its paths add to the blank paths, the loss went
negative and the model was being paid to emit blank on hidden words. Do not go back to that.)
Cross-validation: model `--fold m` also hides the trusted words of fold m (data from scripts/build_vowel_asr.py), so it
can test them blind. That is where the accuracy figure comes from.

Scoring (`score`): each word type gets candidate spellings (proposed, dictionary alternatives, and edits: swap / drop /
insert a vowel, w/y as vowel, drop h). Each occurrence is cut to a five-word window and every candidate's CTC
log-likelihood is computed with the neighbours fixed (hidden neighbours blanked as in training). The best mean
log-likelihood wins; the search repeats from the winner (up to 4 rounds) so several edits can accumulate.
Trusted words are run twice: from their dictionary spelling ("r": does it survive?) and from the bare consonant
skeleton ("sk": is it recovered?). The second is the calibration for the untrusted words.

  python scripts/vowel_asr.py train --fold 0 --out runs/vowel_asr_f0 [--init runs/ctc1b_brh_s3/final]
  python scripts/vowel_asr.py score --fold 0 --model runs/vowel_asr_f0/final --out evals/tts/vowel_asr/scores_f0.json
"""
import argparse, collections, json, random, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
from transformers import Wav2Vec2ForCTC
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "data/tts/vowel_asr"
VOW = "aāiīuūeo"; NEG = -1e4; MIN_N = 3

def rows_of(name, spans):
    out = []
    for l in open(D / f"{name}.jsonl", encoding="utf-8"):
        r = json.loads(l); s = spans.get(r["id"])
        if s and s["spans"]: r["spans"] = s["spans"]; r["T"] = s["T"]; out.append(r)
    return out
def spell(s, vocab): return [vocab[c] for c in s] if s and all(c in vocab for c in s) else None
def hidden(t, fold): return t == "u" or t == f"k{fold}"
def plan(r, fold, vocab, lo=0, hi=None, override=None):
    """words lo..hi of an utterance -> (target ids, [(first, last) frame runs to blank out]).
    Visible words are spelled; a run of hidden words becomes one blanked stretch with a delimiter on each side."""
    hi = len(r["words"]) - 1 if hi is None else hi; items = []
    for i in range(lo, hi + 1):
        w, rom, t = r["words"][i]
        seq = spell(override[i], vocab) if override and i in override else (None if hidden(t, fold) else spell(rom, vocab))
        if seq is not None: items.append(seq); continue
        sp = r["spans"][i]
        if sp is None: continue                                   # no letters the aligner knows: nothing to blank
        if items and isinstance(items[-1], tuple): items[-1] = (items[-1][0], sp[1])
        else: items.append((sp[0], sp[1]))
    ids = []; masks = []
    for k, it in enumerate(items):
        if k: ids.append(vocab["|"])
        if isinstance(it, tuple): masks.append(it)
        else: ids += it
    return ids, masks
def audio(r):
    x, sr = sf.read(ROOT / r["audio"], dtype="float32"); assert sr == 16000
    return (x - x.mean()) / (x.std() + 1e-7)
def batches(rows, max_sec, shuffle, seed=0):
    idx = sorted(range(len(rows)), key=lambda i: rows[i]["duration"]); out = []; cur = []
    for i in idx:
        if cur and rows[i]["duration"] * (len(cur) + 1) > max_sec: out.append(cur); cur = []
        cur.append(i)
    if cur: out.append(cur)
    if shuffle: random.Random(seed).shuffle(out)
    return out
def collate(rows, ix):
    xs = [torch.from_numpy(audio(rows[i])) for i in ix]; n = torch.tensor([len(x) for x in xs])
    x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True); m = (torch.arange(x.shape[1])[None] < n[:, None]).long()
    return x, m, n
def labelled(rows, ix, fold, vocab):
    """batch for the loss: audio, padded targets, and the frames to blank as rows (item, first, last)"""
    x, m, n = collate(rows, ix); pl = [plan(rows[i], fold, vocab) for i in ix]; tl = torch.tensor([len(p[0]) for p in pl])
    pad = torch.zeros(len(ix), max(1, int(tl.max())), dtype=torch.long)
    for b, p in enumerate(pl): pad[b, :tl[b]] = torch.tensor(p[0], dtype=torch.long)
    mk = torch.tensor([(b, a, e) for b, p in enumerate(pl) for a, e in p[1]], dtype=torch.long).reshape(-1, 3)
    return x, m, n, pad, tl, mk
def blank_out(lp, mk):
    """lp [B, T, C]: on the listed frames every token but blank becomes impossible, with no gradient"""
    if not len(mk): return lp
    keep = torch.ones(lp.shape[:2], dtype=torch.bool, device=lp.device)
    for b, a, e in mk.tolist(): keep[b, a:e + 1] = False
    row = torch.full((lp.shape[-1],), NEG, device=lp.device); row[0] = 0.0
    return torch.where(keep[..., None], lp, row)
def ctc(lp, tg, il, tl):   # lp [B, T, C] float32 log-probs -> negative log-likelihood per item
    return F.ctc_loss(lp.transpose(0, 1), tg, il, tl, blank=0, reduction="none", zero_infinity=False)
def forward(model, x, m, n, dev, amp):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(input_values=x.to(dev), attention_mask=m.to(dev)).logits
    return torch.log_softmax(logits.float(), -1), model._get_feat_extract_output_lengths(n)

# ---------------------------------------------------------------- scoring
class AudioBatches(torch.utils.data.Dataset):
    def __init__(self, rows, bl): self.rows = rows; self.bl = bl
    def __len__(self): return len(self.bl)
    def __getitem__(self, k): torch.set_num_threads(1); return collate(self.rows, self.bl[k])
@torch.no_grad()
def emit(model, rows, dev, amp, max_sec):
    model.eval(); E = [None] * len(rows); bl = batches(rows, max_sec, False)   # decoding in workers: in the main thread the full set took ~20 min with the GPU idle
    for ix, xmn in zip(bl, torch.utils.data.DataLoader(AudioBatches(rows, bl), batch_size=None, num_workers=8 if len(bl) > 8 else 0)):
        lp, il = forward(model, *xmn, dev, amp)
        for b, i in enumerate(ix): E[i] = lp[b, :il[b]].cpu()
    return E
def edits(s):
    out = set(); n = len(s)
    for i, c in enumerate(s):
        if c in VOW:
            out.update(s[:i] + v + s[i + 1:] for v in VOW if v != c); out.add(s[:i] + s[i + 1:])
        if c == "w": out.update(s[:i] + v + s[i + 1:] for v in ("o", "ū", "u", "au"))
        if c == "y": out.update(s[:i] + v + s[i + 1:] for v in ("ī", "e", "i", "ai"))
        if c == "h" and i > 0: out.add(s[:i] + s[i + 1:])
    for i in range(1, n + 1):   # a vowel between two consonants, or after a final consonant
        if s[i - 1] not in VOW and (i == n or s[i] not in VOW): out.update(s[:i] + v + s[i:] for v in "aiueo")
    if n and s[0] not in VOW: out.update(v + s for v in "aiu")
    return {x for x in out if x and x != s and any(c in VOW for c in x)}
def score_jobs(E, jobs, dev, bs=4096):
    """jobs: (utterance index, first frame, last frame + 1, target ids, frame runs to blank) -> log-likelihood each"""
    out = np.full(len(jobs), NEG, dtype=np.float32); order = sorted(range(len(jobs)), key=lambda j: jobs[j][2] - jobs[j][1]); row = None
    for a in range(0, len(order), bs):
        ix = order[a:a + bs]; seg = []
        for j in ix:
            u, f0, f1, _, masks = jobs[j]; s = E[u][f0:f1]
            if masks:
                s = s.clone()
                if row is None: row = torch.full((s.shape[-1],), NEG); row[0] = 0.0
                for x, y in masks: s[max(0, x - f0):max(0, y + 1 - f0)] = row
            seg.append(s)
        lp = torch.nn.utils.rnn.pad_sequence(seg, batch_first=True).to(dev); il = torch.tensor([len(s) for s in seg]); tl = torch.tensor([len(jobs[j][3]) for j in ix])
        tg = torch.zeros(len(ix), int(tl.max()), dtype=torch.long)
        for b, j in enumerate(ix): tg[b, :tl[b]] = torch.tensor(jobs[j][3])
        ll = -ctc(lp, tg.to(dev), il, tl).cpu().numpy(); ll[~np.isfinite(ll)] = NEG; out[ix] = np.maximum(ll, NEG)
    return out
def window(r, i, T):
    sp = r["spans"]; lo = max(0, i - 2); hi = min(len(r["words"]) - 1, i + 2)
    while lo < i and sp[lo] is None: lo += 1
    while hi > i and sp[hi] is None: hi -= 1
    return lo, hi, (0 if lo == 0 else max(0, sp[lo][0] - 1)), (T if hi == len(r["words"]) - 1 else min(T, sp[hi][1] + 2))
def climb(E, rows, fold, vocab, dev, starts, extra, occs, rounds=4, log=None):
    """starts {key: (word, start spelling)}, extra {key: [spellings to try in round 1]}, occs {word: [(utt, pos)]} -> {key: result}"""
    table = {k: {} for k in starts}; cur = {k: s for k, (w, s) in starts.items()}; todo = {k: {s, *extra.get(k, []), *edits(s)} for k, (w, s) in starts.items()}; path = {k: [s] for k, (w, s) in starts.items()}
    for rd in range(rounds):
        jobs = []; meta = []
        for k, cands in todo.items():
            w = starts[k][0]
            for c in cands:
                if c in table[k] or spell(c, vocab) is None: continue
                table[k][c] = np.full(len(occs[w]), NEG, dtype=np.float32)
                for o, (u, i) in enumerate(occs[w]):
                    lo, hi, f0, f1 = window(rows[u], i, E[u].shape[0]); ids, masks = plan(rows[u], fold, vocab, lo, hi, override={i: c})
                    jobs.append((u, f0, f1, ids, masks)); meta.append((k, c, o))
        if not jobs: break
        ll = score_jobs(E, jobs, dev)
        for (k, c, o), x in zip(meta, ll): table[k][c][o] = x
        todo = {}
        for k in starts:
            best = max(table[k], key=lambda c: table[k][c].mean()) if table[k] else cur[k]
            if best != cur[k]: cur[k] = best; path[k].append(best); todo[k] = edits(best)
        if log: log(f"  climb round {rd + 1}: {len(jobs)} window scores, {len(todo)} words still moving")
        if not todo: break
    res = {}
    for k, (w, s) in starts.items():
        t = table[k]
        if not t: continue
        M = np.stack(list(t.values())); names = list(t); mean = M.mean(1); order = np.argsort(-mean); win = np.bincount(M.argmax(0), minlength=len(names)) / M.shape[1]
        res[k] = {"w": w, "start": s, "final": cur[k], "path": path[k], "n": M.shape[1],
                  "top": [[names[j], round(float(mean[j] - mean[order[0]]), 3), round(float(win[j]), 3)] for j in order[:6]]}
    return res
def sample_occs(rows, words, cap, seed=0):
    occs = collections.defaultdict(list)
    for u, r in enumerate(rows):
        for i, (w, _, _) in enumerate(r["words"]):
            if w in words and r["spans"][i] is not None: occs[w].append((u, i))
    rnd = random.Random(seed)
    return {w: (rnd.sample(o, cap) if len(o) > cap else o) for w, o in occs.items()}
def run_scoring(model, rows, types, fold, vocab, dev, amp, max_sec, cap, pick=None, log=print):
    E = emit(model, rows, dev, amp, max_sec); log(f"  emissions for {len(rows)} utterances")
    words = {w for w, t in types.items() if hidden(t["t"], fold) and (t["t"] != "u" or t["n"] >= MIN_N)} if pick is None else pick   # untrusted words seen once or twice: one noisy verdict each, and 30k of them made the search crawl
    occs = sample_occs(rows, words, cap); starts = {}; extra = {}
    for w in occs:
        t = types[w]; r0 = t["r"] if spell(t["r"], vocab) is not None else "".join(c for c in t["r"] if c in vocab)
        if r0: starts[w + "\tr"] = (w, r0); extra[w + "\tr"] = t["alt"]
        if t["t"] != "u" and t.get("sk") and t["sk"] != t["r"] and spell(t["sk"], vocab) is not None: starts[w + "\tsk"] = (w, t["sk"])
    return climb(E, rows, fold, vocab, dev, starts, extra, occs, log=log)
def norm(s):   # same word for this test: doubled consonants and apostrophes are not something the vowel search can change
    return "".join(c for i, c in enumerate(s) if c != "'" and not (i and c == s[i - 1] and c not in VOW))
def summary(res, types):
    same = lambda v: norm(v["final"]) == norm(types[v["w"]]["r"])  # noqa: E731
    k = [(key, v) for key, v in res.items() if types[v["w"]]["t"] != "u"]
    r = [same(v) for key, v in k if key.endswith("\tr")]; s = [same(v) for key, v in k if key.endswith("\tsk")]
    wt = lambda sel: sum(types[v["w"]]["n"] * same(v) for key, v in k if key.endswith(sel)) / max(1, sum(types[v["w"]]["n"] for key, v in k if key.endswith(sel)))  # noqa: E731
    return {"hidden_trusted_types": len(r), "dictionary_spelling_survives": round(sum(r) / max(1, len(r)), 3), "survives_token_weighted": round(wt("\tr"), 3),
            "from_skeleton_types": len(s), "skeleton_recovered": round(sum(s) / max(1, len(s)), 3), "recovered_token_weighted": round(wt("\tsk"), 3)}

# ---------------------------------------------------------------- main
ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "score", "dump"]); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--init", default="runs/ctc1b_brh_s3/final"); ap.add_argument("--model"); ap.add_argument("--scores"); ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--max-sec", type=float, default=120)
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--head-lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--cap", type=int, default=48); ap.add_argument("--limit", type=int, default=0, help="smoke test: use this many training utterances"); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
vocab = json.load(open(D / "vocab.json", encoding="utf-8")); types = json.load(open(D / "types.json", encoding="utf-8")); spans = json.load(open(D / "spans.json"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda"
if dev.type != "cuda" and not a.limit: sys.exit("no CUDA: refusing a full run on " + dev.type + " (use --limit for a smoke test)")
out = ROOT / a.out; T0 = time.time()
def log(*x): print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)
train = rows_of("train", spans); devset = rows_of("dev", spans)
if a.limit: train = train[:a.limit]; devset = devset[:max(8, a.limit // 4)]

if a.cmd == "score":
    model = Wav2Vec2ForCTC.from_pretrained(ROOT / a.model).to(dev); rows = train + devset
    res = run_scoring(model, rows, types, a.fold, vocab, dev, amp, a.max_sec * 2, a.cap, log=log)
    sm = summary(res, types); log("summary", json.dumps(sm)); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"fold": a.fold, "model": a.model, "summary": sm, "results": res}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    log("done;", out); sys.exit()

if a.cmd == "dump":
    # every occurrence of every word this model is blind to, with the log-likelihood of each candidate spelling: the raw
    # material for writing each utterance as it was spoken (the decision rule is applied afterwards, off the GPU)
    model = Wav2Vec2ForCTC.from_pretrained(ROOT / a.model).to(dev); rows = train + devset; E = emit(model, rows, dev, amp, a.max_sec * 2); log("emissions done")
    sc = json.load(open(ROOT / a.scores, encoding="utf-8"))["results"]; cache = {}
    def cands(w):
        if w not in cache:
            t = types[w]; c = [t["r"] if spell(t["r"], vocab) is not None else "".join(x for x in t["r"] if x in vocab)]
            for key in (w + "\tr", w + "\tsk"):
                if key in sc: c += [x[0] for x in sc[key]["top"][:6]]
            c += t["alt"][:3]
            if w + "\tr" not in sc and c[0]: c += sorted(edits(c[0]))
            cache[w] = [x for x in dict.fromkeys(c) if spell(x, vocab) is not None][:48]
        return cache[w]
    out.parent.mkdir(parents=True, exist_ok=True); n = 0
    with open(out, "w", encoding="utf-8") as f:
        for b in range(0, len(rows), 400):
            jobs = []; meta = []
            for u in range(b, min(len(rows), b + 400)):
                r = rows[u]
                for i, (w, _, t) in enumerate(r["words"]):
                    if not hidden(t, a.fold) or r["spans"][i] is None: continue
                    cs = cands(w)
                    if not cs: continue
                    lo, hi, f0, f1 = window(r, i, E[u].shape[0]); meta.append((u, i, len(cs)))
                    for c in cs: ids, masks = plan(r, a.fold, vocab, lo, hi, override={i: c}); jobs.append((u, f0, f1, ids, masks))
            ll = score_jobs(E, jobs, dev); k = 0
            for u, i, m in meta:
                f.write(json.dumps({"id": rows[u]["id"], "i": i, "w": rows[u]["words"][i][0], "c": cands(rows[u]["words"][i][0]), "ll": [round(float(x), 2) for x in ll[k:k + m]]}, ensure_ascii=False) + "\n"); k += m; n += 1
            if (b // 400) % 5 == 0: log(f"  dump {min(len(rows), b + 400)}/{len(rows)} utterances, {n} word occurrences")
    log("done;", out, n, "occurrences"); sys.exit()

model = Wav2Vec2ForCTC.from_pretrained(ROOT / a.init, vocab_size=len(vocab), pad_token_id=0, ignore_mismatched_sizes=True, layerdrop=0.0, mask_time_prob=0.05, ctc_zero_infinity=True)
model.freeze_feature_encoder(); model.to(dev)
def usable(r): return sum(not hidden(t, a.fold) and spell(rom, vocab) is not None for _, rom, t in r["words"]) >= 2
train = [r for r in train if usable(r) and r["duration"] <= 22]; devset = [r for r in devset if usable(r)]
bt = batches(train, a.max_sec, True); total = int(len(bt) * a.epochs)
head = [p for n, p in model.named_parameters() if n.startswith("lm_head")]; body = [p for n, p in model.named_parameters() if not n.startswith("lm_head") and p.requires_grad]
opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.head_lr}], weight_decay=0.0, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * max(0.05, 1 - s / total))
shown = sum(not hidden(t, a.fold) for r in train for _, _, t in r["words"]) / sum(len(r["words"]) for r in train)
log(f"fold {a.fold}: {len(train)} utterances, {sum(r['duration'] for r in train) / 3600:.1f} h, {shown:.0%} of words are labels, {len(bt)} batches/epoch, {total} steps, device {dev}")
pool = sorted(w for w, t in types.items() if t["t"] == f"k{a.fold}" and t["n"] >= 8); probe_words = set(random.Random(1).sample(pool, k=min(150, len(pool))))
probe_rows = [r for r in train if any(w in probe_words for w, _, _ in r["words"])][:1500]
def loss_of(batch):
    x, m, n, pad, tl, mk = batch; lp, il = forward(model, x, m, n, dev, amp)
    l = ctc(blank_out(lp, mk), pad.to(dev), il, tl); ok = torch.isfinite(l) & (l < -NEG / 2) & (tl.to(dev) > 0)
    return l, ok, tl
def evaluate(step):
    model.eval(); tot = n = 0
    with torch.no_grad():
        for ix in batches(devset, a.max_sec, False):
            l, ok, tl = loss_of(labelled(devset, ix, a.fold, vocab)); tot += float(l[ok].sum()); n += int(tl[ok.cpu()].sum())
    res = run_scoring(model, probe_rows, types, a.fold, vocab, dev, amp, a.max_sec * 2, 12, pick=probe_words, log=lambda *x: None)
    log(f"eval step {step}: dev loss/token {tot / max(1, n):.3f} | blind probe {json.dumps(summary(res, types))}"); model.train()
class Batches(torch.utils.data.Dataset):   # audio decoding in worker processes: in the main thread it left the GPU idle half the time
    def __init__(self, order): self.order = order
    def __len__(self): return len(self.order)
    def __getitem__(self, k): torch.set_num_threads(1); return labelled(train, self.order[k], a.fold, vocab)
model.train(); step = 0; run = 0.0
while step < total:
    for batch in torch.utils.data.DataLoader(Batches(random.Random(step).sample(bt, len(bt))), batch_size=None, num_workers=a.workers, prefetch_factor=4 if a.workers else None):
        if step >= total: break
        l, ok, tl = loss_of(batch)
        if ok.any():
            loss = (l[ok] / tl.to(dev)[ok]).mean(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            run = 0.95 * run + 0.05 * loss.item() if step else loss.item()
        sched.step(); step += 1
        if step % 20 == 0 or step == 1: print(f"{step}/{total} [loss {run:.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - T0) / step:.2f} s/step]", flush=True)
        if step % a.eval_every == 0 and step < total: evaluate(step)
evaluate(step); (out / "final").mkdir(parents=True, exist_ok=True); model.save_pretrained(out / "final"); json.dump(vocab, open(out / "final/vocab_latin.json", "w", encoding="utf-8"), ensure_ascii=False)
log("done;", out / "final")
