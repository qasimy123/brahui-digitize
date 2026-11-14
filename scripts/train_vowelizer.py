"""Train the text-only vowelizer (brahui/vowelize_nn.py) and measure it on held-out utterances.

Data: utterances with a vowelled spelling per word, {"id", "words": [[arabic, roman, ...], ...]}.
  default  data/tts/vowel_asr/retrans.jsonl   (scripts/vowel_retranscribe.py: vowels as heard by the acoustic model)
  --data data/tts/vowel_asr/train.jsonl        (the Vowelizer's dictionary spellings: for testing this script)
Held out: 5% of utterances by id hash. Reported: exact-word accuracy overall, on word types never seen in training, and
on words that have more than one spelling in the data, next to the baseline "most frequent spelling of this word in
training, else the rule-based Vowelizer".

  uv run python scripts/train_vowelizer.py [--data ...] [--out runs/vowelizer_v1] [--epochs 6]
"""
import argparse, collections, hashlib, json, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import torch, torch.nn.functional as F  # noqa: E402
from brahui.vowelize_nn import Net, source, encode_batch, PAD, BOS, EOS  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/tts/vowel_asr/retrans.jsonl"); ap.add_argument("--out", default="runs/vowelizer_v1"); ap.add_argument("--epochs", type=int, default=16); ap.add_argument("--cap", type=int, default=24, help="samples per word type per epoch: frequent words would otherwise fill every batch and the tail is learned slowly (3 uncapped epochs: 78%)")
ap.add_argument("--hint-drop", type=float, default=0.3); ap.add_argument("--bs", type=int, default=384); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--extra", default="", help="jsonl of extra word-level pairs {w, r} (dictionary entries), trained without context")
a = ap.parse_args(); random.seed(0); torch.manual_seed(0)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
rows = [json.loads(l) for l in open(ROOT / a.data, encoding="utf-8")]
if a.limit: rows = rows[:a.limit]
def held(r): return int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 20 == 0
def samples(rs):   # (sentence words, position, spelling, word)
    return [([w[0] for w in r["words"]], i, w[1], w[0]) for r in rs for i, w in enumerate(r["words"]) if w[1]]
tr = samples([r for r in rows if not held(r)]); te = samples([r for r in rows if held(r)])
if a.extra: tr += [([d["w"]], 0, d["r"], d["w"]) for d in map(json.loads, open(ROOT / a.extra, encoding="utf-8")) if d.get("r")]
lex = collections.defaultdict(collections.Counter)
for _, _, t, w in tr: lex[w][t] += 1
# the hint: the usual spelling, plus a second one when it is a real alternative (>= 10% of the word's occurrences)
hints = {w: [x for k, (x, n) in enumerate(c.most_common(2)) if k == 0 or n / sum(c.values()) >= 0.1] for w, c in lex.items()}
def src_of(x, drop=0.0): return source(x[0], x[1], None if (drop and random.random() < drop) else hints.get(x[3]))
sv = {"<pad>": PAD, "?": 1, **{c: i + 2 for i, c in enumerate(sorted({c for x in tr for c in src_of(x)} | set("<>=/ ")))}}
tv = {"<pad>": PAD, "<s>": BOS, "</s>": EOS, "?": 3, **{c: i + 4 for i, c in enumerate(sorted({c for x in tr for c in x[2]}))}}
cfg = {"d": 192, "heads": 4, "layers": 3, "ff": 512, "drop": 0.1}; net = Net(len(sv), len(tv), **cfg).to(dev)
print(f"train {len(tr)} words, held out {len(te)}; source letters {len(sv)}, target letters {len(tv)}, {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters, device {dev}", flush=True)
bytype = collections.defaultdict(list)
for x in tr: bytype[x[3]].append(x)
def epoch_samples(): return [x for v in bytype.values() for x in (random.sample(v, a.cap) if len(v) > a.cap else v)]
steps = a.epochs * (len(epoch_samples()) // a.bs + 1); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 400) * max(0.02, 1 - s / steps)); step = 0; t0 = time.time()
def decode(strings):
    inv = {i: c for c, i in tv.items()}; out = []; net.eval(); order = sorted(range(len(strings)), key=lambda i: len(strings[i])); res = [None] * len(strings)
    for b in range(0, len(order), 1024):
        ix = order[b:b + 1024]; o = net.greedy(encode_batch([strings[i] for i in ix], sv).to(dev)).tolist()
        for i, x in zip(ix, o): res[i] = "".join(inv[t] for t in (x[:x.index(EOS)] if EOS in x else x) if t > EOS)
    net.train(); return res
for ep in range(a.epochs):
    ep_tr = epoch_samples(); random.shuffle(ep_tr); tr_sorted = [sorted(ep_tr[b:b + a.bs * 50], key=lambda x: sum(map(len, x[0][max(0, x[1] - 2):x[1] + 3]))) for b in range(0, len(ep_tr), a.bs * 50)]; bl = [c[b:b + a.bs] for c in tr_sorted for b in range(0, len(c), a.bs)]; random.shuffle(bl); run = 0.0
    for batch in bl:
        s = encode_batch([src_of(x, a.hint_drop) for x in batch], sv).to(dev); t = encode_batch([x[2] for x in batch], tv, add=True).to(dev)
        loss = F.cross_entropy(net(s, t[:, :-1]).reshape(-1, len(tv)), t[:, 1:].reshape(-1), ignore_index=PAD, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1; run = 0.98 * run + 0.02 * loss.item() if run else loss.item()
        if step % 200 == 0: print(f"{step}/{steps} [loss {run:.3f} {(time.time() - t0) / step:.2f} s/step]", flush=True)
    sub = random.Random(1).sample(te, min(4000, len(te))); pr = decode([src_of(x) for x in sub]); print(f"epoch {ep + 1}: held-out word accuracy {sum(p == x[2] for p, x in zip(pr, sub)) / len(sub):.1%} (sample of {len(sub)})", flush=True)
# ---- final evaluation against the lexicon baseline
from brahui.vowelize import Vowelizer  # noqa: E402
rule = Vowelizer(); pr = decode([src_of(x) for x in te]); base = [lex[x[3]].most_common(1)[0][0] if x[3] in lex else rule.word(x[3])[0].lower() for x in te]
nohint = decode([source(x[0], x[1]) for x in te])      # the same words with the hint withheld: how well it writes a word from its letters alone
def acc(sel, pred): g = [(p, x) for p, x in zip(pred, te) if sel(x)]; return (sum(p == x[2] for p, x in g) / max(1, len(g)), len(g))
multi = {w for w, c in lex.items() if len(c) > 1 and c.most_common(2)[1][1] / sum(c.values()) >= 0.1}
res = {}
for name, sel in (("all words", lambda x: True), ("word type seen in training", lambda x: x[3] in lex), ("word type never seen", lambda x: x[3] not in lex), ("words with two common spellings", lambda x: x[3] in multi)):
    m, n = acc(sel, pr); b, _ = acc(sel, base); h, _ = acc(sel, nohint); res[name] = {"n": n, "model": round(m, 4), "baseline": round(b, 4), "model_without_hint": round(h, 4)}
    print(f"{name:34s} n={n:6d}  model {m:.1%}  lexicon baseline {b:.1%}  model without hint {h:.1%}")
out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); torch.save(net.state_dict(), out / "model.pt")
json.dump({"src": sv, "tgt": tv, "net": cfg, "data": a.data, "eval": res, "lex": hints}, open(out / "config.json", "w", encoding="utf-8"), ensure_ascii=False)
print("saved", out)
