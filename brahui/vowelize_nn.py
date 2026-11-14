"""Text-only vowelizer: Arabic-script Brahui -> vowelled Latin, one word at a time with two words of context on each side.

A small character-level encoder-decoder (about 3M parameters) trained by scripts/train_vowelizer.py on the spellings the
vowel-aware acoustic model heard in real speech (scripts/vowel_retranscribe.py). It closes the gap between TTS training
text (vowels as heard) and synthesis text (MT output, Arabic script, no audio): context picks between homographs and the
character model generalises to words that were never heard.

    from brahui.vowelize_nn import NeuralVowelizer
    NeuralVowelizer("runs/vowelizer_v1").text("...")
"""
import json, math
from pathlib import Path
import torch, torch.nn as nn
from brahui.text import normalize

STRIP = '۔،؟؛!.,?:"\'()…'; PAD, BOS, EOS = 0, 1, 2; CTX = 2

def source(words, i, hint=None):
    """the model's input for word i: two words before, the word in angle brackets, two words after, and then, when the word
    was heard in training, "=" and its one or two usual spellings. The model copies the hint unless the context says
    otherwise, and writes the word from its letters when there is none (hints are dropped at random in training)."""
    s = " ".join(words[max(0, i - CTX):i] + ["<" + words[i] + ">"] + words[i + 1:i + 1 + CTX])
    return s + "=" + "/".join(hint) if hint else s

class Net(nn.Module):
    def __init__(self, n_src, n_tgt, d=192, heads=4, layers=3, ff=512, drop=0.1):
        super().__init__()
        self.d = d; self.src = nn.Embedding(n_src, d, padding_idx=PAD); self.tgt = nn.Embedding(n_tgt, d, padding_idx=PAD)
        self.tr = nn.Transformer(d, heads, layers, layers, ff, drop, batch_first=True, norm_first=True); self.out = nn.Linear(d, n_tgt)
        pe = torch.zeros(256, d); pos = torch.arange(256)[:, None]; div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); self.register_buffer("pe", pe)
    def emb(self, e, x): return e(x) * math.sqrt(self.d) + self.pe[:x.shape[1]]
    def encode(self, s): return self.tr.encoder(self.emb(self.src, s), src_key_padding_mask=s == PAD)
    def decode(self, mem, s, t):
        m = torch.triu(torch.ones(t.shape[1], t.shape[1], dtype=torch.bool, device=t.device), 1)
        return self.out(self.tr.decoder(self.emb(self.tgt, t), mem, tgt_mask=m, tgt_key_padding_mask=t == PAD, memory_key_padding_mask=s == PAD))
    def forward(self, s, t): return self.decode(self.encode(s), s, t)
    @torch.no_grad()
    def greedy(self, s, max_len=32):
        mem = self.encode(s); t = torch.full((s.shape[0], 1), BOS, dtype=torch.long, device=s.device); done = torch.zeros(s.shape[0], dtype=torch.bool, device=s.device)
        for _ in range(max_len):
            nx = self.decode(mem, s, t)[:, -1].argmax(-1); nx[done] = PAD; t = torch.cat([t, nx[:, None]], 1); done |= nx == EOS
            if done.all(): break
        return t[:, 1:]

def encode_batch(strings, table, add=None):
    ids = [[table.get(c, table["?"]) for c in s] for s in strings]
    if add: ids = [[BOS] + x + [EOS] for x in ids]
    n = max(len(x) for x in ids); return torch.tensor([x + [PAD] * (n - len(x)) for x in ids], dtype=torch.long)

class NeuralVowelizer:
    def __init__(self, path, device=None):
        path = Path(path); cfg = json.load(open(path / "config.json", encoding="utf-8")); self.sv = cfg["src"]; self.tv = cfg["tgt"]; self.inv = {i: c for c, i in self.tv.items()}
        self.dev = torch.device(device or "cpu"); self.net = Net(len(self.sv), len(self.tv), **cfg["net"]).to(self.dev).eval()
        self.net.load_state_dict(torch.load(path / "model.pt", map_location=self.dev)); self.cache = {}; self.lex = cfg.get("lex", {})
    def words(self, words):
        """vowelled spelling of every word of one sentence (list of normalized Arabic-script words)"""
        src = [source(words, i, self.lex.get(words[i])) for i in range(len(words))]; todo = [s for s in dict.fromkeys(src) if s not in self.cache]
        for b in range(0, len(todo), 256):
            chunk = todo[b:b + 256]; out = self.net.greedy(encode_batch(chunk, self.sv).to(self.dev)).tolist()
            for s, o in zip(chunk, out): self.cache[s] = "".join(self.inv[i] for i in o[:o.index(EOS)] if i > EOS) if EOS in o else "".join(self.inv[i] for i in o if i > EOS)
        return [self.cache[s] for s in src]
    def text(self, t):
        toks = normalize(t).split(); core = [w.strip(STRIP) for w in toks]; ix = [i for i, w in enumerate(core) if w]
        out = self.words([core[i] for i in ix]); res = list(toks)
        for i, r in zip(ix, out):
            a = toks[i].find(core[i]); res[i] = toks[i][:a] + r + toks[i][a + len(core[i]):]
        return " ".join(res)

DIGRAPHS = ("kh", "gh", "sh", "ch", "th", "ph", "bh", "dh", "jh", "zh", "ṭh", "ḍh", "ṛh"); VOWELS = "aāiīuūeo"
def broken(s):
    """a spelling that cannot be said as written: no vowel at all, an impossible start ("mlīna", "rporaṭng", "hnnāde"), or a run
    of four consonants. Such forms are consonant skeletons the fallback left unvowelled."""
    u = []; i = 0
    while i < len(s):
        if s[i:i + 2] in DIGRAPHS: u.append(s[i:i + 2]); i += 2
        else: u.append(s[i]); i += 1
    u = [x for x in u if x != "'"]; c = "".join("V" if x in VOWELS else "C" for x in u)
    if len(u) < 2: return False
    if "V" not in c or "CCCC" in c: return True
    return c.startswith("CC") and not (u[1] in ("r", "w", "y") or (u[0] == "s" and u[1] in ("t", "ṭ", "k", "p")))

class HeardVowelizer:
    """Lookup front end with the same interface. Per word, the first that applies:
      review    data/lexicon/vowel_overrides.json   a native speaker's corrections (scripts/make_tts_drill.py --apply)
      loan      data/lexicon/vowel_loans.json       Urdu / English / Arabic loans and names with a known pronunciation
      heard     data/lexicon/vowel_heard.json       the spelling most often heard for the word in the recordings
      dict      data/tts/vowel_asr/extra_dict.jsonl dictionary words that never occur in the recordings
      rule      brahui.vowelize.Vowelizer           derived from a dictionary word
      model     runs/vowelizer_v1 without a hint    the letter model writes the word, when the heard or rule spelling is broken
                                                    (see broken()) or the rules had only their fallback guess ("rporaṭng" -> "riporṭing")
    On held-out utterances plain lookup beat the network with hints (91.3% vs 90.6% exact words, 2026-11-13: context did not
    help choose between two spellings), so the network only writes words nothing else knows."""
    def __init__(self, root=None, model="runs/vowelizer_v1"):
        from brahui.vowelize import Vowelizer
        root = Path(root or Path(__file__).resolve().parent.parent); self.rule = Vowelizer(); self.memo = {}
        def table(name): p = root / "data/lexicon" / name; return json.load(open(p, encoding="utf-8")) if p.exists() else {}
        self.review = table("vowel_overrides.json"); self.loans = table("vowel_loans.json")
        self.heard = {w: max(c, key=c.get) for w, c in table("vowel_heard.json").items()}; self.dict = {}
        p = root / "data/tts/vowel_asr/extra_dict.jsonl"
        if p.exists(): self.dict = {d["w"]: d["r"] for d in map(json.loads, open(p, encoding="utf-8"))}
        self.nn = NeuralVowelizer(root / model) if (root / model / "model.pt").exists() else None
    def plain(self, w):
        o = self.nn.net.greedy(encode_batch([source([w], 0)], self.nn.sv).to(self.nn.dev)).tolist()[0]
        return "".join(self.nn.inv[i] for i in (o[:o.index(EOS)] if EOS in o else o) if i > EOS)
    def explain(self, w):
        """-> (spelling, source) for one normalized Arabic-script word"""
        if w not in self.memo: self.memo[w] = self._explain(w)
        return self.memo[w]
    def _explain(self, w):
        for name, t in (("review", self.review), ("loan", self.loans)):
            if w in t: return t[w], name
        s, src = (self.heard[w], "heard") if w in self.heard else (self.dict[w], "dict") if w in self.dict else (None, None)
        if s is None:
            r, how = self.rule.word(w); s = r.lower(); src = "guess" if how == "fallback" else "rule"
        if self.nn and len(w) > 1 and (broken(s) or src == "guess"):
            m = self.plain(w)
            if m and not broken(m): return m, "model"
        return s, src
    def words(self, words): return [self.explain(w)[0] for w in words]
    text = NeuralVowelizer.text

def load(path):
    """"heard" -> HeardVowelizer, a run directory -> NeuralVowelizer"""
    return HeardVowelizer() if str(path).endswith("heard") else NeuralVowelizer(path)
