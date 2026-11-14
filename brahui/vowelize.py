"""Vowelled romanization of Perso-Arabic Brahui for TTS input.

The script leaves short vowels unwritten, so brahui.text.romanize gives a consonant skeleton ("makhafng", "hmo") and the
voice model has to guess the vowels; a native listener heard exactly those errors (word beginnings, short vowels;
blind review 2026-11-11). This module looks words up in vowelled sources instead:

  Shad dictionary (7.8k Arabic-script headwords with romanization), shaigle word list (1.1k),
  Ali & Kobayashi 2024 texts (3.9k sentences of vowelled, morpheme-segmented Brahui, Latin only) and their glossary.

Arabic-script words and vowelled Latin words are reduced to the same key (consonants plus long-vowel positions, short
vowels dropped), so a vowelled form can be found for a word that only ever appeared in Latin. Inflected forms missing
from the sources are split into a known stem and a known suffix. Anything else falls back to romanize(speak=True).

  from brahui.vowelize import Vowelizer;  v = Vowelizer();  v.word("کننگ") -> ("kanning", "lex")
  python -m brahui.vowelize < text      # one line in, one line out

Caveat: A&K recorded Sindh (Garhi Yasin) speakers; vowel qualities can differ from Quetta/Kalat Brahui.
"""
from __future__ import annotations
import collections, json, re, sys, unicodedata
from pathlib import Path
from brahui.text import normalize, romanize, _ROM_PUNCT

ROOT = Path(__file__).resolve().parent.parent

# ---- keys ----
_AR = {"ب": "b", "پ": "p", "ت": "t", "ط": "t", "ٹ": "T", "ث": "s", "س": "s", "ص": "s", "ج": "j", "چ": "C", "ح": "h", "خ": "X", "د": "d", "ڈ": "D",
       "ذ": "z", "ز": "z", "ض": "z", "ظ": "z", "ژ": "Z", "ر": "r", "ڑ": "R", "ش": "S", "غ": "G", "ف": "f", "ق": "k", "ک": "k", "ك": "k", "گ": "g",
       "ل": "l", "ڷ": "l", "م": "m", "ن": "n", "ں": "n", "ہ": "h", "ۀ": "h", "ة": "h", "ح": "h", "ا": "A", "آ": "A", "و": "W", "ؤ": "W", "ی": "Y", "ي": "Y", "ے": "Y", "ئ": "Y"}

def _squash(k: str) -> str:
    return re.sub(r"(.)\1+", r"\1", k)

def key_arabic(w: str) -> str:
    out = []
    for i, c in enumerate(w):
        if c == "ع": out.append("A" if i == 0 else "")
        elif c == "ھ": continue                                   # aspiration: not marked consistently in the Latin sources
        elif c == "ہ" and i == len(w) - 1 and i > 0: out.append("V")   # final he = a short final vowel
        elif c == "ئ" and i == 0: out.append("A")                 # ئس, ئنا: hamza seat for an initial vowel
        elif c in _AR: out.append(_AR[c])
    return "".join(out)   # no squashing: a doubled letter in the script is two sounds with a vowel between (ننا nanā), not a geminate

_LAT = [("kh", "k"), ("gh", "g"), ("bh", "b"), ("ph", "p"), ("th", "t"), ("dh", "d"), ("jh", "j"), ("ch", "C"), ("sh", "S"), ("ai", "Y"), ("ay", "Y"), ("au", "W"), ("aw", "W")]
_LAT1 = {"ā": "A", "ē": "Y", "e": "Y", "ī": "Y", "y": "Y", "ō": "W", "o": "W", "ū": "W", "w": "W", "v": "W", "x": "X", "ɣ": "G", "š": "S", "c": "C", "č": "C", "ž": "Z",
         "ṭ": "T", "ḍ": "D", "ṛ": "R", "ṇ": "n", "ṅ": "n", "ŋ": "n", "ṉ": "n", "ṣ": "s", "ḥ": "h", "ẓ": "z", "ẕ": "z", "ḏ": "z", "ṯ": "s", "ɬ": "l", "ḷ": "l", "q": "k",
         "a": "", "i": "", "u": "", "ʰ": "", "ʿ": "", "ʾ": "", "ʻ": "", "ʼ": "", "'": "", "’": "", "‘": ""}

def _clean_latin(w: str) -> str:
    w = unicodedata.normalize("NFC", w.lower())
    w = w.replace("ã", "an").replace("ẽ", "en").replace("ĩ", "in").replace("õ", "on").replace("ũ", "un").replace("̃", "")
    return re.sub(r"[^a-zāēīōūxɣšcčžṭḍṛṇṅŋṉṣḥẓẕḏṯɬḷʰ'’‘ʿʾʻʼ]", "", w)

def key_latin(w: str, initial: bool = True) -> str:
    """w: one cleaned vowelled word or morpheme. initial=True marks a word-initial vowel the way the script does (alef)."""
    w = re.sub(r"([^aiueoāēīōū])\1", r"\1", w)   # a Latin double consonant is a geminate, written once in the script (kanning = کننگ)
    out = []; i = 0
    if initial and w and w[0] in "aiuāeēīoōū": out.append("A")
    if initial and w[:1] in ("ā",): i = 1
    while i < len(w):
        two = w[i:i + 2]
        hit = next((v for k, v in _LAT if k == two), None)
        if hit is not None and not (two in ("ai", "ay", "au", "aw") and i + 2 < len(w) and w[i + 2] in "aiuāeēīoōū"):
            out.append(hit); i += 2; continue
        ch = w[i]
        if ch == "h" and i == len(w) - 1 and i > 0 and w[i - 1] in "aiu": i += 1; continue   # final -ah = final he
        out.append(_LAT1.get(ch, ch)); i += 1
    if initial and len(w) > 1 and re.search(r"[aiu]h?$", w): out.append("V")   # kamra (کمرہ) must not match kamar (کمر)
    return "".join(out)

def to_scheme(w: str) -> str:
    """Vowelled source spelling -> the letters the TTS models were trained on (kh gh sh ch, macron long vowels, plain e o)."""
    w = _clean_latin(w)
    for a, b in (("x", "kh"), ("ɣ", "gh"), ("š", "sh"), ("č", "ch"), ("c", "ch"), ("ž", "zh"), ("v", "w"), ("ʰ", "h"), ("ē", "e"), ("ō", "o"), ("ɬ", "ḷ"),
                 ("ṇ", "n"), ("ṅ", "n"), ("ŋ", "ng"), ("ṣ", "s"), ("ḥ", "h"), ("ẓ", "z"), ("ẕ", "z"), ("ḏ", "z"), ("ṯ", "s"), ("ʿ", "'"), ("ʾ", "'"), ("ʻ", "'"), ("ʼ", "'"), ("’", "'"), ("‘", "'")):
        w = w.replace(a, b)
    return w.replace("chh", "ch")


# frequent words the sources miss: names and loanwords whose vowels are not in doubt ("ء" as the -e clitic is a guess to confirm)
MANUAL = {"براہوئی": "brāhuī", "یعنی": "yānī", "پاکستان": "pākistān", "پاکستانی": "pākistānī", "بلوچستان": "balochistān", "پروگرام": "progrām", "بالکل": "bilkul",
          "السلام": "assalām", "انشاءاللہ": "inshāallāh", "ماشاءاللہ": "māshāallāh", "ٹائم": "ṭāim", "احمد": "ahmad", "تقریبا": "taqrīban", "مہمان": "mehmān",
          "میڈیا": "mīḍiyā", "خدمت": "khidmat", "فائدہ": "fāida", "ویڈیو": "wīḍiyo", "مہربانی": "mehrbānī", "چینل": "chainal", "تنظیم": "tanzīm", "کلات": "kalāt",
          "ناظرین": "nāzirīn", "مسئلہ": "masala", "بسم": "bismi", "مینگل": "mengal", "ء": "e", "اور": "aur", "و": "o", "اللہ": "allāh", "اٹ": "aṭ"}


class Vowelizer:
    def __init__(self):
        self.direct: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)   # Arabic word -> vowelled forms
        self.bykey: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)    # key -> vowelled whole words
        self.stems: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.sufs: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        L = ROOT / "data/lexicon"
        for line in open(L / "shad_brahui_english_ocr.jsonl", encoding="utf-8"):
            r = json.loads(line); self._pair(r.get("headword", ""), r.get("roman", ""), 3)
        for line in open(L / "shaigle_brh_en.txt", encoding="utf-8"):
            m = re.match(r"^(.+?) \((.+?)\):", line)
            if m: self._pair(m.group(1), m.group(2), 2)
        for line in open(L / "ak_glossary.txt", encoding="utf-8"):
            if line.startswith("#") or not line.strip(): continue
            head = re.split(r"\s+(?:n|adj|adv|vi|vt|itj|pron|num|pp|conj|prt|v|pn|det|cl|q)\.", line.strip())[0]
            for w in head.split():
                self._latin(w, 2)
        for line in open(ROOT / "data/parallel/ak2024_train.jsonl", encoding="utf-8"):
            for w in json.loads(line)["brh_roman"].split():
                self._latin(w, 1)

    def _pair(self, ar: str, lat: str, n: int):
        aw, lw = normalize(ar).split(), lat.replace("-", " ").split()
        if len(aw) != len(lw): return
        for a, l in zip(aw, lw):
            c = to_scheme(l)
            if c and key_arabic(a) == key_latin(_clean_latin(l)): self.direct[a][c] += n
            if c: self.bykey[key_latin(_clean_latin(l))][c] += n; self.stems[key_latin(_clean_latin(l))][c] += n

    def _latin(self, w: str, n: int):
        w = w.strip(".,;:!?()\"“”‘’").replace("=", "-")
        if not w or "." in w.strip("."): parts_all = [p for p in re.split(r"[.]", w) if p]   # asi.elo = two words written together
        else: parts_all = [w]
        for tok in parts_all:
            parts = [p for p in tok.split("-") if _clean_latin(p)]
            if not parts: continue
            whole = _clean_latin("".join(parts)); self.bykey[key_latin(whole)][to_scheme(whole)] += n
            self.stems[key_latin(_clean_latin(parts[0]))][to_scheme(parts[0])] += n
            for j in range(1, len(parts)):
                suf = "".join(parts[j:]); self.sufs[key_latin(_clean_latin(suf), initial=False)][to_scheme(suf)] += n

    def word(self, w: str) -> tuple[str, str]:
        """-> (romanized word, how): how in lex | key | split | fallback"""
        if w in MANUAL: return MANUAL[w], "manual"
        k = key_arabic(w)
        if not k: return romanize(w, speak=True), "fallback"
        if k in self.bykey or w in self.direct:   # corpus frequency decides between homographs (to/tū, kan/kun); a dictionary spelling of this exact word gets a bonus
            c = collections.Counter(self.bykey.get(k, {}))
            for f, n in self.direct.get(w, {}).items(): c[f] += n
            return c.most_common(1)[0][0], "lex" if w in self.direct else "key"
        best = None
        for i in range(len(k) - 1, 0, -1):                      # longest stem first
            a, b = k[:i], k[i:]
            if a in self.stems and b in self.sufs and len(a) >= 2:
                s, sn = self.stems[a].most_common(1)[0]; f, fn = self.sufs[b].most_common(1)[0]
                if fn >= 3: best = s + f; break
        if best: return best, "split"
        return romanize(w, speak=True), "fallback"

    def judge(self, w: str) -> dict:
        """How far to trust word(w): {r, how, conf: low | medium | high, share, alt: other spellings the sources give for the same written word}."""
        rom, how = self.word(w); c = collections.Counter(self.bykey.get(key_arabic(w), {}))
        for f, n in self.direct.get(w, {}).items(): c[f] += n
        share = c[rom] / sum(c.values()) if c.get(rom) else 0.0; alt = [f for f, _ in c.most_common(6) if f != rom]
        if how in ("fallback", "split") or (how != "manual" and share < 0.6): conf = "low"
        elif how in ("manual", "key") or share < 0.85: conf = "medium"
        else: conf = "high"
        return {"r": rom, "how": how, "conf": conf, "share": round(share, 3), "alt": alt}

    def text(self, t: str) -> str:
        out = []
        for word in normalize(t).split():
            core = word.strip(_ROM_PUNCT + "\"'()…")
            if not core or not re.search(r"[؀-ۿ]", core): out.append(romanize(word, speak=True)); continue
            i = word.find(core); out.append(word[:i] + self.word(core)[0] + word[i + len(core):])
        return " ".join(out)


if __name__ == "__main__":
    v = Vowelizer()
    for line in sys.stdin: print(v.text(line.rstrip("\n")))
