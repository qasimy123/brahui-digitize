"""Perso-Arabic Brahui text normalization.

Brahui is written with the Urdu letter set plus ڷ (U+06B7, /ɬ/). Sources mix
Arabic-presentation and Urdu-preferred code points for the same letter
(ي/ی, ك/ک, ه/ہ, ة/ہ ...), optional diacritics, tatweel, and zero-width joiners.
`normalize()` canonicalizes to the Urdu/Brahui code points so that CER/WER and
chrF compare orthography rather than encoding accidents.

Policy (documented in docs/PLAN.md §Phase 0.2):
  * NFC
  * drop tatweel, ZWNJ/ZWJ/ZWSP, BOM, directional marks
  * drop harakat (short-vowel diacritics are unwritten in running Brahui text;
    keeping them would penalize systems for omitting them inconsistently).
    Pass keep_diacritics=True to retain them (e.g. for TTS phonemization).
  * map Arabic/Persian variants to the Urdu forms used by Common Voice/Omnilingual
  * normalize Arabic-Indic / Eastern-Arabic digits to ASCII digits
  * unify punctuation (، ؛ ؟ ۔ and ASCII equivalents) into a small canonical set
  * collapse whitespace
"""
from __future__ import annotations

import re
import unicodedata

# Characters removed outright.
_STRIP = {
    "ـ",  # tatweel/kashida
    "​", "‌", "‍", "‎", "‏",  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    "‪", "‫", "‬", "‭", "‮",  # directional embeddings
    "⁦", "⁧", "⁨", "⁩",
    "﻿",  # BOM
    "­",  # soft hyphen
}

# Arabic harakat and related combining marks (U+064B..U+065F, U+0670, U+06D6..U+06ED).
_DIACRITIC_RE = re.compile(r"[ً-ٰٟۖ-ۭؐ-ؚ]")

# Letter canonicalization: Arabic/Persian presentation -> Urdu/Brahui code points.
_LETTER_MAP = {
    "ي": "ی",  # ي -> ی (Arabic yeh -> Farsi yeh)
    "ى": "ی",  # ى (alef maksura) -> ی
    "ے": "ے",  # ے stays (yeh barree)
    "ك": "ک",  # ك -> ک
    "ه": "ہ",  # ه -> ہ (heh -> heh goal)
    "ة": "ہ",  # ة -> ہ
    "ۀ": "ۂ",  # ۀ -> ہ + hamza (then hamza stripped below if diacritic policy)
    "آ": "آ",  # آ stays
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "ٱ": "ا",  # ٱ -> ا
    "ؤ": "ؤ",  # ؤ stays
    "ھ": "ھ",  # ھ (do chashmee he) stays - phonemic aspiration marker in Urdu loans
    "ۍ": "ی",  # ۍ -> ی (Pashto yeh variants that leak in)
    "ې": "ی",  # ې -> ی
    "ڭ": "ک",  # ڭ -> ک (rare)
    "گ": "گ",  # گ stays
    "ں": "ں",  # ں stays
    "ڷ": "ڷ",  # ڷ stays - the distinctive Brahui lateral fricative
}

_DIGIT_MAP = {}
for i in range(10):
    _DIGIT_MAP[chr(0x0660 + i)] = str(i)  # Arabic-Indic
    _DIGIT_MAP[chr(0x06F0 + i)] = str(i)  # Eastern Arabic-Indic (Urdu/Persian)

# Punctuation: map to a small canonical set. Everything is later optionally stripped for WER.
_PUNCT_MAP = {
    "،": "،", ",": "،",
    "؛": "؛", ";": "؛",
    "؟": "؟", "?": "؟",
    "۔": "۔", ".": "۔", "۝": "۔",
    "!": "!", "ە": "!",  # rarely used
    "‘": "'", "’": "'", "“": '"', "”": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "…": "...",
}

_PUNCT_RE = re.compile(r"[،؛؟۔!\"'()\[\]{}«»\-:/\\.,;?¡¿…_*#@%^&+=<>|~`]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str, *, keep_diacritics: bool = False) -> str:
    """Canonicalize Brahui Perso-Arabic text (see module docstring)."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = "".join(ch for ch in t if ch not in _STRIP)
    t = "".join(_LETTER_MAP.get(ch, ch) for ch in t)
    t = "".join(_DIGIT_MAP.get(ch, ch) for ch in t)
    t = "".join(_PUNCT_MAP.get(ch, ch) for ch in t)
    if not keep_diacritics:
        t = _DIACRITIC_RE.sub("", t)
    t = unicodedata.normalize("NFC", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


_TAG_RE = re.compile(r"<[^<>]{1,30}>")  # transcriber annotations: <hesitation>, <noise>, <laugh> ...


def for_scoring(text: str) -> str:
    """Normalization used for CER/WER and CTC targets: normalize(), drop <annotation> tags, strip punctuation, lowercase Latin."""
    t = normalize(text)
    t = _TAG_RE.sub(" ", t)
    t = _PUNCT_RE.sub(" ", t)
    t = t.lower()
    return _WS_RE.sub(" ", t).strip()


_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def arabic_script_ratio(text: str) -> float:
    """Fraction of non-space characters in Arabic-script blocks. Used as a script-fidelity check."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _ARABIC_RE.match(c)) / len(chars)


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        print(normalize(line.rstrip("\n")))


# ---- Rough Latin rendering of Perso-Arabic Brahui, for people who cannot read the script. ----
# Not a reversible transliteration: short vowels are unwritten in the script, so this is a consonant
# skeleton with long vowels. Good enough to sound a word out. Scheme follows common Urdu/Balochi romanization.
_ROM_CONS = {
    "ب": "b", "پ": "p", "ت": "t", "ٹ": "ṭ", "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ڈ": "ḍ",
    "ذ": "z", "ر": "r", "ڑ": "ṛ", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z",
    "ع": "'", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "ك": "k", "گ": "g", "ل": "l", "ڷ": "ḷ", "م": "m", "ن": "n",
    "ں": "ṉ", "ھ": "h", "ہ": "h", "ۀ": "h", "ة": "h", "ء": "'", "ئ": "'", "ؤ": "'", "ټ": "ṭ", "ډ": "ḍ", "ړ": "ṛ", "ګ": "g",
}
_ROM_DIAC = {"َ": "a", "ِ": "i", "ُ": "u", "ّ": "", "ْ": ""}


def romanize(text: str) -> str:
    vowels = set("aāeiīouū'")
    out = []
    for word in text.split():
        chars = list(word); units = []          # (string, is_vowel)
        for i, c in enumerate(chars):
            first = i == 0; last = i == len(chars) - 1
            prev = chars[i - 1] if i else ""
            if c == "آ":
                units.append(("ā", True))
            elif c == "ا":
                units.append(("a" if first else "ā", True))
            elif c == "و":
                units.append(("w", False) if first or prev in "اآ" else ("o", True))
            elif c in "یي":
                units.append(("y", False) if first or prev in "اآ" else ("ī", True))
            elif c == "ے":
                units.append(("e", True))
            elif c == "ہ" and last and not first:
                units.append(("a", True))        # final he usually marks a vowel
            elif c in _ROM_DIAC:
                if _ROM_DIAC[c]:
                    units.append((_ROM_DIAC[c], True))
            elif c in _ROM_CONS:
                r = _ROM_CONS[c]
                if r in ("h",) and units and not units[-1][1] and units[-1][0] in ("b", "p", "t", "ṭ", "j", "ch", "d", "ḍ", "k", "g", "ṛ"):
                    units[-1] = (units[-1][0] + "h", False)   # aspirates: bh, ph, th, dh, kh...
                elif r == "g" and units and units[-1] == ("n", False):
                    units[-1] = ("ng", False)
                elif units and units[-1] == (r, False):
                    units[-1] = (r + r, False)                # geminate, no vowel inside
                else:
                    units.append((r, False))
            else:
                units.append((c, True))
        # short vowels are unwritten: put "a" between consonants where a reader would otherwise choke
        s = ""; seen_vowel = False
        for i, (u, v) in enumerate(units):
            s += u; seen_vowel |= v
            nxt = units[i + 1] if i + 1 < len(units) else None
            nn = units[i + 2] if i + 2 < len(units) else None
            if not v and nxt and not nxt[1] and (nn is None and not seen_vowel or nn is not None and not nn[1]):
                s += "a"; seen_vowel = True
        out.append(s)
    return " ".join(out)
