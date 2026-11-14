"""brahui.text is applied to every reference and hypothesis before scoring, so its behaviour is pinned here.

    uv run pytest tests
"""
from brahui.text import arabic_script_ratio, for_scoring, normalize, romanize

SENT = "دا المی اف کہ راج اٹی کل اسہ بولی اسے ٹی ہیت وگپ کیر۔"


def test_arabic_variants_map_to_urdu_code_points():
    assert normalize("كتاب") == normalize("کتاب")      # Arabic kaf -> Urdu keheh
    assert normalize("علي") == normalize("علی")        # Arabic yeh -> Farsi yeh
    assert "ك" not in normalize("كتاب") and "ي" not in normalize("علي")


def test_invisible_characters_and_tatweel_are_dropped():
    assert normalize("کـتاب") == "کتاب"
    assert normalize("کت‌اب‍﻿") == "کتاب"


def test_diacritics_dropped_unless_asked():
    assert normalize("کِتاب") == "کتاب"
    assert "ِ" in normalize("کِتاب", keep_diacritics=True)


def test_digits_and_whitespace():
    assert normalize("۱۲۳ ٤٥") == "123 45"
    assert normalize("  دا \t  المی \n") == "دا المی"


def test_normalize_is_idempotent_and_keeps_the_brahui_lateral():
    once = normalize(SENT + " ڷ")
    assert normalize(once) == once
    assert "ڷ" in once


def test_for_scoring_strips_punctuation_and_tags():
    scored = for_scoring("دا المی اف۔ <noise> Hello، دنیا؟")
    assert not any(c in scored for c in "۔،؟<>")
    assert "hello" in scored and "noise" not in scored
    assert for_scoring(for_scoring(SENT)) == for_scoring(SENT)


def test_script_ratio():
    assert arabic_script_ratio(SENT) > 0.95
    assert arabic_script_ratio("hello world") == 0.0


def test_romanize_gives_latin_and_is_deterministic():
    r = romanize(for_scoring(SENT))
    assert r == romanize(for_scoring(SENT))
    assert arabic_script_ratio(r) == 0.0 and r.strip()


def test_romanize_passes_the_urdu_full_stop_through():
    # Known behaviour, pinned so a change is deliberate: the default mode does not convert "۔".
    assert romanize(SENT).endswith("۔")
