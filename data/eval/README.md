# Sealed evaluation sets

Never train, tune prompts, or select models on these. Hill-climbing reads only the *train* splits under data/manifests/.

| file | what |
|---|---|
| cv_test.jsonl | Common Voice 27.0 Brahui official test split (CC0) |
| omni_test.jsonl | Omnilingual ASR corpus brh_Arab test split (CC-BY-4.0) |
| smol_heldout.jsonl | 200 SmolSent en->brh pairs, seeded random hold-out (CC-BY-4.0) |
| bouquet_test.jsonl | BOUQuET brh_Arab->eng_Latn sentence-level test (CC-BY-4.0) |

Verify with `shasum -a 256 -c SHA256SUMS` from this directory.
