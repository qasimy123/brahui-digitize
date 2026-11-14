# Metrics

- **chrfpp** (primary, higher is better): sentence-level chrF++ (sacrebleu CHRF word_order=2) of the hypothesis against the single reference. For en->brh both sides pass through `brahui.text.normalize` (NFC, letter canonicalization, diacritics stripped) so encoding differences don't count as errors. For brh->en, raw English. Corpus chrF++ per direction/split is in each variant's summary.json and is the headline.
- **bleu** (secondary): sentence BLEU with effective order; corpus BLEU in summary.json.
- Directions are scored separately: `en2brh` (SMOL, test = 200 sealed hold-out) and `brh2en` (BOUQuET, test = 854 sealed). Report both; the combined headline is the mean of the two test corpus chrF++ values.
- Few-shot examples come only from train pools (SMOL train + SmolDoc + BOUQuET dev). Test references are never in the model context.
