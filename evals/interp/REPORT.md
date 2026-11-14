# What fine-tuning did inside NLLB-1.3B (base → r5_retrans), 2026-11-06

Data: 504 tri-parallel BOUQuET **dev** sentences (Brahui / Urdu / English; `bouquet_dev_triparallel.jsonl`). Brahui and
Urdu both enter under `urd_Arab`, as in training. Scripts: `scripts/interp/{weight_diff,reps,tag_behavior}.py`.

## Findings

1. **Brahui was merged into Urdu, not written over it.** From encoder layer ~4 up, the gap between the Brahui and
   Urdu sentence centroids falls from ~0.7 (base) to ~0.15 (ft), measured in units of within-language spread
   (`reps.png`, top right). A linear probe can still tell them apart (91–99%), but less easily in the upper layers.
2. **Brahui comprehension is built in encoder layers 4–10.** Parallel brh→en retrieval rises from 0.36 to 0.96
   across those layers (base peaks at 0.67) and stays ≥0.97 from layer 11 on. Representation drift (CKA) is largest
   for Brahui (0.34 at layer 21), moderate for Urdu (0.71) and smallest for English (0.84).
3. **Urdu got better, not worse.** urd→en chrF++ rose 58.4 → 63.6 (n=120, `tag_behavior.json`): positive transfer.
4. **The `urd_Arab` tag now means "write Brahui".** Scored against the Urdu reference, en→`urd_Arab` output falls
   56.9 → 17.6; scored against the Brahui reference, it rises 18.0 → 40.9. brh→en: 21.3 → 52.1.
5. **The weight changes are spread out but fairly low-rank.** Every layer changed by 5.5–7.8% (relative norm),
   rising toward the top of the decoder (`weight_diff.png`). The top 8 directions hold 13% of each change matrix's
   energy (median; random noise would be 0.8%), so a LoRA should capture much of it.
6. **Unseen vocabulary was damaged.** About 230k of 256k tokens never appear in the training data. Their embedding
   rows shrank to 65% of their original length while keeping their direction (cos 0.93). The ~24.5k tokens that do
   appear moved by ~7%. Weight decay at 0.01 looks too small to cause this. The likely cause is Adam amplifying the
   tiny softmax gradients that flow through the tied output layer. It is harmless for the vocab-trimmed serving
   model. For future runs, consider freezing the rows of tokens the data never uses.

## Next

- **Steering test (Mac, ~1 h):** push r5 activations along the Brahui-vs-Urdu direction (layers 4–12), both ways,
  and score dev in both directions. This decides whether r6 is worth GPU money.
- **r6 (`plans/r6.txt`, `train_nllb.py --brh-tag brh_Arab`):** gives Brahui its own tag. Expect a tie to +1 chrF++,
  most likely on en→brh. If r6 wins, the demo, the export, `trim_nllb_vocab.py` and the eval scripts still
  hard-code `urd_Arab` and need a tag setting.
