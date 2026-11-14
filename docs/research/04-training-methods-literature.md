# Literature review: training methods for an unseen, extremely low-resource language (Brahui), 2026-10-04

## 0. Headline
arXiv contains zero Brahui papers, but the resource picture changed in 2025–26: Omnilingual ASR supports `brh_Arab` (13.0 h, CER 8.2); Omnilingual ASR Corpus has ~11 h CC-BY Brahui; Common Voice 27.0 has 9.9 h / 18 speakers; BrahuiSpeech-70H-V2 has 69.5 h; community Whisper-small baseline WER 55.7 / CER 22.8; Brahui G2P PER ~10%.
Balochi lesson: reached MMS via a Bible reading, Omnilingual via commissioned corpus, now has community HF ecosystem (Whisper LoRA, NLLB LoRA, SpeechT5 TTS from 75 clips, OCR) but no peer-reviewed ASR/MT paper.

## 1. Zero/few-shot ASR for unseen languages
- Omnilingual ASR buckets: mean CER 3.1 (>50 h), 3.0 (10–50 h), **18.0 (<10 h)**. ZS model: 26.3 → 17.6 (5 ctx) → 14.4 (10 ctx); SONAR-retrieved context beats random up to 11.2% rel.; frozen encoder worse. https://arxiv.org/abs/2511.09690
- MMS (https://arxiv.org/abs/2305.13516): ~32 h/language Bible readings, 2M-param adapters. MMS Zero-shot (https://arxiv.org/abs/2407.17852): uroman CTC + lexicon + n-gram LM, avg CER 25.2% on 100 unseen languages with ~5k text utterances. ASR2K (https://arxiv.org/abs/2209.02842): ~45% CER with 10k text lines and no audio.
- XEUS (EMNLP 2024). Universal phone: Allosaurus, Wav2Vec2Phoneme (22% PER unseen), Allophant (PHOIBLE-conditioned), ZIPA (ACL 2025; unseen PFER 3.2), POWSM, PhoneticXEUS.
- Whisper new languages: *Learn and Don't Forget* (~12 h; Sorani 110.8 → 32.9 full FT / 41.3 LoRA / 61.4 soft prompt; reused language-token choice swings CER up to 33 pts). LAMA-UT (AAAI 2025): romanized universal transcription + back-transliteration. Script collapse (https://arxiv.org/abs/2604.08786): Urdu script fidelity 6.5% → 97% with script-aware prompting. Pashto: zero-shot Whisper emits Pashto script <0.8%.
- Perso-Arabic normalization: Nisaba (https://arxiv.org/abs/2210.12273), ScriptNormalization (ACL 2023, https://arxiv.org/abs/2305.16407). Issues: ZWNJ, kashida, diacritics, yeh/kaf/heh variants; Brahui adds ڷ and unwritten-vowel instability.
- SSL continued pretraining: 10 h target + 60 h acoustically similar donor ≈ 70 h in-language CPT (https://arxiv.org/abs/2402.02302); CPT on Urdu/Persian/Arabic beats Whisper-large-v3 on Persian (https://arxiv.org/abs/2512.07277); Scottish Gaelic: hybrid HMM + CPT XLS-R 12.8% WER vs Whisper LoRA 19–22% (https://arxiv.org/abs/2506.04915).
- Speech-LLM adapters: SLAM-ASR needs 100–200 h to beat Whisper (https://arxiv.org/abs/2508.05149).

## 2. Cross-lingual transfer selection
- **DonorRank** (Aug 2026, https://arxiv.org/abs/2608.11441): geographic proximity, lexical overlap, donor hours dominate; **genetic similarity least important (1.4–1.9%)**, genetic-only selection sometimes ~100% WER.
- Warlpiri (Interspeech 2026, https://arxiv.org/html/2607.10256): acoustic-embedding similarity ρ=−0.67 with WER; phoneme inventory matters only zero-shot; best source Assamese WER 32.6 vs monolingual 86.9.
- Kalenjin/Luganda (https://arxiv.org/html/2607.04814v2): once ≥1 h target data exists, related vs unrelated pre-adaptation converge within ±5 WER. Garhwali (VarDial 2026): small in-variety data beats large Hindi. CATDS: clip-level acoustic selection from donors only method beating random (https://arxiv.org/abs/2506.22194).
- Dravidian pivot: **Malasar** (https://arxiv.org/html/2411.04573; 4.6 h; Whisper-medium 99.3 → 56.4 direct → **51.9 via Tamil-intermediate FT**). For MT, orthographic resemblance beats phonological. **No paper decouples acoustic transfer (Balochi/Urdu) from morphosyntactic transfer (Dravidian): a genuine gap.**

## 3. Speech translation
- IWSLT 2023–2025: for sources unseen by foundation models, **cascades (FT Whisper/MMS + FT NLLB) won every year**; SeamlessM4T-v2 zero-shot 0–0.9 BLEU on unseen sources, 6.6 after FT vs 32.6 cascade. KIT 2024: full FT 27.7 BLEU vs LoRA 14.1 on Bemba. ~50 h translated speech threshold for good E2E ST. Quechua winner used 50 h TTS-synthesized source speech (2024) while KIT 2025 found under-trained TTS useless; MT-translating ASR transcripts into pseudo-ST worked. MBR cascade+E2E +1–1.6 BLEU.
- Textless: Lee et al. 2022, UnitY, SpeechMatrix, UTUT, Translatotron 3 (needs monolingual speech–text both sides), PFB (unit pretraining + 20–60 h parallel + BT), mHuBERT-147 (FLEURS CER seen 20.8 vs unseen 22.6: units transfer).

## 4. TTS for unseen languages
- MMS-TTS: one VITS per language from a single NT reading, letter input, 100k steps; 85% reach CER ≤5. finetune-hf-vits: 80–150 samples, ~20 min.
- ZMM-TTS (TASLP 2024): XPhoneBERT phoneme input; unseen Italian **15 min → CER 3.9, MOS 3.33**. IMS-Toucan 7,000-language meta-learning (articulatory-feature input, 5–30 min adaptation). G2P: CharsiuG2P/ByT5, XPhoneBERT, UR-BERT (Interspeech 2026), Align2Speak IPA + GRPO.
- LLM-codec TTS: XTTS needed 52–62 h per added language; F5-TTS community Greek 40 h + 20 h English; CosyVoice 300 h German CPT produced meaningless audio and broke other languages. Classics: LRSpeech (minutes paired + hours unpaired, 98% intelligibility), Arabic 1 h → MOS 4.4.
- Cycle consistency: speech chain; **TokenChain** (ICASSP 2026, https://arxiv.org/abs/2510.06201; discrete-token ASR↔TTS chain, −56% WER rel. in transfer); Gronings 24 min TTS aug −25.5% rel.; Frustratingly Easy Data Augmentation (LLM text + IPA-TTS: −1.4 to −14.3 WER abs.).
- Evaluation without listeners: ASR-CER + UTMOS; *Beyond Naturalness* (2026): MOS predictors and LLM judges miss linguistic errors.

## 5. MT for unseen languages
- MTOB (ICLR 2024): Kalamang chrF 15.6 (none) → 29.7 (wordlist) → 35.1 (400 sentences) → 44.7 (+grammar); human 51.6/57.0. Gemini 1.5 eng→kgv 57.0 full-book.
- Aycock et al. (ICLR 2025): parallel-only ≈ full book; NLLB-1.3B FT on 400 sentences 34.2 vs Gemini 26.6; +400 sentences = +4–8 chrF. DiPMT++ (dict + 5k sentences, GPT-4 0 → 32 BLEU). Back to School: 12/16 languages peak with words+sentences only. Manchu (ACL 2025): synthesize 42k pairs via ICL, mT5-small matches Llama-3-70B. Factorial synthetic-data-from-grammar (2026, Kalamang 12.7 → 21.5 chrF++). RL for contextual learning (unseen chrF 0.27 RL vs 0.09 SFT).
- Seed data: SMOL (863 set-cover sentences, +7.6 chrF; LLM drafts needed ~70% post-editing). NLLB-Seed: 6.2k professional sentences beat 10–31k noisy bitext.
- Adding a language to an LLM: Tejaswi et al. (EMNLP 2024): with ~100M tokens, >10k new vocab items give nothing. Goldfish: 125M monolingual models at 5–100 MB beat massively multilingual. Trans-tokenization from a related language (COLM 2024, https://arxiv.org/abs/2408.04303).
- Tiny-scale pivots: Arabic-script biomedical MT with 500 sentences via Arabic/Persian pivots + LoRA merging, Dari 41.0 chrF++ (https://arxiv.org/abs/2607.22300). AmericasNLP 2025: NLLB fine-tunes on 357–15k sentences "hard to beat".
- Evaluation: AfriCOMET shows learned metrics need in-language ratings; chrF stable under shift; BLASER-2 inapplicable; LLM-judge agreement drops in low-resource scripts.

## 6. Joint speech-text models
GSLM, SpiRit-LM, Moshi/Mimi, AudioPaLM, VoxtLM, GLM-4-Voice (trained on text→speech-token synthesized interleaved data, portable recipe), Qwen3-Omni. **No published evidence of cheap new-language addition via speech-token LoRA.** Sylber syllabic units generalize to unseen languages (ICLR 2025).

## 7. Active learning / data value
- **Speakers > hours**: at fixed 5 h, 15 speakers beat 5 by 1.6–20 CER (https://arxiv.org/html/2506.04364).
- Greedy triphone/articulatory set-cover for prompt selection (+55.8% distinct triphones). CV rule of thumb: ~5,000 CC0 sentences ≈ 5.5 h. Lexically rich narratives beat clean audio. *Hard to Be Heard* (ACL 2026): phoneme accuracy is a sigmoid in training frequency → over-sample /ɬ/, retroflexes, vowel length.
- Epistemic-uncertainty selection −27% WER with 45% less labelling. Expert transcription 30–50 h per audio hour; novice-corrects-ASR cuts WER 38.9%; Fotheidil recycles corrections.

## Implications for a novel Brahui recipe
1. Start from the ~90 h that exist; seal CV test + Omnilingual test splits.
2. Omnilingual ASR as acoustic backbone; fine-tune the encoder; use ZS + SONAR-retrieved context as labelling bootstrap.
3. Choose donors by acoustics and loan-lexical overlap (Balochi, Urdu, Sindhi, Saraiki), not Dravidian genealogy; reserve Dravidian transfer for the text/decoder side.
4. Novel angle: decoupled acoustic vs morphosyntactic transfer.
5. Fix the orthography layer first; new language-code embedding; monitor script fidelity; keep a parallel broad-IPA target.
6. Cascade for ST; full FT not LoRA for NLLB; pivot through Urdu.
7. Buy sentences, not grammar prose; SMOL-style set cover; LLM-draft from Urdu + native post-edit; ICL-synthesize 10–40k pairs.
8. Vocab extension ≤10k; init embeddings from Urdu/Balochi; a small monolingual Brahui LM for shallow fusion.
9. TTS: VITS from urd/bcc with 80–150 utterances, phonemic input; gate TTS-for-ASR augmentation on CER; TokenChain-style loop on unpaired audio.
10. Many speakers, short sessions, rare-phoneme quotas.
11. Human-correction loop with uncertainty + diversity selection.
12. chrF++/CER plus 300–500-sentence native-rated set; distrust learned metrics.
