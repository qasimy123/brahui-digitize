# Brahui Speech & Translation: Research Synthesis and Training Plan

*Compiled 2026-10-04. Supporting research with sources: `docs/research/01–04`.*

## 1. Where Brahui actually stands (this changes the whole plan)

Brahui is **not** a zero-resource language anymore. In the last 12 months it got picked up by three big multilingual efforts plus one community contributor. Nobody has combined them yet.

| Asset | Size | Status |
|---|---|---|
| Omnilingual ASR (Meta, Nov 2025) | `brh_Arab` trained, CER 8.2 (7B) | Acoustic backbone (CTC-1B v2, HF format) |
| Omnilingual ASR Corpus | ~11 h spontaneous, speaker IDs | Train/dev/test splits defined; chunked to ≤20 s |
| Common Voice 27.0 | 9.9 h read, 18 speakers, 3,095 sentences | Live; needs sentences + validators |
| BrahuiSpeech-70H-V2 (community) | 69.5 h media speech, transcribed | In the training pool |
| OmniVoice TTS (k2-fsa, Apr 2026) | `brh` trained on 19.9 h | Only TTS with a Brahui ID |
| Google SMOL + GATITOS | 863 sentences + 66 docs + 3,935 lexicon pairs en↔brh | Seed MT data |
| Meta BOUQuET | ~1,350 multi-way parallel sentences | MT eval |
| BECO text corpora (MDC) | ~0.7M tokens (Talár, literature, research) | LM / back-translation (terms to accept on MDC) |
| FineWeb-2 + FinePDFs | ~10k docs, ~38 MB | LM pretraining (mostly Urdu; GlotLID filter) |
| Brahui G2P (community) | Perso-Arabic → IPA, PER ~10% | TTS front end |
| UD_Brahui-Kholum | 64 sentences | Morphology sanity check |
| Bible NT (1998) + Quran translation | PDF scans | OCR → verse-aligned parallel |
| Brahvi-Whisper-Small baseline | WER 55.7 / CER 22.8 | Public yardstick to beat (18.9 CER on our CV test) |

Missing: Brahui↔English or Brahui↔Urdu parallel corpus beyond SMOL; Brahui TTS; Brahui in any MT engine (NLLB, Google, Microsoft); any peer-reviewed Brahui speech/NLP paper. **You can publish the first one.**

## 2. What the 2025–26 literature says works (and doesn't)

1. **Cascades beat end-to-end** for unseen source languages at every IWSLT 2023–2026. SeamlessM4T zero-shot is ~0 BLEU on unseen sources; frontier LLM prompting lost to fine-tuned NLLB (10.8 vs 19.5 BLEU in 2026). E2E speech translation needs ≥50 h of translated speech.
2. **Full fine-tuning beats LoRA** for adding languages to Whisper/NLLB/Seamless (Bemba 27.7 vs 14.1 BLEU; Pashto Whisper LoRA 33 pp worse). Whisper is the wrong backbone for Perso-Arabic minority languages anyway (emits the right script <1% of the time zero-shot).
3. **Donor-language choice is about acoustics and shared vocabulary, not genealogy.** DonorRank (2026): genetic relatedness explains ~1.5% of transfer; geography, loan-lexical overlap and donor hours dominate. Brahui's ~55% Balochi/Persian/Urdu loan vocabulary and shared script make Urdu/Balochi/Sindhi the right acoustic donors. The only positive Dravidian pivot result (Malasar via Tamil, −4.5 WER) was on the decoder side.
4. **Parallel sentences beat grammar books.** MTOB follow-ups (ICLR 2025, Kanuri 2025): grammar prose adds ~0 chrF; 400 sentences give 30–35 chrF; NLLB-1.3B fine-tuned on those same sentences beats Gemini in-context. 6.2k professional sentences beat 30k noisy pairs.
5. **~10 h is the TTS knee** for LLM/diffusion TTS (F5: 1 h → MUSHRA 34, 10 h → 62). Phoneme-input VITS-class models work from 15 min to 1 h. Phonemized input beat romanized by 10–27% MOS on Urdu. Brahui's unwritten short vowels make G2P the central TTS problem.
6. **Speakers > hours.** 15 speakers × 20 min beats 5 × 1 h by up to 20 CER. Phoneme accuracy is a sigmoid in frequency, so rare segments (/ɬ/ ڷ, retroflexes, vowel length) must be over-sampled deliberately.
7. **Learned metrics don't exist for Brahui.** BLASER-2, COMET, MetricX, UTMOS all lack it. Use chrF++, CER, script-fidelity, OmniASR-WER-of-TTS, plus a 300–500 sentence native-rated set.

## 3. The novel approach: Decoupled Acoustic/Linguistic Transfer with a Phonemic Pivot (DALT-P)

Three ideas, each individually motivated by the literature, that nobody has combined for a language in Brahui's exact position (Iranian-area phonology and script, Dravidian grammar, ~90 h audio, <1M words):

### 3.1 Decouple the two transfer axes
Every existing recipe transfers from one "source language". Brahui needs two:
- **Acoustic/script axis ← Urdu, Balochi, Sindhi, Saraiki.** The OmniASR encoder (already trained on all of these plus Brahui) handles sounds and letters. Fine-tune the encoder + CTC head on pooled Brahui audio with donor clips selected by acoustic similarity (CATDS-style), not by family.
- **Morphosyntactic axis ← Tamil/Kannada/Malayalam (agglutinative Dravidian) + Brahui text.** Build the language model / MT decoder side with trans-tokenization from Urdu embeddings (shared lexicon) and then continued pretraining on Brahui text mixed with a small Dravidian agglutination curriculum (synthetic morphological paradigms generated from Bray's and Barjasteh Delforooz's case/verb tables).
- Fuse them: CTC acoustic model + external Brahui LM via shallow fusion (Goldfish-style 125M monolingual LM beats multilingual LLMs at this data scale), then distil into the OmniASR LLM-decoder variant once pseudo-labels are clean.

This directly tests DonorRank's claim that acoustic and syntactic transfer have different optimal donors. It is publishable on its own.

### 3.2 Phonemic pivot as the canonical internal representation
Perso-Arabic Brahui drops short vowels; the Latin Brolikva orthography writes them but almost nobody uses it. Instead of picking one script:
- Train a **Brahui diacritizer/phonemizer** (Phonikud-style, seeded from the existing G2P model + Bray's fully-vowelled dictionary entries + Brolikva Wikipedia/Talár Roman text which is effectively vowelled).
- ASR emits **graphemes and broad IPA jointly** (two CTC heads, shared encoder). The IPA head gets ensembled with ZIPA universal phone recognition, which gives you a Brahui ASR signal that is independent of orthographic inconsistency.
- MT operates on normalized Perso-Arabic (Nisaba/ScriptNormalization rules + ڷ canonicalization).
- TTS consumes IPA, never raw script. This is what let ZMM-TTS/IMS-Toucan reach CER ~4 from 15 min.
- The IPA layer also makes Brolikva ⇄ Perso-Arabic transliteration free, and lets you accept either script from users.

### 3.3 Closed data loop: TokenChain-style ASR⇄TTS + MT back-translation
You have ~90 h of transcribed audio and ~0.7M tokens of text with no pairing. The 2025–26 results that actually moved low-resource numbers were cycle-consistency methods:
- **ASR⇄TTS chain** on discrete tokens (TokenChain, ICASSP 2026, −56% WER in transfer): TTS synthesizes the BECO/FineWeb text in many voices → ASR trains on it, gated by ASR-CER of the synthetic audio (KIT 2025 showed ungated synthetic speech hurts). ASR pseudo-labels the 70 h → TTS trains on cleaned pairs.
- **MT back-translation** over the monolingual Brahui text, with a frontier LLM as the first teacher (it translates Brahui→English at chrF++ 57 with no Brahui training, from regional-language knowledge and context) and NLLB as the student. Proven 2026-10-19: 70k synthetic pairs took NLLB-600M en→brh from 20.7 to 38.3 chrF++, above the teacher. The loop continues without humans: better teacher passes, iterated back-translation from human English, bigger students. Design and levers: `docs/SYNTHETIC_BOOTSTRAP.md`.
- *(Optional, not on the critical path; see `docs/PLAN_95.md`)* Human-in-the-loop correction rather than transcription: uncertainty + speaker/topic diversity selection picks ~45% fewer clips for the same WER gain; corrections are recycled into the next round (Fotheidil).

### 3.4 Resulting system
```
English text/speech ──► (Whisper/any EN ASR) ──► NLLB-3.3B+brh ──► IPA phonemizer ──► OmniVoice-LoRA / VITS ──► Brahui speech
Brahui speech ──► OmniASR encoder (FT) ──► {grapheme CTC, IPA CTC} + Brahui LM fusion ──► NLLB brh→eng ──► English
```
Both directions share the encoder, the phonemizer and the NLLB model. Everything runs on one A100-80GB.

## 4. Phased plan

### Phase 0 — Foundation (weeks 1–3, ~$50 GPU)
1. Download: Omnilingual corpus `brh_Arab`, CV 27.0 Brahui (MDC account), SMOL/GATITOS en_brh, BOUQuET, BECO corpora, FineWeb-2/FinePDFs brh_Arab, UD treebank, G2P model.
2. Build `brahui-text`: Perso-Arabic normalizer (NFC, kashida/ZWNJ/diacritic policy, yeh/kaf/heh canonicalization, ڷ U+06B7), Brolikva⇄Arabic transliterator, GlotLID filter.
3. Seal evaluation sets: CV test (882 clips), Omnilingual test (106 segments), BOUQuET brh, SMOL held-out 200 sentences. Never train on them.
4. Baselines: run `omniASR_LLM_7B` and `omniASR_CTC_1B_v2` on all Brahui test sets (CER/WER/script fidelity); run NLLB via Urdu token and GPT/Gemini/Claude with GATITOS shots on SMOL held-out (chrF++); run OmniVoice brh zero-shot and measure OmniASR-WER + MMS-LID.

### Phase 1 — ASR that works (weeks 3–6, ~$100)
1. Fine-tune `omniASR_CTC_1B_v2` (HF conversion) on Common Voice + Omnilingual chunks + BrahuiSpeech-70H (~90 h), lr 1e-5, 5k steps, one A100 80 GB (`scripts/gpu/train_asr.sh`). Target: CER <8 on CV test, <10 on Omnilingual test (paper's 10–50 h bucket mean is 3.0).
2. Add the second IPA CTC head using G2P-converted transcripts; ensemble with ZIPA.
3. Train a Goldfish-style 125M Brahui LM on BECO + FineWeb + FinePDFs (~1M tokens is tiny; use byte/char-level, 1–2 h on one GPU). Shallow-fuse; measure gain.
4. Pseudo-label everything unlabeled (70 h, PTV Bolan/YouTube pulls after Demucs + MossFormer2 + pyannote + MMS-LID filter). Optional (not required by `docs/PLAN_95.md`): uncertainty-select 2–3 h for human correction (Label Studio with the CTC model as pre-annotator).

### Phase 2 — MT (weeks 5–9, ~$150)
1. OCR the Bible NT and Quran translation PDFs with Gemini-2.5 or Qaari fine-tuned on ~300 hand-corrected Brahui lines; verse-align to English/Urdu/Arabic → ~8–15k parallel verses (religious domain; use for pretraining, not eval).
2. Full fine-tune NLLB-200 1.3B (then 3.3B) with new `brh_Arab` token from `urd_Arab` on SMOL + verses + dictionary entries. Evaluate chrF++ on SMOL held-out and BOUQuET.
3. Iterative back-translation: LLM-teacher translation of all Brahui text (done: 70k pairs), then rounds of student back-translation over human English text (conversational corpora first), consensus/context teacher passes, and distillation into NLLB 1.3B/3.3B or a small LLM. This is the main MT engine; see `docs/SYNTHETIC_BOOTSTRAP.md`.
4. Optional (not required; `docs/PLAN_95.md` reaches the MT target without it). SMOL-style set-cover: select 2,000 English sentences maximizing token coverage of Brahui web text; a native speaker *speaks* each one in Brahui (recording tool in this repo). One session yields en→brh speech-translation references, the TTS voice (Phase 3.2), and, via the fine-tuned ASR, text pairs. Valuable when a speaker is available, chiefly for the voice and for verification; the data loop above does not depend on it.
5. Optional: QLoRA Gemma 3 12B / Qwen3-8B on the same data for an instruction-following Brahui assistant; compare against NLLB on chrF++ (expect NLLB to win on translation).

### Phase 3 — TTS (weeks 8–12, ~$150)
1. Build the diacritizer/phonemizer: train on Bray's vowelled dictionary + Brolikva text + G2P outputs; evaluate PER on 500 hand-checked words.
2. The Phase 2.3 recordings are the single-speaker set: 3–5 h, quiet room, 16 kHz+, phonetically balanced by construction of the sentence list (quotas for ڷ, retroflexes, long/short vowel minimal pairs). This speaker is the product voice.
3. Two TTS tracks in parallel: (a) VITS from `mms-tts-urd-script_arabic` with IPA input and extended vocab (permissive-ish, robust, 36M params); (b) OmniVoice LoRA on `brh` with the same data (higher quality, CC-BY-NC). Pick by OmniASR-WER + 20-listener MOS panel.
4. Chatterbox-Multilingual LoRA as a third option.

### Phase 4 — Closed loop and release (weeks 12–20)
1. TokenChain-style ASR⇄TTS cycle: synthesize BECO text in N voices → ASR augmentation gated on CER; ASR pseudo-labels → TTS data. Re-run Phases 1 and 3 models.
2. Speech translation cascade evaluation on a FLEURS-Brahui set you create (see §5.2).
3. Release: ASR, MT, TTS checkpoints on HF; FLORES+ Brahui via OLDI; sentences + prompts to Common Voice; a paper (first Brahui speech/MT paper; the decoupled-transfer experiment is the scientific contribution).

## 5. Data gathering plan

### 5.1 Speech (target +40 h read, +40 h spontaneous, 150+ speakers in year 1)
- **Common Voice first.** Write/translate 5,000 CC0 sentences (≤15 words, phoneme-targeted; ~5.5 h of audio). Recruit validators explicitly (validation lags recording by ~40%). File the Spontaneous Speech request with 60 Brahui prompts.
- **Broadcast partnership** is what moved Pashto from 1.5 h to 147 h: pitch PTV Bolan / "Brahui Time" / Radio Chiltan FM 88 for an on-air recording appeal pointing to Common Voice.
- **Student drives** at University of Balochistan Brahui Dept and Brahui Academy Quetta: LIG-Aikuma on Android for offline image-prompted monologues (Vaani recipe: ≤15 min per speaker, 16 kHz mono, local images) and respeaking of spontaneous clips. Record verbal consent in Brahui at session start.
- **Diaspora** (Karachi, UAE): WhatsApp-group recruitment into Common Voice; short sessions, many speakers.
- **Mining** PTV Bolan live stream / YouTube talk channels (Khudi Talks, Brahui TV): yt-dlp → Demucs → MossFormer2 → pyannote → MMS-LID (brh vs bcc/bgp/urd) → pseudo-label.
- Pay: no published Pakistani rates; plan PKR 1,000–2,000 per validated hour-equivalent, capped per speaker.

### 5.2 Benchmarks you should create (cheap, high leverage)
- **FLORES+ Brahui via OLDI**: human-translate 2,009 sentences (no MT post-editing allowed), CC-BY-SA, submit as HF PR. Then read them aloud → FLEURS-Brahui. This puts Brahui into the standard 200-language eval grid and into the next NLLB/Seamless training runs.
- 300–500 sentence native-rated MT adequacy set; 20-listener TTS MOS panel.

### 5.3 Text (target 5M tokens)
- OCR Brahui Academy PDFs (with permission), Bible NT, Quran translation, Bray 1909 (PD). Qaari-0.1 or Gemini-2.5 with a 300-line Brahui fine-tune set.
- Crawl Talár (Nastaliq + Roman), Balochistan Post, Facebook groups; GlotLID filter; dedupe.
- Wikipedia Incubator Wp/brh: 553 Brolikva articles; expanding it feeds both text and visibility.
- Dictionaries → lexicon for GATITOS-style MT prompting and for the phonemizer.

### 5.4 Funding
ELDP 2027 round (with UoB as partner; up to €300k MDP), Wikimedia Rapid Fund ($500–5k, individuals, rolling), Pakistan HEC/Ignite via a university partner. Lacuna Fund and Mozilla programs are Africa-focused in 2026.

## 6. Budget and compute

| Item | Estimate |
|---|---|
| GPU (one A100-80GB, ~$1.2–1.6/h RunPod/Lambda; 300–400 h over 5 months) | $400–650 |
| Native-speaker recording (2,000 sentences, doubles as the TTS voice) + FLORES+ translation (2,009 sentences) | in-house speaker; FLORES+ depends on rates |
| OCR API (Gemini) | <$50 |
| Common Voice / LIG-Aikuma / Label Studio | free |

Every fine-tune in the plan fits one A100-80GB; the whole first iteration of all models is roughly $10–60 of compute. Apple Silicon handles inference (mlx-audio runs OmniVoice and Whisper).

## 7. Risks
- OmniVoice quality insufficient for one speaker → VITS or Chatterbox track.
- Orthographic inconsistency across sources → the normalizer + IPA pivot is the mitigation; measure script fidelity separately from WER.
- Dialect skew (Sarawani standard vs Jhalawani/Rakhshani) → tag speakers by district from day one.
- Validation bottleneck on Common Voice → recruit validators before recorders.

## 8. First five actions
1. Create MDC and HF accounts; download CV 27.0 brh, Omnilingual brh_Arab, SMOL, BECO. Email BrahuiSpeech-70H author.
2. Run OmniASR 7B + CTC-1B baselines on CV test and Omnilingual test; record CER/WER/script fidelity.
3. Run OmniVoice brh zero-shot on 50 sentences; measure OmniASR-WER.
4. Write the Perso-Arabic normalizer and seal the test sets in `data/eval/`.
5. Draft 60 Spontaneous Speech prompts and 500 CC0 sentences; recruit 5 validators.
