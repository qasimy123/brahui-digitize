# Gathering Brahui speech + text data: practical methods (2026-10-04)

**Headline: Brahui is already live on Mozilla Common Voice** (Scripted Speech 27.0: 5,095 clips / 9.93 h / 18 speakers / 3,095 sentences). You skip onboarding and go straight to (a) adding CC0 sentences, (b) recording/validation drives, (c) requesting Spontaneous Speech for Brahui.

## 1. Mozilla Common Voice (2025–2026 state)
- Survived the 2024 Mozilla Foundation layoffs. Since Oct 2025 datasets distributed via **Mozilla Data Collective** (MDC), account + ToS required. Still adding languages: v23.0 jumped 137→286 languages; v27.0 (2026-09-11) 295 languages / 42,593 h. Quarterly releases (Mar/Jun/Sep/Dec). https://github.com/common-voice/cv-dataset
- Onboarding (for reference): 300 mandatory Pontoon strings; sentences before recording opens: Band A 750, Band B 2,000, Band C 5,000; CC0/public domain only (self-written sentences qualify); each sentence needs 2 upvotes; bulk .txt submission and Sentence Extractor exist. https://github.com/common-voice/common-voice/blob/main/docs/LANGUAGE.md , https://github.com/common-voice/common-voice/blob/main/docs/SENTENCES.md
- **Spontaneous Speech** (launched 2025): contributors answer prompts (10–25 s), others transcribe + validate. Enable for a language: GitHub issue "Spontaneous Speech New Language" with ≥60 culturally appropriate questions, no PII, prompts reviewed by ≥2 speakers, cannot be changed once loaded (example https://github.com/common-voice/common-voice/issues/5423). Spontaneous Speech 2.0 (Dec 2025) 62 languages.
- Pakistani model case, **Pashto**: 1.5 h/5 speakers (CV14) → 147 h/1,483 speakers (CV23). The 108× jump came from one VOA Pashto broadcast + Facebook + tutorial videos + ~27k Wikipedia sentences ≤15 words; prompts targeted frequently-dropped letters; validation lagged recording by ~40%. Whisper-base FT: 13.4% WER vs 99% zero-shot. https://arxiv.org/abs/2603.27021
- Common Voice funded programs have been Africa-only; no Asia program found.

## 2. Alternative collection platforms
| Tool | Fit | Source |
|---|---|---|
| LIG-Aikuma v3.1 (Android, AGPL) | Offline fieldwork: free recording, **respeaking**, oral translation, elicitation from text/image/video | https://lig-aikuma.imag.fr/ |
| Karya (open-source Android, paid crowd) | 200k+ workers, 115+ languages; 11-language monologue project: 5,000 villagers, 80 districts, <3 months; linguist-validated sentence corpus then manual re-verification | https://www.karya.in/case-studies/monologue-speech-data-collection/ |
| Project Vaani (IISc/ARTPARK/Google) | Reference recipe for image-prompted spontaneous speech: ~1,000 speakers/district, ≤15 min each, ~200 h/district, 16 kHz mono, mic <2 ft, 1,700–2,000 district-specific images; 5–10% transcribed by local linguists, <5% WER second check, batch re-validation if >10% QC fail; CC BY 4.0 | https://arxiv.org/html/2603.28714v2 |
| Gram Vaani IVR | Phone-in community radio → 1,108 h spontaneous Hindi; IVR works for low-literacy rural users | https://www.isca-archive.org/interspeech_2022/bhanushali22_interspeech.pdf |
| Bloom (SIL) | `sil-ai/bloom-speech` 56 languages, no Brahui yet | https://huggingface.co/datasets/sil-ai/bloom-speech |
| GRN | Brahui "Words of Life" + "Good News" (~1.4 h); ML licensing unverified | https://globalrecordings.net/en/language/brh |
| Appen / Defined.ai / Shaip / Datatang | No off-the-shelf Brahui | |
| WhatsApp voice notes | No published project; feasible, needs per-sender consent capture | |

## 3. Mining existing audio
- PTV Bolan: 1 h live Brahui every morning + daily news; live stream https://ptv.com.pk/ptvcorporate/livestreaming ; "Brahui Time" https://www.facebook.com/Brahuitime/ ; PTV Flix archive. Radio Pakistan Quetta: no systematic online archive found.
- YouTube pipeline template: YODAS (ESPnet) used CC-licensed YouTube only; use yt-dlp + MMS-LID/GlotLID + pyannote; expect heavy Urdu/Balochi code-switching.
- Law: Pakistan Copyright Ordinance 1962 has fair dealing for research, **no TDM exception**; PECA 2016 ss.3–5; **no data-protection law in force as of 2026** (PDP Bill stalled). Practical rule: broadcast/YouTube audio → unlabelled pretraining only, never redistribute; redistributable data must be consented.
- Consent: record verbal consent in Brahui at session start; BAS Munich template https://www.phonetik.uni-muenchen.de/Bas/BasTemplateInformedConsent_en.pdf ; Common Voice CC0 click-through is the simplest model.

## 4. Elicitation design (value per recorded hour)
1. Read speech on Common Voice: write/translate CC0 sentences yourself; phoneme-targeted sentences for weak segments; ≤15 words.
2. **FLORES+ via OLDI**: translate 997 dev + 1,012 devtest English sentences; human translation only, no MT post-editing; dataset card with ISO 639-3 `brh`, Glottocode `brah1256`, script `Arab`; ≥1 extra native reviewer; HF pull request with DCO; CC BY-SA 4.0. Read them aloud → FLEURS-aligned Brahui ASR/ST benchmark. https://oldi.org/guidelines
3. Image-prompted spontaneous speech (Vaani recipe), paired with Common Voice Spontaneous Speech 60-prompt format.
4. Respeaking (LIG-Aikuma) of spontaneous audio for alignability.
5. Bible/Quran parallel reading if text can be cleared.

## 5. Text gathering
- OCR: Tesseract/EasyOCR/Kraken fail on Nastaliq. Gemini-2.5-Pro WER 0.133 on Urdu Newspaper Benchmark; GPT-4o fine-tuned on 500 samples +6 pts (https://arxiv.org/abs/2505.13943). Open: **Qaari-0.1** (Qwen2-VL-2B, WER 0.048 synthetic Nastaliq) https://huggingface.co/oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct . Brahui ڷ needs fine-tuning; budget a few hundred hand-corrected lines.
- Public domain: Bray 1909 on archive.org; TUFS Brahui texts PDF https://tufs.repo.nii.ac.jp/record/2000650/files/B532_AAL66_BrahuiTexts.pdf
- Brahui Academy digital library (permission needed); UoB Brahui Dept "Brahui Digital Archive", journal Al-Burz.
- Web crawl: GlotLID `brh_Arab`; GlotSparse `brh_Arab`.
- Input: Gboard supports Brahui; Wikipedia Incubator Wp/brh open.

## 6. Community model
- Pashto lesson: a "language technology broker" with media contacts + Facebook + tutorial videos, then earned broadcast media, beat every technical optimisation; recruit validators explicitly.
- Partners: Brahui Academy, UoB Brahui Dept (7,000 alumni), PTV Bolan/Brahui Time, Karachi/UAE diaspora WhatsApp groups.
- Pay: no published Pakistani rates; planning guess PKR 1,000–2,000 per validated hour-equivalent; cap per-speaker minutes (Vaani ≤15 min) for diversity.
- QC: double validation; automatic SNR/silence thresholds; >10% failure → batch re-validation.
- Licensing: CC0 for anything flowing into Common Voice; CC BY for the rest; FLORES+ is CC BY-SA.

## 7. Funding (2026)
- ELDP 2026 round closed 1 Oct 2026; next cycle 2027; Small Grants €10k, MDP up to €300k; community members eligible. https://linguistlist.org/issues/37/2122/
- Lacuna Fund NLP 2026: Africa/Latin America focus; Pakistan likely ineligible.
- Wikimedia Rapid Fund $500–5,000, individuals eligible, rolling. https://meta.wikimedia.org/wiki/Grants:Programs/Wikimedia_Community_Fund/Rapid_Fund
- Pakistan: HEC NRPU; Ignite Emerging Tech & AI Fund (needs university partner).
- Google/Mozilla/UNESCO: no open channel in 2026. Meta NLLB community: contribute via OLDI.
