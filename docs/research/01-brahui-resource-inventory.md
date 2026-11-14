# Brahui (brh) Digital Resource Inventory — October 2026

Legend: **[V]** verified directly by fetching the page/API; **[S]** from search snippets only; **[NV]** could not be verified; **NOT FOUND** checked and absent.

## 1. Speech data

| Resource | Status | Volume | URL |
|---|---|---|---|
| **Mozilla Common Voice** | [V] YES, brh added in CV 23.0 (2025-09-05) | CV 27.0 (2026-09-11): 5,095 clips, 9.93 h total, 9.91 h validated, 18 speakers, avg 7.0 s; train 1,703 / dev 506 / test 882; 3,095 sentences; CC0 | https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/scripted-speech/cv-corpus-27.0-2026-09-11.json ; MDC page https://datacollective.mozillafoundation.org/datasets/cmn2aefkq01b8mm07dblhdy6q |
| Pontoon (CV UI localization) | [V] brh locale exists; Common Voice project ~76% localized | | https://pontoon.mozilla.org/brh/ |
| **TBOGamer22/BrahuiSpeech-70H-V2** (HF) | [V] largest Brahui speech set | 15,626 clips, 69 h 32 m, 16 kHz WAV, Perso-Arabic transcripts; real-world media speech; mixed human + Whisper-assisted transcription, 2,507 clips re-adjudicated; single split; **no license declared**; ~8 GB | https://huggingface.co/datasets/TBOGamer22/BrahuiSpeech-70H-V2 |
| TBOGamer22/Brahvi-Whisper-Small | [V] whisper-small fine-tune (2026-09-16); WER 55.71%, CER 22.83%; uses Urdu language token | | https://huggingface.co/TBOGamer22/Brahvi-Whisper-Small |
| TBOGamer22/nemo-brahvi-g2p-conformer | [V] Perso-Arabic → broad IPA G2P; PER ~9.4–10%; MIT; lexicon not released | | https://huggingface.co/TBOGamer22/nemo-brahvi-g2p-conformer |
| **Meta MMS** | [V] brh in ASR (mms-1b-all, Latin-script Bible-derived vocab) and LID-4017; **not in TTS** | | https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html |
| **Meta Omnilingual ASR** (Nov 2025) | [V] `brh_Arab` is a trained language; results CSV: 13.0 h training, CER 8.2 (7B LLM) | Corpus `brh_Arab`: 264 train / 73 dev / 106 test rows, ~88 s each ≈ 10–11 h prompted spontaneous speech, speaker IDs, CC-BY-4.0 | https://github.com/facebookresearch/omnilingual-asr ; https://huggingface.co/datasets/facebook/omnilingual-asr-corpus |
| **OmniVoice TTS** (k2-fsa, Apr 2026) | [V] `brh` listed with 19.89 h training audio; code Apache-2.0, weights CC-BY-NC | | https://github.com/k2-fsa/OmniVoice/blob/main/docs/languages.md |
| Google FLEURS | NOT FOUND | | |
| Global Recordings Network | [V] "Words of Life" (52:12) and "Good News" (32:37, Sarawani) | ~1.4 h | https://globalrecordings.net/en/language/brh |
| Bible audio (FCBH / Bible.is) | [V via ScriptureEarth] NT audio 2022, Pakistan Bible Society; direct URL [NV]; licensing for ML unverified | ~18–20 h if full NT | https://scriptureearth.org/00i-Scripture_Index.php?sortby=lang&name=brh |
| JESUS Film in Brahui | [V] 128 min dubbed | | https://www.jesusfilm.org/watch/jesus.html/brahui.html |
| ELAR | [V] Nourzaei 2023, ~1 h folktales, Iranian Brahui, open access | | https://www.elararchive.org/dk0731/ |
| Pakistan Multilingual Speech Corpus (Habib Univ, 2026) | [V] no Brahui (Urdu/Punjabi/Pashto/Sindhi/Saraiki/Balochi/Kashmiri), CC BY 4.0 | | https://zenodo.org/records/19323537 |
| Pakistani university speech datasets | NOT FOUND | | |
| Broadcast | [S] Radio Pakistan Quetta AM 750; Radio Chiltan FM 88; PTV Bolan (1 h live Brahui every morning + news; "Brahui Time"); VSH News off air since 2020 | | https://www.youtube.com/@ptvbolan5760 ; https://ptv.com.pk/ptvcorporate/livestreaming |
| YouTube | [V via Playboard] ~78 channels; Khudi Talks (41K subs, 1.3K videos), Abid Baloch (32K), Brahvi Music Channel (30K), Brahui TV, Brahui AI Media, ADAB RANG, B4 Brahui, Fitrat Tv Kalat | mostly music/drama; talk content exists | https://playboard.co/en/search?q=brahui |

## 2. Text data

| Resource | Status | Detail | URL |
|---|---|---|---|
| Brahui Wikipedia | [V] Incubator only (Wp/brh since 2012), 553 articles, **Latin Brolikva** | | https://incubator.wikimedia.org/wiki/Wp/brh |
| NLLB-200 / FLORES-200 / MADLAD-400 / Glot500 | NOT FOUND | | |
| **GlotLID v3** | [V] `brh_Arab` F1 0.992; no brh_Latn | | https://github.com/cisnlp/GlotLID |
| GlotCC-V1 | [V] brh-Arab 125 kB (tiny) | | https://huggingface.co/datasets/cis-lmu/GlotCC-V1 |
| **FineWeb-2** | [V] brh_Arab: 6,795 train + 89 test docs, 9.65 MB | | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/tree/main/data/brh_Arab |
| **FinePDFs** | [V] brh_Arab: 3,153 train + 33 test docs, 28.3 MB | | https://huggingface.co/datasets/HuggingFaceFW/finepdfs/tree/main/data/brh_Arab |
| **BECO Brahui corpora** (Mozilla Data Collective, 2026) | Talar ~150k words; literature ~355k tokens; research ~185k tokens (≈0.7M tokens); CC-BY-NC-SA-4.0 | | https://mozilladatacollective.com/datasets/cmlgx0s7j000jmg07pr4o635t |
| Bible (text) | [V] NT with Portions, Pakistan Bible Society 1998, Nastaliq PDF scans only (OCR needed); copyright PBS; not on eBible | | https://find.bible/bibles/BRHGEN/ |
| Quran translations | [V] King Fahd Complex translation (Abdul Karim Murad, ~1992, 814 pp. PDF) | OCR needed | https://islamwww.com/brahui_quran/ |
| Dictionaries | [V] Bray *The Brahui Language* Pt I (1909, PD), Pt III Etymological Vocabulary (1934); Andronov 1980; Sarparra English–Brahui (1995/2022) scans; Shahwani Brahvi-Urdu Lughat (1995) | | https://archive.org/details/in.ernet.dli.2015.232478 ; https://archive.org/details/english-to-brahui-dictionary |
| Grammar / phonology | Barjasteh Delforooz 2020 (IJALS, OA PDF, full phoneme inventory, case tables, interlinear texts); Elfenbein 1997/1998; Emeneau 1937/1962 | best free G2P reference | https://ijals.usb.ac.ir/article_5661_6be1605a6671b6661f41e7ebde97c472.pdf |
| **Universal Dependencies** | [V] UD_Brahui-Kholum (v2.18, 2026-05): 64 sentences / 819 tokens, mixed Latin/Arabic; CC BY-SA 4.0 | | https://universaldependencies.org/treebanks/brh_kholum/index.html |
| Newspapers | [V] Talár (talarbrahui.com), Nastaliq + Roman section, posts 2022–2026; Balochistan Post | | https://talarbrahui.com/ |
| Wiktionary | [V] 179 lemmas, 207-item Swadesh list | | https://en.wiktionary.org/wiki/Category:Brahui_lemmas |
| Tatoeba / OPUS / Google Translate / Microsoft Translator | NOT FOUND (Google added Balochi June 2024, not Brahui) | | |
| CLDR | [V] CLDR 49 (2026) added brh.xml (Arabic, RTL, exemplar incl. ڷ), mostly unconfirmed | | https://raw.githubusercontent.com/unicode-org/cldr/main/common/main/brh.xml |
| Keyboards | Keyman: NOT FOUND for l:brh (a "Rachitrali-Brahui" listing was reported separately, unverified); Gboard supports Brahui (Arabic, Latin, transliteration); hobby GitHub keyboards | | https://support.google.com/gboard/answer/6380730 |

## 3. Parallel data

| Resource | Status | Detail | URL |
|---|---|---|---|
| **Google SMOL** (2025) | [V] en_brh: SmolDoc 66 docs, SmolSent 863 sentences, GATITOS 3,935 word/phrase pairs; CC-BY-4.0 | | https://huggingface.co/datasets/google/smol |
| **BOUQuET** (Meta) | [V] brh_Arab among 275 varieties; dev 504 + test 854 source sentences | | https://huggingface.co/datasets/facebook/bouquet |
| Bible / Quran | PDF scans (OCR) → verse-aligned parallel with English/Urdu/Arabic | | |
| Grammar-book examples | Bray 1909, Barjasteh Delforooz 2020 | | |

## 4. Orthography

- **Perso-Arabic (dominant)**: Urdu letter set plus ڷ (U+06B7) for /ɬ/; retroflexes ٹ ڈ ڑ, ژ, ں, ہ/ھ, ے. Brahui Language Board 2008: 35 phonemes, 27 "pure" consonants. Short vowels unwritten → G2P ambiguity (the main TTS problem).
- **Brolikva (Latin)**: BLB/UoB 2008–09; ş=ش, ź=ژ, ģ=غ, ŧ=ٹ, đ=ڈ, ŕ=ڑ, ļ=ڷ, ń=ŋ, x=خ, c=چ, j=ج. Used by Incubator Wikipedia, Talár Roman section, half of UD treebank. Rare in practice.
- **Phonology**: 28 consonants incl. /ɬ/, retroflex ḍ ṭ ṛ ṇ, uvular x γ; vowels a i u / ā ī ū ē ō; no phonemic aspiration.
- **Dialects**: Sarawani (standard, Kalat/Mastung/Quetta), Jhalawani (Khuzdar), Chaghi/Rakhshani. Mutually intelligible.

## 5. People / orgs / prior computational work

- Brahui Academy Quetta (1959/66; 600+ works; brahuiacademy.com digital library; info@brahuiacademy.pk).
- Brahui Language Board (UoB, 2008). Brahui Research Institute (brahui.org).
- UoB Dept of Brahui (1992; journal *Al-Burz*; Liaquat Ali Sani, Abdul Razzaq Sabir, Shabir Ahmed Shahwani). 7,000 alumni.
- Linguists: Maryam Nourzaei, Behrooz Barjasteh Delforooz (Uppsala), Elena Bashir (Chicago).
- Computational work: arXiv has zero Brahui papers. Artifacts: UD_Brahui-Kholum (2026), Talha Bin Omar's BrahuiSpeech-70H / Whisper / G2P (2026), Omnilingual ASR, SMOL, OmniVoice, BECO corpora, CLDR. No Brahui TTS/MT/OCR paper.

## Summary
Usable now: ~90 h transcribed speech (70 h BrahuiSpeech + 9.9 h Common Voice + ~11 h Omnilingual), ~10k web docs + ~0.7M tokens curated text, ~4.9k professional parallel items (SMOL) + BOUQuET, 64-sentence treebank, G2P model, CLDR locale, GlotLID. Gaps: no open NT text, no permissive TTS, no MT engine, no Brahui–Urdu parallel corpus, no Latin-script LID. Biggest licence to clear: BrahuiSpeech-70H-V2.
