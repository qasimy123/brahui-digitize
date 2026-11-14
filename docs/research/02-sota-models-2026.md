# SOTA models for English ↔ Brahui speech (ASR, MT, TTS) — surveyed 2026-10-04

## 1. ASR / speech encoders

**Primary: Meta Omnilingual ASR** (2025-11-10, Apache-2.0). https://github.com/facebookresearch/omnilingual-asr , paper https://arxiv.org/abs/2511.09690
- Family: `omniASR_W2V_{300M,1B,3B,7B}` SSL encoders; `omniASR_CTC_*` (+`_v2`); `omniASR_LLM_*` (+`_v2`, `_Unlimited_*_v2` long-form); `omniASR_LLM_7B_ZS` zero-shot (7.81B).
- Inference VRAM (bf16, A100): CTC-300M ~2 GiB, CTC-1B ~3, CTC-7B ~15, LLM-7B ~17, ZS ~20.
- Brahui (`brh_Arab`) is a *trained* language: 13.0 h, CER 8.2 (7B LLM). Neighbours: bcc 10.8 h/CER 1.0 (suspicious, read-speech), bgp 8.4 h/24.5, snd 12.6 h/6.6, urd 237.5 h/3.5, skr 1.5 h/17.0.
- Zero-shot in-context (Table 11): 32 held-out languages CER 26.3 (CTC) → 17.6 (5 ctx) → 14.4 (10 ctx); SONAR-retrieved examples beat random; frozen encoder fails. API: `ASRInferencePipeline(model_card="omniASR_LLM_7B_ZS").transcribe_with_context(audio, context_examples=[[ContextExample(path, text), ...]])`.
- Fine-tuning recipe for 5–10 h (paper §5.7.5; `workflows/recipes/wav2vec2/asr/README.md`): CTC-300M/1B seeded from OmniASR CTC, lr 1e-5, ~5K steps. Sorani Kurdish (Arabic script, 9.1 h) CER 4.47. Small CTC fine-tunes beat the 7B LLM on most languages. LoRA on `omniASR_LLM_7B_v2` on one GPU done by an MLC-SLM 2026 system (https://arxiv.org/abs/2607.12468).
- Perso-Arabic script robustness: PashtoASR benchmark (https://arxiv.org/pdf/2604.04598): Whisper emitted Pashto script in <0.8% of utterances (WER 90–297%); MMS-1B, SeamlessM4T-v2, OmniASR-CTC-300M kept >93% script fidelity.

**Alternatives**
- MMS-1B-all (CC-BY-NC-4.0): adapter recipe ~4 h, lr 1e-3, <30 min, ~16 GB (https://huggingface.co/blog/mms_adapters). Shipped `brh` adapter is Latin-script; would need a new Perso-Arabic adapter. MMS-Zeroshot (`mms-meta/mms-zeroshot-300m`, uroman + lexicon/n-gram LM, https://arxiv.org/abs/2407.17852).
- w2v-BERT 2.0 (600M, 143+ langs), XEUS (577M, 4,057 langs, CC-BY-NC, https://arxiv.org/abs/2407.00837): bare encoders; superseded by OmniASR W2V for Brahui.
- Whisper large-v3/turbo: no Brahui token. "Learn and Don't Forget" (Interspeech 2024, https://arxiv.org/html/2407.06800v2): ~12 h → Sorani WER 110→32.9 full FT; full FT beats LoRA. Pashto 113 h: full FT 21.2% WER, LoRA 33 pp worse (https://arxiv.org/abs/2604.06507). Reused language-token choice swings CER up to 33 pts → train a new code embedding. Wrong backbone for Brahui.
- ML-SUPERB 2.0 winner (TalTech 2025): fine-tuned SeamlessM4T + MMS adapters + MMS-zeroshot (https://arxiv.org/abs/2506.01458).
- Not useful: Qwen3-ASR (52 langs, no ur/bal/brh), Canary/Parakeet (European), Granite Speech, Kyutai STT, Voxtral, Gemini audio.
- Universal phone recognizers: ZIPA (ACL 2025, CC-BY-4.0, `anyspeech/zipa-large-crctc-500k`, unseen-language PFER 3.2; https://arxiv.org/abs/2505.23170); PhoneticXEUS (https://arxiv.org/abs/2603.29042); POWSM; Allosaurus.

## 2. Speech translation / speech LLMs

No ST or omni model covers Brahui or Balochi. Cascade wins at this scale.
- SeamlessM4T v2 Large (2.3B, CC-BY-NC): Urdu speech-in/out; no Brahui. Fine-tune CLI requires existing code → borrow `urd`. GMU IWSLT 2025: ASR-finetune encoder first, then E2E ST (https://arxiv.org/html/2505.21781v1).
- IWSLT 2026 findings (https://aclanthology.org/2026.iwslt-1.39/): cascades still beat E2E; frontier LLM prompting (GPT-5, Gemini 3, Claude) lost to fine-tuned NLLB (10.8 vs 19.5 BLEU); baseline ranking: mono-FT SeamlessM4T > OmniASR-1B + NLLB cascade > frozen-Gemma-4 LoRA speech-LLM. IWSLT 2025: TTS-synthesized speech for ST augmentation ineffective; ≥50 h translated speech needed for usable E2E.
- Hibiki / Hibiki-Zero (Kyutai): English-output only, irrelevant for EN→Brahui.
- Proprietary realtime (OpenAI gpt-realtime, Gemini Live 3.8): Urdu yes, Brahui no, not fine-tunable.
- Open omni: Qwen3-Omni-30B-A3B (Apache; 19 speech-in incl. Urdu; LoRA via ms-swift), Qwen3.5-Omni (Mar 2026; 29 TTS langs incl. Urdu; open-weight variant unverified), Ultravox v0.7 (MIT; cleanest projector codebase, https://github.com/fixie-ai/ultravox). No paper demonstrates projector-only addition of a truly unseen language from a few hours. LoASR-Bench: speech-LLMs err 0.38 non-Latin vs 0.26 Latin (https://arxiv.org/html/2603.20042).

## 3. TTS

- **OmniVoice** (k2-fsa, Apr 2026; 0.6B on Qwen3-0.6B-Base; diffusion-LM NAR; zero-shot cross-lingual cloning; 646 langs incl. `brh` 19.89 h). Code Apache-2.0, weights CC-BY-NC. Fine-tune: `examples/run_finetune.sh`, `run_finetune_lora.sh`, JSONL `{"audio_path","text","language_id"}`. PashtoTTS-Bench (https://arxiv.org/abs/2605.26978): 24–27% WER on Pashto (~11 h), best of tested. Brahui quality unverified. https://huggingface.co/k2-fsa/OmniVoice
- Data scaling (IN-F5, https://arxiv.org/html/2505.20693v1): F5-TTS per language 100 h MUSHRA 64.3, 10 h 61.5, 1 h 33.7 / WER 59%; unseen Bhojpuri/Tulu via 1 h + synthetic self-training → MUSHRA 82–94. ~10 h is the knee for AR/diffusion; VITS tolerates ~1 h.
- MMS-TTS (VITS 36M/lang, CC-BY-NC): no brh; `facebook/mms-tts-urd-script_arabic` exists. Recipe https://github.com/ylacombe/finetune-hf-vits (80–150 samples; includes script to create checkpoint for unsupported ISO code).
- Others: F5-TTS (char vocab extendable; community Arabic ~20 h); Cross-Lingual F5-TTS 2 (Sep 2026, https://arxiv.org/abs/2609.15184); Chatterbox Multilingual (MIT, 0.5B, 23 langs; community new-script LoRA https://huggingface.co/reenigne314/chatterbox-indic-lora) = best permissive option; Fun-CosyVoice3 (no South Asian); Qwen3-TTS (Apache, 10 langs); VoxCPM2 (Apache, 2B, 30 langs); Higgs Audio v3 TTS 4B (non-commercial, 102 langs incl. Urdu/Pashto/Sindhi); XTTS-v2 (CPML). ElevenLabs v3: Urdu/Pashto/Sindhi/Persian, no Brahui.
- Phoneme route: espeak-ng has ur/sd/fa, no brh; epitran has urd-Arab, pbu-Arab, no brh → write a Brahui epitran map. URDUTTS (LREC 2026): phonemized input beat romanized by 9.6% (VITS) / 26.5% (GlowTTS) MOS. Phonikud (Hebrew diacritizer→IPA) is the template for a Brahui diacritizer (https://github.com/thewh1teagle/phonikud). Align2Speak (ICASSP 2026): IPA-token TTS + ASR-reward GRPO from unpaired text (https://arxiv.org/abs/2509.21718).

## 4. Machine translation

- Coverage: NLLB-200/FLORES-200, MADLAD-400 — no brh/bal. Google Translate added Balochi (June 2024), not Brahui. Qwen3: Urdu/Sindhi/Persian. Gemma 3 / Apertus Brahui inclusion unverified.
- Grammar-book-in-context evidence turned negative: Aycock et al. (ICLR 2025, https://arxiv.org/html/2409.19151v2): grammar explanations add ≈0 chrF++; NLLB-1.3B fine-tuned on same parallel sentences matched/beat Gemini (34.2 vs 26.6). XRI Kanuri (https://arxiv.org/html/2508.06665): parallel-only beat grammar+parallel. DiPMT++ (dict + 5k sentences, 0→16 BLEU), LingoLLM. 2026: RL with chrF reward teaches LLMs to use in-context linguistic material (https://arxiv.org/abs/2606.06428).
- Recipe: (1) fine-tune NLLB-200 1.3B/3.3B adding `brh_Arab` token initialized from `urd_Arab` on SMOL + Bible/Quran alignments (guide https://cointegrated.medium.com/a37fc706b865); (2) DiPMT++ prompting with GATITOS brh + SmolSent as retrieved shots; (3) iterative back-translation over ~0.7M-token BECO corpora, Urdu/Balochi/Persian-only mixtures (broad mixing interferes, https://arxiv.org/abs/2408.12780); (4) QLoRA CPT+SFT of Gemma 3 12B / Qwen3-8B following UrduLLaMA (https://arxiv.org/abs/2502.16961).

## 5. Data-efficient recipes and tooling

- Self-training: AI4Bharat Hindi pipeline (ASR-confidence + SONAR filtering); Pseudo2Real task arithmetic (https://arxiv.org/abs/2510.08047); Easper no-code iterative Whisper FT from ELAN (https://arxiv.org/abs/2608.11629).
- TTS→ASR augmentation: "Frustratingly Easy Data Augmentation" −14.3 pp WER (https://arxiv.org/abs/2509.15373); phoneme-guided TTS aug −19.3% rel.; self-refining ASR–TTS loop (https://arxiv.org/abs/2506.11130).
- Forced alignment: torchaudio MMS_FA (uroman, word-level OK for Arabic script); best path CTC-segmentation / NeMo Forced Aligner with your fine-tuned OmniASR CTC model.
- Diarization/VAD/enhancement: pyannote community-1 (CC-BY-4.0); NVIDIA Sortformer; Silero VAD v6 (MIT); Demucs `--two-stems=vocals` then ClearerVoice-Studio MossFormer2_SE_48K; avoid Resemble Enhance for training data (hallucinates).
- LID: `facebook/mms-lid-4017` has brh, bcc, bgp, bgn, urd, snd (CC-BY-NC); GlotLID v3 for text.

## 6. Benchmarks and evaluation

FLEURS has no Brahui; build FLEURS-Brahui via FLORES+ (OLDI). BLASER 2.0 cannot score Brahui (SONAR lacks it). COMET/xCOMET/MetricX have no Brahui; WMT24++ fell back to chrF++. Use chrF++ primary, BLEU secondary, COMET on English side only, plus native-speaker adequacy/fluency. TTS: UTMOSv2/DNSMOS/NISQA correlate poorly on low-resource languages (TTSDS2); use OmniASR brh WER/CER + script fidelity + MMS-LID + small MOS panel.

## 7. Compute and cost

| Task | VRAM | Time |
|---|---|---|
| MMS-1B adapter (~4 h data) | ~16 GB | <30 min |
| OmniASR CTC-300M/1B FT (5–10 h data) | 80 GB w/ grad-accum | several hours (estimate) |
| Whisper-small full FT (8 h) | T4-class | 5–10 h |
| F5-TTS LoRA | 24 GB | ~1,250 updates |
| OmniVoice LoRA | ~24 GB (estimate) | hours |
| QLoRA 8B / 12B LLM, few 100k tokens | 10–14 / ~24 GB | <1 h |

Prices 2026-10-04: RunPod A100 80GB $1.59 secure / $1.19–1.39 community, H100 SXM $3.49/$2.69, B200 $6.79/$5.98. Lambda A100 $1.99–2.79, H100 $3.29–4.29, B200 $6.69–6.99. Vast.ai A100 ~$0.53–0.67, H100 ~$0.90–2.50 (unverified). A full iteration of all fine-tunes fits one A100-80GB for roughly $10–60. mlx-audio runs Whisper/Qwen3-ASR/OmniVoice inference on Apple Silicon.

## Unverified
Per-language Brahui CER for OmniASR beyond the CSV figure; exact Omnilingual-corpus Brahui hours (10–11 h estimated); OmniVoice Brahui audio quality; Google Translate 2025–26 additions; Apertus/Gemma Brahui pretraining; Qwen3.5-Omni open-weight variant; uroman handling of ڷ; Vast.ai prices; ML-SUPERB 2.0 Brahui inclusion.
