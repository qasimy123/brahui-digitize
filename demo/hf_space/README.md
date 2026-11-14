---
title: Brahui → English
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: Speak Brahui, read it back in English
---

Speech → text → English for Brahui, an endangered Dravidian language of Balochistan.

- **ASR**: omniASR-CTC-1B fine-tuned on ~80 h of Brahui read speech (stage 3).
- **MT**: NLLB-200 1.3B fine-tuned brahui→English on a synthetic, LLM-retranslated parallel corpus (round 5).

Everything here was built without native-speaker annotation: see the project repo for the synthetic-bootstrap method.
