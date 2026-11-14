# v3: direction-specific system prompts (en->brh: v2 glossary prompt; brh->en: v0 plain prompt)

Round 2 (Opus): v2 vs v0 — en2brh +0.44 test [-0.80,+1.62] / +0.63 train [+0.02,+1.22]; brh2en -1.00 test [-1.55,-0.49] / -0.74 train.
So the glossary is a small consistent gain for generation *into* Brahui and a consistent loss for reading Brahui,
regardless of the brh->en section. One change: stop sharing the prompt; give each direction the prompt that won for it.
Harness change: run_eval.py gained --prompt-brh2en (direction-specific system prompt). Expected: en2brh ≈ v2, brh2en ≈ v0.
