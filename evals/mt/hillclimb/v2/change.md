# v2: drop the brh->en reading-hints section from v1

Round 1 result (Opus 5.5, paired bootstrap on sealed test):
- en2brh +0.38 [-0.87, +1.66] (noise), brh2en -0.55 [-1.09, -0.05] (regression, just outside noise).
- train: en2brh +0.26 (noise), brh2en -0.48 (noise).

Hypothesis: the glossary is neutral-to-slightly-positive for generation into Brahui, but the brh->en section
(forced lowercase speaker labels, 'match register and length', look-alike warnings) made Opus terser / more literal
than the English references, costing chrF++ on the long BOUQuET test. One change: remove that section entirely and
let Opus translate into English as in v0. Expected: brh2en returns to v0 level; en2brh unchanged.
