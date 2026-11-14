# v1 change: native Brahui vocabulary conventions

Source: `evals/mt/hillclimb/baseline/train_failures_for_analyzer.json` (40 train cases per direction), checked against the train pools `data/manifests/smol_sent_train.jsonl` + `data/manifests/bouquet_sent_dev.jsonl` (1167 rows). No eval/test rows were read.

## Bucket counts (primary cause per case)

| Bucket | en->brh | brh->en |
|---|---|---|
| (a) genuine mistranslation / wrong meaning | 0 | 18 (model misreads core Brahui words: چاہنگ as "tea", سلنگ, دڑینگ, نرنگ, رائی, سُو, گُڈ, زوت, گجر ...) |
| (b) correct meaning, different wording than the single reference | 18 (of which ~8 have garbled/over-elaborated references) | 18 |
| (c) orthographic/spelling convention | 0 as primary; see lexical sub-bucket below | 0 |
| (c') lexical register: Urdu/Persian word where the reference uses a native Brahui word or a transliterated English loan | 17 | (same root cause as (a)) |
| (d) register/length | 1 | 0 |
| (e) untranslated/transliterated chunks | 0 | 0 |
| (f) Brahui grammar (singular imperative where refs use -بو; verb number agreement; negative past form) | 4 | - |
| (g) English-side issues (speaker-tag casing <Woman:> vs <female:>, <Comment 3:> vs <comment 3:>) | - | 3 (secondary on top of other buckets) |
| reference contains content absent from the Brahui source (unfixable) | - | 4 |

## Hypothesis

The largest fixable bucket in both directions is the same: Sonnet treats Brahui as Urdu-with-extras. En->brh it reaches for Urdu core vocabulary (دل، کچھ، جلد، ہمیشہ، بندوق، چوری، ناشتہ، دوسرا، پورا، بس) and formal Urdu/Arabic coinages for technical terms (رکن، محقق، طبی، فارغ التحصیل، بل، مدیر), while the references use native Brahui words (اُست، گڑاس، زو، مدامی، توفک، دُزی، نہاری، ایلو، سرجم، بیرہ) and keep technical terms as transliterated English (ممبر، ڈاکٹر، کلینیکل، گریجویٹ، انوائس، ایڈیٹر). Counts in the train pools back this up: جوان 63 vs اچھا 0, بھلو 41 vs بڑا 0, گڑاس 41 vs کچھ 0, تون 142 vs ساتھ 0, ہم 65 vs بھی 1, خن 62 vs آنکھ 0, دیر 18 vs پانی 0, زو/زوئس 45 vs جلد 0, ارفنگ 30 vs پوچھ 0, ٹیسٹ 7 vs امتحان 3, ڈاکٹر 10 vs حکیم/طبیب 1. Brh->en, the same gap shows as misreading those native words (چاہنگ -> "tea", سلنگ -> "standing", دڑینگ -> "arrived", زوت -> "quickly distributed"). A compact, verified glossary of ~60 core words stated as rules should move chrF++ directly, because chrF++ is character-level and a wrong lexeme costs every n-gram of that word. The 15 retrieved few-shots cannot teach this reliably since any given word appears in them only by chance.

## What changed in `evals/mt/prompts/system_v1.md`

1. New section "Brahui vocabulary: native words, not Urdu": a glossary of native nouns, adjectives/quantifiers, function words, verbs and copula forms, each with the Urdu word to avoid; plus the rule that modern/technical/job terms are transliterated English, not Persianised coinages.
2. Small "grammar and formatting" block, all verified in the references: polite/plural imperative in -بو for instructions (کبو appears 72x; hypotheses used bare کر), spaced case suffixes, Western digits, ۔ and ،, ڷ.
3. Brh->en block: read the glossary by its Brahui meaning (three explicit false-friend warnings), lowercase role speaker labels (74 tagged rows in bouquet dev; refs use <female:>, <comment 3:>, <yong:>), keep source register/length.

No failing sentence or reference was copied into the prompt; only individual words.

## Expected effect

- en->brh (baseline 32.6): main target. 17/40 analysed failures are primarily this bucket and several (b) cases have one or two Urdu lexemes as a secondary cost. Expect +1 to +3 chrF++. The -بو imperative rule touches every instructional sentence (smol has many how-to lines).
- brh->en (baseline 48.6): modest gain. Fixes a few genuine misreadings per 100 sentences and the speaker-tag casing on every tagged dialogue line. Expect +0.5 to +1.5.

## Risks

- The glossary biases toward the smol/bouquet translators' idiolect; some items (توفک, مخسرہ, سلنگ, چاہنگ) have 1-5 attestations. If a reference uses the Urdu word in a given sentence (دل does appear 42x vs اُست 46x; نوا 46x vs پوسکن 16x), the rule costs a little. I excluded the near-tie items (new, heart is kept because the hypotheses picked دل every time while the failing refs used اُست).
- Listing Urdu words "not to use" could in principle prime them; wording keeps the native word first and bold-free.
- Longer system prompt (~2.5x) slightly increases the chance the model paraphrases or adds commentary; the "output only the requested JSON" instruction is unchanged.
- The -بو rule could misfire on second-person singular narrative imperatives (fiction dialogue); scoped to "instructions and requests addressed to the reader".
