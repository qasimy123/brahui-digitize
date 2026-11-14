export const meta = {
  name: 'brahui-retranslate-curated',
  description: 'Retranslate the 22,776-sentence curated real-Brahui corpus with Opus retrieval-by-grep teacher agents, plus an agreement check against the old translation',
  phases: [
    { title: 'Translate', detail: '760 chunks x 30 sentences; Opus 5.5 xhigh; greps Shad (Arabic headwords), gold corpus, glossed corpus', model: 'opus' },
    { title: 'Verify', detail: 'Opus low effort: does the new translation agree in meaning with the old single-pass one?', model: 'opus' },
  ],
}

const T_SCHEMA = {
  type: 'object',
  properties: { translations: { type: 'array', items: { type: 'object',
    properties: { id: { type: 'string' }, en: { type: 'string' }, hard_words: { type: 'string' } }, required: ['id', 'en', 'hard_words'] } } },
  required: ['translations'],
}
const V_SCHEMA = {
  type: 'object',
  properties: { verdicts: { type: 'array', items: { type: 'object',
    properties: { id: { type: 'string' }, agree: { type: 'boolean' }, note: { type: 'string' } }, required: ['id', 'agree', 'note'] } } },
  required: ['verdicts'],
}

const paths = Array.from({ length: args.count }, (_, i) => `data/tmp/retrans_chunks/chunk_${String(i).padStart(4, '0')}.json`)

const tPrompt = (path, i) => `You are a professional Brahui -> English translator working with dictionaries and a glossed corpus. Translate every sentence in the JSON file ${path} (fields: id, brahui = Perso-Arabic script Brahui, mostly transcripts of spontaneous speech (brahuispeech, media, cv, omni sources): expect spelling variation, Urdu/Balochi loans, false starts and no punctuation; context = the preceding clips of the same recording; old_en = an earlier machine translation, often wrong in details: use it only as a hint of the topic, never copy it).

Resources (read with Read/Grep):
- data/lexicon/shad_brh_en.txt : Shad 2021 Brahui -> English dictionary, 7,816 entries, format "<Perso-Arabic headword> (<roman>) [pos]: English; Urdu synonyms". Grep the Arabic headword (strip suffixes first; try the stem with and without ہ/ی/ے endings, and with ك/ک, ي/ی variants); also grep the roman stem.
- data/lexicon/shaigle_brh_en.txt : 1,100 everyday words (Arabic, roman, English).
- data/lexicon/flywheel_brh_en.txt : ~13k surface forms (inflected, as written) resolved by earlier teacher passes, format "form | gloss [sentences]; ... | n=<sentences> tag". Grep the exact written form here FIRST; treat a gloss as a strong hint to verify against Shad/grammar, not as ground truth (several senses can be listed). data/lexicon/flywheel_unresolved.txt lists forms earlier passes could not settle: give those extra greps.
- data/lexicon/ak_glossary.txt and data/lexicon/ak_grammar_outline.txt : romanized glossary (3,418) and an outline grammar with the transcription key (read the grammar once, fully: case suffixes ئے/-e D/A, نا/-nā GEN, ٹی/-ṭī LOC, آن/-ān ABL, اٹ/-aṭ INS, آ/-ā ALL, تون/-ton COM; verb endings; negation).
- data/parallel/gold_train.jsonl : 1,992 human-translated pairs in Perso-Arabic (fields brh, en). Grep a word here to see it used and translated.
- data/parallel/ak2024_train.jsonl : 3,931 glossed sentences (romanized brh_roman, gloss, en). Grep the romanized stem for morphology and dialect words.
- data/lexicon/sarparra_en_brh.txt : English -> romanized Brahui (4,403), to confirm a guess in reverse.

Method for EVERY sentence: (1) segment words into stem + suffixes; (2) grep the dictionaries and corpora for each stem you are not certain of (2-3 spellings each); (3) resolve case, tense, negation, person from the grammar; (4) use the context clips for referents and who is speaking; (5) write a complete, faithful English sentence: every fact in the source and nothing more; for speech, keep the speaker's meaning and register, do not polish away hedges or repetitions that carry meaning, and translate the intelligible part of a garbled stretch rather than inventing. Never transliterate an unknown word as if it were a name unless the context shows it is one. Expect several greps per sentence; this chunk should take many tool calls.

Return via StructuredOutput: {"translations": [{"id", "en", "hard_words": "stem=meaning (source) pairs you looked up, comma-separated"}]}, exactly one entry per input sentence, same ids, same order. Chunk ${i} of ${args.count}.`

const vPrompt = (path, tr) => `You are checking machine translations of Brahui speech transcripts. The file ${path} has, per id, the Brahui source and old_en (an earlier translation). Below are NEW translations for the same ids. For each id, decide whether NEW and old_en preserve the same meaning: the same facts, agents, polarity and events, allowing paraphrase, register and wording differences. Mark agree=false when a fact, referent, number, polarity or event differs, or when one of them is garbled. Put a few words of reason in note. You do not need to read the Brahui; judge the two English texts. Return via StructuredOutput {"verdicts": [{"id","agree","note"}]} for every id.

NEW translations:
${tr.translations.map(t => `${t.id}: ${t.en}`).join('\n')}`

const results = await pipeline(paths,
  (path, _it, i) => agent(tPrompt(path, i), { label: `translate:${i}`, phase: 'Translate', schema: T_SCHEMA, effort: 'xhigh', model: 'opus' }),
  (tr, path, i) => tr && tr.translations && tr.translations.length
    ? agent(vPrompt(path, tr), { label: `verify:${i}`, phase: 'Verify', schema: V_SCHEMA, effort: 'low', model: 'opus' }).then(v => ({ i, n: tr.translations.length, agree: v ? v.verdicts.filter(x => x.agree).length : null }))
    : null)

const done = results.filter(Boolean)
const n = done.reduce((s, r) => s + r.n, 0)
const agree = done.reduce((s, r) => s + (r.agree || 0), 0)
log(`${done.length}/${paths.length} chunks translated, ${n} sentences, ${agree} agree with the old translation`)
return { chunks_done: done.length, chunks_total: paths.length, sentences: n, agree }