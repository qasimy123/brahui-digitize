export const meta = {
  name: 'brahui-retranslate-fineweb-a',
  description: 'Retranslate FineWeb-2 Brahui web prose (chunks 0-761 of 1524) with Opus retrieval-by-grep teacher agents',
  phases: [{ title: 'Translate', detail: '762 chunks x 30 sentences; Opus 5.5 xhigh; greps Shad (Arabic headwords), dev-free gold corpus, glossed corpus', model: 'opus' }],
}
const T_SCHEMA = { type: 'object', properties: { translations: { type: 'array', items: { type: 'object',
  properties: { id: { type: 'string' }, en: { type: 'string' }, hard_words: { type: 'string' } }, required: ['id', 'en', 'hard_words'] } } }, required: ['translations'] }

const paths = Array.from({ length: args.end - args.start }, (_, k) => `${args.dir}/chunk_${String(args.start + k).padStart(4, '0')}.json`)

const tPrompt = (path, i) => `You are a professional Brahui -> English translator working with dictionaries and a glossed corpus. Translate every sentence in the JSON file ${path} (fields: id, brahui = Perso-Arabic Brahui sentences from web pages (FineWeb-2: news, essays, religious and literary prose, some boilerplate or lists; occasionally Urdu/Balochi/Persian text that is not Brahui), context = the preceding sentences of the same page, old_en = an earlier machine translation, often wrong in details: use it only as a topic hint, never copy it).

Resources (Read/Grep):
- data/lexicon/shad_brh_en.txt : Shad 2021 Brahui -> English dictionary, 7,816 entries, "<Perso-Arabic headword> (<roman>) [pos]: English; Urdu synonyms". Grep the Arabic stem (strip suffixes; try variants with ہ/ی/ے endings, ك/ک, ي/ی) and the roman stem.
- data/lexicon/shaigle_brh_en.txt : 1,100 everyday words (Arabic, roman, English).
- data/lexicon/flywheel_brh_en.txt : ~13k surface forms (inflected, as written) resolved by earlier teacher passes, format "form | gloss [sentences]; ... | n=<sentences> tag". Grep the exact written form here FIRST; treat a gloss as a strong hint to verify against Shad/grammar, not as ground truth (several senses can be listed). data/lexicon/flywheel_unresolved.txt lists forms earlier passes could not settle: give those extra greps.
- data/lexicon/ak_glossary.txt + data/lexicon/ak_grammar_outline.txt : romanized glossary (3,418) and the outline grammar with the transcription key (read the grammar once, fully: case suffixes ئے/-e D/A, نا/-nā GEN, ٹی/-ṭī LOC, آن/-ān ABL, اٹ/-aṭ INS, آ/-ā ALL, تون/-ton COM; verb endings; negation).
- data/parallel/gold_train_nodev.jsonl : 1,690 human-translated pairs in Perso-Arabic (brh, en): grep a word to see it used and translated. Never open gold_train.jsonl.
- data/parallel/ak2024_train.jsonl : 3,931 glossed sentences (romanized brh_roman, gloss, en): grep a romanized stem for morphology and dialect words.
- data/lexicon/sarparra_en_brh.txt : English -> romanized Brahui (4,403), to confirm a guess in reverse.

Method for EVERY sentence: (1) segment words into stem + suffixes; (2) grep the dictionaries and corpora for each stem you are not certain of (2-3 spellings each); (3) resolve case, tense, negation, person; (4) use the page context for referents and topic; (5) write a complete, faithful English sentence in the register of the source (news, essay, scripture, list): every fact in the source and nothing more; keep names, numbers, dates and quotations exact; translate the intelligible part of a garbled stretch rather than inventing; if a sentence is not Brahui (Urdu, Balochi, Persian, Arabic) translate it anyway and start hard_words with "NOT_BRAHUI:"; if it is boilerplate (menu, tag list), translate it literally. Never transliterate an unknown word as a name unless the context shows it is one. Expect several greps per sentence; many tool calls are intended.

Return via StructuredOutput: {"translations": [{"id", "en", "hard_words": "stem=meaning (source) pairs you looked up, comma-separated"}]}, exactly one entry per input sentence, same ids, same order. Chunk ${args.start + i}.`

phase('Translate')
const results = await pipeline(paths, (path, _it, i) =>
  agent(tPrompt(path, i), { label: `translate:${args.start + i}`, phase: 'Translate', schema: T_SCHEMA, effort: 'xhigh', model: 'opus' }))
const ok = results.filter(Boolean)
log(`${ok.length}/${paths.length} chunks; ${ok.reduce((n, r) => n + r.translations.length, 0)} translations`)
return { chunks_done: ok.length, chunks_total: paths.length, sentences: ok.reduce((n, r) => n + r.translations.length, 0) }