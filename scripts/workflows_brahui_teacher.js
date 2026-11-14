export const meta = {
  name: 'brahui-teacher-validate',
  description: 'Translate the 200-sentence dev2 slice with retrieval-by-grep teacher agents (validation of the workflow teacher)',
  phases: [{ title: 'Translate', detail: 'one Opus 5.5 agent per 20-sentence chunk; greps dictionaries and glossed corpus per token', model: 'opus' }],
}

const OUT_SCHEMA = {
  type: 'object',
  properties: {
    translations: {
      type: 'array',
      items: {
        type: 'object',
        properties: { id: { type: 'string' }, en: { type: 'string' }, hard_words: { type: 'string' } },
        required: ['id', 'en', 'hard_words'],
      },
    },
  },
  required: ['translations'],
}

const prompt = (path, i) => `You are a professional Brahui -> English translator working with a dictionary and a glossed corpus. Translate every sentence in the JSON file ${path} (fields: id, brahui = romanized Brahui in the Ali & Kobayashi transcription with '-' at suffix boundaries and '=' at clitics, context = the preceding sentences of the same story, story = title).

Resources in this repo (read with Read/Grep; all romanized unless stated; the transcription key is in data/lexicon/ak_grammar_outline.txt):
- data/lexicon/ak_grammar_outline.txt : outline grammar + transcription key (read once, fully).
- data/lexicon/ak_glossary.txt : 3,418 glossary entries for this exact corpus (romanized headwords, loan-language tags).
- data/lexicon/shad_brh_en.txt : Shad 2021 Brahui -> English dictionary, 7,816 entries, format "<Perso-Arabic headword> (<roman>) [pos]: English; Urdu synonyms". Grep the roman form (ignore macrons/diacritics when grepping: try the plain ASCII stem).
- data/lexicon/sarparra_en_brh.txt : English -> romanized Brahui dictionary, 4,403 entries ("English: brahui1, brahui2"). Use in reverse to confirm a Brahui word.
- data/parallel/ak2024_train.jsonl : 3,931 glossed sentences from the same book (fields brh_roman, gloss, en). Grep a stem here to see it glossed in context; this is the strongest evidence for dialect words and morphology.
- data/parallel/gold_train.jsonl : 1,992 human-translated pairs (Perso-Arabic brh, en).

Method, for EVERY sentence: (1) segment each word into stem + suffixes; (2) for each stem you are not certain of, grep the glossary, Shad, and ak2024_train (try 2-3 spellings: with and without doubled consonants, with the ASCII vowel); (3) resolve case suffixes (-e D/A, -nā GEN, -ṭī LOC, -ān ABL, -aṭ INS, -ā ALL, -ton COM), tense, negation, person from the grammar outline; (4) use the context to fix pronoun gender and referents (Brahui marks no gender; English must), who is speaking, and tense; (5) compose a complete, faithful English sentence: every fact in the source, nothing more, keep quotation marks and direct speech as in the source. Never transliterate an unknown word; choose the reading the evidence supports. Do not summarize. Expect ~5 greps per sentence; this chunk should take many tool calls, that is intended.

Return, via the StructuredOutput tool, {"translations": [{"id", "en", "hard_words": "stem=meaning (source) pairs you had to look up, comma-separated"}]} with exactly one entry per input sentence, same ids, same order. Chunk ${i}.`

phase('Translate')
const results = await pipeline(args, (path, _item, i) =>
  agent(prompt(path, i), { label: `chunk:${i}`, phase: 'Translate', schema: OUT_SCHEMA, effort: 'xhigh', model: 'opus' }))
const ok = results.filter(Boolean)
log(`${ok.length}/${args.length} chunks returned; ${ok.reduce((n, r) => n + r.translations.length, 0)} translations`)
return { chunks: ok.length, translations: ok.flatMap(r => r.translations) }