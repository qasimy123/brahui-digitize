// Rough Latin rendering of Perso-Arabic Brahui, for people who cannot read the script. Port of brahui/text.py romanize():
// not reversible (short vowels are unwritten), a consonant skeleton with long vowels plus an "a" where a reader would choke.
const CONS = { 'ب':'b','پ':'p','ت':'t','ٹ':'ṭ','ث':'s','ج':'j','چ':'ch','ح':'h','خ':'kh','د':'d','ڈ':'ḍ','ذ':'z','ر':'r','ڑ':'ṛ','ز':'z','ژ':'zh',
  'س':'s','ش':'sh','ص':'s','ض':'z','ط':'t','ظ':'z','ع':"'",'غ':'gh','ف':'f','ق':'q','ک':'k','ك':'k','گ':'g','ل':'l','ڷ':'ḷ','م':'m','ن':'n',
  'ں':'ṉ','ھ':'h','ہ':'h','ۀ':'h','ة':'h','ء':"'",'ئ':"'",'ؤ':"'",'ټ':'ṭ','ډ':'ḍ','ړ':'ṛ','ګ':'g' };
const DIAC = { 'َ':'a', 'ِ':'i', 'ُ':'u', 'ّ':'', 'ْ':'' };
const ASP = new Set(['b','p','t','ṭ','j','ch','d','ḍ','k','g','ṛ']);
export function romanize(text) {
  const out = [];
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const chars = [...word]; const units = [];  // [string, isVowel]
    chars.forEach((c, i) => {
      const first = i === 0, last = i === chars.length - 1, prev = i ? chars[i - 1] : '';
      if (c === 'آ') units.push(['ā', true]);
      else if (c === 'ا') units.push([first ? 'a' : 'ā', true]);
      else if (c === 'و') units.push(first || 'اآ'.includes(prev) ? ['w', false] : ['o', true]);
      else if (c === 'ی' || c === 'ي') units.push(first || 'اآ'.includes(prev) ? ['y', false] : ['ī', true]);
      else if (c === 'ے') units.push(['e', true]);
      else if (c === 'ہ' && last && !first) units.push(['a', true]);
      else if (c in DIAC) { if (DIAC[c]) units.push([DIAC[c], true]); }
      else if (c in CONS) {
        const r = CONS[c], u = units[units.length - 1];
        if (r === 'h' && u && !u[1] && ASP.has(u[0])) u[0] += 'h';
        else if (r === 'g' && u && u[0] === 'n' && !u[1]) u[0] = 'ng';
        else if (u && u[0] === r && !u[1]) u[0] = r + r;
        else units.push([r, false]);
      } else units.push([c, true]);
    });
    let s = '', seen = false;
    units.forEach(([u, v], i) => {
      s += u; seen = seen || v;
      const nxt = units[i + 1], nn = units[i + 2];
      if (!v && nxt && !nxt[1] && ((nn === undefined && !seen) || (nn !== undefined && !nn[1]))) { s += 'a'; seen = true; }
    });
    out.push(s);
  }
  return out.join(' ');
}
