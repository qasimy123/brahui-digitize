// CTC prefix beam search with a character n-gram LM for the browser demo. Mirrors scripts/ctc_lm_decode.py (BackoffLM +
// prefix_beam) and reads the .blm table written by scripts/export_char_lm.py. 300M student on cv_dev: greedy CER 5.1 -> 3.6
// with alpha 0.5 / beta 1.0 (2026-10-25), i.e. the training-free way to close the gap to the 1B teacher.
const NEG = -1e30;
const lse = (a, b) => a === NEG ? b : b === NEG ? a : (a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b)));
function fnv1a(s) {  // over UTF-16 code units, same as the Python exporter
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { const u = s.charCodeAt(i); h ^= u & 0xff; h = Math.imul(h, 0x01000193) >>> 0; h ^= u >> 8; h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}
export async function loadCharLM(url, progress) {
  const buf = await (await fetch(url)).arrayBuffer(); const dv = new DataView(buf);
  if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== 'BLM1') throw new Error('bad LM file');
  const N = dv.getUint8(4), ncb = dv.getUint32(5, true); let o = 9;
  const chars = new TextDecoder('utf-16le').decode(new Uint8Array(buf, o, ncb)); o += ncb;
  const n = dv.getUint32(o, true); o += 4;
  const keys = new Uint32Array(buf.slice(o, o + 4 * n)); o += 4 * n;
  const lp = new Float32Array(buf.slice(o, o + 4 * n)); o += 4 * n; const bo = new Float32Array(buf.slice(o, o + 4 * n));
  const find = g => { const k = fnv1a(g); let lo = 0, hi = n - 1; while (lo <= hi) { const m = (lo + hi) >> 1; if (keys[m] < k) lo = m + 1; else if (keys[m] > k) hi = m - 1; else return m; } return -1; };
  const cache = new Map(); const logUni = Math.log(1 / chars.length);
  return { N, chars, bos: '\x02', eos: '\x03', size: n,
    logp(h, c) {
      h = h.slice(-(N - 1)); const key = h + '\x00' + c; const hit = cache.get(key); if (hit !== undefined) return hit;
      let b = 0, r;
      for (;;) { const i = find(h + c); if (i >= 0) { r = b + lp[i]; break; } if (!h) { r = b + logUni; break; } const j = find(h); if (j >= 0) b += bo[j]; h = h.slice(1); }
      if (cache.size > 2e6) cache.clear(); cache.set(key, r); return r;
    } };
}
// logits: Float32Array [T*V] raw (log-softmax applied here). id2ch: Map id -> single char. Returns the best prefix string.
export function ctcPrefixBeam(logits, T, V, id2ch, blank, lm, { alpha = 0.5, beta = 1.0, beam = 16, cand = 8, prune = -12 } = {}) {
  const bos = lm ? lm.bos.repeat(lm.N - 1) : '';
  let beams = new Map([['', [0.0, NEG, 0.0]]]);  // prefix -> [p_blank, p_nonblank, lm_score]
  const row = new Float32Array(V), idx = new Int32Array(V);
  for (let t = 0; t < T; t++) {
    const off = t * V; let m = -Infinity; for (let v = 0; v < V; v++) if (logits[off + v] > m) m = logits[off + v];
    let s = 0; for (let v = 0; v < V; v++) s += Math.exp(logits[off + v] - m); const lz = m + Math.log(s);
    for (let v = 0; v < V; v++) { row[v] = logits[off + v] - lz; idx[v] = v; }
    // top-`cand` ids by partial selection
    const top = []; const used = new Uint8Array(V);
    for (let k = 0; k < cand; k++) { let bi = -1, bv = -Infinity; for (let v = 0; v < V; v++) if (!used[v] && row[v] > bv) { bv = row[v]; bi = v; } if (bi < 0 || bv <= prune) break; used[bi] = 1; top.push(bi); }
    if (!used[blank]) top.push(blank);
    const nb = new Map();
    const add = (pfx, pb, pnb, lmsc) => { const e = nb.get(pfx); if (e) { e[0] = lse(e[0], pb); e[1] = lse(e[1], pnb); } else nb.set(pfx, [pb, pnb, lmsc]); };
    for (const [pfx, [pb, pnb, lmsc]] of beams) {
      const tot = lse(pb, pnb);
      for (const i of top) {
        const p = row[i];
        if (i === blank) { add(pfx, tot + p, NEG, lmsc); continue; }
        const ch = id2ch.get(i); if (ch === undefined) continue;
        const last = pfx.length ? pfx[pfx.length - 1] : null;
        if (ch === last) { add(pfx, NEG, pnb + p, lmsc); if (pb > NEG) add(pfx + ch, NEG, pb + p, lmsc + (lm ? alpha * lm.logp(bos + pfx, ch) : 0)); }
        else add(pfx + ch, NEG, tot + p, lmsc + (lm ? alpha * lm.logp(bos + pfx, ch) : 0));
      }
    }
    beams = new Map([...nb.entries()].sort((a, b) => (lse(b[1][0], b[1][1]) + b[1][2] + beta * b[0].length) - (lse(a[1][0], a[1][1]) + a[1][2] + beta * a[0].length)).slice(0, beam));
  }
  let best = '', bs = -Infinity;
  for (const [pfx, [pb, pnb, lmsc]] of beams) { const sc = lse(pb, pnb) + lmsc + (lm ? alpha * lm.logp(bos + pfx, lm.eos) : 0) + beta * pfx.length; if (sc > bs) { bs = sc; best = pfx; } }
  return best;
}
