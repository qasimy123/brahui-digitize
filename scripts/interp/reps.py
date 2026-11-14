"""Experiments 2 + 3: what the encoder does with Brahui vs Urdu vs English, layer by layer, base vs fine-tuned.

Input: 504 tri-parallel BOUQuET dev sentences. Brahui and Urdu both go in under the urd_Arab tag (that is how the
fine-tune was trained), English under eng_Latn. For every encoder layer we mean-pool token states per sentence.

  extract: run both models, save evals/interp/reps_{base,ft}.pt   [layers+1, N, d] per language
  analyze: language separation (brh vs urd), parallel retrieval (brh/urd -> en), base->ft drift (CKA) per language

  uv run python scripts/interp/reps.py extract
  uv run --with matplotlib --with scikit-learn python scripts/interp/reps.py analyze
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import BRH_TAG, ENG_TAG, OUT, URD_TAG, load, triparallel  # noqa: E402

LANGS = {"brh": BRH_TAG, "urd": URD_TAG, "en": ENG_TAG}


@torch.no_grad()
def encode(model, tok, texts, tag, bs=16):
    tok.src_lang = tag
    chunks = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        hs = model.get_encoder()(**enc, output_hidden_states=True).hidden_states  # embeddings + 24 layers
        m = enc["attention_mask"].clone()
        m[:, 0] = 0  # drop the language tag
        m[torch.arange(m.shape[0]), enc["attention_mask"].sum(1) - 1] = 0  # drop </s>
        m = m.unsqueeze(-1).float()
        chunks.append(torch.stack([(h.float() * m).sum(1) / m.sum(1) for h in hs]).cpu())
    return torch.cat(chunks, 1).half()  # [L+1, N, d]


def extract():
    rows = triparallel()
    for which in ["base", "ft"]:
        p = OUT / f"reps_{which}.pt"
        if p.exists():
            continue
        model, tok = load(which)
        reps = {lang: encode(model, tok, [r[lang] for r in rows], tag) for lang, tag in LANGS.items()}
        torch.save(reps, p)
        print(which, {k: tuple(v.shape) for k, v in reps.items()})
        del model
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None


def retrieval(a, b):
    """Top-1 accuracy of finding b[i] for each a[i] by cosine, after per-language mean-centering (removes the language offset)."""
    a = a - a.mean(0); b = b - b.mean(0)
    a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float((np.argmax(a @ b.T, 1) == np.arange(len(a))).mean())


def cka(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    return float(np.linalg.norm(x.T @ y) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def separation(a, b):
    """Distance between the brh and urd centroids, in units of average within-language spread (a Fisher-style ratio)."""
    gap = np.linalg.norm(a.mean(0) - b.mean(0))
    spread = (np.linalg.norm(a - a.mean(0), axis=1).mean() + np.linalg.norm(b - b.mean(0), axis=1).mean()) / 2
    return float(gap / spread)


def probe(a, b, seed=0):
    """Cross-validated linear probe brh vs urd. Trained on parallel pairs split by sentence, so content can't leak across folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([a, b]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]; g = np.r_[np.arange(len(a)), np.arange(len(b))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
    return float(cross_val_score(clf, X, y, groups=g, cv=GroupKFold(5)).mean())


def analyze():
    R = {w: {k: v.float().numpy() for k, v in torch.load(OUT / f"reps_{w}.pt").items()} for w in ["base", "ft"]}
    L = R["base"]["brh"].shape[0]
    res = {w: {m: [] for m in ["ret_brh_en", "ret_urd_en", "ret_brh_urd", "sep_brh_urd", "probe_brh_urd", "cos_brh_urd_centroid"]} for w in R}
    res["drift_cka"] = {lang: [] for lang in LANGS}
    for l in range(L):
        for w, r in R.items():
            b, u, e = r["brh"][l], r["urd"][l], r["en"][l]
            res[w]["ret_brh_en"].append(retrieval(b, e))
            res[w]["ret_urd_en"].append(retrieval(u, e))
            res[w]["ret_brh_urd"].append(retrieval(b, u))
            res[w]["sep_brh_urd"].append(separation(b, u))
            res[w]["probe_brh_urd"].append(probe(b, u))
            mb, mu = b.mean(0), u.mean(0)
            res[w]["cos_brh_urd_centroid"].append(float(mb @ mu / np.linalg.norm(mb) / np.linalg.norm(mu)))
        for lang in LANGS:
            res["drift_cka"][lang].append(cka(R["base"][lang][l], R["ft"][lang][l]))
        print(l, {w: {k: round(v[-1], 3) for k, v in res[w].items()} for w in ["base", "ft"]},
              {k: round(v[-1], 3) for k, v in res["drift_cka"].items()}, flush=True)
    (OUT / "reps_analysis.json").write_text(json.dumps(res, indent=1))
    plot(res, L)


def plot(res, L):
    import matplotlib.pyplot as plt
    x = range(L)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a = ax[0, 0]
    for w, ls in [("base", "--"), ("ft", "-")]:
        a.plot(x, res[w]["ret_brh_en"], ls, c="C0", label=f"brh→en {w}")
        a.plot(x, res[w]["ret_urd_en"], ls, c="C1", label=f"urd→en {w}")
        a.plot(x, res[w]["ret_brh_urd"], ls, c="C2", label=f"brh→urd {w}")
    a.set_title("Shared meaning space: parallel-sentence retrieval top-1\n(mean-pooled encoder states, language-centred)")
    a.set_ylim(0, 1.02); a.legend(fontsize=8)
    a = ax[0, 1]
    for w, ls in [("base", "--"), ("ft", "-")]:
        a.plot(x, res[w]["sep_brh_urd"], ls, c="C3", label=f"{w}")
    a.set_title("Brahui vs Urdu centroid gap / within-language spread\n(higher = model keeps them apart; both tagged urd_Arab)"); a.legend()
    a = ax[1, 0]
    for w, ls in [("base", "--"), ("ft", "-")]:
        a.plot(x, res[w]["probe_brh_urd"], ls, c="C4", label=f"{w}")
    a.set_title("Linear probe Brahui vs Urdu (5-fold, by sentence)"); a.set_ylim(0.45, 1.02); a.legend()
    a = ax[1, 1]
    for i, lang in enumerate(LANGS):
        a.plot(x, res["drift_cka"][lang], c=f"C{i + 5}", label=lang)
    a.set_title("How much fine-tuning changed each language's representations\n(linear CKA base vs ft; 1 = unchanged geometry)")
    a.set_ylim(0, 1.02); a.legend()
    for a in ax.flat:
        a.set_xlabel("encoder layer (0 = embeddings)"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "reps.png", dpi=130)


if __name__ == "__main__":
    {"extract": extract, "analyze": analyze}[sys.argv[1]]()
