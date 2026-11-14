"""Translate Brahui units to English with the Claude API (Opus 5.5, default effort), cheaply:
a fixed few-shot block in the system prompt with prompt caching, N sentences per request, structured JSON
output, and the Message Batches API (50% off) for the bulk run. Output rows match scripts/build_parallel.py
so scripts/train_nllb.py can read them.

Usage:
  uv run python scripts/translate_api.py sample  --units data/parallel/fineweb_units.jsonl --n 200
      synchronous run on the first N units; prints measured tokens and $ per sentence  -> <out>
  uv run python scripts/translate_api.py submit  --units ... --budget-usd 85 [--per-sent-usd X]
      submits a batch over the first K units that fit the budget (K from the sample's measured cost
      unless --n is given); records the batch id in <out>.batches.json
  uv run python scripts/translate_api.py collect --units ...
      fetches finished batches, appends rows to <out>, reports cost from usage
Defaults: --out data/parallel/synthetic_fineweb_brh_en.jsonl, --per-request 40, --model claude-opus-5-5
Needs ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evals" / "mt"))
from run_eval import load_cases, SEED  # noqa: E402

PRICE = {"claude-opus-5-5": (4.0, 20.0, 0.20, 8.0)}  # $/M: input, output, cache read (0.05x), 1h-TTL cache write (2x)
SYSTEM = (ROOT / "evals/mt/prompts/system_v0.md").read_text(encoding="utf-8") + (
    "\n\nThe Brahui input is written news and web prose, or a transcript of speech: expect spelling variation and "
    "occasional noise. Translate the meaning faithfully and completely, sentence by sentence; do not summarize or add "
    "commentary; if part of a sentence is unintelligible, translate the intelligible part.")
SCHEMA = {"type": "object", "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
          "required": ["translations"], "additionalProperties": False}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if Path(p).exists() else []


def shot_block(n=40):
    """Fixed, seeded few-shot block (gold brh->en pairs) so every request shares one cached prefix."""
    _, pools = load_cases()
    pool = sorted(pools["brh2en"], key=lambda p: p["id"])
    shots = random.Random(SEED).sample(pool, n)
    lines = ["## Reference translations (Brahui -> English)", ""]
    for p in shots:
        lines += [f"Brahui: {p['brh']}", f"English: {p['en']}", ""]
    return "\n".join(lines)


def user_msg(batch):
    lines = [f"## Translate each of the following {len(batch)} Brahui sentences into English.",
             "Return JSON {\"translations\": [...]} with exactly one string per input, in order.", ""]
    lines += [f"{i}. {u['brh']}" for i, u in enumerate(batch, 1)]
    return "\n".join(lines)


def params(model, batch, shots):
    return dict(model=model, max_tokens=8000,
                system=[{"type": "text", "text": SYSTEM},
                        {"type": "text", "text": shots, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                messages=[{"role": "user", "content": user_msg(batch)}],
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}})


def cost(usage, model, batch_api=False):
    pi, po, pr, pw = PRICE[model]
    c = (usage.input_tokens * pi + usage.output_tokens * po + (usage.cache_read_input_tokens or 0) * pr
         + (usage.cache_creation_input_tokens or 0) * pw) / 1e6
    return c / 2 if batch_api else c


def parse(msg, n):
    if msg.stop_reason == "refusal":
        return None, "refusal"
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        t = json.loads(text)["translations"]
    except Exception as e:
        return None, f"bad json: {e}"
    return (t, None) if len(t) == n else (None, f"{len(t)} translations for {n}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sample", "submit", "collect"])
    ap.add_argument("--units", required=True); ap.add_argument("--out", default=str(ROOT / "data/parallel/synthetic_fineweb_brh_en.jsonl"))
    ap.add_argument("--model", default="claude-opus-5-5"); ap.add_argument("--per-request", type=int, default=40)
    ap.add_argument("--n", type=int); ap.add_argument("--budget-usd", type=float, default=85.0); ap.add_argument("--per-sent-usd", type=float)
    a = ap.parse_args()
    env = ROOT / ".env"  # ANTHROPIC_API_KEY=... (gitignored); the shell env wins if set
    if env.exists():
        import os
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))
    client = anthropic.Anthropic()
    units = read_jsonl(a.units); out = Path(a.out); done = {r["id"] for r in read_jsonl(out)}
    shots = shot_block(); state_p = Path(str(out) + ".batches.json"); state = json.loads(state_p.read_text()) if state_p.exists() else {"batches": [], "sample": {}}
    pending = [u for u in units if u["id"] not in done and u["id"] not in {i for b in state["batches"] for i in b["ids"]}]

    if a.cmd == "sample":
        todo = pending[:a.n or 200]; rows, tot, usage_sum = [], 0.0, {}
        for i in range(0, len(todo), a.per_request):
            batch = todo[i:i + a.per_request]
            msg = client.messages.create(**params(a.model, batch, shots))
            t, err = parse(msg, len(batch)); c = cost(msg.usage, a.model); tot += c
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                usage_sum[k] = usage_sum.get(k, 0) + (getattr(msg.usage, k) or 0)
            print(f"  req {i//a.per_request+1}: {len(batch)} sents, usage={msg.usage.input_tokens}/{msg.usage.output_tokens} "
                  f"cache r/w={msg.usage.cache_read_input_tokens}/{msg.usage.cache_creation_input_tokens} ${c:.3f} {err or ''}", flush=True)
            if t:
                rows += [{**u, "en": s.strip(), "model": msg.model} for u, s in zip(batch, t)]
        with open(out, "a", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        per = tot / max(1, len(rows)); state["sample"] = {"rows": len(rows), "usd_sync": round(tot, 4), "per_sent_sync": per, "per_sent_batch": per / 2, "usage": usage_sum}
        state_p.write_text(json.dumps(state, indent=2))
        print(f"sample: {len(rows)} rows, ${tot:.2f} sync = ${per*1000:.2f}/1000 sents sync, ${per*500:.2f}/1000 via Batch API")
        for r in rows[:3]: print("  ", r["brh"][:90], "->", r["en"][:110])

    elif a.cmd == "submit":
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        per = a.per_sent_usd or state.get("sample", {}).get("per_sent_batch")
        if not per: sys.exit("no per-sentence cost known: run `sample` first or pass --per-sent-usd")
        k = a.n or int(a.budget_usd / per); todo = pending[:k]
        reqs = [Request(custom_id=f"b{i//a.per_request:05d}", params=MessageCreateParamsNonStreaming(**params(a.model, todo[i:i + a.per_request], shots)))
                for i in range(0, len(todo), a.per_request)]
        print(f"submitting {len(todo)} sentences in {len(reqs)} requests, est ${len(todo)*per:.0f} (budget ${a.budget_usd})", flush=True)
        b = client.messages.batches.create(requests=reqs)
        state["batches"].append({"id": b.id, "ids": [u["id"] for u in todo], "n_req": len(reqs), "per_request": a.per_request, "collected": False})
        state_p.write_text(json.dumps(state, indent=2)); print("batch", b.id, b.processing_status)

    else:
        idx = {u["id"]: u for u in units}
        for bst in state["batches"]:
            if bst["collected"]:
                continue
            b = client.messages.batches.retrieve(bst["id"])
            print(bst["id"], b.processing_status, b.request_counts)
            if b.processing_status != "ended":
                continue
            ids = bst["ids"]; pr = bst["per_request"]; rows, tot, bad = [], 0.0, 0
            for res in client.messages.batches.results(bst["id"]):
                j = int(res.custom_id[1:]); batch = [idx[i] for i in ids[j*pr:(j+1)*pr]]
                if res.result.type != "succeeded":
                    bad += 1; print("  ", res.custom_id, res.result.type); continue
                msg = res.result.message; tot += cost(msg.usage, a.model, batch_api=True)
                t, err = parse(msg, len(batch))
                if err:
                    bad += 1; print("  ", res.custom_id, err); continue
                rows += [{**u, "en": s.strip(), "model": msg.model} for u, s in zip(batch, t)]
            with open(out, "a", encoding="utf-8") as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            bst["collected"] = True; bst["rows"] = len(rows); bst["usd"] = round(tot, 2); bst["failed_requests"] = bad
            state_p.write_text(json.dumps(state, indent=2))
            print(f"  collected {len(rows)} rows, {bad} failed requests, ${tot:.2f}")


if __name__ == "__main__":
    main()
