#!/usr/bin/env python3
"""bench.py: single-stream decode benchmark against the rank-0 SGLang server.

Streams /v1/chat/completions and reports TTFT, decode tok/s measured from the first
to the last streamed token, and SGLang's speculative stats (accept length = tokens
committed per target forward). Stdlib only, so it runs on any Spark host.

usage: bench.py [--n N] [--max-tokens T] [--ctx-tokens C] [--prompt NAME] [--port P]
"""
import argparse, http.client, json, os, statistics, time

PROMPTS = {
    "code": "Write a Python implementation of a thread-safe LRU cache with TTL expiry, "
            "with type hints, docstrings, and a short pytest test suite.",
    "essay": "Explain how a modern CPU's out-of-order execution engine works, from fetch and "
             "decode through register renaming, the reorder buffer, and retirement.",
    "math": "Find all integer solutions of x^2 - 7y^2 = 1 with 0 < x < 10^6 and explain the "
            "method you used.",
}
FILLER = ("The quick brown fox jumps over the lazy dog while the engineer reviews the logs. ")


def run_once(port, prompt, max_tokens, ctx_tokens, thinking):
    content = prompt
    if ctx_tokens:
        # ~17 tokens per filler sentence; prepend a long preamble to test decode at context.
        reps = max(1, ctx_tokens // 17)
        # random nonce up front so the radix cache cannot reuse an earlier run's prefix
        content = (f"Session {os.urandom(8).hex()}. Background notes (ignore):\n" + FILLER * reps
                   + "\n\nTask: " + prompt)
    body = {
        "model": "mimo-v2.6-pro",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_spec_tokens_details": True,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=float(os.environ.get("BENCH_TIMEOUT", "3600")))
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
    t_first = t_last = None
    usage = spec = None
    chunks = 0
    buf = b""
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        obj = json.loads(data)
        now = time.perf_counter()
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content"):
                chunks += 1
                if t_first is None:
                    t_first = now
                t_last = now
        if obj.get("usage"):
            usage = obj["usage"]
        for key in ("spec_tokens_details",):
            if obj.get(key):
                spec = obj[key]
        for ch in obj.get("choices") or []:
            if ch.get("spec_tokens_details"):
                spec = ch["spec_tokens_details"]
        if (obj.get("sglext") or {}).get("spec_tokens_details"):
            spec = obj["sglext"]["spec_tokens_details"]
    conn.close()
    n_out = usage["completion_tokens"] if usage else None
    n_in = usage["prompt_tokens"] if usage else None
    ttft = t_first - t0
    dec = (n_out - 1) / (t_last - t_first) if n_out and t_last > t_first else float("nan")
    return dict(prompt_tokens=n_in, completion_tokens=n_out, ttft_s=ttft,
                decode_tok_s=dec, chunks=chunks, spec=spec, usage=usage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--ctx-tokens", type=int, default=0)
    ap.add_argument("--prompt", default="all")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--raw", action="store_true", help="print the raw usage/spec objects")
    a = ap.parse_args()
    names = list(PROMPTS) if a.prompt == "all" else [a.prompt]
    rates = []
    for name in names:
        for i in range(a.n):
            r = run_once(a.port, PROMPTS[name], a.max_tokens, a.ctx_tokens, a.thinking)
            sp = r["spec"] or {}
            acc = sp.get("spec_accept_length")
            print(f"{name:6s} #{i} in={r['prompt_tokens']} out={r['completion_tokens']} "
                  f"ttft={r['ttft_s']:.2f}s decode={r['decode_tok_s']:.2f} tok/s "
                  f"accept_len={acc if acc is None else round(acc, 3)} "
                  f"verify_ct={sp.get('spec_verify_ct')}", flush=True)
            if a.raw:
                print("   usage:", json.dumps(r["usage"])[:400])
                print("   spec :", json.dumps(r["spec"])[:400])
            rates.append(r["decode_tok_s"])
    if len(rates) > 1:
        print(f"decode tok/s: median {statistics.median(rates):.2f}  "
              f"mean {statistics.mean(rates):.2f}  min {min(rates):.2f}  max {max(rates):.2f}")


if __name__ == "__main__":
    main()
