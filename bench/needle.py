#!/usr/bin/env python3
"""needle.py [ctx_tokens] [trials]: long-context retrieval check against the rank-0 server (127.0.0.1:30000).

Five random facts are hidden at depths 2/25/50/75/97% of a filler document of ~ctx_tokens tokens; the model
(greedy, thinking off) must list all five. Every prefill chunk's attention over the earlier chunks is on the
path, so a wrong KV page mapping or a broken prefill kernel shows up as missing or wrong facts. Also prints
the total request time (dominated by the prefill). Stdlib only: run it on rank 0, e.g.
    ssh spark-1 "python3 - 120000 1" < bench/needle.py
"""
import http.client, json, os, random, sys, time

ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rng = random.Random(os.urandom(8))
WORDS = ("amber basil cobalt delta ember fjord granite harbor indigo juniper kestrel lantern meadow nickel "
         "orchid pepper quartz raven saffron tundra umber velvet walnut xenon yarrow zephyr").split()
FILL = [
    "The committee reviewed the quarterly logistics report and noted minor delays in the northern depot.",
    "Engineers replaced the coolant pump on line four after the vibration sensor flagged an anomaly.",
    "A short rain shower passed over the harbor in the afternoon, and the ferries kept their schedule.",
    "The archive team digitized another shelf of maritime charts from the early twentieth century.",
    "Students in the evening class practiced numerical integration with the trapezoid rule.",
]
total = 0
for trial in range(trials):
    facts = {f"crate-{rng.randint(100, 999)}": f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(10, 99)}"
             for _ in range(5)}
    n = max(50, ctx // 19)
    doc = [rng.choice(FILL) for _ in range(n)]
    for depth, key in zip((0.02, 0.25, 0.5, 0.75, 0.97), facts):
        doc.insert(int(n * depth), f"Inventory note: the shelf label of {key} is {facts[key]}.")
    question = ("\n\nList the shelf label of each of these crates, one per line as 'crate: label': "
                + ", ".join(sorted(facts)))
    body = {"model": "mimo-v2.6-pro", "max_tokens": 200, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"Ref {os.urandom(6).hex()}.\n" + " ".join(doc) + question}]}
    conn = http.client.HTTPConnection("127.0.0.1", 30000, timeout=7200)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = json.loads(conn.getresponse().read())
    dt = time.perf_counter() - t0
    text = resp["choices"][0]["message"]["content"] or ""
    found = sum(1 for v in facts.values() if v in text)
    total += found
    print(f"trial {trial}: prompt_tokens={resp['usage']['prompt_tokens']} time={dt:.1f}s recalled {found}/5", flush=True)
    if found < 5:
        print("  expected:", facts, "\n  got:", text[:400].replace("\n", " | "), flush=True)
print(f"recalled {total}/{5 * trials}")
