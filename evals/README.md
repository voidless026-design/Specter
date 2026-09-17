# Retrieval evals

`ev eval` measures whether E.V.'s retrieval actually works, so that changing
it is a matter of numbers rather than opinion.

```
ev eval                         # run the golden set, print a report, save it
ev eval -v                      # show every question as it runs
ev eval --floor 0.25            # try a different relevance floor
ev eval --reranker cross-encoder
ev eval --compare evals/results/20260916-203759-lexical-baseline.json
```

It ingests `corpus/` into a **throwaway** store — it never touches what you
have actually learned — and runs every question in `golden.json`.

Everything in `corpus/` is **invented for the eval**, including the files
named `my-shed.md`, `my-vehicle.md` and `my-preferences.md`. They exist so
the `personal` namespace has something to route to and rank. None of it is
anyone's real notes.

## What the buckets test

| Bucket | Questions | The failure it catches |
|---|---|---|
| `store` | 47 | The answer is in the corpus. Did the right document come back, and how high? |
| `brain` | 16 | The corpus says nothing about it. Did retrieval **abstain**? |
| `followup` | 10 | A fragment like "what about its melting point". Did rewriting recover the subject? |
| `adversarial` | 12 | A decoy document shares the question's keywords. Did the reranker keep it out of the top 3? |

Everything is scored at **document** level, not chunk level. Chunk ids move
every time the chunker changes, and an eval that needs rewriting whenever you
touch chunking gets ignored inside a month.

## Reading abstention precision

Precision is *of the times retrieval stayed quiet, how often was that right*.
It counts a wrong abstention on an answerable question against you, so it
falls if you raise the floor too far. That is deliberate: it is the number
that tracks whether E.V. can be trusted, in both directions.

Recall is *of the times it should have abstained, how often did it*. Without
a cross-encoder this one is weak, and that is expected — see below.

## Calibrating

The floor is the main dial. Sweep it:

```
for f in 0.06 0.10 0.15 0.20 0.30; do ev eval --floor $f --no-save; done
```

The lexical floor in `rerank.py` was set this way — 0.10 gives store
recall@10 1.000 and abstention precision 1.000, while 0.20 drops precision to
0.867 because answerable questions start abstaining too.

RRF `k` and the per-document cap were swept as well and are **flat** on this
corpus at every value from 10 to 200. That is an honest result, not a tuned
one: this corpus is small enough that the reranker dominates. Re-sweep on a
larger library before trusting 60.

## The baseline in `results/`, and what it does not tell you

`20260916-203759-lexical-baseline.json` was produced on a machine that could
not reach huggingface.co, so **both neural models were unavailable**:

- embeddings fell back to `hashing:512` — lexical hashing, not semantics
- reranking fell back to word overlap — no cross-encoder

```
store recall@10        1.000   [PASS]
abstention precision   1.000   [PASS]
p50 latency            21.5ms  [PASS]
abstention recall      0.562
adversarial decoys     7 of 12
```

The two PASSes are real but they are **not** the Definition of Done. Those
numbers are for the intended configuration — `BAAI/bge-m3` plus
`BAAI/bge-reranker-v2-m3` — and they can only be produced on a machine that
can download them.

The two weak numbers say exactly what the missing piece costs:

- **abstention recall 0.562.** Seven brain-only questions came back with
  something, all scoring 0.10–0.375 on coincidental word overlap: "which
  planet has the shortest day in the **solar** system" matched the off-grid
  **solar** document. A cross-encoder reads the question and the passage
  together and scores that near zero.
- **7 of 12 adversarial questions had a decoy in the top 3.** In most the
  right document was still ranked first with the decoy second, but `adv-001`
  ("how do I tie a bowline knot") genuinely ranked a boat-mooring document
  above the knot one. Word overlap cannot tell those apart.

Both are the same finding, and it is the one the spec predicted: the
reranker is the load-bearing component.

## On your machine

```
pip install -e '.[local-embeddings]'
ev eval --label first-run
```

The first run downloads roughly 2.5 GB of models. Expect p50 latency to rise
substantially — the cross-encoder is the slow stage — and expect abstention
recall and adversarial performance to rise with it. Then tune
`relevance_floor` against *your* numbers; the 0.30 defaults for the neural
rerankers are starting points, not measurements.

## Adding questions

Add to `golden.json`. `expect` and `avoid` name corpus file stems. The test
suite checks that every name resolves to a real document, that follow-ups
carry history, that adversarial entries name a decoy, and that brain-only
questions claim no document — so a malformed entry fails CI rather than
quietly scoring nothing.
