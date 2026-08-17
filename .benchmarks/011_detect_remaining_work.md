# Benchmark 011 — `detect_remaining_work`: keyword vs centroid

Measured 2026-08-17 on M5 Pro / 48 GB. All data generated locally with `gemma4:26b`
via Ollama (127.0.0.1:11434); no cloud calls.

## Why

`detect_remaining_work` (`crates/auto_prompt/src/auto_prompt.rs:2831`) decides whether the
worker AI left unfinished work — one of the three `DecisionSource` paths that resolve
**without** an LLM call (`ConfidenceGate`, `Handbrake`, `RuleRemainingWork`). It is currently
substring matching over nine fixed patterns plus an "actionable" check.

That is the same technique class as the `lexical` router in katgpt-rs, which measured **44%**
against a hashed centroid's 89% on that repo's intent corpus. This benchmark asks whether the
same substitution pays off here.

## Data

- **Train:** 91 messages (54 remaining / 37 done) — `011_detect_remaining_work_train.json`
- **Held-out:** 60 messages (30 / 30) — `011_detect_remaining_work_heldout.json`
- Disjoint by construction (held-out texts filtered out of train).
- Both sets deliberately include the two hard classes: unfinished work described *without* any
  trigger keyword ("the migration script failed", "I haven't updated the unit tests"), and
  completed work that *does* contain trigger keywords ("No next steps are required.").

## Result (held-out, n=60)

| Method | Accuracy | Precision | Recall | F1 | Latency/msg |
|---|---|---|---|---|---|
| 1. keyword (current impl) | 0.700 | 0.833 | **0.500** | 0.625 | 3 µs |
| 2. **hashed n-gram centroid (modelless)** | **0.850** | 0.784 | **0.967** | **0.866** | 164 µs* |
| 3. embedding centroid (nomic-embed-text) | 0.833 | 0.857 | 0.800 | 0.828 | 16 ms |

\* unoptimised Python prototype; a Rust port should land well under 20 µs.

**Headline: the current matcher misses half of all genuinely unfinished work** (recall 0.500 —
15 false negatives out of 30). The modelless centroid cuts that to a single miss (recall 0.967)
and beats the embedding centroid while being ~100× cheaper than it, and ~1,200–3,000× cheaper
than the local-tier LLM call it would otherwise defer to (200–500 ms, see benchmark in
`crates/auto_prompt/src/local_mlx.rs` header).

## The tradeoff to decide

Precision drops 0.833 → 0.784 (3 → 8 false positives). The two error types are not symmetric
for this system:

- **False negative** — agent stops while work remains → *the chain dies*. This is the failure
  the whole auto_prompt loop exists to prevent.
- **False positive** — agent continues when done → one wasted turn, and the override prompt
  already tells the worker "if the work is already done or this is a false positive, stop."

For a system whose purpose is continuous operation, trading 5 extra recoverable false positives
for 14 fewer chain-killing false negatives is the right direction.

## Method caveat (read before promoting)

Train and held-out sets come from the **same generator** (`gemma4:26b`). The centroid may be
partly learning that model's phrasing rather than the underlying signal, which would inflate all
three scores in its favour. Before default-on, re-measure against **real** stop-messages
harvested from actual auto_prompt sessions. `n=60` is also small — treat the gaps as directional,
not precise.

## GOAT status

- **G1 correctness** — ✅ centroid classifies the held-out set; deterministic, no RNG.
- **G2 perf** — ⚠️ 55× *slower* than the keyword scan in absolute terms (3 µs → 164 µs), but
  negligible against the LLM call it avoids. Needs the Rust port before claiming the gate.
- **G3 no-regression** — ⬜ not run; would replace the body of `detect_remaining_work`.
- **G4 alloc-free** — ⬜ the prototype allocates a 1024-dim vector per message; a Rust port
  should use a fixed-size stack array.
- **G5 quality** — ⚠️ the numbers above, subject to the generator caveat.

**Recommendation: not yet default-on.** Port to Rust behind a config flag, re-measure on real
session data, then decide.
