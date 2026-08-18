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

---

## Addendum (same day) — Rust port, and a coverage gap this benchmark had

Ported to `crates/auto_prompt/src/remaining_work.rs`. Two changes from the prototype:

**FNV-1a instead of BLAKE2b.** The crate has no hashing dependency, and `DefaultHasher` is
explicitly not stable across Rust releases — the centroids are derived from the hash, so a
change would silently invalidate them. FNV-1a is written out in the module and pinned by a
test against reference vectors. Re-running the sweep under FNV also moved the numbers:

| DIM | acc | prec | rec | F1 | FP / FN |
|---|---|---|---|---|---|
| 128 | 0.717 | 0.676 | 0.833 | 0.746 | 12 / 5 |
| 256 | 0.783 | 0.743 | 0.867 | 0.800 | 9 / 4 |
| 512 | 0.867 | 0.806 | 0.967 | 0.879 | 7 / 1 |
| **1024** | **0.917** | **0.857** | **1.000** | **0.923** | **5 / 0** |
| 2048 | 0.900 | 0.853 | 0.967 | 0.906 | 5 / 1 |
| 512, unigrams only | 0.817 | 0.732 | 1.000 | 0.845 | 11 / 0 |

DIM=1024 is the pick. Note the gap between the BLAKE2b run (0.850) and the FNV run (0.917) at
the same width is pure bucket-collision luck — at n=60 that difference is noise, not a reason
to prefer either hash on quality.

**The gap: this benchmark never tested short messages.** A unit test written against the port
failed on a 15-word completion ("Done — the endpoint now returns the correct status code and
the regression test covers it."), which scored +0.02 *toward* "remaining" — wrong, on a margin
indistinguishable from noise. Checking the corpus explains it: the shortest training example is
**24 words** (median 32) and the held-out set contains nothing under 25. Every number in the
table above therefore describes messages of ≥25 words only, and says nothing about the short
ones real agents certainly emit.

Accuracy by length on the held-out set: 25–49 words **0.912** (52/57), ≥50 words **1.000** (3/3),
under 25 words **no data**.

The port answers this by abstaining rather than bluffing: `indicates_remaining_work` returns
`Option<bool>`, with `None` for empty input or anything under `MIN_WORDS = 24`. Callers fall
back to the keyword path, which reads short explicit statements perfectly well. That also
suggests the eventual shape is a **union** of the two — keyword for short and explicit,
centroid for long and implicit — which is the fusion a Super-GOAT claim would need to prove.

### GOAT status after the port

- **G1 correctness** — ✅ deterministic (no RNG, no clock, no network); hash pinned by test;
  7 unit tests including the real held-out false-negative cases.
- **G2 perf** — ⬜ still unmeasured in Rust. The Python prototype's 164 µs is not the number to
  quote; needs a criterion bench before the gate is claimable.
- **G3 no-regression** — ✅ 176 passed, 0 failed, clippy clean. `detect_remaining_work` is
  **untouched** — the module is not wired in yet.
- **G4 alloc-free** — ⚠️ the feature vector is a fixed `[f32; 1024]` on the stack, but bigram
  extraction still allocates a `String` per pair and a `Vec<&str>` per message. Fixable with a
  streaming hash over the word pair; not done.
- **G5 quality** — ⚠️ the table above, subject to the single-generator caveat and now the
  explicit short-message blind spot.

**Still not default-on, and still not wired in.** Next: criterion bench (G2), kill the per-bigram
allocation (G4), harvest real session messages including short ones (G5).

---

## Addendum 2 — G2 and G4 closed

**G4 — allocation removed.** `features` no longer builds a cleaned `String`, a `Vec<&str>` of
words, or a `String` per bigram. It walks the input once, locating ASCII-alphanumeric runs by
index and hashing them in place with lowercasing folded into the fold step. The only storage is
the fixed `[f32; 1024]` return array, so classification never touches the heap.

The bigram hash is now streamed (`a`, then `b'_'`, then `b`) instead of hashing a built
`"{a}_{b}"`. FNV-1a is a rolling hash, so the two are bit-identical — pinned by
`bigram_hash_matches_concatenation`, which is what makes this an optimisation rather than a
retrain: **the committed centroids describe exactly the same feature space as before.**

**G2 — measured** (criterion, `cargo bench -p auto_prompt`, M5 Pro):

| Input | Time |
|---|---|
| typical stop-message (46 words) | **2.16 µs** |
| long stop-message (92 words) | **2.59 µs** |
| short message (abstains via `MIN_WORDS`) | **5.57 ns** |

The Python prototype's 164 µs quoted above was 76x pessimistic; do not cite it. In context:

| Path | Cost per agent stop |
|---|---|
| keyword scan (current) | ~3 µs (Python-measured; same order) |
| **centroid (this module)** | **2.2 µs** |
| local tier LLM call (T1, thinking off) | ~200,000 µs |
| cloud orchestration call | ~1,000,000+ µs |

So the centroid is roughly free next to the keyword scan it would join, and ~90,000x cheaper
than the T1 call it can pre-empt. Scaling is sub-linear in message length (2x the words for
1.2x the time) because the abstain check and normalisation are fixed cost.

### GOAT status after addendum 2

- **G1 correctness** — ✅ deterministic; hash pinned to reference vectors; bigram identity pinned.
- **G2 perf** — ✅ 2.16 µs typical, criterion-measured.
- **G3 no-regression** — ✅ 178 passed, 0 failed, clippy clean. `detect_remaining_work` untouched.
- **G4 alloc-free** — ✅ no heap allocation on the classification path.
- **G5 quality** — ⚠️ **the one gate still open.** Single-generator corpus, n=60, and no data at
  all below 24 words.

**G5 is now the only thing between this and a promotion decision**, and it needs real session
data rather than more synthetic messages. The union with the keyword matcher (keyword for short
and explicit, centroid for long and implicit) should be measured in the same pass — that union,
not this module alone, is what a Super-GOAT claim would rest on.
