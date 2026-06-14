# Plan 06: GLM-5.1 Empty-Stream Root Cause Fix

## Status

- [x] Task 1: Investigate orchestration streaming/retry code path
- [x] Task 2: Confirm root cause (timeout/empty-stream on large context → hard `Err` propagates, no recovery)
- [x] Task 3: Design fix (unify `Err` + synthetic-failure recovery)
- [x] Task 4: Implement — convert primary-call `Err` to synthetic failure in `decide_with_llm`
- [x] Task 5: Remove the now-dead outer `Err` arm
- [x] Task 6: Add unit tests for the new recovery-on-error path
- [x] Task 7: Validate — `cargo check`, `cargo clippy`, `cargo test`
- [x] Task 8: Commit with conventional message

---

## Problem

GLM-5.1 (default orchestration model) returns an empty stream when the orchestration context is large (observed at 73K+ tokens). Plan 005's compaction **mitigates** this by shrinking payloads before the LLM call (`compact_at` = 35k), but it does not fix the **root cause**: when the orchestration call *does* fail (timeout or empty stream), the auto-prompt chain dies instead of recovering.

Concretely, a single orchestration failure on a big context kills the whole autonomous chain — even when the last assistant message clearly describes remaining work and the plan has unchecked tasks.

## Root Cause (confirmed from source — not a hypothesis)

Trace through `decide_with_llm` → `routing::route_and_call` → `call_language_model`:

1. `call_language_model` (`auto_prompt.rs:1983`) wraps the *entire* stream lifecycle in one `future::select(completion_future, timer(60s))` (line 2144). On a large prompt, GLM-5.1's time-to-first-token can exceed 60s → the timer fires → the function `bail!("...timed out after 60 seconds")` and returns **`Err`**.
2. A stream that completes with zero Text + zero errors returns **`Ok(synthetic stop, confidence 0.0)`** instead.
3. Back in `decide_with_llm`, these two equivalent failures are handled **asymmetrically**:
   - **Synthetic `Ok`** (empty Text): `is_synthetic_failure` is detected (line 864: `confidence <= Some(0.3) && reason.starts_with("model")`) → enters the recovery block (line 1109) → **3x lightweight retry with reduced context** + `detect_remaining_work` safety net + `detect_remaining_plan_tasks` fallback. **Recovers.**
   - **`Err`** (timeout / hard streaming error): falls to the outer `Err(err)` arm (line 1446) → `write_error_log` + `Err(err)` propagated. **No retry. No safety net. Chain dies.**

The 73K+ empty-stream bug is the **`Err` path** — the timeout. The exact same underlying problem ("model couldn't produce output on this payload") is recovered gracefully on the `Ok` path but fatally on the `Err` path.

## Proposed Fix (implemented)

Unify the two failure paths: convert the primary call's `Err` into a synthetic-failure response at the **top** of `decide_with_llm`, so the existing recovery machinery handles it unchanged. This is minimal (one localized change + dead-code removal), DRY (reuses the 300-line recovery block), and symmetric (both failure modes now recover identically).

### Changes

**`crates/auto_prompt/src/auto_prompt.rs`**

1. **Top of `decide_with_llm`** (after `route_and_call`): match the result; on `Err`, synthesize an `AutoPromptResponse` with:
   - `should_continue: false`
   - `confidence: 0.0`  (≤ 0.3 → triggers `is_synthetic_failure`)
   - `reason: "model orchestration call failed (timeout/empty stream): {err:#}"`  (starts with `"model"` → triggers `is_synthetic_failure`)
   - `next_prompt: None`, `all_plan_done: false`, `thread_summary: None`, `handover: None`
   - Call `write_error_log(...)` first so the original error is still recorded for diagnostics.
   - Then proceed with the existing `Ok((raw, response))` logic unchanged.

2. **Remove the now-unreachable outer `Err(err)` arm** (old line 1446) — the error is converted at the top, so this branch can never fire.

### Why this works

The synthesized response satisfies both halves of the `is_synthetic_failure` predicate (confidence 0.0 ≤ 0.3, reason starts with "model"), so `evaluate_response` returns `WantsStop` with `is_synthetic_failure=true`, which routes into the **existing** recovery block:
- 3x lightweight retry against the cloud model using *reduced* context (last assistant message + incomplete plan names only) — small payload, so it succeeds where the full-context call timed out.
- If all retries fail, `detect_remaining_work` extracts actionable work from the last assistant message.
- Else `detect_remaining_plan_tasks` continues the next unchecked plan task.
- Only if all safety nets are exhausted does the chain stop (gracefully, with a reason).

No new recovery code is written — the bug was that the `Err` path bypassed the recovery that the `Ok` path already had.

### Non-goals (deliberately out of scope)

- **Not** raising the 60s timeout — a too-large context will always be slow; the right answer is plan 005's compaction (prevention) + this fix (recovery), not waiting longer.
- **Not** switching off GLM-5.1 — it's the configured orchestration model; the fix is model-agnostic.
- **Not** touching `routing.rs` — local-tier routing already escalates correctly; the gap is only in `decide_with_llm`'s `Err` handling.

## Validation

- `cargo check -p auto_prompt` — clean
- `cargo clippy -p auto_prompt` — no warnings
- `cargo test -p auto_prompt --lib` — all pass, including new unit tests for the `Err`-to-recovery path

## Refs

- Predecessor: plan 005 (context compaction) — mitigates payload size; this plan fixes recovery on the residual failures.
- Commit: (filled at commit time)
- Key code: `auto_prompt.rs:1983` (`call_language_model` + 60s timeout), `auto_prompt.rs:864` (`is_synthetic_failure`), `auto_prompt.rs:1109` (recovery block), `auto_prompt.rs:1446` (old dead `Err` arm).
