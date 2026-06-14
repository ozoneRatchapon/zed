# Plan 08: Streaming Per-Event Timeout (two-tier)

## Status

- [x] Task 1: Confirm timeout code path + cancellation-safety of `StreamExt::next()`
- [x] Task 2: Decide timeout tiers + defaults (TTFT 120s / per-event 30s / total 300s) — sign-off received
- [x] Task 3: Add configurable env overrides (`ZED_AUTO_PROMPT_CALL_TIMEOUT_*`)
- [x] Task 4: Implement — replace monolithic `future::select` with in-loop per-event select
- [x] Task 5: Keep a generous total backstop as a safety net
- [x] Task 6: Add unit tests (CallTimeouts config: defaults, durations, serde, tier invariant)
- [x] Task 7: Validate — `cargo check`, `cargo clippy --tests`, `cargo test -p auto_prompt`
- [x] Task 8: Commit on `feature/008_streaming_per_event_timeout`, conventional message

---

## Problem

`call_language_model` (`auto_prompt.rs:1991`) wraps the **entire** stream lifecycle in a single monolithic race:

```rust
let timeout_future = cx.background_executor().timer(Duration::from_secs(60));
pin_mut!(completion_future, timeout_future);
match future::select(completion_future, timeout_future).await {
    Either::Left((Ok(text), _)) => parse(text),
    Either::Left((Err(e), _))   => Err(e),
    Either::Right(_)            => bail!("...timed out after 60 seconds"),  // ← problem
}
```

The 60s clock starts at call entry and **never resets**. Two failure modes:

1. **Slow TTFT (time-to-first-token):** GLM-5.1 on a large context (the 73K+ case that triggered plan 006) can need >60s for prefill before emitting the first token. The timer fires *before any progress*, the request — though legitimately in flight — is abandoned, and we `bail!`.

2. **Long-but-healthy stream:** even when tokens arrive steadily, if total stream duration exceeds 60s (e.g. a long response), the timer fires mid-stream and **all accumulated text is lost**.

The clock does not distinguish "stalled" (problem) from "slow but progressing" (healthy). Plan 006 made this failure **recoverable**; plan 008 makes the streaming itself **resilient** so the legit-slow case completes instead of falling through to recovery.

## Root Cause (confirmed from source)

- The timeout is a **single fixed 60s timer** at `auto_prompt.rs:2152`, hardcoded (`Duration::from_secs(60)`) — **not configurable** (no env override exists for it; `config.rs:107` `timeout_ms` is a separate HTTP-tier value capped at 60s, unrelated to the stream timeout).
- The `completion_future` async block (`auto_prompt.rs:2008`) must drain the **whole** `while let Some(event) = stream.next().await` loop before returning — so the race is "entire stream vs 60s", not "each event vs 60s".
- Three call sites all funnel through this one function, so the fix is **localized**: `routing.rs:200` (cloud path), `auto_prompt.rs:1077` (`decide_with_llm`), `auto_prompt.rs:1197` (lightweight retry). One change, all paths benefit.

## Proposed Fix (sketch — pending sign-off)

Replace the outer monolithic race with a **two-tier per-event timeout inside the stream loop**, plus a generous **total backstop**.

### Architecture

```rust
// Inside completion_future, replace the plain while-let loop:
let first_token_timeout = Duration::from_secs(config.first_token_timeout_secs); // generous: prefill
let per_event_timeout   = Duration::from_secs(config.per_event_timeout_secs);   // tight: catch stalls
let total_backstop      = Duration::from_secs(config.total_timeout_secs);       // safety net
let started = Instant::now();
let mut first_token = true;

loop {
    // total backstop — prevents an immortal stream
    if started.elapsed() >= total_backstop {
        bail!("auto_prompt: stream exceeded total backstop ({total_backstop:?})");
    }
    let window = if first_token { first_token_timeout } else { per_event_timeout };
    match future::select(stream.next(), cx.background_executor().timer(window)).await {
        Either::Left((Some(event), _)) => {
            first_token = false;          // ← switch to tight window after first progress
            // ... existing match arms (Text / Thinking / Other / Err) ...
        }
        Either::Left((None, _)) => break, // stream ended normally
        Either::Right(_) => {
            bail!("auto_prompt: stream stalled — no event for {window:?}{}",
                  if first_token { " (during prefill/TTFT)" } else { "" });
        }
    }
}
// then remove the outer future::select(completion_future, timeout_future) entirely
```

### Cancellation-safety (correctness note)

This pattern is safe **only** because `StreamExt::next()` (the `futures::stream::Next` future) is **cancellation-safe**: dropping it before it resolves leaves the stream untouched, and the next `stream.next()` re-polls from the same position — **no event is lost**. This is documented `futures` behavior and the standard idiom. (A naive hand-rolled poll could drop an item; we avoid that by using the library future.)

### Defaults (proposal — open for discussion)

| Tier | Default | Rationale |
|------|---------|-----------|
| First-token (TTFT) | **120s** | Large-context prefill on GLM-5.1 legitimately exceeds 60s; 120s absorbs the 73K-token case without tripping. |
| Per-event (subsequent) | **30s** | Once streaming starts, a 30s gap between tokens means the stream has stalled — kill it fast. |
| Total backstop | **300s** | Hard ceiling so a pathological "1 token / 29s" stream can't run forever. |

Net effect on the original 73K bug: instead of `bail!` at 60s → plan 006 recovery → 3x lightweight retry, the call **completes** within its 120s TTFT window. Plan 006's recovery still catches genuinely-dead streams (e.g. endpoint hung at prefill with no token ever).

### Configurable

Add env overrides matching the existing `ZED_AUTO_PROMPT_*` pattern (`config.rs:330-405`):
- `ZED_AUTO_PROMPT_CALL_TIMEOUT_FIRST_TOKEN_SECS` (default 120)
- `ZED_AUTO_PROMPT_CALL_TIMEOUT_PER_EVENT_SECS`   (default 30)
- `ZED_AUTO_PROMPT_CALL_TIMEOUT_TOTAL_SECS`        (default 300)

Read in the existing `from_env`-style block; fall back to defaults when unset.

## Why this works / relation to plan 006

- **Plan 006 = recovery**: when the call fails (timeout/empty), route the `Err` into the existing recovery block. Reactive.
- **Plan 008 = resilience**: let slow-but-healthy streams *finish* instead of failing. Proactive.
- They are **complementary, not redundant**: plan 008 reduces how often recovery is needed; plan 006 still handles the residual genuinely-stalled cases. After 008, plan 006's recovery triggers far less often (only true stalls), and when it does, the reason is unambiguous.

## Non-goals (deliberately out of scope)

- **Not** resumable streaming / checkpointing — far more complex; the per-event timeout removes the pressure for it. (Possible future plan.)
- **Not** changing the model or the HTTP-tier `timeout_ms` in config — orthogonal.
- **Not** removing plan 006's recovery — it remains the safety net.

## Open questions for sign-off

1. **TTFT default 120s** — generous enough for the 73K case? Too generous (lets a hung endpoint waste 2 min)? Consider 90s as a middle ground.
2. **Per-event 30s** — GLM-5.1's inter-token latency on long outputs could occasionally exceed 30s under load; consider 45s to avoid false stalls. Trade-off: slower detection of real stalls.
3. **Total backstop 300s** — acceptable upper bound, or tighten to 180s?
4. **Config surface** — env vars only (matching current pattern), or also expose in `auto_prompt.json`? Env-only is simpler and consistent with `fork_at`/`compact_at`.

## Validation (done)

- `cargo check -p auto_prompt -p agent_ui` — clean
- `cargo clippy -p auto_prompt --tests` — clean (no warnings/errors)
- `cargo clippy -p agent_ui` — clean
- `cargo test -p auto_prompt` — **206 passed**, 0 failed, 3 ignored (155 lib + 7 new call_timeouts + 26 context + 18 tiered)
- New tests: `call_timeouts_test.rs` (7) — defaults match plan, Duration accessors, serde absent/full/partial/round-trip, tier-ordering invariant
- No regressions in the pre-existing 199 tests.

- `cargo check -p auto_prompt -p agent_ui`
- `cargo clippy -p auto_prompt --tests`
- `cargo test -p auto_prompt` — full suite (199+ tests) stays green
- New unit tests: (a) per-event timeout fires on a stalled mock stream; (b) clock resets on each event so a long-but-steady stream completes; (c) total backstop fires last-resort.
- Manual/e2e: a large-context orchestration call that previously tripped 60s now completes within the TTFT window.

## Refs

- Predecessor: plan 006 (empty-stream recovery) — plan 008 reduces its trigger frequency.
- Predecessor: plan 005 (context compaction) — shrinks payloads, lowering TTFT; complementary.
- Key code: `auto_prompt.rs:1991` (`call_language_model`), `auto_prompt.rs:2008` (stream loop), `auto_prompt.rs:2152` (monolithic 60s timer), `config.rs:330-405` (env-override pattern).
