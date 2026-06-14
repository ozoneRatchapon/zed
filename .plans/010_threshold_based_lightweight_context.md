# Plan 010 — Threshold-Based Lightweight Orchestration Context

## Motivation

The primary continue/stop decision call (`decide_with_llm` → `call_language_model`)
sends the **full** serialized `AutoPromptContext` as the user message — up to ~80K
tokens on long threads (the plan-006 case was ~73K tokens, ~290K chars). That is
expensive, slow, and the main source of the timeout/overflow pressure that plans
005 (compaction), 006 (recovery), and 008 (per-event timeout) all work to mitigate.

Katopz's `lightweight_context.rs` takes the aggressive approach: **always** replace
the primary context with a ~500-token summary. The fork's current design deliberately
keeps full context for decision quality. This plan takes the middle ground.

## Current Behavior

- `decide()` serializes `AutoPromptContext` → `context_json` (full), packs it into
  `LlmCallData`.
- `decide_with_llm()` calls `call_language_model(model, system_prompt, &data.context_json, …)`,
  which puts `context_json` verbatim as the user message (`auto_prompt.rs:2006`).
- The **retry** path (~lines 1150–1408) already rebuilds a small `lightweight_ctx`
  via `build_lightweight_retry_context` (last 3 paragraphs + `build_plan_landscape`)
  and passes *that* to the same `call_language_model`. So the building blocks already
  exist — they are only used on retry, never on the primary call.

## Fix

Threshold-based hybrid: when `context_json.len()` exceeds a configurable threshold,
the **primary** call uses a lightweight context (reusing `build_lightweight_retry_context`)
instead of the full payload. Below the threshold, behavior is unchanged (full context).

This preserves full decision quality on normal-sized iterations and only trades
quality for cost/latency exactly when the context is large enough to cause the
problems plans 005/006/008 fight.

## Approach

1. Add a pure selector that decides full-vs-lightweight from `context_json.len()`
   and the threshold, returning the effective context string plus a mode flag.
2. Wire it into `decide_with_llm` just before the primary `call_language_model`
   call; pass the effective string to the call and reflect the mode in the
   decision log.
3. Make the threshold configurable (config field + env override, matching plan
   008's `ZED_AUTO_PROMPT_CALL_TIMEOUT_*_SECS` pattern). Default 120_000 chars
   (~30K tokens).
4. Reuse `build_lightweight_retry_context` for v1 (DRY). If decision quality
   suffers when lightweight engages, the improvement path is a richer dedicated
   builder (toward katopz's full-replacement format) — deferred per "try hybrid
   first."

## Tasks

- [x] Create feature branch `feature/010_threshold_based_lightweight_context`
- [x] Add config field `lightweight_context_threshold_chars` (default 120_000) + env
      override `ZED_AUTO_PROMPT_LIGHTWEIGHT_CONTEXT_THRESHOLD_CHARS`
- [x] Add `LlmCallData.lightweight_context_threshold_chars`, populate from config in `decide()`
- [x] Add pure helper `select_primary_context(context_json, last_assistant_message, title, threshold) -> PrimaryContext` (enum `{ Full, Lightweight(String) }`)
- [x] In `decide_with_llm`, select effective context before the primary call; pass it
      to `call_language_model`; reflect mode in the decision log + a clear `log::info!`
      when lightweight engages
- [x] Add hermetic unit tests (under-threshold → Full; over-threshold → Lightweight;
      boundary equality → Full; lightweight payload equals
      `build_lightweight_retry_context` output)
- [x] `cargo check -p auto_prompt` clean
- [x] `cargo clippy -p auto_prompt --tests` clean
- [x] `cargo test -p auto_prompt` all green (no regressions vs 215 baseline)
- [x] Commit: `feat(auto_prompt): threshold-based lightweight context for primary call — plan 010`

## Validation

- `cargo check -p auto_prompt`: clean
- `cargo clippy -p auto_prompt --tests`: clean
- `cargo test -p auto_prompt`: **220 passed** (was 215 → +5 new), 0 failed, 3 ignored
- Manual (optional): set `ZED_AUTO_PROMPT_LIGHTWEIGHT_CONTEXT_THRESHOLD_CHARS=1` to
  force lightweight on every call, run a real chain, confirm decisions still behave
  and the decision log shows the lightweight mode.

## Risk

Medium (behavioral change, bounded).

- **Decision quality when lightweight engages** — the core tradeoff. The retry-context
  format (last 3 paragraphs + plan landscape) is sparser than full context. Mitigation:
  conservative default threshold (~30K tokens), fully tunable/disable-able via env
  (set very high to effectively disable), clear logging when it engages.
- **Primary + retry both lightweight** — if a lightweight primary call synthesizes a
  failure, the retry path also uses lightweight → two sparse calls. Bounded by the
  existing `detect_remaining_work` safety net.
- **Decision-log fidelity** — must log the *effective* (sent) context / mode, not just
  the full `context_json`, or debugging will mismatch what the model actually saw.
- Normal iterations (context under threshold) are completely unaffected — the change
  only triggers on large contexts.
