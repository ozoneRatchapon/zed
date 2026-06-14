# Plan 05: Context Compaction and Structured Handover

## Status

- [x] Task 1: Add `compact_at` / `fork_at` config fields with env-var overrides
- [x] Task 2: Implement pure-Rust in-place compaction (`compaction.rs`)
- [x] Task 3: Implement `HandoverBlock` YAML serialization (`handover.rs`)
- [x] Task 4: Wire compaction into `decide()` before the LLM call
- [x] Task 5: Wire handover prepending into `decide_with_llm()` next_prompt
- [x] Task 6: Switch `dispatch_action()` fork ceiling to `fork_at.max(same_thread_token_threshold)`
- [x] Task 7: Fix infinite loop in `next_compactable_index` (idempotency guard)
- [x] Task 8: Unit tests — compaction 19, handover 11
- [x] Task 9: Validate — `cargo check` clean, `cargo test` 152 passed
- [x] Task 10: Commit with conventional message

---

## Problem

The orchestrator LLM hit timeouts on real auto-prompt chains. Root cause: the single 50k hard-fork limit let context balloon to **168K-token payloads** (measured on real threads) before anything forked. The orchestration model's request window blew past its timeout on these payloads.

Two failure modes in the old single-threshold model:

1. **Too late** — `same_thread_token_threshold` (50k) was the only ceiling, and the orchestrator often ran well past it before the fork kicked in.
2. **Context too rich to fork cleanly** — when fork finally triggered, the new thread started from a flat continuation string, losing structured state (what is done, in-progress, blocked).

## Design

Two-threshold model that shrinks context *before* it becomes un-handleable, and emits *structured* context when forking:

| Threshold | Default | Trigger | Action |
|-----------|---------|---------|--------|
| `compact_at` | 35,000 tokens | `approximate_token_count > compact_at` | Summarize old tool calls + assistant chunks **in-place** (pure Rust, no LLM call) |
| `fork_at` | 70,000 tokens | `actual_input_tokens >= fork_at` | Force thread fork; orchestrator emits structured `HandoverBlock` |

### Compaction (`compact_at`)

- **Tool results** → one-line metadata (status + truncated args)
- **Assistant chunks** → truncate to `ASSISTANT_TRUNCATE_BYTES` (200) + `[...]` sentinel
- **Recent messages preserved** via `keep_recent` policy (no compaction of the tail)
- Runs synchronously inside `decide()` before the LLM call, so the orchestrator never sees un-compacted payloads above the threshold

### Handover (`fork_at`)

When fork is imminent, the orchestrator LLM emits a `HandoverBlock`. Serialized as hand-rolled YAML wrapped in `<handover>...</handover>` fences:

- `original_intent`
- `completed:`
- `in_progress:`
- `blocked_on:`
- `files_touched:`
- `active_plans:`
- Empty fields omitted entirely (no `key: null` noise)

If the LLM omits a handover, a minimal fallback block is synthesized from `first_user_message` + last assistant message + plan landscape, so forks degrade gracefully.

### Legacy fallback

`same_thread_token_threshold` (default 50,000) is preserved unchanged. The actual fork ceiling is `fork_at.max(same_thread_token_threshold)`, so old configs that only set the legacy field behave byte-identically.

## New Modules

- **`crates/auto_prompt/src/compaction.rs`** — 591 lines, 19 unit tests. Pure-Rust in-place compaction (`next_compactable_index`, `truncate_assistant`, keep-recent policy).
- **`crates/auto_prompt/src/handover.rs`** — 293 lines, 11 unit tests. `HandoverBlock` YAML rendering + `prepend_handover()` + fallback synthesizer. (The `HandoverBlock` struct itself lives in `context.rs`.)

## Wiring

- **`crates/auto_prompt/src/auto_prompt.rs`**
  - `decide()`: runs compaction when `approximate_token_count > compact_at` (line ~705)
  - `decide_with_llm()`: sets `fork_imminent = ctx_tokens >= fork_at` (line ~727); if the orchestrator emitted a `HandoverBlock`, prepends it to `next_prompt` via `handover::prepend_handover()` (line ~842); else synthesizes a minimal continuation referencing the block
- **`crates/agent_ui/src/auto_prompt/mod.rs`**
  - `dispatch_action()`: fork ceiling is now `fork_at.max(same_thread_token_threshold)` (line ~209-215)
- **`crates/auto_prompt/src/config.rs`** — new fields with env-var overrides:
  - `compact_at` (`ZED_AUTO_PROMPT_COMPACTION_COMPACT_AT`, default 35,000)
  - `fork_at` (`ZED_AUTO_PROMPT_FORK_AT`, default 70,000)
  - `same_thread_token_threshold` preserved (default 50,000)

## Bug Fix (included)

**Infinite loop in `next_compactable_index`**: truncation produced content of length `ASSISTANT_TRUNCATE_BYTES + len(" [...]") = 205` bytes, which exceeded the 200-byte eligibility threshold, so the same assistant message was re-truncated on every pass — forever.

Fix: idempotency guard — skip assistant messages whose trimmed content already ends with the `[...]` sentinel.

```rust
// compaction.rs
if msg.content.trim_end().ends_with("[...]") {
    continue; // already truncated, don't re-process
}
```

## Validation

- `cargo check -p auto_prompt` — clean, no warnings
- `cargo check -p agent_ui` — clean
- `cargo test -p auto_prompt --lib` — **152 passed, 0 failed, 0.00s**
  - `compaction::` — 19/19
  - `handover::` — 11/11

## Refs

- Commit: `0b666f78e5` — `feat(auto_prompt): context compaction and structured handover — plan 005`
- Predecessor: plan 004 (tiered local-LLM routing) — `feature/004_tiered_local_llm`
