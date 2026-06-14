# Plan 009 — Duplicate JSON Keys Recovery

## Problem

GLM-5.1 sometimes returns JSON with duplicate object keys, e.g.:

```json
{"thread_summary": "Plan 257 fully implemented.", "confidence": 0.75, "thread_summary": null}
```

`serde_json` strictly rejects duplicate keys when deserializing into a struct
(`Error: duplicate field `thread_summary``). The fork's `parse_response`
handles this parse failure by synthesizing a **stop** response:

```rust
AutoPromptResponse { should_continue: false, confidence: Some(0.0),
    reason: Some("unparseable response ..."), ... }
```

**This halts the auto-prompt chain prematurely with no recovery.** Plan 006's
recovery does NOT catch it because `is_synthetic_failure` requires the reason
to start with `"model"` (the parse-error reason starts with `"unparseable"`).

## Root Cause

`serde_json::from_str::<AutoPromptResponse>` → `Err("duplicate field ...")` →
synthetic stop → `is_synthetic_failure = false` (reason prefix mismatch) →
chain death.

## Fix

When `serde_json` fails with a `"duplicate field"` error, rebuild the JSON
object keeping only the **first occurrence** of each key, then re-parse. Only
if dedup-recovery also fails do we fall back to the existing synthetic stop.

First-occurrence semantics is intentional: GLM-5.1 emits the real value first
(`"thread_summary": "..."`) then a trailing redundant null
(`"thread_summary": null`). Keeping the first preserves the real value.

## Approach

Manual byte-level scan of the top-level JSON object that:
- Tracks string state (escaped quotes) to avoid false splits inside strings.
- Tracks brace/bracket depth so commas inside nested objects/arrays don't
  fool the top-level entry splitter.
- Uses `serde_json::from_str` to resolve escape sequences in keys.
- Drops duplicates with a debug log; keeps first occurrence.

Reference: katopz commit `b928dabaee` — same idea, re-implemented in the
fork's style with a cleaner structure (dedicated `skip_whitespace`,
`scan_json_string`, `scan_json_value` helpers) and without the operator-
precedence bug in katopz's whitespace-skip loop.

## Tasks

- [x] Create feature branch `feature/009_duplicate_json_keys_recovery`
- [x] Add `deduplicate_and_parse` + `rebuild_deduplicated_json` + scan helpers
- [x] Wire dedup-retry into `parse_response`'s `Err` arm (before synthetic stop)
- [x] Add hermetic unit tests (dedup logic + parse_response end-to-end)
- [x] `cargo check -p auto_prompt` clean
- [x] `cargo clippy -p auto_prompt --tests` clean
- [x] `cargo test -p auto_prompt` all green (no regressions)
- [x] Commit: `fix(auto_prompt): recover from duplicate JSON keys — plan 009`

## Validation

- `cargo check -p auto_prompt`: clean (exit 0)
- `cargo clippy -p auto_prompt --tests`: clean, zero warnings in auto_prompt code
- `cargo test -p auto_prompt`: **215 passed**, 0 failed, 3 ignored
  - 7 new dedup tests + 2 new parse_response tests = 9 new (lib: 155 -> 164)
- No regressions in the pre-existing 206 tests

## Risk

Low. The dedup path is only triggered on `"duplicate field"` errors — normal
responses are unaffected. If dedup itself fails, behavior is unchanged
(existing synthetic stop). The scan helpers are pure functions with no I/O.
