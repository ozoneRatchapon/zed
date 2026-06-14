//! Tests for `CallTimeouts` (Plan 008) — the two-tier per-event streaming
//! timeout config that replaces the old monolithic 60s ceiling.
//!
//! These are pure, hermetic unit tests — no network, no LanguageModel, no async
//! runtime, no global-state mutation. The streaming-loop behavior itself (a timer
//! racing a real stream) is integration-tier and belongs in the e2e suite; here
//! we lock down the config surface: defaults, Duration accessors, serde
//! round-trip/partial-parse, and the tier-ordering invariant.
//!
//! Env-var overrides (`ZED_AUTO_PROMPT_CALL_TIMEOUT_*`) follow the established
//! `ZED_AUTO_PROMPT_*` loader pattern and are intentionally not exercised here
//! (global env mutation is non-hermetic under cargo's parallel test threads —
//! consistent with the rest of the crate's test suite).

use std::time::Duration;

use auto_prompt::{AutoPromptConfig, CallTimeouts};

// ── defaults ─────────────────────────────────────────────────────────────────

#[test]
fn call_timeouts_defaults_match_plan_008() -> anyhow::Result<()> {
    let t = CallTimeouts::default();
    assert_eq!(t.first_token_secs, 120, "TTFT window default must be 120s");
    assert_eq!(t.per_event_secs, 30, "per-event window default must be 30s");
    assert_eq!(t.total_secs, 300, "total backstop default must be 300s");
    Ok(())
}

#[test]
fn call_timeouts_duration_methods() -> anyhow::Result<()> {
    let t = CallTimeouts::default();
    assert_eq!(t.first_token(), Duration::from_secs(120));
    assert_eq!(t.per_event(), Duration::from_secs(30));
    assert_eq!(t.total(), Duration::from_secs(300));
    Ok(())
}

// ── serde ────────────────────────────────────────────────────────────────────

#[test]
fn call_timeouts_absent_in_json_uses_defaults() -> anyhow::Result<()> {
    // An old config file with no call_timeouts field must fall back to Plan 008
    // defaults via #[serde(default)].
    let json = r#"{}"#;
    let config: AutoPromptConfig = serde_json::from_str(json)?;
    assert_eq!(config.call_timeouts, CallTimeouts::default());
    Ok(())
}

#[test]
fn call_timeouts_parses_full_override() -> anyhow::Result<()> {
    let json = r#"{
        "call_timeouts": {
            "first_token_secs": 90,
            "per_event_secs": 45,
            "total_secs": 180
        }
    }"#;
    let config: AutoPromptConfig = serde_json::from_str(json)?;
    assert_eq!(config.call_timeouts.first_token_secs, 90);
    assert_eq!(config.call_timeouts.per_event_secs, 45);
    assert_eq!(config.call_timeouts.total_secs, 180);
    Ok(())
}

#[test]
fn call_timeouts_partial_override_fills_defaults() -> anyhow::Result<()> {
    // Only one field supplied — the rest must come from their serde defaults.
    let json = r#"{ "call_timeouts": { "per_event_secs": 60 } }"#;
    let config: AutoPromptConfig = serde_json::from_str(json)?;
    assert_eq!(config.call_timeouts.first_token_secs, 120, "unset field defaults");
    assert_eq!(config.call_timeouts.per_event_secs, 60, "supplied field honored");
    assert_eq!(config.call_timeouts.total_secs, 300, "unset field defaults");
    Ok(())
}

#[test]
fn call_timeouts_serde_round_trip() -> anyhow::Result<()> {
    let original = CallTimeouts {
        first_token_secs: 200,
        per_event_secs: 50,
        total_secs: 600,
    };
    let json = serde_json::to_string(&original)?;
    let parsed: CallTimeouts = serde_json::from_str(&json)?;
    assert_eq!(original, parsed);
    Ok(())
}

// ── tier-ordering invariant ──────────────────────────────────────────────────

#[test]
fn call_timeouts_defaults_satisfy_tier_invariant() -> anyhow::Result<()> {
    // The two-tier design relies on: first_token >= per_event (the prefill
    // window is at least as generous as the inter-token window) and total >
    // first_token (the backstop must outlive a single TTFT window). Locking
    // this in prevents a future default change from silently breaking the
    // "slow TTFT completes instead of failing" guarantee that is the whole
    // point of Plan 008.
    let t = CallTimeouts::default();
    assert!(
        t.first_token() >= t.per_event(),
        "TTFT window ({:?}) must be >= per-event window ({:?})",
        t.first_token(),
        t.per_event()
    );
    assert!(
        t.total() > t.first_token(),
        "total backstop ({:?}) must exceed TTFT window ({:?})",
        t.total(),
        t.first_token()
    );
    Ok(())
}
