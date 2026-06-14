//! Tests for the tiered routing classifier and config parsing.
//!
//! These are pure unit tests — no network, no real LanguageModel, no mlx_lm.server.

use auto_prompt::{
    AutoPromptConfig, CloudFallbackConfig, LocalRoutingConfig, OrchestrationProvider, TierConfig,
};
use auto_prompt::context::StopPhase;
use auto_prompt::routing::{classify_from_fields, DecisionClass};

// ── classify_from_fields ─────────────────────────────────────────────────────

#[test]
fn classify_routine_by_default() {
    let class = classify_from_fields(1, 20, &StopPhase::Working, "{}");
    assert_eq!(class, DecisionClass::Routine);
}

#[test]
fn classify_critical_on_final_iteration() {
    // iteration 19, max 20 → 19 >= 20-1 → Critical
    let class = classify_from_fields(19, 20, &StopPhase::Working, "{}");
    assert_eq!(class, DecisionClass::Critical);
}

#[test]
fn classify_critical_on_beyond_final_iteration() {
    let class = classify_from_fields(25, 20, &StopPhase::Working, "{}");
    assert_eq!(class, DecisionClass::Critical);
}

#[test]
fn classify_judgment_on_pre_stop() {
    let class = classify_from_fields(5, 20, &StopPhase::PreStop, "{}");
    assert_eq!(class, DecisionClass::Judgment);
}

#[test]
fn classify_judgment_on_truncated_context() {
    let ctx = r#"{"was_truncated": true, "messages": []}"#;
    let class = classify_from_fields(5, 20, &StopPhase::Working, ctx);
    assert_eq!(class, DecisionClass::Judgment);
}

#[test]
fn classify_routine_when_not_truncated() {
    let ctx = r#"{"was_truncated": false}"#;
    let class = classify_from_fields(5, 20, &StopPhase::Working, ctx);
    assert_eq!(class, DecisionClass::Routine);
}

#[test]
fn classify_pre_stop_beats_truncated_check() {
    // Pre-stop check comes first in the classifier.
    let class = classify_from_fields(5, 20, &StopPhase::PreStop, r#"{"was_truncated": true}"#);
    assert_eq!(class, DecisionClass::Judgment);
}

#[test]
fn classify_critical_beats_pre_stop() {
    // Critical check (final iteration) comes first.
    let class = classify_from_fields(19, 20, &StopPhase::PreStop, "{}");
    assert_eq!(class, DecisionClass::Critical);
}

#[test]
fn classify_handles_zero_max_iterations() {
    // Edge case: max_iterations=0 → saturating_sub gives 0 → any iteration >= 0 → Critical.
    let class = classify_from_fields(0, 0, &StopPhase::Working, "{}");
    assert_eq!(class, DecisionClass::Critical);
}

// ── Config parsing (serde defaults, backward-compat) ─────────────────────────

#[test]
fn config_parses_minimal_json_without_new_fields() {
    // An old config file without any tiered-routing fields should parse fine.
    let json = r#"{
        "max_iterations": 15,
        "max_context_tokens": 40000,
        "backoff_base_ms": 1000,
        "same_thread_token_threshold": 30000
    }"#;
    let config: AutoPromptConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.max_iterations, 15);
    assert_eq!(config.orchestration_provider, OrchestrationProvider::Cloud);
    assert!(config.local_routing.is_none());
    assert!(config.verdict_log_path.is_none());
}

#[test]
fn config_parses_empty_object() {
    let json = r#"{}"#;
    let config: AutoPromptConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.orchestration_provider, OrchestrationProvider::Cloud);
    assert!(config.local_routing.is_none());
}

#[test]
fn config_parses_full_tiered_config() {
    let json = r#"{
        "orchestration_provider": "auto",
        "local_routing": {
            "enabled": true,
            "health_check_ms": 750,
            "t1": {
                "endpoint": "http://127.0.0.1:9001/v1",
                "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "confidence_threshold": 0.9,
                "timeout_ms": 3000
            },
            "t2": {
                "endpoint": "http://127.0.0.1:9002/v1",
                "model": "mlx-community/gemma-4-E4B-it-qat-4bit",
                "confidence_threshold": 0.65,
                "timeout_ms": 20000
            },
            "cloud_fallback": {
                "enabled": true,
                "on": ["critical", "low_confidence"]
            }
        },
        "verdict_log_path": "/tmp/auto_prompt_verdicts.jsonl"
    }"#;
    let config: AutoPromptConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.orchestration_provider, OrchestrationProvider::Auto);

    let local = config.local_routing.expect("local_routing should be set");
    assert!(local.enabled);
    assert_eq!(local.health_check_ms, 750);
    assert_eq!(local.t1.endpoint, "http://127.0.0.1:9001/v1");
    assert!((local.t1.confidence_threshold - 0.9).abs() < 1e-9);
    assert_eq!(local.t2.model, "mlx-community/gemma-4-E4B-it-qat-4bit");
    assert!((local.t2.confidence_threshold - 0.65).abs() < 1e-9);
    assert_eq!(local.cloud_fallback.on.len(), 2);
    assert_eq!(
        config.verdict_log_path,
        Some(std::path::PathBuf::from("/tmp/auto_prompt_verdicts.jsonl"))
    );
}

#[test]
fn config_local_only_provider() {
    let json = r#"{"orchestration_provider": "local_only"}"#;
    let config: AutoPromptConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.orchestration_provider, OrchestrationProvider::LocalOnly);
}

#[test]
fn config_local_routing_disabled_by_default() {
    // Even with orchestration_provider=auto, local_routing defaults to None.
    let json = r#"{"orchestration_provider": "auto"}"#;
    let config: AutoPromptConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.orchestration_provider, OrchestrationProvider::Auto);
    assert!(config.local_routing.is_none());
}

#[test]
fn config_default_construction() {
    let config = AutoPromptConfig::default();
    assert_eq!(config.orchestration_provider, OrchestrationProvider::Cloud);
    assert!(config.local_routing.is_none());
    assert_eq!(config.max_iterations, 20);
}

#[test]
fn local_routing_default_has_sensible_tier_values() {
    let local = LocalRoutingConfig::default();
    assert!(!local.enabled, "should default to disabled");
    assert!((local.t1.confidence_threshold - 0.85).abs() < 1e-9);
    assert!((local.t2.confidence_threshold - 0.70).abs() < 1e-9);
    assert!(local.t1.timeout_ms < local.t2.timeout_ms, "T1 should be faster");
    assert!(local.cloud_fallback.enabled);
}

#[test]
fn cloud_fallback_default_triggers() {
    let cf = CloudFallbackConfig::default();
    assert!(cf.enabled);
    assert!(cf.on.iter().any(|t| t == "critical"));
    assert!(cf.on.iter().any(|t| t == "low_confidence"));
}

#[test]
fn tier_config_serde_uses_defaults_for_omitted_fields() {
    let json = r#"{
        "endpoint": "http://localhost:9999/v1",
        "model": "test-model"
    }"#;
    let tier: TierConfig = serde_json::from_str(json).unwrap();
    assert_eq!(tier.endpoint, "http://localhost:9999/v1");
    assert_eq!(tier.model, "test-model");
    assert!((tier.confidence_threshold - 0.75).abs() < 1e-9, "should use serde default");
    assert_eq!(tier.timeout_ms, 10_000);
}
