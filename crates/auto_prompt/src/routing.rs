//! Tiered routing for auto_prompt orchestration calls.
//!
//! Wraps the single `call_language_model` site in `decide_with_llm`. When local
//! routing is disabled (or the config field is absent), routes straight to the
//! existing cloud path — byte-identical behavior to pre-change. When enabled,
//! classifies the current decision and routes to a local mlx_lm.server tier
//! (T1=Llama-1B for routine, T2=Gemma-4-E4B for judgment), escalating to cloud
//! on low confidence, server-down, or critical decision class.
//!
//! Everything downstream of the LLM call (parsing, the 14 rules via
//! `evaluate_response`, logging) is reused unchanged — this module only picks
//! *which model to invoke*.

use anyhow::Result;
use gpui::AsyncApp;

use crate::config::{AutoPromptConfig, LocalRoutingConfig, OrchestrationProvider, TierConfig};
use crate::context::AutoPromptResponse;
use crate::local_mlx;
use crate::{LlmCallData, call_language_model};

/// Coarse classification of the current orchestration decision.
/// Drives tier selection. Computed by cheap string inspection on the context —
/// no extra LLM call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionClass {
    /// Routine "should I continue?" check → T1 (Llama-1B).
    Routine,
    /// Judgment-heavy (pre-stop verification, thread summary) → T2 (Gemma-4-E4B).
    Judgment,
    /// High-stakes (final iteration) → cloud, never local.
    Critical,
}

/// Which tier to try.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    T1,
    T2,
}

/// The single entry point. Called from `decide_with_llm` instead of
/// `call_language_model` directly. Returns the same shape so the rest of
/// `decide_with_llm` is unchanged.
pub async fn route_and_call(
    data: &LlmCallData,
    cx: &AsyncApp,
) -> Result<(String, AutoPromptResponse)> {
    let config = match crate::load_config_cached() {
        Ok(c) => c,
        Err(err) => {
            log::warn!("[auto_prompt::routing] config load failed: {err} — using cloud");
            return cloud_call(data, cx).await;
        }
    };

    match config.orchestration_provider {
        OrchestrationProvider::Cloud => cloud_call(data, cx).await,
        OrchestrationProvider::Auto | OrchestrationProvider::LocalOnly => {
            let local = match config.local_routing.as_ref() {
                Some(lr) if lr.enabled => lr,
                _ => {
                    log::info!(
                        "[auto_prompt::routing] local_routing not enabled — using cloud (provider={:?})",
                        config.orchestration_provider
                    );
                    return cloud_call(data, cx).await;
                }
            };

            let class = classify_decision(data, &config);
            log::info!(
                "[auto_prompt::routing] provider={:?} decision_class={:?}",
                config.orchestration_provider,
                class
            );

            match class {
                DecisionClass::Critical => cloud_or_local_fail(data, &config, cx).await,
                DecisionClass::Judgment => {
                    route_with_escalation(data, local, &config, cx, Tier::T2).await
                }
                DecisionClass::Routine => {
                    route_with_escalation(data, local, &config, cx, Tier::T1).await
                }
            }
        }
    }
}

/// Try the starting tier, then escalate to the next tier, then cloud.
/// T1 → T2 → cloud for Routine; T2 → cloud for Judgment.
async fn route_with_escalation(
    data: &LlmCallData,
    local: &LocalRoutingConfig,
    config: &AutoPromptConfig,
    cx: &AsyncApp,
    start: Tier,
) -> Result<(String, AutoPromptResponse)> {
    let tiers = match start {
        Tier::T1 => [Some(Tier::T1), Some(Tier::T2)],
        Tier::T2 => [Some(Tier::T2), None],
    };

    for tier in tiers.into_iter().flatten() {
        match try_tier(data, local, config, tier).await? {
            Some(resp) => {
                log::info!("[auto_prompt::routing] tier {tier:?} produced usable verdict");
                return Ok(resp);
            }
            None => {
                log::info!(
                    "[auto_prompt::routing] tier {tier:?} declined (low conf/server down) — escalating"
                );
            }
        }
    }
    cloud_or_local_fail(data, config, cx).await
}

/// Try a single tier. Returns None if the tier should be skipped (server down)
/// or its verdict was below the confidence threshold.
async fn try_tier(
    data: &LlmCallData,
    local: &LocalRoutingConfig,
    config: &AutoPromptConfig,
    tier: Tier,
) -> Result<Option<(String, AutoPromptResponse)>> {
    let tier_cfg: &TierConfig = match tier {
        Tier::T1 => &local.t1,
        Tier::T2 => &local.t2,
    };

    // Health check — skip tier if server is down.
    if !local_mlx::is_alive(&tier_cfg.endpoint, local.health_check_ms).await {
        log::info!(
            "[auto_prompt::routing] tier {tier:?} endpoint {} not responding — skipping",
            tier_cfg.endpoint
        );
        return Ok(None);
    }

    // Call the local endpoint.
    let outcome = local_mlx::call(
        &tier_cfg.endpoint,
        &tier_cfg.model,
        &data.system_prompt,
        &data.context_json,
        tier_cfg.timeout_ms,
    )
    .await;

    let (raw, response) = match outcome {
        Ok(pair) => pair,
        Err(err) => {
            log::warn!("[auto_prompt::routing] tier {tier:?} call failed: {err:#} — escalating");
            return Ok(None);
        }
    };

    let conf = response.confidence.unwrap_or(0.0);
    if conf >= tier_cfg.confidence_threshold {
        log_verdict(config, tier, conf, &data.system_prompt, false);
        Ok(Some((raw, response)))
    } else {
        log::info!(
            "[auto_prompt::routing] tier {tier:?} confidence {conf:.3} < threshold {:.3} — escalating",
            tier_cfg.confidence_threshold
        );
        log_verdict(config, tier, conf, &data.system_prompt, true);
        Ok(None)
    }
}

/// Cloud path — the existing call_language_model. When `LocalOnly` and cloud
/// fallback is disabled, synthesize a stop instead of calling cloud.
async fn cloud_or_local_fail(
    data: &LlmCallData,
    config: &AutoPromptConfig,
    cx: &AsyncApp,
) -> Result<(String, AutoPromptResponse)> {
    if matches!(
        config.orchestration_provider,
        OrchestrationProvider::LocalOnly
    ) && !cloud_fallback_enabled(&config.local_routing)
    {
        log::warn!(
            "[auto_prompt::routing] LocalOnly + cloud fallback disabled — failing closed to Stop"
        );
        return Ok(synthetic_stop(
            "local-only mode: all local tiers exhausted, cloud fallback disabled",
        ));
    }
    log::info!("[auto_prompt::routing] escalating to cloud");
    cloud_call(data, cx).await
}

/// Direct cloud call — identical to pre-change behavior.
async fn cloud_call(data: &LlmCallData, cx: &AsyncApp) -> Result<(String, AutoPromptResponse)> {
    call_language_model(
        &data.model,
        &data.system_prompt,
        &data.context_json,
        &data.call_timeouts,
        cx,
    )
    .await
}

fn cloud_fallback_enabled(local: &Option<LocalRoutingConfig>) -> bool {
    local
        .as_ref()
        .map(|lr| lr.cloud_fallback.enabled)
        .unwrap_or(true)
}

/// Build a synthetic stop response (confidence 0, should_continue=false).
/// Used when LocalOnly mode exhausts local tiers and can't fall back to cloud.
fn synthetic_stop(reason: &str) -> (String, AutoPromptResponse) {
    let response = AutoPromptResponse {
        should_continue: false,
        next_prompt: None,
        reason: Some(reason.to_string()),
        all_plan_done: false,
        confidence: Some(0.0),
        thread_summary: None,
        handover: None,
    };
    let raw = serde_json::to_string(&response).unwrap_or_else(|_| "{}".into());
    (raw, response)
}

/// Classify the current decision from context inspection. No LLM call.
pub fn classify_decision(data: &LlmCallData, config: &AutoPromptConfig) -> DecisionClass {
    classify_from_fields(
        data.iteration_count,
        config.max_iterations,
        &data.stop_phase,
        &data.context_json,
    )
}

/// Pure classification logic, separated from LlmCallData so it is unit-testable
/// without constructing an Arc<dyn LanguageModel>.
pub fn classify_from_fields(
    iteration_count: u32,
    max_iterations: u32,
    stop_phase: &crate::context::StopPhase,
    context_json: &str,
) -> DecisionClass {
    // Final iteration is high-stakes — always cloud.
    if iteration_count >= max_iterations.saturating_sub(1) {
        return DecisionClass::Critical;
    }
    // Pre-stop verification is quality-sensitive — T2 (reasoning).
    if *stop_phase == crate::context::StopPhase::PreStop {
        return DecisionClass::Judgment;
    }
    // New-thread summary generation (compression) is quality-sensitive — T2.
    if context_json.contains("\"was_truncated\": true") {
        return DecisionClass::Judgment;
    }
    // Default: most "should I continue?" checks are routine binary verdicts — T1.
    DecisionClass::Routine
}

/// Append a JSON-lines verdict record when verdict_log_path is configured.
/// Best-effort: log-and-continue on I/O errors so logging never breaks routing.
fn log_verdict(
    config: &AutoPromptConfig,
    tier: Tier,
    confidence: f64,
    _system_prompt: &str,
    escalated: bool,
) {
    use std::io::Write;
    let Some(path) = config.verdict_log_path.as_ref() else {
        return;
    };
    let record = serde_json::json!({
        "ts": chrono::Local::now().to_rfc3339(),
        "tier": format!("{tier:?}"),
        "confidence": confidence,
        "escalated": escalated,
    });
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let line = format!("{}\n", record);
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(mut f) => {
            let _ = f.write_all(line.as_bytes());
        }
        Err(err) => {
            log::warn!("[auto_prompt::routing] verdict log write failed: {err}");
        }
    }
}
