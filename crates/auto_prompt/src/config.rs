use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for the auto-prompt hook.
///
/// Loaded from `~/.config/zed/auto_prompt.json` or environment variables.
/// The LLM used is whatever Zed has configured as the default model.
///
/// Enable/disable is controlled by the UI toggle in the agent panel toolbar,
/// not by this config file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoPromptConfig {
    /// Optional system prompt to use when calling the LLM.
    /// Defaults to a built-in prompt that instructs the model to return JSON.
    #[serde(default)]
    pub system_prompt: Option<String>,

    /// Maximum number of auto-prompt iterations before hard-stopping the loop.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,

    /// Token count threshold (approximate) at which context is considered too large
    /// and the system forces a "continue" prompt instead of asking the LLM.
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,

    /// Base delay in milliseconds for exponential backoff on errors.
    /// Actual delay = backoff_base_ms * 2^retry_count (capped at 60s).
    #[serde(default = "default_backoff_base_ms")]
    pub backoff_base_ms: u64,

    /// Maximum number of pre-stop verification attempts before forcing a stop.
    /// When the LLM says stop, we verify (plans done, diagnostics clean, git committed).
    /// If verification fails, we retry up to this many times before forcing stop.
    #[serde(default = "default_max_verification_attempts")]
    pub max_verification_attempts: u32,

    /// Maximum number of automatic retry attempts for LLM orchestration call failures.
    /// When the auto-prompt's own LLM call fails (network/timeout/parse), it will
    /// retry with exponential backoff up to this many times before showing "Retry" button.
    #[serde(default = "default_max_llm_retries")]
    pub max_llm_retries: u32,

    /// Token count threshold below which auto-prompt continues in the same thread
    /// instead of creating a new thread with summary. When the conversation's
    /// approximate token count is below this value, the next_prompt is injected
    /// as a user message in the current thread, preserving full context.
    #[serde(default = "default_same_thread_token_threshold")]
    pub same_thread_token_threshold: usize,

    /// Which provider the orchestrator uses for the "should I continue?" decision.
    /// Default: Cloud (current behavior — uses Zed's configured default model).
    /// Set to "auto" for tiered local-LLM routing (T1 → T2 → cloud fallback).
    /// Set to "local_only" to never call cloud (offline mode).
    #[serde(default)]
    pub orchestration_provider: OrchestrationProvider,

    /// Tiered local-LLM routing config. None (or absent) = local routing disabled.
    /// Only consulted when `orchestration_provider` is not `Cloud`.
    #[serde(default)]
    pub local_routing: Option<LocalRoutingConfig>,

    /// Optional path to append a JSON-lines verdict log for each orchestration call.
    /// Used for Phase 2 calibration of confidence thresholds. None = no logging.
    #[serde(default)]
    pub verdict_log_path: Option<PathBuf>,
}

// ── Tiered routing types ─────────────────────────────────────────────────────

/// Which model provider the orchestrator consults for the continuation decision.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OrchestrationProvider {
    /// Always use Zed's configured default model (current behavior).
    #[default]
    Cloud,
    /// Never call cloud. On local failure, fails closed to Stop.
    LocalOnly,
    /// Tiered: local T1 → local T2 → cloud based on decision class + confidence.
    Auto,
}

/// Configuration for a single local MLX tier (OpenAI-compatible endpoint).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TierConfig {
    /// Endpoint base URL, e.g. "http://127.0.0.1:8081/v1".
    pub endpoint: String,
    /// Model identifier served by `mlx_lm.server` on this endpoint.
    pub model: String,
    /// Minimum confidence (0.0–1.0) required to trust this tier's verdict.
    /// Below this, the router escalates to the next tier.
    #[serde(default = "default_tier_confidence")]
    pub confidence_threshold: f64,
    /// HTTP timeout in milliseconds for a single orchestration call.
    #[serde(default = "default_tier_timeout_ms")]
    pub timeout_ms: u64,
}

/// When to escalate from local tiers back to the cloud provider.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloudFallbackConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Triggers: "critical" (high-stakes decision), "low_confidence",
    /// "server_down", "parse_error".
    #[serde(default = "default_cloud_fallback_triggers")]
    pub on: Vec<String>,
}

/// Tiered local-LLM routing configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalRoutingConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Timeout for health-check probes (GET /v1/models) on tier endpoints.
    #[serde(default = "default_health_check_ms")]
    pub health_check_ms: u64,
    /// Tier 1 — speed (e.g. Llama-3.2-1B). Used for routine decisions.
    pub t1: TierConfig,
    /// Tier 2 — reasoning (e.g. Gemma-4-E4B). Used for judgment decisions.
    pub t2: TierConfig,
    #[serde(default)]
    pub cloud_fallback: CloudFallbackConfig,
}

impl Default for LocalRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            health_check_ms: default_health_check_ms(),
            t1: TierConfig {
                endpoint: "http://127.0.0.1:8081/v1".into(),
                model: "mlx-community/Llama-3.2-1B-Instruct-4bit".into(),
                confidence_threshold: 0.85,
                timeout_ms: 4_000,
            },
            t2: TierConfig {
                endpoint: "http://127.0.0.1:8082/v1".into(),
                model: "mlx-community/gemma-4-E4B-it-qat-4bit".into(),
                confidence_threshold: 0.70,
                timeout_ms: 15_000,
            },
            cloud_fallback: CloudFallbackConfig::default(),
        }
    }
}

impl Default for CloudFallbackConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            on: default_cloud_fallback_triggers(),
        }
    }
}

// ── serde default fns ────────────────────────────────────────────────────────

fn default_true() -> bool {
    true
}

fn default_max_iterations() -> u32 {
    20
}

fn default_max_context_tokens() -> usize {
    80_000
}

fn default_backoff_base_ms() -> u64 {
    2_000
}

fn default_max_verification_attempts() -> u32 {
    2
}

fn default_max_llm_retries() -> u32 {
    3
}

fn default_same_thread_token_threshold() -> usize {
    50_000
}

fn default_tier_confidence() -> f64 {
    0.75
}

fn default_tier_timeout_ms() -> u64 {
    10_000
}

fn default_health_check_ms() -> u64 {
    500
}

fn default_cloud_fallback_triggers() -> Vec<String> {
    vec![
        "critical".into(),
        "low_confidence".into(),
        "server_down".into(),
        "parse_error".into(),
    ]
}

impl Default for AutoPromptConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            max_iterations: default_max_iterations(),
            max_context_tokens: default_max_context_tokens(),
            backoff_base_ms: default_backoff_base_ms(),
            max_verification_attempts: default_max_verification_attempts(),
            max_llm_retries: default_max_llm_retries(),
            same_thread_token_threshold: default_same_thread_token_threshold(),
            orchestration_provider: OrchestrationProvider::default(),
            local_routing: None,
            verdict_log_path: None,
        }
    }
}

impl AutoPromptConfig {
    /// Returns the path to the config file: `~/.config/zed/auto_prompt.json`
    pub fn config_path() -> Result<PathBuf> {
        let config_dir = paths::config_dir();
        Ok(config_dir.join("auto_prompt.json"))
    }

    /// Load config from file, falling back to environment variables.
    pub fn load() -> Result<Self> {
        log::info!("[auto_prompt::config] Loading config...");
        let path = Self::config_path()?;
        log::info!("[auto_prompt::config] Config path: {:?}", path);

        if path.exists() {
            log::info!("[auto_prompt::config] Config file exists, loading from file");
            let content = std::fs::read_to_string(&path)?;
            let config: Self = serde_json::from_str(&content)?;
            log::info!(
                "[auto_prompt::config] Loaded from file: max_iterations={}, provider={:?}",
                config.max_iterations,
                config.orchestration_provider
            );
            return Ok(config);
        }

        log::info!(
            "[auto_prompt::config] Config file not found, loading from environment variables"
        );
        let config = Self::from_env();
        log::info!(
            "[auto_prompt::config] Loaded from env: max_iterations={}",
            config.max_iterations
        );
        Ok(config)
    }

    /// Build config from environment variables.
    fn from_env() -> Self {
        let system_prompt = std::env::var("ZED_AUTO_PROMPT_SYSTEM_PROMPT").ok();

        let max_iterations = std::env::var("ZED_AUTO_PROMPT_MAX_ITERATIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_max_iterations);

        let max_context_tokens = std::env::var("ZED_AUTO_PROMPT_MAX_CONTEXT_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_max_context_tokens);

        let backoff_base_ms = std::env::var("ZED_AUTO_PROMPT_BACKOFF_BASE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_backoff_base_ms);

        let max_verification_attempts = std::env::var("ZED_AUTO_PROMPT_MAX_VERIFICATION_ATTEMPTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_max_verification_attempts);

        let max_llm_retries = std::env::var("ZED_AUTO_PROMPT_MAX_LLM_RETRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_max_llm_retries);

        let same_thread_token_threshold =
            std::env::var("ZED_AUTO_PROMPT_SAME_THREAD_TOKEN_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(default_same_thread_token_threshold);

        let orchestration_provider = std::env::var("ZED_AUTO_PROMPT_ORCHESTRATION_PROVIDER")
            .ok()
            .and_then(|v| match v.to_ascii_lowercase().as_str() {
                "auto" => Some(OrchestrationProvider::Auto),
                "local_only" | "local-only" | "localonly" => {
                    Some(OrchestrationProvider::LocalOnly)
                }
                "cloud" => Some(OrchestrationProvider::Cloud),
                _ => None,
            })
            .unwrap_or_default();

        let verdict_log_path = std::env::var("ZED_AUTO_PROMPT_VERDICT_LOG_PATH")
            .ok()
            .map(PathBuf::from);

        Self {
            system_prompt,
            max_iterations,
            max_context_tokens,
            backoff_base_ms,
            max_verification_attempts,
            max_llm_retries,
            same_thread_token_threshold,
            orchestration_provider,
            local_routing: None,
            verdict_log_path,
        }
    }

    /// Calculate backoff delay for a given retry count.
    /// Capped at 60 seconds.
    pub fn backoff_delay_ms(&self, retry_count: u32) -> u64 {
        let capped_retry = retry_count.min(5);
        let delay = self.backoff_base_ms * 2u64.pow(capped_retry);
        delay.min(60_000)
    }

    /// Write current config to the config file.
    pub fn save(&self) -> Result<()> {
        let path = Self::config_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;

        // Invalidate cache so next load picks up the new config
        crate::invalidate_config_cache();

        Ok(())
    }
}
