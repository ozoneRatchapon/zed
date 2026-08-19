//! OpenAI-compatible HTTP client for local `mlx_lm.server` endpoints.
//!
//! This is a contained side-channel that bypasses Zed's `LanguageModel` trait
//! intentionally — registering a new provider would touch the language_model
//! crate, settings UI, and provider registry, which is out of scope for
//! tiered routing. Instead, we call the mlx_lm.server `/v1/chat/completions`
//! endpoint directly and feed the result through the existing `parse_response`,
//! producing the same `AutoPromptResponse` shape the cloud path produces.

use std::time::Duration;

use anyhow::{Context as _, Result};

use crate::context::AutoPromptResponse;

/// Health-check probe: GET /v1/models with a short timeout.
/// Returns false on any network error or non-2xx status.
pub async fn is_alive(endpoint: &str, timeout_ms: u64) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(timeout_ms))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let url = format!("{endpoint}/models");
    match client.get(&url).send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// Call a local mlx_lm.server endpoint with the same (system_prompt, context_json)
/// pair the cloud path uses. Returns the raw text plus the parsed AutoPromptResponse.
///
/// Reuses `crate::parse_response` so malformed local output synthesizes a
/// confidence-0 stop (same behavior as the cloud path), which naturally fails
/// the tier threshold check and escalates.
pub async fn call(
    endpoint: &str,
    model: &str,
    system_prompt: &str,
    context_json: &str,
    timeout_ms: u64,
) -> Result<(String, AutoPromptResponse)> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(timeout_ms))
        .build()
        .context("local_mlx: failed to build HTTP client")?;

    // workspace `reqwest` (zed-reqwest) has no `json` feature — serialize manually.
    //
    // `reasoning_effort: "none"` disables thinking on servers that honour it
    // (Ollama does; mlx_lm.server ignores unknown fields). Orchestration wants
    // a small JSON verdict, so chain-of-thought is pure latency here — measured
    // 0.6s -> 0.2s (qwen3:0.6b) and 1.6s -> 0.5s (gemma4:26b) on an M5 Pro.
    // It also removes a failure mode: with thinking ON and a capped response,
    // the budget is spent on `reasoning` and `content` comes back EMPTY, which
    // parse_response turns into a confidence-0 stop — escalating every call.
    let body = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user",   "content": context_json },
        ],
        "stream": false,
        "temperature": 0.0,
        "reasoning_effort": "none",
    });
    let body_bytes = serde_json::to_vec(&body).context("local_mlx: encode request body")?;

    let url = format!("{endpoint}/chat/completions");
    let start = std::time::Instant::now();
    let resp = client
        .post(&url)
        .header("content-type", "application/json")
        .body(body_bytes)
        .send()
        .await
        .with_context(|| format!("local_mlx: POST {url} failed"))?;

    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("local_mlx: {url} returned {status}");
    }

    let resp_bytes = resp
        .bytes()
        .await
        .context("local_mlx: read response body")?;
    let resp_json: serde_json::Value =
        serde_json::from_slice(&resp_bytes).context("local_mlx: parse response JSON")?;

    // Refuse a verdict the server reached on a truncated prompt.
    //
    // Ollama's OpenAI-compatible surface hard-caps the prompt at its default
    // num_ctx and drops the overflow SILENTLY — no error, no warning, and every
    // documented way of raising it is ignored on this endpoint (`options.num_ctx`,
    // top-level `num_ctx`, `context_length` all measured as no-ops; the prompt
    // still lands at 16387 tokens). Only the native /api/chat route honours it.
    // A decision made on a truncated orchestration context is worse than no
    // decision, so return Err and let the router escalate.
    if let Some(reported) = resp_json["usage"]["prompt_tokens"].as_u64() {
        let sent_chars = (system_prompt.len() + context_json.len()) as u64;
        if let Some(floor) = truncation_floor_tokens(sent_chars)
            && reported < floor
        {
            anyhow::bail!(
                "local_mlx: {url} reported {reported} prompt tokens for {sent_chars} chars sent \
                 (floor {floor}) — the server truncated the context, so its verdict is unusable"
            );
        }
    }

    let text = resp_json["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| {
            anyhow::anyhow!("local_mlx: missing choices[0].message.content in response")
        })?
        .to_owned();

    let elapsed_ms = start.elapsed().as_millis();
    log::info!(
        "local_mlx: call completed in {elapsed_ms}ms, model={model}, {} chars response",
        text.len()
    );

    // Reuse the existing parser — local models emit the same JSON schema
    // because we send them the same system_prompt. On parse failure,
    // parse_response synthesizes a confidence-0 stop (escalates naturally).
    let parsed = crate::parse_response(&text)?;
    Ok((text, parsed))
}

/// Fewest prompt tokens a server could honestly report for `sent_chars`.
///
/// English prose runs about 4 chars/token, so 8 is a deliberately generous
/// floor — roughly half the expected count — chosen so ordinary tokeniser
/// variation never trips the check and only genuine dropping does.
///
/// Returns `None` for small payloads, where the ratio is noisy (short prompts
/// carry proportionally more punctuation and markup) and truncation is not a
/// risk worth flagging anyway.
fn truncation_floor_tokens(sent_chars: u64) -> Option<u64> {
    const MIN_CHARS_TO_CHECK: u64 = 20_000;
    const GENEROUS_CHARS_PER_TOKEN: u64 = 8;
    (sent_chars >= MIN_CHARS_TO_CHECK).then(|| sent_chars / GENEROUS_CHARS_PER_TOKEN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_payloads_are_not_checked() {
        assert_eq!(truncation_floor_tokens(0), None);
        assert_eq!(truncation_floor_tokens(19_999), None);
    }

    #[test]
    fn floor_is_half_the_expected_token_count() {
        // 80k chars is ~20k tokens at 4 chars/token; the floor sits at 10k so
        // normal variation passes and only real dropping fails.
        assert_eq!(truncation_floor_tokens(80_000), Some(10_000));
    }

    #[test]
    fn catches_the_measured_ollama_v1_truncation() {
        // Measured 2026-08-19: ~180k chars of context sent to Ollama's
        // /v1/chat/completions came back reporting 16387 prompt tokens, with
        // the planted fact dropped and the answer wrong.
        let floor = truncation_floor_tokens(180_000).expect("large payload is checked");
        assert!(
            16_387 < floor,
            "16387 tokens for 180k chars must read as truncated (floor {floor})"
        );
    }

    #[test]
    fn honest_full_context_passes() {
        // Same content through the native API reported 54190 tokens — well
        // above the floor, so the guard must not fire.
        let floor = truncation_floor_tokens(180_000).expect("large payload is checked");
        assert!(54_190 >= floor, "an honest full-context reply must pass");
    }
}
