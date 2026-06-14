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
    let body = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user",   "content": context_json },
        ],
        "stream": false,
        "temperature": 0.0,
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
