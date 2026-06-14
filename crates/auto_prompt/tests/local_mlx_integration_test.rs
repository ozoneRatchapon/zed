//! Integration tests for the local MLX client.
//!
//! These require a running `mlx_lm.server` and are therefore `#[ignore]` by default.
//! Run them explicitly:
//!
//!     cargo test -p auto_prompt --test local_mlx_integration_test -- --ignored
//!
//! Prerequisites:
//!     pipx install mlx-lm
//!     mlx_lm.server --model mlx-community/Llama-3.2-1B-Instruct-4bit --port 8081 &

use auto_prompt::local_mlx;

const T1_ENDPOINT: &str = "http://127.0.0.1:8081/v1";
const T1_MODEL: &str = "mlx-community/Llama-3.2-1B-Instruct-4bit";

#[test]
#[ignore]
fn health_check_detects_live_server() {
    let alive = futures::executor::block_on(local_mlx::is_alive(T1_ENDPOINT, 2000));
    assert!(alive, "expected mlx_lm.server to be alive at {T1_ENDPOINT}");
}

#[test]
#[ignore]
fn health_check_returns_false_for_dead_endpoint() {
    let alive =
        futures::executor::block_on(local_mlx::is_alive("http://127.0.0.1:59999/v1", 500));
    assert!(!alive, "nothing should be listening on port 59999");
}

#[test]
#[ignore]
fn call_returns_parsed_response_for_valid_prompt() {
    let system_prompt = r#"You are an orchestration helper. Respond ONLY with JSON:
{"should_continue": bool, "next_prompt": string|null, "reason": string|null, "all_plan_done": bool, "confidence": float, "thread_summary": string|null}"#;

    let context_json = r#"{
        "session_id": "test-session",
        "messages": [{"role": "user", "content": "Hello"}],
        "used_tools": false,
        "stop_reason": "end_turn",
        "iteration_count": 1
    }"#;

    let result = futures::executor::block_on(local_mlx::call(
        T1_ENDPOINT,
        T1_MODEL,
        system_prompt,
        context_json,
        30_000,
    ));

    let (raw, response) = result.expect("local_mlx::call should succeed against a live server");
    assert!(!raw.is_empty(), "raw response should not be empty");
    // The response should have parsed the confidence field (either from valid JSON
    // or as a synthetic 0.0 stop if the model didn't emit valid JSON).
    let conf = response.confidence.expect("confidence should be set");
    assert!(conf >= 0.0 && conf <= 1.0, "confidence {conf} out of [0,1]");
}
