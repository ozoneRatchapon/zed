//! Rolling compaction of old tool calls and assistant chunks (Plan 005).
//!
//! When the orchestrator's input context grows past `compact_at` tokens, this
//! module reclaims space by replacing verbose old messages with short summaries
//! in-place. Tool calls become one-line metadata. Assistant chunks beyond the
//! recent window get truncated. User and Plan messages are never touched.
//!
//! Compaction is pure-Rust string manipulation in its default configuration
//! (no LLM call), so it runs in microseconds. The result is observable via
//! `AutoPromptContext.compaction_log` and the `was_truncated` flag, both of
//! which the Plan 004 router uses as escalation signals.
//!
//! Algorithm:
//!   while tokens > target_budget:
//!     pick next compactable message index (oldest, lowest-priority)
//!     summarize it in-place
//!     record entry in compaction_log
//!   set was_truncated = !compaction_log.is_empty()
//!
//! Priority order (most-reclaim-first):
//!   1. Completed/Failed tool calls (biggest payload per message)
//!   2. Old assistant chunks (truncatable, lower value than recent ones)
//!   3. (User and Plan messages are NEVER compacted)

use crate::config::CompactionConfig;
use crate::context::{
    AutoPromptContext, CompactionEntry, CompactionStrategy, ContextMessage, ContextMessageRole,
};

/// Cap on `compaction_log` entries retained on the context. Older entries
/// roll off FIFO. Full logs go to disk via `verdict_log_path`.
const MAX_COMPACTION_LOG_ENTRIES: usize = 50;

/// Default byte limit for truncated assistant chunks. ~50 tokens.
const ASSISTANT_TRUNCATE_BYTES: usize = 200;

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct CompactionStats {
    /// Number of messages summarized this pass.
    pub messages_compacted: usize,
    /// Total bytes reclaimed (original_len - summary_len, summed).
    pub bytes_reclaimed: usize,
    /// Token count of the context after compaction.
    pub resulting_token_count: usize,
    /// Breakdown by strategy.
    pub strategy_breakdown: StrategyBreakdown,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct StrategyBreakdown {
    pub tool_call_metadata: usize,
    pub assistant_truncated: usize,
    pub assistant_llm_abstracted: usize,
}

/// Run compaction on `ctx` in-place until `estimate_token_count() <= target_budget`
/// or there is nothing left we're allowed to compact.
///
/// `policy` controls which messages are eligible and the recent-preserve window.
pub fn compact_old_messages(
    ctx: &mut AutoPromptContext,
    target_token_budget: usize,
    policy: &CompactionConfig,
) -> CompactionStats {
    let mut stats = CompactionStats::default();

    loop {
        if ctx.estimate_token_count() <= target_token_budget {
            break;
        }

        let Some((idx, strategy)) = next_compactable_index(&ctx.messages, policy.keep_recent)
        else {
            break; // Nothing eligible left.
        };

        let original = ctx.messages[idx].content.clone();
        let role = ctx.messages[idx].role.clone();
        let (summary_text, _used_strategy) =
            summarize_message(&ctx.messages[idx], strategy.clone());
        let reclaimed = original.len().saturating_sub(summary_text.len());

        // Record audit entry (FIFO cap).
        let entry = CompactionEntry {
            message_index: idx,
            role: role_name(&role).to_string(),
            original_bytes: original.len(),
            summary_bytes: summary_text.len(),
            strategy: strategy.clone(),
        };
        push_compaction_log(ctx, entry);

        // Apply the summary in-place.
        ctx.messages[idx].content = summary_text;

        // Update stats.
        stats.messages_compacted += 1;
        stats.bytes_reclaimed += reclaimed;
        match strategy {
            CompactionStrategy::ToolCallMetadata => {
                stats.strategy_breakdown.tool_call_metadata += 1
            }
            CompactionStrategy::AssistantTruncated => {
                stats.strategy_breakdown.assistant_truncated += 1
            }
            CompactionStrategy::AssistantLlmAbstracted => {
                stats.strategy_breakdown.assistant_llm_abstracted += 1
            }
        }

        // Recompute token count after each compaction so the loop guard is accurate.
        ctx.approximate_token_count = ctx.estimate_token_count();
    }

    ctx.was_truncated = !ctx.compaction_log.is_empty();
    ctx.approximate_token_count = ctx.estimate_token_count();
    stats.resulting_token_count = ctx.approximate_token_count;
    stats
}

/// Find the next message index eligible for compaction.
///
/// Eligibility rules:
/// - Skip the last `preserve_last_n` messages (always keep recent).
/// - Tool messages with status `completed` or `failed` are highest priority.
/// - Assistant messages (beyond the preserve window) are next.
/// - User and Plan messages are never eligible.
fn next_compactable_index(
    messages: &[ContextMessage],
    preserve_last_n: usize,
) -> Option<(usize, CompactionStrategy)> {
    let preserve_from = messages.len().saturating_sub(preserve_last_n);

    // First pass: completed/failed tool calls.
    for (i, msg) in messages.iter().enumerate().take(preserve_from) {
        if !matches!(msg.role, ContextMessageRole::Tool) {
            continue;
        }
        if is_terminal_tool_call(&msg.content) {
            return Some((i, CompactionStrategy::ToolCallMetadata));
        }
    }

    // Second pass: assistant chunks.
    for (i, msg) in messages.iter().enumerate().take(preserve_from) {
        if !matches!(msg.role, ContextMessageRole::Assistant) {
            continue;
        }
        // Skip assistant messages that are already small — nothing to gain.
        if msg.content.len() <= ASSISTANT_TRUNCATE_BYTES {
            continue;
        }
        // Idempotency guard: never re-truncate a chunk we already shortened.
        // Without this, the truncation suffix (" [...]") pushes content above
        // ASSISTANT_TRUNCATE_BYTES on the next pass and we loop forever.
        if msg.content.trim_end().ends_with("[...]") {
            continue;
        }
        return Some((i, CompactionStrategy::AssistantTruncated));
    }

    None
}

/// Check if a serialized tool-call string represents a terminal state
/// (`[Tool: ... (completed)]` or `(failed)`). Only terminal tool calls
/// are eligible for compaction — pending/in-progress ones may still update.
fn is_terminal_tool_call(content: &str) -> bool {
    content.contains("(completed)") || content.contains("(failed)")
}

/// Produce a summary for a single message based on the chosen strategy.
///
/// Returns `(summary_text, strategy_actually_used)`. The strategy may differ
/// from the input when an LLM abstract was requested but unavailable — we
/// fall back to truncation rather than failing.
fn summarize_message(
    msg: &ContextMessage,
    strategy: CompactionStrategy,
) -> (String, CompactionStrategy) {
    match strategy {
        CompactionStrategy::ToolCallMetadata => (summarize_tool_call(&msg.content), strategy),
        CompactionStrategy::AssistantTruncated => (
            truncate_assistant(&msg.content, ASSISTANT_TRUNCATE_BYTES),
            strategy,
        ),
        // LLM abstracts are not yet wired (Phase 2 of Plan 005); degrade to truncation.
        CompactionStrategy::AssistantLlmAbstracted => (
            truncate_assistant(&msg.content, ASSISTANT_TRUNCATE_BYTES),
            CompactionStrategy::AssistantTruncated,
        ),
    }
}

/// Compress a serialized tool call (as produced by `serialize_tool_call` in
/// context.rs) into a single line of metadata.
///
/// Input shape (typical):
///   [Tool: terminal (completed)]
///   Input: { ... }
///   Output: { ... }
///
/// Output shape:
///   [compacted:Tool terminal exit=completed args="..."]
///
/// Preserves the tool name, status, and a short hint of the input (cmd line
/// or first arg). Drops the output entirely — it's the largest payload and
/// the orchestrator rarely needs old tool output verbatim.
fn summarize_tool_call(content: &str) -> String {
    let first_line = content.lines().next().unwrap_or("").trim();
    // first_line looks like: [Tool: terminal (completed)]
    let (label, status) = parse_tool_label_line(first_line);

    // Try to find an "Input:" or "cmd:" hint on subsequent lines.
    let arg_hint = content
        .lines()
        .skip(1)
        .find_map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix("Input:")
                .map(|s| s.trim().to_string())
                .or_else(|| trimmed.strip_prefix("cmd:").map(|s| s.trim().to_string()))
        })
        .unwrap_or_default();

    let short_hint = truncate_to_chars(&arg_hint, 80);
    format!("[compacted:Tool {label} status={status} args=\"{short_hint}\"]")
}

/// Parse a `[Tool: terminal (completed)]` line into (label, status).
/// Falls back to ("unknown", "unknown") on parse failure.
fn parse_tool_label_line(line: &str) -> (String, String) {
    // Strip wrapping brackets.
    let inner = line.trim_start_matches('[').trim_end_matches(']').trim();
    // Shape: "Tool: <label> (<status>)"
    let Some(after_tool) = inner.strip_prefix("Tool:") else {
        return (line.to_string(), "unknown".into());
    };
    let after_tool = after_tool.trim();

    // Split on the last "(...)" — that's the status.
    if let Some(open) = after_tool.rfind(" (") {
        let label = after_tool[..open].trim().to_string();
        let rest = &after_tool[open + 2..];
        let status = rest.trim_end_matches(')').to_string();
        (label, status)
    } else {
        (after_tool.to_string(), "unknown".into())
    }
}

/// Truncate an assistant chunk to `limit` chars on a UTF-8 boundary.
/// Appends "[...]" so the orchestrator knows it was shortened.
fn truncate_assistant(content: &str, limit: usize) -> String {
    let trimmed = content.trim();
    if trimmed.chars().count() <= limit {
        return trimmed.to_string();
    }
    let truncated: String = trimmed.chars().take(limit).collect();
    format!("{truncated} [...]")
}

/// Truncate to a char-count limit on a UTF-8 boundary, no suffix.
fn truncate_to_chars(s: &str, limit: usize) -> String {
    if s.chars().count() <= limit {
        return s.to_string();
    }
    s.chars().take(limit).collect()
}

/// Get a stable string name for a message role, for audit log entries.
fn role_name(role: &ContextMessageRole) -> &'static str {
    match role {
        ContextMessageRole::User => "user",
        ContextMessageRole::Assistant => "assistant",
        ContextMessageRole::Tool => "tool",
        ContextMessageRole::Plan => "plan",
    }
}

/// Push a compaction-log entry, enforcing FIFO cap to avoid unbounded growth
/// across iterations within a single chain (max 20 iterations × N compactions).
fn push_compaction_log(ctx: &mut AutoPromptContext, entry: CompactionEntry) {
    if ctx.compaction_log.len() >= MAX_COMPACTION_LOG_ENTRIES {
        ctx.compaction_log.remove(0);
    }
    ctx.compaction_log.push(entry);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool_call(label: &str, status: &str, input: &str, output: &str) -> ContextMessage {
        let content = format!("[Tool: {label} ({status})]\nInput: {input}\nOutput: {output}");
        ContextMessage {
            role: ContextMessageRole::Tool,
            content,
        }
    }

    fn make_assistant(text: &str) -> ContextMessage {
        ContextMessage {
            role: ContextMessageRole::Assistant,
            content: text.to_string(),
        }
    }

    fn make_user(text: &str) -> ContextMessage {
        ContextMessage {
            role: ContextMessageRole::User,
            content: text.to_string(),
        }
    }

    fn make_ctx_with_messages(messages: Vec<ContextMessage>) -> AutoPromptContext {
        let mut ctx = AutoPromptContext {
            current_datetime: "2026-01-01T00:00:00Z".into(),
            current_paths: vec![],
            session_id: "test".into(),
            title: None,
            messages,
            used_tools: true,
            entry_count: 0,
            current_plan: vec![],
            plan_files: vec![],
            doc_files: vec![],
            stop_reason: "end_turn".into(),
            had_error: false,
            approximate_token_count: 0,
            actual_input_tokens: None,
            iteration_count: 1,
            stop_phase: crate::context::StopPhase::Working,
            verification_count: 0,
            was_truncated: false,
            compaction_log: vec![],
            fork_imminent: false,
            plan_has_checkboxes: false,
            first_plan_filename: "plan.md".into(),
            plan_number: "00".into(),
            first_user_message: None,
            last_assistant_message: None,
        };
        ctx.approximate_token_count = ctx.estimate_token_count();
        ctx
    }

    fn policy(keep_recent: usize) -> CompactionConfig {
        CompactionConfig {
            enabled: true,
            compact_at: 0,
            compact_target: 0,
            keep_recent,
            use_llm_for_assistant_abstracts: false,
        }
    }

    #[test]
    fn compaction_reclaims_bytes_from_tool_calls() {
        let big_output = "x".repeat(5000);
        let messages = vec![
            make_tool_call("terminal", "completed", "cargo build", &big_output),
            make_user("recent 1"),
            make_user("recent 2"),
        ];
        let mut ctx = make_ctx_with_messages(messages);
        let tokens_before = ctx.estimate_token_count();

        let stats = compact_old_messages(&mut ctx, 1, &policy(2));

        assert!(stats.messages_compacted >= 1);
        assert!(
            stats.bytes_reclaimed > 4000,
            "should reclaim most of the 5KB output"
        );
        assert!(ctx.estimate_token_count() < tokens_before);
        assert!(ctx.was_truncated);
        assert_eq!(ctx.compaction_log.len(), 1);
    }

    #[test]
    fn compaction_never_touches_user_messages() {
        let messages = vec![
            make_user("important user intent that must survive"),
            make_user("recent"),
            make_user("recent2"),
        ];
        let mut ctx = make_ctx_with_messages(messages);

        let _stats = compact_old_messages(&mut ctx, 1, &policy(2));

        assert_eq!(
            ctx.messages[0].content,
            "important user intent that must survive"
        );
        assert!(!ctx.was_truncated);
    }

    #[test]
    fn compaction_preserves_keep_recent_window() {
        let big = "x".repeat(3000);
        let messages = vec![
            make_tool_call("terminal", "completed", "cmd1", &big),
            make_tool_call("terminal", "completed", "cmd2", &big),
            make_tool_call("terminal", "completed", "cmd3", &big), // last 2 should be preserved
            make_tool_call("terminal", "completed", "cmd4", &big),
        ];
        let mut ctx = make_ctx_with_messages(messages);

        let _stats = compact_old_messages(&mut ctx, 1, &policy(2));

        // Last 2 tool calls (indices 2, 3) must be untouched.
        assert!(ctx.messages[2].content.contains("Output: xxx"));
        assert!(ctx.messages[3].content.contains("Output: xxx"));
        // Earlier ones compacted.
        assert!(ctx.messages[0].content.starts_with("[compacted:Tool"));
        assert!(ctx.messages[1].content.starts_with("[compacted:Tool"));
    }

    #[test]
    fn compaction_skips_pending_tool_calls() {
        let big = "x".repeat(3000);
        let messages = vec![
            make_tool_call("terminal", "pending", "cmd1", &big),
            make_user("recent1"),
            make_user("recent2"),
        ];
        let mut ctx = make_ctx_with_messages(messages);

        let stats = compact_old_messages(&mut ctx, 1, &policy(2));

        // Pending tool call should NOT be compacted.
        assert_eq!(stats.messages_compacted, 0);
        assert!(ctx.messages[0].content.contains("Output: xxx"));
    }

    #[test]
    fn compaction_idempotent_second_run_is_noop_when_under_budget() {
        let big = "x".repeat(3000);
        let messages = vec![
            make_tool_call("terminal", "completed", "cmd1", &big),
            make_user("r1"),
            make_user("r2"),
        ];
        let mut ctx = make_ctx_with_messages(messages);

        let _first = compact_old_messages(&mut ctx, 1, &policy(2));
        let log_len_after_first = ctx.compaction_log.len();
        let tokens_after_first = ctx.estimate_token_count();

        let second = compact_old_messages(&mut ctx, tokens_after_first, &policy(2));

        assert_eq!(second.messages_compacted, 0);
        assert_eq!(ctx.compaction_log.len(), log_len_after_first);
    }

    #[test]
    fn compaction_falls_through_to_assistant_after_tool_calls() {
        let big = "x".repeat(3000);
        let messages = vec![
            make_tool_call("terminal", "completed", "cmd", &big),
            make_assistant(&format!("Long reasoning about the build. {big}")),
            make_user("r1"),
            make_user("r2"),
        ];
        let mut ctx = make_ctx_with_messages(messages);

        let stats = compact_old_messages(&mut ctx, 1, &policy(2));

        // Tool call compacted first, then assistant chunk.
        assert!(stats.messages_compacted >= 2);
        assert!(stats.strategy_breakdown.tool_call_metadata >= 1);
        assert!(stats.strategy_breakdown.assistant_truncated >= 1);
        assert!(ctx.messages[1].content.ends_with("[...]"));
    }

    #[test]
    fn compaction_skips_already_short_assistant_chunks() {
        let messages = vec![
            make_assistant("short message"), // under ASSISTANT_TRUNCATE_BYTES
            make_user("r1"),
            make_user("r2"),
        ];
        let mut ctx = make_ctx_with_messages(messages);

        let stats = compact_old_messages(&mut ctx, 1, &policy(2));

        assert_eq!(stats.messages_compacted, 0);
    }

    #[test]
    fn compaction_sets_was_truncated_flag() {
        let big = "x".repeat(3000);
        let messages = vec![
            make_tool_call("terminal", "completed", "cmd", &big),
            make_user("r1"),
            make_user("r2"),
        ];
        let mut ctx = make_ctx_with_messages(messages);

        assert!(!ctx.was_truncated);
        let _stats = compact_old_messages(&mut ctx, 1, &policy(2));
        assert!(ctx.was_truncated);
    }

    #[test]
    fn compaction_log_capped_at_max_entries() {
        let big = "x".repeat(500);
        let mut messages: Vec<ContextMessage> = Vec::new();
        // Push 60 tool calls + 2 recent users (preserved).
        for _ in 0..60 {
            messages.push(make_tool_call("terminal", "completed", "cmd", &big));
        }
        messages.push(make_user("r1"));
        messages.push(make_user("r2"));

        let mut ctx = make_ctx_with_messages(messages);
        let _stats = compact_old_messages(&mut ctx, 1, &policy(2));

        assert!(
            ctx.compaction_log.len() <= MAX_COMPACTION_LOG_ENTRIES,
            "compaction_log must be capped, got {}",
            ctx.compaction_log.len()
        );
    }

    #[test]
    fn parse_tool_label_line_handles_completed() {
        let (label, status) = parse_tool_label_line("[Tool: terminal (completed)]");
        assert_eq!(label, "terminal");
        assert_eq!(status, "completed");
    }

    #[test]
    fn parse_tool_label_line_handles_failed() {
        let (label, status) = parse_tool_label_line("[Tool: terminal (failed)]");
        assert_eq!(label, "terminal");
        assert_eq!(status, "failed");
    }

    #[test]
    fn parse_tool_label_line_falls_back_on_malformed() {
        let (label, _status) = parse_tool_label_line("garbage");
        assert_eq!(label, "garbage");
    }

    #[test]
    fn summarize_tool_call_extracts_label_and_status() {
        let content = "[Tool: terminal (completed)]\nInput: cargo build\nOutput: x".repeat(1);
        let summary = summarize_tool_call(&content);
        assert!(summary.starts_with("[compacted:Tool"));
        assert!(summary.contains("terminal"));
        assert!(summary.contains("status=completed"));
        assert!(summary.contains("cargo build"));
        // Output should NOT be in the summary.
        assert!(!summary.contains("Output:"));
    }

    #[test]
    fn truncate_assistant_appends_ellipsis() {
        let long = "a".repeat(500);
        let truncated = truncate_assistant(&long, 50);
        assert!(truncated.ends_with("[...]"));
        assert!(truncated.chars().count() < 60);
    }

    #[test]
    fn truncate_assistant_returns_short_input_unchanged() {
        let short = "hello";
        let result = truncate_assistant(short, 100);
        assert_eq!(result, "hello");
    }

    #[test]
    fn is_terminal_tool_call_recognizes_completed() {
        assert!(is_terminal_tool_call("[Tool: terminal (completed)]\n..."));
    }

    #[test]
    fn is_terminal_tool_call_recognizes_failed() {
        assert!(is_terminal_tool_call("[Tool: terminal (failed)]\n..."));
    }

    #[test]
    fn is_terminal_tool_call_rejects_pending() {
        assert!(!is_terminal_tool_call("[Tool: terminal (pending)]\n..."));
    }

    #[test]
    fn role_name_maps_each_variant() {
        assert_eq!(role_name(&ContextMessageRole::User), "user");
        assert_eq!(role_name(&ContextMessageRole::Assistant), "assistant");
        assert_eq!(role_name(&ContextMessageRole::Tool), "tool");
        assert_eq!(role_name(&ContextMessageRole::Plan), "plan");
    }
}
