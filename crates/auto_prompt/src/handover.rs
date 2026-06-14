//! Structured handover for thread forks (Plan 005).
//!
//! When a thread is about to fork (`actual_input_tokens >= fork_at`), the
//! orchestrator LLM may emit a `HandoverBlock`. This module serializes that
//! block into a YAML-ish text wrapped in `<handover>...</handover>` fences,
//! which is prepended to the new thread's first user message so the worker
//! LLM starts with structured context instead of a flat continuation string.
//!
//! Design notes:
//! - We hand-roll the YAML output (not pulling in a yaml crate) to keep the
//!   dependency surface zero and the output deterministic.
//! - Field order is fixed: intent, completed, in-progress, blocked, next-step,
//!   files, plans. Empty fields are omitted entirely (no `key: null` noise).
//! - The fallback synthesizer is invoked when the LLM omitted a handover;
//!   it produces a minimal block from `first_user_message` + last assistant
//!   message + plan landscape, so forks degrade gracefully.

use crate::context::HandoverBlock;

impl HandoverBlock {
    /// Render the handover as a YAML-ish block wrapped in `<handover>` fences.
    /// Returns empty string when the block is entirely empty.
    pub fn to_yaml_block(&self) -> String {
        if self.is_empty() {
            return String::new();
        }

        let mut lines: Vec<String> = Vec::new();
        lines.push("<handover>".into());

        if let Some(intent) = self.original_intent.as_ref().filter(|s| !s.trim().is_empty()) {
            lines.push(format!("original_intent: {}", escape_yaml_scalar(intent)));
        }
        if !self.completed.is_empty() {
            lines.push("completed:".into());
            for item in &self.completed {
                lines.push(format!("  - {}", escape_yaml_scalar(item)));
            }
        }
        if !self.in_progress.is_empty() {
            lines.push("in_progress:".into());
            for item in &self.in_progress {
                lines.push(format!("  - {}", escape_yaml_scalar(item)));
            }
        }
        if !self.blocked_on.is_empty() {
            lines.push("blocked_on:".into());
            for item in &self.blocked_on {
                lines.push(format!("  - {}", escape_yaml_scalar(item)));
            }
        }
        if let Some(next) = self.next_step.as_ref().filter(|s| !s.trim().is_empty()) {
            lines.push(format!("next_step: {}", escape_yaml_scalar(next)));
        }
        if !self.files_touched.is_empty() {
            lines.push("files_touched:".into());
            for item in &self.files_touched {
                lines.push(format!("  - {}", escape_yaml_scalar(item)));
            }
        }
        if !self.active_plans.is_empty() {
            lines.push("active_plans:".into());
            for item in &self.active_plans {
                lines.push(format!("  - {}", escape_yaml_scalar(item)));
            }
        }

        lines.push("</handover>".into());
        lines.join("\n")
    }

    /// True when every field is empty/None. Used to skip emitting empty blocks.
    pub fn is_empty(&self) -> bool {
        self.original_intent.as_ref().map(|s| s.trim().is_empty()).unwrap_or(true)
            && self.completed.is_empty()
            && self.in_progress.is_empty()
            && self.blocked_on.is_empty()
            && self.next_step.as_ref().map(|s| s.trim().is_empty()).unwrap_or(true)
            && self.files_touched.is_empty()
            && self.active_plans.is_empty()
    }
}

/// Prepend a handover block to a prompt. If the handover is empty, the prompt
/// is returned unchanged. Used by `decide_with_llm` before the action ships.
pub fn prepend_handover(prompt: &str, handover: Option<&HandoverBlock>) -> String {
    match handover {
        Some(h) if !h.is_empty() => {
            let block = h.to_yaml_block();
            if block.is_empty() {
                prompt.to_string()
            } else {
                format!("{block}\n\n{prompt}")
            }
        }
        _ => prompt.to_string(),
    }
}

/// Synthesize a minimal fallback handover when the orchestrator omitted one.
/// Uses first user message as intent, last assistant message as in-progress,
/// and the supplied plan filenames as active_plans. Better than nothing.
pub fn synthesize_fallback(
    first_user_message: Option<&str>,
    last_assistant_message: Option<&str>,
    active_plan_paths: &[String],
) -> HandoverBlock {
    let mut block = HandoverBlock::default();

    if let Some(msg) = first_user_message.map(str::trim).filter(|s| !s.is_empty()) {
        block.original_intent = Some(truncate_to_chars(msg, 240));
    }

    if let Some(last) = last_assistant_message.map(str::trim).filter(|s| !s.is_empty()) {
        // Heuristic: treat the last assistant message as the in-progress state.
        // It's often a status update. Truncate hard — the new thread can ask
        // for details; the goal is just to point it in the right direction.
        block.in_progress.push(truncate_to_chars(last, 400));
    }

    if !active_plan_paths.is_empty() {
        block.active_plans = active_plan_paths.to_vec();
    }

    block
}

/// Escape a string for use as a YAML scalar. We prefer double-quoted style
/// only when the string contains characters that would be ambiguous bare
/// (`:`, leading `-`, leading/trailing whitespace, newlines). Otherwise we
/// emit bare scalars for readability.
fn escape_yaml_scalar(s: &str) -> String {
    let trimmed = s.trim();
    let needs_quoting = trimmed.is_empty()
        || trimmed.contains(':')
        || trimmed.contains('\n')
        || trimmed.contains('"')
        || trimmed.starts_with('-')
        || trimmed.starts_with(' ')
        || trimmed.ends_with(' ')
        || trimmed.starts_with('[')
        || trimmed.starts_with('{')
        || trimmed.starts_with('#')
        || trimmed.starts_with('&')
        || trimmed.starts_with('*')
        || trimmed.starts_with('!')
        || trimmed.starts_with('|')
        || trimmed.starts_with('>')
        || trimmed.starts_with('@')
        || trimmed.starts_with('`');

    if !needs_quoting {
        return trimmed.to_string();
    }

    // Double-quote with C-style escapes.
    let escaped = trimmed
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{escaped}\"")
}

/// Truncate to a char-count limit on a UTF-8 char boundary, appending "...".
fn truncate_to_chars(s: &str, limit: usize) -> String {
    if s.chars().count() <= limit {
        return s.to_string();
    }
    let truncated: String = s.chars().take(limit).collect();
    format!("{truncated}...")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_block_renders_as_empty_string() {
        let block = HandoverBlock::default();
        assert_eq!(block.to_yaml_block(), "");
        assert!(block.is_empty());
    }

    #[test]
    fn renders_full_block_with_fences() {
        let block = HandoverBlock {
            original_intent: Some("Fix the login bug".into()),
            completed: vec!["Reproduced the issue".into()],
            in_progress: vec!["Editing auth.rs".into()],
            blocked_on: vec![],
            next_step: Some("Run cargo test".into()),
            files_touched: vec!["src/auth.rs".into()],
            active_plans: vec!["042_login_bug".into()],
        };
        let yaml = block.to_yaml_block();
        assert!(yaml.starts_with("<handover>"));
        assert!(yaml.ends_with("</handover>"));
        assert!(yaml.contains("original_intent: Fix the login bug"));
        assert!(yaml.contains("completed:"));
        assert!(yaml.contains("  - Reproduced the issue"));
        assert!(yaml.contains("next_step: Run cargo test"));
    }

    #[test]
    fn omits_empty_fields() {
        let block = HandoverBlock {
            original_intent: Some("Do thing".into()),
            completed: vec![],
            in_progress: vec![],
            blocked_on: vec![],
            next_step: None,
            files_touched: vec![],
            active_plans: vec![],
        };
        let yaml = block.to_yaml_block();
        assert!(yaml.contains("original_intent"));
        assert!(!yaml.contains("completed:"));
        assert!(!yaml.contains("in_progress:"));
        assert!(!yaml.contains("next_step:"));
    }

    #[test]
    fn quotes_scalars_with_colons() {
        let block = HandoverBlock {
            original_intent: Some("Fix URL: https://example.com".into()),
            ..Default::default()
        };
        let yaml = block.to_yaml_block();
        assert!(yaml.contains("original_intent: \"Fix URL: https://example.com\""));
    }

    #[test]
    fn prepend_returns_prompt_unchanged_when_handover_none() {
        let prompt = "do the thing";
        let result = prepend_handover(prompt, None);
        assert_eq!(result, "do the thing");
    }

    #[test]
    fn prepend_returns_prompt_unchanged_when_handover_empty() {
        let prompt = "do the thing";
        let result = prepend_handover(prompt, Some(&HandoverBlock::default()));
        assert_eq!(result, "do the thing");
    }

    #[test]
    fn prepend_prefixes_block_when_populated() {
        let prompt = "do the thing";
        let block = HandoverBlock {
            original_intent: Some("intent".into()),
            ..Default::default()
        };
        let result = prepend_handover(prompt, Some(&block));
        assert!(result.starts_with("<handover>"));
        assert!(result.contains("do the thing"));
    }

    #[test]
    fn synthesize_fallback_uses_first_user_message_as_intent() {
        let block = synthesize_fallback(
            Some("please fix the bug in auth"),
            Some("I edited auth.rs and ran tests"),
            &["042_login".into()],
        );
        assert_eq!(block.original_intent.as_deref(), Some("please fix the bug in auth"));
        assert_eq!(block.in_progress.len(), 1);
        assert!(block.in_progress[0].contains("edited auth.rs"));
        assert_eq!(block.active_plans, vec!["042_login".to_string()]);
    }

    #[test]
    fn synthesize_fallback_returns_empty_block_when_all_inputs_empty() {
        let block = synthesize_fallback(None, None, &[]);
        assert!(block.is_empty());
    }

    #[test]
    fn truncate_handles_multibyte_chars() {
        let s = "日本語のテスト文字列"; // 10 chars
        let truncated = truncate_to_chars(s, 3);
        assert_eq!(truncated.chars().count(), 6); // 3 chars + "..."
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn truncate_returns_input_when_under_limit() {
        let s = "short";
        let truncated = truncate_to_chars(s, 100);
        assert_eq!(truncated, "short");
    }
}
