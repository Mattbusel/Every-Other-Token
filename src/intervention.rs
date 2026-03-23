//! Token Intervention Mode.
//!
//! Allows a user (or an automated controller) to pause token generation at
//! any position, inject a corrected or alternative token, and then continue
//! generation from that injected token — measuring the causal effect of the
//! intervention on the downstream output.
//!
//! # Concepts
//!
//! An *intervention* is a triple `(position, original_token, injected_token)`.
//! After injection, generation continues causally from the injected token,
//! producing a *counterfactual* stream that diverges from the original.
//! [`InterventionHistory`] records both streams so the causal effect can be
//! measured and displayed.
//!
//! # Causal effect measurement
//!
//! The causal effect at position `k > position` is measured as token-level
//! edit distance normalised by stream length.  A score of 0 means the
//! intervention had no downstream effect; a score of 1 means every subsequent
//! token changed.
//!
//! # TUI token editor
//!
//! [`TokenEditor`] is a lightweight terminal widget that renders the current
//! token stream with cursor navigation and edit mode.  It does not depend on
//! any external TUI framework — it emits ANSI escape sequences directly —
//! so it works in any terminal that supports VT100.
//!
//! # Usage
//!
//! ```rust,no_run
//! use every_other_token::intervention::{InterventionHistory, Intervention, TokenEditor};
//!
//! let original = vec!["The".to_string(), " cat".to_string(), " sat".to_string()];
//! let mut history = InterventionHistory::new(original.clone());
//!
//! // Inject " dog" at position 1 (replacing " cat").
//! let intervention = Intervention::new(1, " cat", " dog");
//! let counterfactual = vec!["The".to_string(), " dog".to_string(), " slept".to_string()];
//! history.apply(intervention, counterfactual);
//!
//! println!("Causal effect: {:.2}", history.causal_effect());
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ── Single intervention ───────────────────────────────────────────────────────

/// A single token-level intervention: replacing one token with another.
#[derive(Debug, Clone, PartialEq)]
pub struct Intervention {
    /// Zero-based token position where the injection occurred.
    pub position: usize,
    /// The token that was originally at this position.
    pub original_token: String,
    /// The token that was injected in its place.
    pub injected_token: String,
    /// Wall-clock timestamp of the intervention (Unix ms).
    pub timestamp_ms: u64,
}

impl Intervention {
    /// Construct an intervention.
    pub fn new(
        position: usize,
        original_token: impl Into<String>,
        injected_token: impl Into<String>,
    ) -> Self {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            position,
            original_token: original_token.into(),
            injected_token: injected_token.into(),
            timestamp_ms,
        }
    }

    /// Whether the intervention actually changed the token (not a no-op).
    pub fn is_effective(&self) -> bool {
        self.original_token != self.injected_token
    }
}

// ── Intervention history ──────────────────────────────────────────────────────

/// Complete record of an interventional experiment.
///
/// Stores the original token stream, all injections that were applied, and
/// the final counterfactual stream that resulted from those injections.
#[derive(Debug, Clone)]
pub struct InterventionHistory {
    /// The original, unmodified token stream.
    pub original_stream: Vec<String>,
    /// All interventions applied, in order.
    pub interventions: Vec<Intervention>,
    /// The counterfactual stream after applying all interventions.
    ///
    /// Before any intervention is applied this is identical to `original_stream`.
    pub final_stream: Vec<String>,
}

impl InterventionHistory {
    /// Create an empty history for the given original stream.
    pub fn new(original_stream: Vec<String>) -> Self {
        let final_stream = original_stream.clone();
        Self {
            original_stream,
            interventions: Vec::new(),
            final_stream,
        }
    }

    /// Apply an intervention: record it and replace `final_stream` with the
    /// counterfactual continuation provided by the caller.
    ///
    /// The counterfactual stream should start at position 0 (full stream)
    /// and include the injected token at `intervention.position`.
    pub fn apply(&mut self, intervention: Intervention, counterfactual_stream: Vec<String>) {
        self.interventions.push(intervention);
        self.final_stream = counterfactual_stream;
    }

    /// Apply an in-place intervention without a new counterfactual stream.
    ///
    /// Useful for editing the stream token by token before re-generating.
    pub fn patch_token(&mut self, position: usize, new_token: impl Into<String>) {
        let new_token = new_token.into();
        if position < self.final_stream.len() {
            let old = self.final_stream[position].clone();
            let iv = Intervention::new(position, old, new_token.clone());
            self.interventions.push(iv);
            self.final_stream[position] = new_token;
        }
    }

    /// Causal effect score: token-level normalised edit distance between
    /// the original and final streams, considering only positions *after*
    /// the first intervention.
    ///
    /// Returns `0.0` if no interventions have been applied.
    pub fn causal_effect(&self) -> f64 {
        if self.interventions.is_empty() {
            return 0.0;
        }
        let first_pos = self
            .interventions
            .iter()
            .map(|iv| iv.position)
            .min()
            .unwrap_or(0);

        let orig = &self.original_stream[first_pos.min(self.original_stream.len())..];
        let final_ = &self.final_stream[first_pos.min(self.final_stream.len())..];

        if orig.is_empty() && final_.is_empty() {
            return 0.0;
        }

        let max_len = orig.len().max(final_.len());
        let differing = orig
            .iter()
            .zip(final_.iter())
            .filter(|(a, b)| a != b)
            .count()
            + orig.len().abs_diff(final_.len()); // insertion/deletion penalty

        differing as f64 / max_len as f64
    }

    /// Summary of the full intervention chain as a human-readable string.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Interventions: {}, causal_effect: {:.4}\n",
            self.interventions.len(),
            self.causal_effect()
        ));
        for (i, iv) in self.interventions.iter().enumerate() {
            s.push_str(&format!(
                "  [{}] pos={} {:?} → {:?}\n",
                i, iv.position, iv.original_token, iv.injected_token
            ));
        }
        s
    }

    /// Plain text of the original stream.
    pub fn original_text(&self) -> String {
        self.original_stream.join("")
    }

    /// Plain text of the final (counterfactual) stream.
    pub fn final_text(&self) -> String {
        self.final_stream.join("")
    }
}

// ── Token editor (TUI widget) ─────────────────────────────────────────────────

/// Lightweight terminal token editor.
///
/// Renders the current token stream with a visible cursor and optional edit
/// mode.  All output is ANSI escape sequences; no external TUI dependency.
///
/// # Controls (conceptual — integrate with your event loop)
///
/// | Key | Action |
/// |-----|--------|
/// | `←` / `→` | Move cursor left / right by one token |
/// | `Home` / `End` | Jump to first / last token |
/// | `Enter` | Enter edit mode at cursor position |
/// | `Esc` | Exit edit mode without committing |
/// | `Tab` | Commit current edit and advance cursor |
pub struct TokenEditor {
    /// Current token stream (mutable).
    tokens: Vec<String>,
    /// Zero-based index of the currently highlighted token.
    cursor: usize,
    /// Whether the editor is in edit mode (cursor token is being typed).
    edit_mode: bool,
    /// Buffer for in-progress edit (only meaningful when `edit_mode` is true).
    edit_buffer: String,
    /// History of all committed edits.
    edits: Vec<(usize, String, String)>,
}

impl TokenEditor {
    /// Construct a new editor for the given token stream.
    pub fn new(tokens: Vec<String>) -> Self {
        Self {
            tokens,
            cursor: 0,
            edit_mode: false,
            edit_buffer: String::new(),
            edits: Vec::new(),
        }
    }

    /// Move the cursor left by one token.
    pub fn cursor_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    /// Move the cursor right by one token.
    pub fn cursor_right(&mut self) {
        if self.cursor + 1 < self.tokens.len() {
            self.cursor += 1;
        }
    }

    /// Jump the cursor to the first token.
    pub fn cursor_home(&mut self) {
        self.cursor = 0;
    }

    /// Jump the cursor to the last token.
    pub fn cursor_end(&mut self) {
        if !self.tokens.is_empty() {
            self.cursor = self.tokens.len() - 1;
        }
    }

    /// Enter edit mode at the current cursor position.
    pub fn enter_edit_mode(&mut self) {
        if self.cursor < self.tokens.len() {
            self.edit_mode = true;
            self.edit_buffer = self.tokens[self.cursor].clone();
        }
    }

    /// Append a character to the edit buffer (while in edit mode).
    pub fn edit_push(&mut self, c: char) {
        if self.edit_mode {
            self.edit_buffer.push(c);
        }
    }

    /// Remove the last character from the edit buffer (backspace).
    pub fn edit_pop(&mut self) {
        if self.edit_mode {
            self.edit_buffer.pop();
        }
    }

    /// Commit the current edit and exit edit mode.
    ///
    /// Returns `Some((position, original, new_token))` if the token changed.
    pub fn commit_edit(&mut self) -> Option<(usize, String, String)> {
        if !self.edit_mode {
            return None;
        }
        self.edit_mode = false;
        let new_token = self.edit_buffer.clone();
        let original = self.tokens[self.cursor].clone();
        if new_token != original {
            self.tokens[self.cursor] = new_token.clone();
            let edit = (self.cursor, original, new_token);
            self.edits.push(edit.clone());
            Some(edit)
        } else {
            None
        }
    }

    /// Cancel the current edit without committing.
    pub fn cancel_edit(&mut self) {
        self.edit_mode = false;
        self.edit_buffer.clear();
    }

    /// Render the token stream to a string with ANSI highlighting.
    ///
    /// - Normal tokens: default terminal colours.
    /// - Cursor token (not in edit mode): cyan underline `\x1b[4;36m`.
    /// - Cursor token (in edit mode):     yellow background `\x1b[43m` showing
    ///   the edit buffer instead of the original token.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for (i, token) in self.tokens.iter().enumerate() {
            if i == self.cursor {
                if self.edit_mode {
                    out.push_str(&format!("\x1b[43m{}\x1b[0m", self.edit_buffer));
                } else {
                    out.push_str(&format!("\x1b[4;36m{}\x1b[0m", token));
                }
            } else {
                out.push_str(token);
            }
        }
        out
    }

    /// Current token stream (with any committed edits applied).
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// All committed edits as `(position, original, new)` triples.
    pub fn edits(&self) -> &[(usize, String, String)] {
        &self.edits
    }

    /// Convert committed edits to a vec of [`Intervention`] values.
    pub fn to_interventions(&self) -> Vec<Intervention> {
        self.edits
            .iter()
            .map(|(pos, orig, new_tok)| Intervention::new(*pos, orig.clone(), new_tok.clone()))
            .collect()
    }

    /// Plain text of the current stream.
    pub fn text(&self) -> String {
        self.tokens.join("")
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns `true` if the editor contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Current cursor position.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Whether edit mode is currently active.
    pub fn is_editing(&self) -> bool {
        self.edit_mode
    }
}

// ── Causal effect comparison ──────────────────────────────────────────────────

/// Compare two token streams and return per-position agreement flags.
///
/// `agreements[i]` is `true` if `stream_a[i] == stream_b[i]`.
/// Positions beyond the shorter stream are treated as disagreements.
pub fn per_position_agreement(stream_a: &[String], stream_b: &[String]) -> Vec<bool> {
    let len = stream_a.len().max(stream_b.len());
    (0..len)
        .map(|i| {
            match (stream_a.get(i), stream_b.get(i)) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            }
        })
        .collect()
}

/// Compute a position-to-effect map showing, for each token position `p`,
/// the fraction of downstream positions (> `p`) that changed after the
/// intervention.
///
/// This quantifies how much *influence* each position exerts on the future.
pub fn causal_influence_map(
    original: &[String],
    counterfactual: &[String],
) -> HashMap<usize, f64> {
    let len = original.len().max(counterfactual.len());
    let mut map = HashMap::new();

    for p in 0..len {
        let downstream_len = len.saturating_sub(p + 1);
        if downstream_len == 0 {
            map.insert(p, 0.0);
            continue;
        }
        let changed: usize = ((p + 1)..len)
            .filter(|&i| {
                original.get(i).map(String::as_str) != counterfactual.get(i).map(String::as_str)
            })
            .count();
        map.insert(p, changed as f64 / downstream_len as f64);
    }

    map
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(s: &str) -> String {
        s.to_string()
    }

    fn toks(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn intervention_is_effective() {
        let iv = Intervention::new(0, "cat", "dog");
        assert!(iv.is_effective());
        let noop = Intervention::new(0, "cat", "cat");
        assert!(!noop.is_effective());
    }

    #[test]
    fn history_causal_effect_zero_when_no_change() {
        let orig = toks(&["a", "b", "c"]);
        let mut history = InterventionHistory::new(orig.clone());
        let counterfactual = orig.clone();
        history.apply(Intervention::new(0, "a", "a"), counterfactual);
        assert!((history.causal_effect() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn history_causal_effect_full_change() {
        let orig = toks(&["a", "b", "c"]);
        let mut history = InterventionHistory::new(orig);
        let counterfactual = toks(&["X", "Y", "Z"]);
        history.apply(Intervention::new(0, "a", "X"), counterfactual);
        let effect = history.causal_effect();
        assert!(effect > 0.0, "full token change should produce non-zero effect: {effect}");
    }

    #[test]
    fn history_patch_token() {
        let orig = toks(&["hello", " world"]);
        let mut history = InterventionHistory::new(orig);
        history.patch_token(1, " rust");
        assert_eq!(history.final_stream[1], " rust");
        assert_eq!(history.interventions.len(), 1);
    }

    #[test]
    fn history_no_interventions_returns_zero_effect() {
        let orig = toks(&["a", "b"]);
        let history = InterventionHistory::new(orig);
        assert_eq!(history.causal_effect(), 0.0);
    }

    #[test]
    fn token_editor_cursor_navigation() {
        let mut ed = TokenEditor::new(toks(&["a", "b", "c"]));
        assert_eq!(ed.cursor(), 0);
        ed.cursor_right();
        assert_eq!(ed.cursor(), 1);
        ed.cursor_left();
        assert_eq!(ed.cursor(), 0);
        ed.cursor_end();
        assert_eq!(ed.cursor(), 2);
        ed.cursor_home();
        assert_eq!(ed.cursor(), 0);
    }

    #[test]
    fn token_editor_cursor_clamps_at_bounds() {
        let mut ed = TokenEditor::new(toks(&["a"]));
        ed.cursor_left(); // already at 0
        assert_eq!(ed.cursor(), 0);
        ed.cursor_right(); // already at last
        assert_eq!(ed.cursor(), 0);
    }

    #[test]
    fn token_editor_commit_edit() {
        let mut ed = TokenEditor::new(toks(&["cat", " sat"]));
        ed.enter_edit_mode();
        ed.edit_push('d');
        ed.edit_push('o');
        ed.edit_push('g');
        // Simulate clearing original and typing new token.
        ed.edit_buffer = "dog".to_string();
        let result = ed.commit_edit();
        assert!(result.is_some());
        let (pos, orig, new_tok) = result.unwrap();
        assert_eq!(pos, 0);
        assert_eq!(orig, "cat");
        assert_eq!(new_tok, "dog");
        assert_eq!(ed.tokens()[0], "dog");
    }

    #[test]
    fn token_editor_cancel_edit() {
        let mut ed = TokenEditor::new(toks(&["cat"]));
        ed.enter_edit_mode();
        ed.edit_buffer = "something_else".to_string();
        ed.cancel_edit();
        assert!(!ed.is_editing());
        assert_eq!(ed.tokens()[0], "cat"); // unchanged
    }

    #[test]
    fn token_editor_to_interventions() {
        let mut ed = TokenEditor::new(toks(&["a", "b"]));
        ed.edits.push((0, "a".to_string(), "X".to_string()));
        let ivs = ed.to_interventions();
        assert_eq!(ivs.len(), 1);
        assert_eq!(ivs[0].position, 0);
    }

    #[test]
    fn per_position_agreement_identical() {
        let s = toks(&["a", "b", "c"]);
        let ag = per_position_agreement(&s, &s);
        assert!(ag.iter().all(|&a| a));
    }

    #[test]
    fn per_position_agreement_all_different() {
        let s1 = toks(&["a", "b"]);
        let s2 = toks(&["x", "y"]);
        let ag = per_position_agreement(&s1, &s2);
        assert!(ag.iter().all(|&a| !a));
    }

    #[test]
    fn causal_influence_map_no_change() {
        let s = toks(&["a", "b", "c"]);
        let map = causal_influence_map(&s, &s);
        for (_, v) in map.iter().enumerate() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn causal_influence_map_full_change_at_zero() {
        let orig = toks(&["a", "b", "c"]);
        let counter = toks(&["X", "Y", "Z"]);
        let map = causal_influence_map(&orig, &counter);
        // Position 0: 2 downstream tokens both changed → 1.0
        assert!((map[&0] - 1.0).abs() < 1e-10, "map[0]={}", map[&0]);
    }

    #[test]
    fn render_contains_token_text() {
        let ed = TokenEditor::new(toks(&["hello", " world"]));
        let rendered = ed.render();
        assert!(rendered.contains("hello"));
        assert!(rendered.contains("world"));
    }

    #[test]
    fn history_summary_not_empty() {
        let mut history = InterventionHistory::new(toks(&["a"]));
        history.apply(Intervention::new(0, "a", "b"), toks(&["b"]));
        let s = history.summary();
        assert!(!s.is_empty());
        assert!(s.contains("Interventions: 1"));
    }
}
