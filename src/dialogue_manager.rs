//! Multi-turn dialogue state manager.
//!
//! Tracks conversation history, manages context window budgets, and provides
//! serialization for persistence across sessions.

use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

/// The role of a participant in a dialogue turn.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::System => write!(f, "system"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

// ---------------------------------------------------------------------------
// Turn
// ---------------------------------------------------------------------------

/// A single turn in a multi-turn dialogue.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Turn {
    pub role: Role,
    pub content: String,
    pub tokens: usize,
    pub timestamp_ms: u64,
}

impl Turn {
    /// Create a new turn with the current wall-clock timestamp.
    pub fn new(role: Role, content: impl Into<String>, tokens: usize) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self { role, content: content.into(), tokens, timestamp_ms }
    }

    /// Create a turn with an explicit timestamp (useful for tests and replay).
    pub fn with_timestamp(
        role: Role,
        content: impl Into<String>,
        tokens: usize,
        timestamp_ms: u64,
    ) -> Self {
        Self { role, content: content.into(), tokens, timestamp_ms }
    }
}

// ---------------------------------------------------------------------------
// DialogueState
// ---------------------------------------------------------------------------

/// The complete mutable state of an ongoing dialogue.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DialogueState {
    /// All turns ever recorded (including trimmed-out turns).
    pub history: Vec<Turn>,
    /// Indices into `history` that form the current active context window.
    pub active_indices: Vec<usize>,
    /// Maximum token budget for the context window.
    pub max_tokens: usize,
    /// Running total of tokens in the active context window.
    pub tokens_used: usize,
}

impl DialogueState {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            history: Vec::new(),
            active_indices: Vec::new(),
            max_tokens,
            tokens_used: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// DialogueManager
// ---------------------------------------------------------------------------

/// Manages multi-turn dialogue state including budgeting and summarisation.
pub struct DialogueManager {
    state: DialogueState,
}

impl DialogueManager {
    /// Create a manager with the given token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self { state: DialogueState::new(max_tokens) }
    }

    /// Restore a manager from a previously saved `DialogueState`.
    pub fn from_state(state: DialogueState) -> Self {
        Self { state }
    }

    /// Access the underlying state (e.g. for persistence).
    pub fn state(&self) -> &DialogueState {
        &self.state
    }

    /// Add a new turn to the dialogue.
    ///
    /// The turn is appended to `history` and its index is added to the active
    /// context window.  Token count is updated accordingly.
    pub fn add_turn(&mut self, role: Role, content: impl Into<String>, tokens: usize) {
        let turn = Turn::new(role, content, tokens);
        let idx = self.state.history.len();
        self.state.history.push(turn);
        self.state.active_indices.push(idx);
        self.state.tokens_used += tokens;
    }

    /// Remove the oldest non-System turns from the active context window until
    /// the token budget is satisfied.
    ///
    /// Returns the number of turns removed.
    pub fn trim_to_budget(&mut self, max_tokens: usize) -> usize {
        self.state.max_tokens = max_tokens;
        let mut removed = 0;

        loop {
            if self.state.tokens_used <= max_tokens {
                break;
            }
            // Find the oldest non-System active index.
            let pos = self
                .state
                .active_indices
                .iter()
                .position(|&idx| self.state.history[idx].role != Role::System);

            match pos {
                None => break, // Only System turns left; cannot trim further.
                Some(pos) => {
                    let idx = self.state.active_indices.remove(pos);
                    let tokens = self.state.history[idx].tokens;
                    self.state.tokens_used =
                        self.state.tokens_used.saturating_sub(tokens);
                    removed += 1;
                }
            }
        }
        removed
    }

    /// Return a placeholder summary of the oldest `n` turns in the full history.
    ///
    /// In a production system this would call an LLM; here it concatenates
    /// truncated content with role labels.
    pub fn summarize_oldest(&self, n: usize) -> String {
        let turns: Vec<&Turn> = self.state.history.iter().take(n).collect();
        if turns.is_empty() {
            return String::from("[No turns to summarize]");
        }
        let mut parts = Vec::with_capacity(turns.len());
        for t in turns {
            let snippet = if t.content.len() > 80 {
                format!("{}…", &t.content[..80])
            } else {
                t.content.clone()
            };
            parts.push(format!("[{}]: {}", t.role, snippet));
        }
        format!("Summary of {} turn(s): {}", n, parts.join(" | "))
    }

    /// Return the current active turns in chronological order.
    pub fn context_window(&self) -> Vec<&Turn> {
        let mut indices = self.state.active_indices.clone();
        indices.sort_unstable();
        indices
            .into_iter()
            .map(|idx| &self.state.history[idx])
            .collect()
    }

    /// Return the fraction of the token budget currently consumed.
    ///
    /// Returns `0.0` if `max_tokens` is zero.
    pub fn token_utilization(&self) -> f64 {
        if self.state.max_tokens == 0 {
            return 0.0;
        }
        self.state.tokens_used as f64 / self.state.max_tokens as f64
    }

    /// Total turns in the full history (including trimmed ones).
    pub fn history_len(&self) -> usize {
        self.state.history.len()
    }

    /// Turns currently in the active context window.
    pub fn active_len(&self) -> usize {
        self.state.active_indices.len()
    }
}

// ---------------------------------------------------------------------------
// DialoguePersistence
// ---------------------------------------------------------------------------

/// Serialize/deserialize a `DialogueState` to and from a JSON string.
pub struct DialoguePersistence;

impl DialoguePersistence {
    /// Serialize the given state to a JSON string.
    pub fn serialize(state: &DialogueState) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(state)
    }

    /// Deserialize a state from a JSON string previously produced by [`serialize`].
    pub fn deserialize(json: &str) -> Result<DialogueState, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> DialogueManager {
        DialogueManager::new(1000)
    }

    #[test]
    fn add_turn_updates_token_count() {
        let mut dm = make_manager();
        dm.add_turn(Role::User, "Hello", 10);
        dm.add_turn(Role::Assistant, "Hi there!", 15);
        assert_eq!(dm.state().tokens_used, 25);
        assert_eq!(dm.history_len(), 2);
        assert_eq!(dm.active_len(), 2);
    }

    #[test]
    fn context_window_returns_active_turns() {
        let mut dm = make_manager();
        dm.add_turn(Role::System, "You are helpful.", 5);
        dm.add_turn(Role::User, "What is 2+2?", 8);
        dm.add_turn(Role::Assistant, "4", 2);
        let window = dm.context_window();
        assert_eq!(window.len(), 3);
        assert_eq!(window[0].role, Role::System);
        assert_eq!(window[2].content, "4");
    }

    #[test]
    fn trim_to_budget_removes_oldest_non_system() {
        let mut dm = make_manager();
        dm.add_turn(Role::System, "System prompt.", 50);
        dm.add_turn(Role::User, "Turn 1", 400);
        dm.add_turn(Role::User, "Turn 2", 400);
        dm.add_turn(Role::Assistant, "Response", 300);

        // Budget of 800 should force removal of the first User turn (400 tokens).
        let removed = dm.trim_to_budget(800);
        assert!(removed >= 1);
        assert!(dm.state().tokens_used <= 800);
        // System turn must survive.
        let window = dm.context_window();
        assert!(window.iter().any(|t| t.role == Role::System));
    }

    #[test]
    fn trim_does_not_remove_system_turns() {
        let mut dm = DialogueManager::new(100);
        dm.add_turn(Role::System, "Stay.", 90);
        // Budget is already exceeded but only System turn exists.
        let removed = dm.trim_to_budget(50);
        assert_eq!(removed, 0);
        assert_eq!(dm.active_len(), 1);
    }

    #[test]
    fn summarize_oldest_returns_non_empty_string() {
        let mut dm = make_manager();
        dm.add_turn(Role::User, "First message", 5);
        dm.add_turn(Role::Assistant, "First reply", 5);
        let summary = dm.summarize_oldest(2);
        assert!(summary.contains("First message"));
        assert!(summary.contains("First reply"));
    }

    #[test]
    fn summarize_oldest_empty() {
        let dm = make_manager();
        let s = dm.summarize_oldest(5);
        assert!(s.contains("No turns"));
    }

    #[test]
    fn token_utilization_correct() {
        let mut dm = DialogueManager::new(200);
        dm.add_turn(Role::User, "hi", 100);
        let u = dm.token_utilization();
        assert!((u - 0.5).abs() < 1e-9);
    }

    #[test]
    fn token_utilization_zero_budget() {
        let dm = DialogueManager::new(0);
        assert_eq!(dm.token_utilization(), 0.0);
    }

    #[test]
    fn persistence_round_trip() {
        let mut dm = DialogueManager::new(500);
        dm.add_turn(Role::System, "sys", 10);
        dm.add_turn(Role::User, "hello", 20);
        dm.add_turn(Role::Assistant, "world", 30);

        let json = DialoguePersistence::serialize(dm.state()).unwrap();
        let restored = DialoguePersistence::deserialize(&json).unwrap();

        assert_eq!(restored.history.len(), 3);
        assert_eq!(restored.tokens_used, 60);
        assert_eq!(restored.max_tokens, 500);
        assert_eq!(restored.history[1].role, Role::User);
    }

    #[test]
    fn from_state_restores_correctly() {
        let mut dm1 = DialogueManager::new(300);
        dm1.add_turn(Role::User, "test", 50);
        let state = dm1.state().clone();

        let dm2 = DialogueManager::from_state(state);
        assert_eq!(dm2.state().tokens_used, 50);
        assert_eq!(dm2.history_len(), 1);
    }

    #[test]
    fn turn_with_timestamp() {
        let t = Turn::with_timestamp(Role::Tool, "result", 7, 123456789);
        assert_eq!(t.timestamp_ms, 123456789);
        assert_eq!(t.tokens, 7);
    }
}
