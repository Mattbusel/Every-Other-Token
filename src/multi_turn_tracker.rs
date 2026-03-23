//! Multi-turn conversation state tracking.
//!
//! Tracks conversation turns across multiple interactions, with topic analysis,
//! context windowing, and conversation summarization.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TurnRole
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TurnRole {
    User,
    Assistant,
    System,
    Tool,
}

// ---------------------------------------------------------------------------
// Turn
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Turn {
    pub id: u64,
    pub role: TurnRole,
    pub content: String,
    pub timestamp_ms: u64,
    pub metadata: HashMap<String, String>,
    pub token_count: usize,
}

impl Turn {
    fn estimate_tokens(content: &str) -> usize {
        // Rough approximation: ~4 chars per token
        (content.len() / 4).max(1)
    }
}

// ---------------------------------------------------------------------------
// ConversationState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationState {
    Active,
    Paused,
    Completed,
    Abandoned,
}

// ---------------------------------------------------------------------------
// TurnError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnError {
    MaxTurnsExceeded,
    InvalidStateTransition,
    EmptyContent,
}

impl std::fmt::Display for TurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TurnError::MaxTurnsExceeded => write!(f, "Maximum number of turns exceeded"),
            TurnError::InvalidStateTransition => write!(f, "Invalid state transition"),
            TurnError::EmptyContent => write!(f, "Turn content must not be empty"),
        }
    }
}

impl std::error::Error for TurnError {}

// ---------------------------------------------------------------------------
// TopicTracker
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TopicTracker {
    pub topics: Vec<String>,
    pub weights: Vec<f64>,
}

impl TopicTracker {
    pub fn new() -> Self {
        Self {
            topics: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Extract key terms from `text` and update topic weights with exponential decay.
    pub fn update(&mut self, text: &str) {
        // Apply decay to existing weights
        for w in self.weights.iter_mut() {
            *w *= 0.85;
        }

        // Extract candidate terms: capitalized words or words longer than 6 chars
        let candidates: Vec<String> = text
            .split_whitespace()
            .filter_map(|word| {
                let clean: String = word
                    .chars()
                    .filter(|c| c.is_alphabetic())
                    .collect();
                if clean.len() < 2 {
                    return None;
                }
                let is_capitalized = clean.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
                let is_long = clean.len() > 6;
                if is_capitalized || is_long {
                    Some(clean.to_lowercase())
                } else {
                    None
                }
            })
            .collect();

        for term in candidates {
            if let Some(pos) = self.topics.iter().position(|t| t == &term) {
                self.weights[pos] += 1.0;
            } else {
                self.topics.push(term);
                self.weights.push(1.0);
            }
        }

        // Remove very-low-weight topics
        let threshold = 0.01;
        let mut i = 0;
        while i < self.topics.len() {
            if self.weights[i] < threshold {
                self.topics.remove(i);
                self.weights.remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Return the topic with the highest weight.
    pub fn primary_topic(&self) -> Option<&str> {
        self.weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| self.topics[i].as_str())
    }
}

impl Default for TopicTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ConversationSummary
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub total_turns: usize,
    pub user_turns: usize,
    pub assistant_turns: usize,
    pub total_tokens: usize,
    pub primary_topic: Option<String>,
    pub duration_ms: u64,
}

// ---------------------------------------------------------------------------
// MultiTurnConversation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MultiTurnConversation {
    pub id: String,
    pub turns: Vec<Turn>,
    pub state: ConversationState,
    pub topic_tracker: TopicTracker,
    pub created_at_ms: u64,
    pub max_turns: usize,
}

impl MultiTurnConversation {
    pub fn new(id: String, max_turns: usize) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            id,
            turns: Vec::new(),
            state: ConversationState::Active,
            topic_tracker: TopicTracker::new(),
            created_at_ms: now,
            max_turns,
        }
    }

    /// Add a turn to the conversation. Returns the new turn ID on success.
    pub fn add_turn(
        &mut self,
        role: TurnRole,
        content: String,
        timestamp_ms: u64,
    ) -> Result<u64, TurnError> {
        if content.trim().is_empty() {
            return Err(TurnError::EmptyContent);
        }
        if self.state != ConversationState::Active {
            return Err(TurnError::InvalidStateTransition);
        }
        if self.turns.len() >= self.max_turns {
            return Err(TurnError::MaxTurnsExceeded);
        }

        let token_count = Turn::estimate_tokens(&content);
        self.topic_tracker.update(&content);

        let id = self.turns.len() as u64;
        self.turns.push(Turn {
            id,
            role,
            content,
            timestamp_ms,
            metadata: HashMap::new(),
            token_count,
        });
        Ok(id)
    }

    /// Return the last `n` turns.
    pub fn last_n_turns(&self, n: usize) -> Vec<&Turn> {
        let start = self.turns.len().saturating_sub(n);
        self.turns[start..].iter().collect()
    }

    /// Return all turns with the given role.
    pub fn turns_by_role(&self, role: &TurnRole) -> Vec<&Turn> {
        self.turns.iter().filter(|t| &t.role == role).collect()
    }

    /// Return the most recent turns that fit within `max_tokens`.
    pub fn context_window(&self, max_tokens: usize) -> Vec<&Turn> {
        let mut budget = max_tokens;
        let mut result: Vec<&Turn> = Vec::new();
        for turn in self.turns.iter().rev() {
            if turn.token_count > budget {
                break;
            }
            budget -= turn.token_count;
            result.push(turn);
        }
        result.reverse();
        result
    }

    /// Produce a summary of this conversation.
    pub fn conversation_summary(&self) -> ConversationSummary {
        let total_turns = self.turns.len();
        let user_turns = self.turns.iter().filter(|t| t.role == TurnRole::User).count();
        let assistant_turns = self.turns.iter().filter(|t| t.role == TurnRole::Assistant).count();
        let total_tokens = self.turns.iter().map(|t| t.token_count).sum();
        let primary_topic = self.topic_tracker.primary_topic().map(String::from);
        let duration_ms = if let (Some(first), Some(last)) = (self.turns.first(), self.turns.last()) {
            last.timestamp_ms.saturating_sub(first.timestamp_ms)
        } else {
            0
        };
        ConversationSummary {
            total_turns,
            user_turns,
            assistant_turns,
            total_tokens,
            primary_topic,
            duration_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// MultiTurnManager
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct MultiTurnManager {
    pub conversations: HashMap<String, MultiTurnConversation>,
    next_id: u64,
}

impl MultiTurnManager {
    pub fn new() -> Self {
        Self {
            conversations: HashMap::new(),
            next_id: 0,
        }
    }

    /// Create a new conversation and return its ID.
    pub fn create(&mut self, max_turns: usize) -> String {
        let id = format!("conv-{}", self.next_id);
        self.next_id += 1;
        self.conversations
            .insert(id.clone(), MultiTurnConversation::new(id.clone(), max_turns));
        id
    }

    pub fn get(&self, id: &str) -> Option<&MultiTurnConversation> {
        self.conversations.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut MultiTurnConversation> {
        self.conversations.get_mut(id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ts() -> u64 {
        1_000_000
    }

    #[test]
    fn test_add_turns() {
        let mut conv = MultiTurnConversation::new("test".into(), 10);
        let id = conv.add_turn(TurnRole::User, "Hello world".into(), ts()).unwrap();
        assert_eq!(id, 0);
        let id2 = conv.add_turn(TurnRole::Assistant, "Hi there!".into(), ts() + 100).unwrap();
        assert_eq!(id2, 1);
        assert_eq!(conv.turns.len(), 2);
    }

    #[test]
    fn test_max_turns_exceeded() {
        let mut conv = MultiTurnConversation::new("test".into(), 2);
        conv.add_turn(TurnRole::User, "turn one".into(), ts()).unwrap();
        conv.add_turn(TurnRole::Assistant, "turn two".into(), ts()).unwrap();
        let err = conv.add_turn(TurnRole::User, "turn three".into(), ts()).unwrap_err();
        assert_eq!(err, TurnError::MaxTurnsExceeded);
    }

    #[test]
    fn test_empty_content_error() {
        let mut conv = MultiTurnConversation::new("test".into(), 10);
        let err = conv.add_turn(TurnRole::User, "   ".into(), ts()).unwrap_err();
        assert_eq!(err, TurnError::EmptyContent);
    }

    #[test]
    fn test_last_n_returns_correct() {
        let mut conv = MultiTurnConversation::new("test".into(), 20);
        for i in 0..10u64 {
            conv.add_turn(TurnRole::User, format!("message {}", i), ts() + i * 100).unwrap();
        }
        let last3 = conv.last_n_turns(3);
        assert_eq!(last3.len(), 3);
        assert!(last3[2].content.contains("9"));
    }

    #[test]
    fn test_context_window_respects_budget() {
        let mut conv = MultiTurnConversation::new("test".into(), 20);
        // each short message ~1 token, longer messages more
        conv.add_turn(TurnRole::User, "a".repeat(40), ts()).unwrap(); // ~10 tokens
        conv.add_turn(TurnRole::User, "b".repeat(40), ts() + 1).unwrap(); // ~10 tokens
        conv.add_turn(TurnRole::User, "c".repeat(40), ts() + 2).unwrap(); // ~10 tokens
        // Budget of 15 tokens should get at most 1-2 turns from the end
        let window = conv.context_window(15);
        let total: usize = window.iter().map(|t| t.token_count).sum();
        assert!(total <= 15);
    }

    #[test]
    fn test_summary_counts() {
        let mut conv = MultiTurnConversation::new("test".into(), 20);
        conv.add_turn(TurnRole::User, "Hello assistant".into(), ts()).unwrap();
        conv.add_turn(TurnRole::Assistant, "Hello user, how can I help?".into(), ts() + 500).unwrap();
        conv.add_turn(TurnRole::User, "Tell me about Rust programming".into(), ts() + 1000).unwrap();
        let summary = conv.conversation_summary();
        assert_eq!(summary.total_turns, 3);
        assert_eq!(summary.user_turns, 2);
        assert_eq!(summary.assistant_turns, 1);
        assert!(summary.total_tokens > 0);
        assert_eq!(summary.duration_ms, 1000);
    }

    #[test]
    fn test_topic_tracker_updates() {
        let mut tracker = TopicTracker::new();
        tracker.update("Discussing Rust programming language features");
        tracker.update("Rust ownership system is interesting");
        assert!(!tracker.topics.is_empty());
        let primary = tracker.primary_topic();
        assert!(primary.is_some());
    }

    #[test]
    fn test_multi_turn_manager_create_get() {
        let mut mgr = MultiTurnManager::new();
        let id = mgr.create(10);
        assert!(mgr.get(&id).is_some());
        let conv = mgr.get_mut(&id).unwrap();
        conv.add_turn(TurnRole::User, "test message".into(), ts()).unwrap();
        assert_eq!(mgr.get(&id).unwrap().turns.len(), 1);
    }
}
