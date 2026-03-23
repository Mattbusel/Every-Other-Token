//! Context Window Manager
//!
//! Manages a fixed-size token budget for LLM context, with priority-based eviction
//! and canonical assembly ordering.

use std::collections::HashMap;

// ── BlockType ────────────────────────────────────────────────────────────────

/// The type of a context block, determining its canonical position in assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockType {
    /// System prompt — always first.
    System,
    /// Historical conversation turns.
    History,
    /// Retrieved documents or context.
    Document,
    /// The current user query — always last.
    Query,
}

// ── ContextBlock ─────────────────────────────────────────────────────────────

/// A single block of content within the context window.
#[derive(Debug, Clone)]
pub struct ContextBlock {
    /// Unique identifier for this block.
    pub id: u64,
    /// Token strings making up this block.
    pub content: Vec<String>,
    /// Priority (0 = lowest, 255 = highest). Higher-priority blocks survive eviction.
    pub priority: u8,
    /// Type of block for canonical ordering.
    pub block_type: BlockType,
}

impl ContextBlock {
    /// Number of tokens in this block.
    pub fn token_count(&self) -> usize {
        self.content.len()
    }
}

// ── ContextWindowConfig ───────────────────────────────────────────────────────

/// Configuration for a [`ContextWindow`].
#[derive(Debug, Clone)]
pub struct ContextWindowConfig {
    /// Maximum tokens the model can accept.
    pub max_tokens: usize,
    /// Tokens reserved for the model's output.
    pub reserved_for_output: usize,
    /// Tokens consumed by the system prompt (already counted outside of blocks).
    pub system_tokens: usize,
}

impl ContextWindowConfig {
    /// Available budget = max_tokens - reserved_for_output - system_tokens.
    pub fn available_budget(&self) -> usize {
        self.max_tokens
            .saturating_sub(self.reserved_for_output)
            .saturating_sub(self.system_tokens)
    }
}

impl Default for ContextWindowConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            reserved_for_output: 512,
            system_tokens: 128,
        }
    }
}

// ── ContextError ─────────────────────────────────────────────────────────────

/// Errors from context window operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    /// The budget would be exceeded even after all possible evictions.
    BudgetExceeded { needed: usize, available: usize },
    /// A single block exceeds the entire available budget.
    BlockTooLarge,
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextError::BudgetExceeded { needed, available } => {
                write!(f, "budget exceeded: needed {needed} tokens, only {available} available")
            }
            ContextError::BlockTooLarge => {
                write!(f, "block is too large to fit in the context window")
            }
        }
    }
}

impl std::error::Error for ContextError {}

// ── ContextStats ─────────────────────────────────────────────────────────────

/// Statistics about the current context window state.
#[derive(Debug, Clone, Default)]
pub struct ContextStats {
    /// Total number of blocks currently held.
    pub total_blocks: usize,
    /// Count of blocks by type.
    pub by_type: HashMap<BlockType, usize>,
    /// Fraction of available budget currently used ([0.0, 1.0]).
    pub utilization: f64,
    /// Total number of blocks evicted since window creation.
    pub evictions: u64,
}

// ── ContextWindow ─────────────────────────────────────────────────────────────

/// Manages a fixed-size token budget with priority-based eviction.
///
/// Blocks are added and eviction of lowest-priority [`BlockType::History`] blocks
/// is performed automatically when budget runs low. Assembly returns blocks in
/// canonical order: System → Document → History → Query.
pub struct ContextWindow {
    config: ContextWindowConfig,
    blocks: Vec<ContextBlock>,
    used_tokens: usize,
    evictions: u64,
}

impl ContextWindow {
    /// Create a new context window with the given configuration.
    pub fn new(config: ContextWindowConfig) -> Self {
        Self {
            config,
            blocks: Vec::new(),
            used_tokens: 0,
            evictions: 0,
        }
    }

    /// Available budget remaining after accounting for currently-held blocks.
    pub fn remaining_budget(&self) -> usize {
        self.config.available_budget().saturating_sub(self.used_tokens)
    }

    /// Fraction of the budget currently used ([0.0, 1.0]).
    pub fn utilization(&self) -> f64 {
        let budget = self.config.available_budget();
        if budget == 0 {
            return 1.0;
        }
        (self.used_tokens as f64 / budget as f64).min(1.0)
    }

    /// Add a block to the context window.
    ///
    /// If the block fits without eviction, it is inserted directly.
    /// If the budget is tight, the lowest-priority History blocks are evicted
    /// (ties broken by order: earlier blocks evicted first) until sufficient
    /// space is freed. Non-History blocks are never evicted.
    ///
    /// Returns [`ContextError::BlockTooLarge`] if the block itself exceeds the
    /// total available budget, or [`ContextError::BudgetExceeded`] if there
    /// are not enough evictable History blocks to make room.
    pub fn add_block(&mut self, block: ContextBlock) -> Result<(), ContextError> {
        let budget = self.config.available_budget();
        let needed = block.token_count();

        if needed > budget {
            return Err(ContextError::BlockTooLarge);
        }

        // Evict lowest-priority History blocks if needed.
        while self.used_tokens + needed > budget {
            // Find the lowest-priority History block (smallest priority value).
            // In case of ties, remove the one with the smallest index (oldest).
            let evict_idx = self
                .blocks
                .iter()
                .enumerate()
                .filter(|(_, b)| b.block_type == BlockType::History)
                .min_by_key(|(i, b)| (b.priority, *i))
                .map(|(i, _)| i);

            match evict_idx {
                Some(idx) => {
                    let evicted = self.blocks.remove(idx);
                    self.used_tokens = self.used_tokens.saturating_sub(evicted.token_count());
                    self.evictions += 1;
                }
                None => {
                    // No History blocks left to evict; still not enough room.
                    return Err(ContextError::BudgetExceeded {
                        needed,
                        available: budget.saturating_sub(self.used_tokens),
                    });
                }
            }
        }

        self.used_tokens += needed;
        self.blocks.push(block);
        Ok(())
    }

    /// Assemble all blocks in canonical order: System → Document → History → Query.
    ///
    /// Within each type, blocks appear in the order they were added (surviving eviction).
    pub fn assemble(&self) -> Vec<String> {
        let order = [BlockType::System, BlockType::Document, BlockType::History, BlockType::Query];
        let mut result = Vec::new();
        for block_type in &order {
            for block in self.blocks.iter().filter(|b| &b.block_type == block_type) {
                result.extend(block.content.iter().cloned());
            }
        }
        result
    }

    /// Remove all blocks and reset counters.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.used_tokens = 0;
    }

    /// Return current statistics.
    pub fn stats(&self) -> ContextStats {
        let mut by_type: HashMap<BlockType, usize> = HashMap::new();
        for block in &self.blocks {
            *by_type.entry(block.block_type).or_insert(0) += 1;
        }
        ContextStats {
            total_blocks: self.blocks.len(),
            by_type,
            utilization: self.utilization(),
            evictions: self.evictions,
        }
    }

    /// Number of tokens currently used.
    pub fn used_tokens(&self) -> usize {
        self.used_tokens
    }

    /// All blocks currently held (in insertion order).
    pub fn blocks(&self) -> &[ContextBlock] {
        &self.blocks
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(max: usize) -> ContextWindowConfig {
        ContextWindowConfig {
            max_tokens: max,
            reserved_for_output: 0,
            system_tokens: 0,
        }
    }

    fn make_block(id: u64, tokens: usize, priority: u8, block_type: BlockType) -> ContextBlock {
        ContextBlock {
            id,
            content: vec!["tok".to_string(); tokens],
            priority,
            block_type,
        }
    }

    // 1. Empty window has 0 utilization
    #[test]
    fn test_empty_utilization() {
        let w = ContextWindow::new(make_config(100));
        assert_eq!(w.utilization(), 0.0);
    }

    // 2. Adding a block increases used_tokens
    #[test]
    fn test_add_block_increases_used() {
        let mut w = ContextWindow::new(make_config(100));
        w.add_block(make_block(1, 10, 5, BlockType::Query)).unwrap();
        assert_eq!(w.used_tokens(), 10);
    }

    // 3. Utilization is fractional
    #[test]
    fn test_utilization_fractional() {
        let mut w = ContextWindow::new(make_config(100));
        w.add_block(make_block(1, 50, 5, BlockType::Query)).unwrap();
        assert!((w.utilization() - 0.5).abs() < 1e-9);
    }

    // 4. Block too large returns BlockTooLarge
    #[test]
    fn test_block_too_large() {
        let mut w = ContextWindow::new(make_config(10));
        let err = w.add_block(make_block(1, 11, 5, BlockType::Query)).unwrap_err();
        assert_eq!(err, ContextError::BlockTooLarge);
    }

    // 5. Exact budget fills without error
    #[test]
    fn test_exact_budget() {
        let mut w = ContextWindow::new(make_config(10));
        w.add_block(make_block(1, 10, 5, BlockType::Query)).unwrap();
        assert_eq!(w.utilization(), 1.0);
    }

    // 6. Evicts lowest-priority history block
    #[test]
    fn test_evicts_lowest_priority_history() {
        let mut w = ContextWindow::new(make_config(20));
        w.add_block(make_block(1, 10, 1, BlockType::History)).unwrap();
        w.add_block(make_block(2, 10, 9, BlockType::History)).unwrap();
        // Adding 5 more should evict block 1 (priority 1)
        w.add_block(make_block(3, 5, 5, BlockType::Query)).unwrap();
        assert_eq!(w.evictions, 1);
        assert!(w.blocks().iter().any(|b| b.id == 2));
        assert!(!w.blocks().iter().any(|b| b.id == 1));
    }

    // 7. BudgetExceeded when no history to evict
    #[test]
    fn test_budget_exceeded_no_history() {
        let mut w = ContextWindow::new(make_config(10));
        w.add_block(make_block(1, 8, 5, BlockType::Query)).unwrap();
        let err = w.add_block(make_block(2, 5, 5, BlockType::Document)).unwrap_err();
        assert!(matches!(err, ContextError::BudgetExceeded { .. }));
    }

    // 8. Canonical assembly order: System → Document → History → Query
    #[test]
    fn test_assemble_order() {
        let mut w = ContextWindow::new(make_config(100));
        w.add_block(ContextBlock {
            id: 1,
            content: vec!["Q".to_string()],
            priority: 5,
            block_type: BlockType::Query,
        }).unwrap();
        w.add_block(ContextBlock {
            id: 2,
            content: vec!["S".to_string()],
            priority: 5,
            block_type: BlockType::System,
        }).unwrap();
        w.add_block(ContextBlock {
            id: 3,
            content: vec!["D".to_string()],
            priority: 5,
            block_type: BlockType::Document,
        }).unwrap();
        w.add_block(ContextBlock {
            id: 4,
            content: vec!["H".to_string()],
            priority: 5,
            block_type: BlockType::History,
        }).unwrap();
        let assembled = w.assemble();
        assert_eq!(assembled, vec!["S", "D", "H", "Q"]);
    }

    // 9. Assemble returns empty when no blocks
    #[test]
    fn test_assemble_empty() {
        let w = ContextWindow::new(make_config(100));
        assert!(w.assemble().is_empty());
    }

    // 10. Stats total_blocks count
    #[test]
    fn test_stats_total_blocks() {
        let mut w = ContextWindow::new(make_config(100));
        w.add_block(make_block(1, 5, 5, BlockType::Query)).unwrap();
        w.add_block(make_block(2, 5, 5, BlockType::History)).unwrap();
        assert_eq!(w.stats().total_blocks, 2);
    }

    // 11. Stats by_type counts
    #[test]
    fn test_stats_by_type() {
        let mut w = ContextWindow::new(make_config(100));
        w.add_block(make_block(1, 5, 5, BlockType::History)).unwrap();
        w.add_block(make_block(2, 5, 5, BlockType::History)).unwrap();
        w.add_block(make_block(3, 5, 5, BlockType::Document)).unwrap();
        let s = w.stats();
        assert_eq!(*s.by_type.get(&BlockType::History).unwrap_or(&0), 2);
        assert_eq!(*s.by_type.get(&BlockType::Document).unwrap_or(&0), 1);
    }

    // 12. Eviction count tracked correctly
    #[test]
    fn test_eviction_count() {
        let mut w = ContextWindow::new(make_config(10));
        w.add_block(make_block(1, 6, 1, BlockType::History)).unwrap();
        w.add_block(make_block(2, 6, 1, BlockType::History)).unwrap();
        assert_eq!(w.evictions, 1);
        assert_eq!(w.stats().evictions, 1);
    }

    // 13. Multiple evictions chain correctly
    #[test]
    fn test_multiple_evictions() {
        let mut w = ContextWindow::new(make_config(15));
        w.add_block(make_block(1, 5, 1, BlockType::History)).unwrap();
        w.add_block(make_block(2, 5, 2, BlockType::History)).unwrap();
        w.add_block(make_block(3, 5, 3, BlockType::History)).unwrap();
        // Now at budget. Add 10-token block → evict 2 History blocks
        w.add_block(make_block(4, 10, 5, BlockType::Query)).unwrap();
        assert_eq!(w.evictions, 2);
    }

    // 14. Clear resets everything
    #[test]
    fn test_clear() {
        let mut w = ContextWindow::new(make_config(100));
        w.add_block(make_block(1, 10, 5, BlockType::Query)).unwrap();
        w.clear();
        assert_eq!(w.used_tokens(), 0);
        assert!(w.blocks().is_empty());
        assert_eq!(w.utilization(), 0.0);
    }

    // 15. available_budget respects reserved_for_output and system_tokens
    #[test]
    fn test_available_budget() {
        let cfg = ContextWindowConfig {
            max_tokens: 100,
            reserved_for_output: 20,
            system_tokens: 10,
        };
        assert_eq!(cfg.available_budget(), 70);
    }

    // 16. Remaining budget decreases after adding block
    #[test]
    fn test_remaining_budget() {
        let mut w = ContextWindow::new(make_config(100));
        assert_eq!(w.remaining_budget(), 100);
        w.add_block(make_block(1, 30, 5, BlockType::Query)).unwrap();
        assert_eq!(w.remaining_budget(), 70);
    }

    // 17. Priority tie-breaking evicts oldest (lowest index) block first
    #[test]
    fn test_eviction_tie_breaks_oldest_first() {
        let mut w = ContextWindow::new(make_config(20));
        w.add_block(make_block(10, 10, 5, BlockType::History)).unwrap();
        w.add_block(make_block(11, 10, 5, BlockType::History)).unwrap();
        // Add 5 more: evicts block 10 (same priority, inserted first)
        w.add_block(make_block(12, 5, 5, BlockType::Query)).unwrap();
        assert!(!w.blocks().iter().any(|b| b.id == 10));
        assert!(w.blocks().iter().any(|b| b.id == 11));
    }

    // 18. Non-History blocks are never evicted
    #[test]
    fn test_non_history_not_evicted() {
        let mut w = ContextWindow::new(make_config(10));
        w.add_block(make_block(1, 8, 1, BlockType::Document)).unwrap();
        let err = w.add_block(make_block(2, 5, 5, BlockType::Query)).unwrap_err();
        assert!(matches!(err, ContextError::BudgetExceeded { .. }));
        // Document block still present
        assert!(w.blocks().iter().any(|b| b.id == 1));
    }

    // 19. Zero-token block is valid
    #[test]
    fn test_zero_token_block() {
        let mut w = ContextWindow::new(make_config(10));
        w.add_block(make_block(1, 0, 5, BlockType::Query)).unwrap();
        assert_eq!(w.used_tokens(), 0);
    }

    // 20. Utilization is 1.0 when budget is 0
    #[test]
    fn test_utilization_zero_budget() {
        let cfg = ContextWindowConfig {
            max_tokens: 10,
            reserved_for_output: 5,
            system_tokens: 5,
        };
        let w = ContextWindow::new(cfg);
        assert_eq!(w.utilization(), 1.0);
    }

    // 21. BlockTooLarge error message is non-empty
    #[test]
    fn test_error_display() {
        let e = ContextError::BudgetExceeded { needed: 10, available: 5 };
        assert!(!e.to_string().is_empty());
        assert!(!ContextError::BlockTooLarge.to_string().is_empty());
    }

    // 22. Multiple System blocks assemble consecutively
    #[test]
    fn test_multiple_system_blocks() {
        let mut w = ContextWindow::new(make_config(100));
        w.add_block(ContextBlock {
            id: 1,
            content: vec!["S1".to_string()],
            priority: 5,
            block_type: BlockType::System,
        }).unwrap();
        w.add_block(ContextBlock {
            id: 2,
            content: vec!["S2".to_string()],
            priority: 5,
            block_type: BlockType::System,
        }).unwrap();
        let assembled = w.assemble();
        assert_eq!(&assembled[..2], &["S1", "S2"]);
    }
}
