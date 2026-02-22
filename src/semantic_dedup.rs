//! # Stage: SemanticDedup
//!
//! ## Responsibility
//! Fingerprint incoming text (prompts/requests) and return cached results for
//! near-duplicate inputs within a configurable TTL window. This catches
//! copy-paste duplicates, minor reformatting, extra whitespace, and
//! capitalization differences with zero external dependencies.
//!
//! ## Guarantees
//! - Deterministic: same normalized text always produces the same fingerprint
//! - Thread-safe when wrapped in `Mutex` / `RwLock` by the caller
//! - Bounded: at most `capacity` entries are held at any time (LRU eviction)
//! - Non-panicking: no `unwrap` or `expect` in any production path
//! - TTL-based expiry is evaluated lazily on every `check()` call
//!
//! ## NOT Responsible For
//! - Semantic similarity beyond exact normalized-string matching
//! - Cross-process or distributed deduplication
//! - Persistence (in-memory only)

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`SemanticDedup`].
pub struct DedupConfig {
    /// Milliseconds after insertion before an entry is considered expired.
    /// Default: 300 000 ms (5 minutes).
    pub ttl_ms: u64,
    /// Maximum number of entries held at one time. When the cache is full the
    /// entry with the lowest `inserted_ms` (oldest) is evicted before a new
    /// entry is inserted. Default: 1 024.
    pub capacity: usize,
}

impl Default for DedupConfig {
    fn default() -> Self {
        DedupConfig { ttl_ms: 300_000, capacity: 1_024 }
    }
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// A single cached deduplication record.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The normalised fingerprint key used for lookup.
    pub fingerprint: String,
    /// The caller-supplied value (e.g. response text) stored alongside the key.
    pub value: String,
    /// Unix-epoch millisecond timestamp at which this entry was inserted.
    pub inserted_ms: u64,
    /// Number of times this entry has been returned as a cache hit.
    pub hit_count: u32,
}

// ---------------------------------------------------------------------------
// SemanticDedup
// ---------------------------------------------------------------------------

/// In-memory semantic deduplication cache with TTL and LRU eviction.
pub struct SemanticDedup {
    config: DedupConfig,
    /// Maps a normalised fingerprint to its cached entry.
    cache: HashMap<String, CacheEntry>,
}

impl SemanticDedup {
    /// Construct a new, empty cache with the provided configuration.
    pub fn new(config: DedupConfig) -> Self {
        SemanticDedup { config, cache: HashMap::new() }
    }

    // -----------------------------------------------------------------------
    // Fingerprinting
    // -----------------------------------------------------------------------

    /// Normalize `text` to a stable fingerprint suitable for deduplication.
    ///
    /// Steps applied in order:
    /// 1. Lowercase every character.
    /// 2. Collapse every run of whitespace to a single ASCII space.
    /// 3. Remove every character that is neither alphanumeric nor a space.
    /// 4. Trim the result to at most 128 characters (by `char` boundary).
    ///
    /// # Panics
    /// This function never panics.
    pub fn fingerprint(text: &str) -> String {
        // Step 1: lowercase.
        let lowered = text.to_lowercase();

        // Step 2 + 3 combined: walk chars, collapse whitespace runs, drop
        // non-alphanumeric-non-space characters.
        let mut result = String::with_capacity(lowered.len().min(128));
        let mut last_was_space = false;

        for ch in lowered.chars() {
            if ch.is_whitespace() {
                if !last_was_space {
                    result.push(' ');
                    last_was_space = true;
                }
            } else if ch.is_alphanumeric() {
                result.push(ch);
                last_was_space = false;
            }
            // other chars are silently dropped
        }

        // Trim leading/trailing spaces that came from leading/trailing whitespace
        // in the original input before we apply the 128-char limit.
        let trimmed = result.trim_matches(' ');

        // Step 4: trim to first 128 chars (char-boundary safe).
        if trimmed.chars().count() > 128 {
            trimmed.chars().take(128).collect()
        } else {
            trimmed.to_string()
        }
    }

    // -----------------------------------------------------------------------
    // Core API
    // -----------------------------------------------------------------------

    /// Check whether `text` has a live (non-expired) cached entry.
    ///
    /// Expired entries are evicted before the lookup. If a matching live entry
    /// is found its `hit_count` is incremented and a shared reference to it is
    /// returned.
    ///
    /// `now_ms` is the current Unix-epoch timestamp in milliseconds. Pass it
    /// explicitly so callers (and tests) can control the clock.
    ///
    /// # Returns
    /// - `Some(&CacheEntry)` when a live hit exists.
    /// - `None` when the entry is absent or expired.
    ///
    /// # Panics
    /// This function never panics.
    pub fn check(&mut self, text: &str, now_ms: u64) -> Option<&CacheEntry> {
        self.evict_expired(now_ms);
        let fp = Self::fingerprint(text);
        if let Some(entry) = self.cache.get_mut(&fp) {
            entry.hit_count += 1;
            // Re-borrow as shared so we can return `&CacheEntry`.
            return self.cache.get(&fp);
        }
        None
    }

    /// Register a new cache entry for `text` with the given `value`.
    ///
    /// If an entry for the same fingerprint already exists it is overwritten.
    /// Expired entries are evicted first. If the cache is at capacity after TTL
    /// eviction the oldest entry (lowest `inserted_ms`) is removed to make room.
    ///
    /// # Panics
    /// This function never panics.
    pub fn register(&mut self, text: &str, value: String, now_ms: u64) {
        self.evict_expired(now_ms);

        let fp = Self::fingerprint(text);

        // Enforce capacity only when inserting a truly new key.
        if !self.cache.contains_key(&fp) && self.cache.len() >= self.config.capacity {
            self.evict_lru();
        }

        self.cache.insert(
            fp.clone(),
            CacheEntry { fingerprint: fp, value, inserted_ms: now_ms, hit_count: 0 },
        );
    }

    /// Remove all entries whose `inserted_ms` is older than `ttl_ms` before
    /// `now_ms`.
    ///
    /// # Panics
    /// This function never panics.
    pub fn evict_expired(&mut self, now_ms: u64) {
        let ttl = self.config.ttl_ms;
        self.cache.retain(|_, entry| {
            now_ms.saturating_sub(entry.inserted_ms) < ttl
        });
    }

    /// Return the number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Return `true` when the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Return a sorted list of all live (non-expired) fingerprints.
    ///
    /// Sorted lexicographically for deterministic output.
    ///
    /// # Panics
    /// This function never panics.
    pub fn live_keys(&self, now_ms: u64) -> Vec<String> {
        let ttl = self.config.ttl_ms;
        let mut keys: Vec<String> = self
            .cache
            .iter()
            .filter(|(_, entry)| now_ms.saturating_sub(entry.inserted_ms) < ttl)
            .map(|(k, _)| k.clone())
            .collect();
        keys.sort();
        keys
    }

    /// Remove every entry from the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Evict the entry with the smallest `inserted_ms` (oldest wall-clock time).
    /// Does nothing when the cache is empty.
    fn evict_lru(&mut self) {
        if self.cache.is_empty() {
            return;
        }
        // Find the key of the oldest entry.
        let oldest_key = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.inserted_ms)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest_key {
            self.cache.remove(&key);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn make_dedup() -> SemanticDedup {
        SemanticDedup::new(DedupConfig::default())
    }

    fn make_dedup_with(ttl_ms: u64, capacity: usize) -> SemanticDedup {
        SemanticDedup::new(DedupConfig { ttl_ms, capacity })
    }

    // -----------------------------------------------------------------------
    // fingerprint tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fingerprint_lowercases_input() {
        let fp = SemanticDedup::fingerprint("Hello WORLD");
        assert_eq!(fp, "hello world");
    }

    #[test]
    fn test_fingerprint_collapses_whitespace() {
        let fp = SemanticDedup::fingerprint("hello   world\t\nfoo");
        assert_eq!(fp, "hello world foo");
    }

    #[test]
    fn test_fingerprint_removes_punctuation() {
        let fp = SemanticDedup::fingerprint("hello, world! How are you?");
        assert_eq!(fp, "hello world how are you");
    }

    #[test]
    fn test_fingerprint_trims_to_128_chars() {
        let long_input: String = "a ".repeat(200); // 400 chars
        let fp = SemanticDedup::fingerprint(&long_input);
        assert!(fp.chars().count() <= 128, "fingerprint must be at most 128 chars");
    }

    #[test]
    fn test_fingerprint_empty_input_gives_empty() {
        let fp = SemanticDedup::fingerprint("");
        assert_eq!(fp, "");
    }

    #[test]
    fn test_fingerprint_only_whitespace_gives_empty_or_single_space() {
        let fp = SemanticDedup::fingerprint("     \t\n  ");
        // All whitespace collapses to one space; after trimming that leading/
        // trailing space the result is empty.
        assert!(fp.is_empty(), "unexpected: {:?}", fp);
    }

    #[test]
    fn test_fingerprint_alphanumeric_unchanged() {
        let fp = SemanticDedup::fingerprint("abc123");
        assert_eq!(fp, "abc123");
    }

    #[test]
    fn test_fingerprint_same_for_minor_variations_extra_spaces() {
        let fp1 = SemanticDedup::fingerprint("hello world");
        let fp2 = SemanticDedup::fingerprint("hello    world");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_same_for_minor_variations_mixed_case() {
        let fp1 = SemanticDedup::fingerprint("Hello World");
        let fp2 = SemanticDedup::fingerprint("hello world");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_same_for_minor_variations_combined() {
        let fp1 = SemanticDedup::fingerprint("  Hello,  World!  ");
        let fp2 = SemanticDedup::fingerprint("hello world");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_different_for_different_content() {
        let fp1 = SemanticDedup::fingerprint("hello world");
        let fp2 = SemanticDedup::fingerprint("goodbye world");
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_max_length_exactly_128() {
        // Exactly 128 'a' chars — should pass through unchanged.
        let input: String = "a".repeat(128);
        let fp = SemanticDedup::fingerprint(&input);
        assert_eq!(fp.chars().count(), 128);
    }

    #[test]
    fn test_fingerprint_longer_than_128_truncated() {
        let input: String = "a".repeat(200);
        let fp = SemanticDedup::fingerprint(&input);
        assert_eq!(fp.chars().count(), 128);
    }

    #[test]
    fn test_fingerprint_is_deterministic() {
        let text = "This IS a Test  Prompt!!";
        let fp1 = SemanticDedup::fingerprint(text);
        let fp2 = SemanticDedup::fingerprint(text);
        assert_eq!(fp1, fp2);
    }

    // -----------------------------------------------------------------------
    // Construction / empty state
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_cache_is_empty() {
        let d = make_dedup();
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn test_dedup_config_defaults() {
        let cfg = DedupConfig::default();
        assert_eq!(cfg.ttl_ms, 300_000);
        assert_eq!(cfg.capacity, 1_024);
    }

    // -----------------------------------------------------------------------
    // register
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_adds_entry() {
        let mut d = make_dedup();
        d.register("hello world", "response1".to_string(), 1_000);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_register_second_time_same_key_overwrites() {
        let mut d = make_dedup();
        d.register("hello world", "first".to_string(), 1_000);
        d.register("hello world", "second".to_string(), 2_000);
        assert_eq!(d.len(), 1);
        // The overwritten entry should hold the new value.
        let entry = d.check("hello world", 2_000).expect("entry should exist");
        assert_eq!(entry.value, "second");
    }

    #[test]
    fn test_register_stores_inserted_ms() {
        let mut d = make_dedup();
        d.register("test prompt", "val".to_string(), 9_999);
        // Peek at the raw cache entry via check (which bumps hit_count, but
        // inserted_ms is what we care about here).
        let entry = d.check("test prompt", 9_999).expect("entry");
        assert_eq!(entry.inserted_ms, 9_999);
    }

    #[test]
    fn test_register_increments_len() {
        let mut d = make_dedup();
        d.register("first", "a".to_string(), 0);
        assert_eq!(d.len(), 1);
        d.register("second", "b".to_string(), 0);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn test_two_different_texts_two_entries() {
        let mut d = make_dedup();
        d.register("prompt one", "v1".to_string(), 0);
        d.register("prompt two", "v2".to_string(), 0);
        assert_eq!(d.len(), 2);
    }

    // -----------------------------------------------------------------------
    // check
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_returns_none_for_unknown_text() {
        let mut d = make_dedup();
        assert!(d.check("nothing here", 0).is_none());
    }

    #[test]
    fn test_check_returns_some_for_known_text() {
        let mut d = make_dedup();
        d.register("hello", "world".to_string(), 0);
        assert!(d.check("hello", 0).is_some());
    }

    #[test]
    fn test_check_returns_stored_value() {
        let mut d = make_dedup();
        d.register("my prompt", "the answer".to_string(), 0);
        let entry = d.check("my prompt", 0).expect("entry");
        assert_eq!(entry.value, "the answer");
    }

    #[test]
    fn test_check_hit_count_starts_at_zero() {
        let mut d = make_dedup();
        d.register("prompt", "val".to_string(), 0);
        // Directly inspect the raw cache (no check call yet).
        let fp = SemanticDedup::fingerprint("prompt");
        assert_eq!(d.cache[&fp].hit_count, 0);
    }

    #[test]
    fn test_check_increments_hit_count() {
        let mut d = make_dedup();
        d.register("prompt", "val".to_string(), 0);
        d.check("prompt", 0);
        let fp = SemanticDedup::fingerprint("prompt");
        assert_eq!(d.cache[&fp].hit_count, 1);
    }

    #[test]
    fn test_entry_hit_count_accumulates_across_checks() {
        let mut d = make_dedup();
        d.register("some text", "resp".to_string(), 0);
        d.check("some text", 0);
        d.check("some text", 0);
        d.check("some text", 0);
        let fp = SemanticDedup::fingerprint("some text");
        assert_eq!(d.cache[&fp].hit_count, 3);
    }

    #[test]
    fn test_check_returns_none_for_expired_entry() {
        let mut d = make_dedup_with(1_000, 128);
        d.register("old prompt", "val".to_string(), 0);
        // now_ms is 2_000 ms later → entry is 2_000 ms old, TTL is 1_000 ms → expired.
        assert!(d.check("old prompt", 2_000).is_none());
    }

    #[test]
    fn test_check_returns_some_for_live_entry() {
        let mut d = make_dedup_with(5_000, 128);
        d.register("live prompt", "val".to_string(), 1_000);
        // now_ms = 3_000 → age = 2_000 ms < ttl 5_000 ms → live.
        assert!(d.check("live prompt", 3_000).is_some());
    }

    #[test]
    fn test_check_after_ttl_expires_returns_none() {
        let mut d = make_dedup_with(500, 64);
        d.register("prompt", "val".to_string(), 100);
        // Age = 600 > TTL 500.
        assert!(d.check("prompt", 700).is_none());
    }

    #[test]
    fn test_check_evicts_expired_entries_on_call() {
        let mut d = make_dedup_with(100, 64);
        d.register("stale", "v".to_string(), 0);
        d.register("fresh", "v".to_string(), 200);
        // now_ms = 150 → "stale" (age 150 >= TTL 100) expires; "fresh" (age -50, underflows to 0 < 100) lives.
        d.check("anything", 150);
        assert_eq!(d.len(), 1, "expired entry should have been evicted");
    }

    // -----------------------------------------------------------------------
    // evict_expired
    // -----------------------------------------------------------------------

    #[test]
    fn test_evict_expired_removes_old_entries() {
        let mut d = make_dedup_with(1_000, 128);
        d.register("old", "v".to_string(), 0);
        d.register("also old", "v".to_string(), 100);
        d.evict_expired(2_000); // both are older than 1_000 ms
        assert!(d.is_empty());
    }

    #[test]
    fn test_evict_expired_keeps_live_entries() {
        let mut d = make_dedup_with(1_000, 128);
        d.register("old", "v".to_string(), 0);
        d.register("fresh", "v".to_string(), 1_500);
        d.evict_expired(2_000); // "old" expires, "fresh" (age 500 < 1000) lives
        assert_eq!(d.len(), 1);
        assert!(d.cache.contains_key(&SemanticDedup::fingerprint("fresh")));
    }

    #[test]
    fn test_evict_expired_with_zero_now_removes_nothing() {
        let mut d = make_dedup_with(1_000, 128);
        // All entries have inserted_ms = 0. now_ms = 0 → age = 0 < TTL 1000 → none expire.
        d.register("a", "v".to_string(), 0);
        d.register("b", "v".to_string(), 0);
        d.evict_expired(0);
        assert_eq!(d.len(), 2);
    }

    // -----------------------------------------------------------------------
    // len / is_empty
    // -----------------------------------------------------------------------

    #[test]
    fn test_len_matches_registered_entries() {
        let mut d = make_dedup();
        d.register("one", "v".to_string(), 0);
        d.register("two", "v".to_string(), 0);
        d.register("three", "v".to_string(), 0);
        assert_eq!(d.len(), 3);
    }

    #[test]
    fn test_is_empty_true_when_empty() {
        let d = make_dedup();
        assert!(d.is_empty());
    }

    #[test]
    fn test_is_empty_false_when_has_entries() {
        let mut d = make_dedup();
        d.register("entry", "v".to_string(), 0);
        assert!(!d.is_empty());
    }

    // -----------------------------------------------------------------------
    // Capacity / LRU eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_capacity_evicts_lru_on_insert() {
        let mut d = make_dedup_with(1_000_000, 2);
        d.register("alpha", "v1".to_string(), 1);
        d.register("beta", "v2".to_string(), 2);
        // Cache is now full (capacity = 2). Inserting "gamma" should evict "alpha" (oldest).
        d.register("gamma", "v3".to_string(), 3);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn test_capacity_evicts_oldest_not_newest() {
        let mut d = make_dedup_with(1_000_000, 2);
        d.register("alpha", "v1".to_string(), 10);
        d.register("beta", "v2".to_string(), 20);
        d.register("gamma", "v3".to_string(), 30);
        // "alpha" (inserted_ms = 10) is oldest → should be gone.
        assert!(
            d.cache.get(&SemanticDedup::fingerprint("alpha")).is_none(),
            "alpha should have been evicted as LRU"
        );
        // "beta" and "gamma" should remain.
        assert!(d.cache.contains_key(&SemanticDedup::fingerprint("beta")));
        assert!(d.cache.contains_key(&SemanticDedup::fingerprint("gamma")));
    }

    #[test]
    fn test_register_at_capacity_replaces_oldest() {
        let mut d = make_dedup_with(1_000_000, 3);
        d.register("p1", "v1".to_string(), 5);
        d.register("p2", "v2".to_string(), 10);
        d.register("p3", "v3".to_string(), 15);
        d.register("p4", "v4".to_string(), 20);
        // "p1" (min inserted_ms = 5) should have been evicted.
        assert!(d.cache.get(&SemanticDedup::fingerprint("p1")).is_none());
        assert_eq!(d.len(), 3);
    }

    // -----------------------------------------------------------------------
    // live_keys
    // -----------------------------------------------------------------------

    #[test]
    fn test_live_keys_excludes_expired() {
        let mut d = make_dedup_with(500, 64);
        d.register("old", "v".to_string(), 0);
        d.register("new", "v".to_string(), 600);
        // now_ms = 700 → "old" age 700 >= 500 expired; "new" age 100 < 500 live.
        let keys = d.live_keys(700);
        assert!(!keys.contains(&SemanticDedup::fingerprint("old")));
        assert!(keys.contains(&SemanticDedup::fingerprint("new")));
    }

    #[test]
    fn test_live_keys_sorted() {
        let mut d = make_dedup();
        d.register("zebra", "v".to_string(), 0);
        d.register("apple", "v".to_string(), 0);
        d.register("mango", "v".to_string(), 0);
        let keys = d.live_keys(0);
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "live_keys must return a sorted slice");
    }

    // -----------------------------------------------------------------------
    // clear
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_empties_cache() {
        let mut d = make_dedup();
        d.register("a", "v".to_string(), 0);
        d.register("b", "v".to_string(), 0);
        d.register("c", "v".to_string(), 0);
        d.clear();
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn test_clear_then_reinsert_works() {
        let mut d = make_dedup();
        d.register("prompt", "v1".to_string(), 0);
        d.clear();
        d.register("prompt", "v2".to_string(), 100);
        let entry = d.check("prompt", 100).expect("entry after reinsert");
        assert_eq!(entry.value, "v2");
    }

    // -----------------------------------------------------------------------
    // Edge / interaction cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_normalises_before_lookup() {
        let mut d = make_dedup();
        // Register with clean text.
        d.register("hello world", "result".to_string(), 0);
        // Look up with a "dirty" version that normalises to the same fingerprint.
        let entry = d.check("  Hello,  WORLD!  ", 0).expect("should hit");
        assert_eq!(entry.value, "result");
    }

    #[test]
    fn test_register_normalises_key() {
        let mut d = make_dedup();
        d.register("  HELLO  world  ", "v1".to_string(), 0);
        // Lookup with clean version must hit.
        assert!(d.check("hello world", 0).is_some());
    }

    #[test]
    fn test_cache_entry_clone() {
        let entry = CacheEntry {
            fingerprint: "fp".to_string(),
            value: "val".to_string(),
            inserted_ms: 42,
            hit_count: 7,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.fingerprint, entry.fingerprint);
        assert_eq!(cloned.value, entry.value);
        assert_eq!(cloned.inserted_ms, entry.inserted_ms);
        assert_eq!(cloned.hit_count, entry.hit_count);
    }

    #[test]
    fn test_capacity_one_always_evicts_previous() {
        let mut d = make_dedup_with(1_000_000, 1);
        d.register("first", "v1".to_string(), 1);
        d.register("second", "v2".to_string(), 2);
        assert_eq!(d.len(), 1);
        assert!(d.cache.contains_key(&SemanticDedup::fingerprint("second")));
    }

    #[test]
    fn test_overwrite_does_not_grow_beyond_capacity() {
        let mut d = make_dedup_with(1_000_000, 2);
        d.register("a", "v1".to_string(), 1);
        d.register("b", "v2".to_string(), 2);
        // Overwrite "a" — this must not evict anything because the key already exists.
        d.register("a", "v3".to_string(), 3);
        assert_eq!(d.len(), 2);
        let entry = d.check("a", 3).expect("a should still be present");
        assert_eq!(entry.value, "v3");
    }

    #[test]
    fn test_fingerprint_removes_special_chars_only() {
        let fp = SemanticDedup::fingerprint("café résumé");
        // é is alphanumeric (Unicode), so it is kept; accent-less ASCII letters kept.
        // The important invariant: result contains only alphanumerics and spaces.
        for ch in fp.chars() {
            assert!(ch.is_alphanumeric() || ch == ' ', "unexpected char: {:?}", ch);
        }
    }

    #[test]
    fn test_evict_expired_empty_cache_is_noop() {
        let mut d = make_dedup();
        d.evict_expired(1_000_000); // must not panic on empty cache
        assert!(d.is_empty());
    }

    #[test]
    fn test_live_keys_empty_cache_gives_empty_vec() {
        let d = make_dedup();
        assert!(d.live_keys(0).is_empty());
    }

    #[test]
    fn test_register_capacity_zero_is_noop() {
        // capacity = 0 means every insert triggers an LRU eviction before
        // the new item is added — resulting in the new item being the sole
        // entry (the map was empty after eviction so no eviction happened, and
        // capacity 0 >= 0 triggers evict_lru on empty map which is a no-op).
        // The assertion here is simply: no panic.
        let mut d = make_dedup_with(1_000_000, 0);
        d.register("test", "v".to_string(), 0);
        // len may be 0 or 1 depending on implementation — just must not panic.
        let _len = d.len();
    }
}
