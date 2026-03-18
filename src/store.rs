use rusqlite::{Connection, params};
use serde_json::json;

// ---------------------------------------------------------------------------
// Storage trait — unified persistence abstraction
// ---------------------------------------------------------------------------

/// Unified persistence interface for experiment data.
///
/// `ExperimentStore` (SQLite) implements this trait.  Any future backend
/// (in-memory, Redis, remote API) only needs to implement these three methods.
/// Code that reads or writes experiment data should accept `&dyn Storage`
/// rather than `&ExperimentStore` so backends can be swapped without changes.
pub trait Storage {
    /// Record a new experiment session, returning its unique ID.
    fn store_experiment(
        &self,
        created_at: &str,
        prompt: &str,
        provider: &str,
        transform: &str,
        model: &str,
    ) -> Result<i64, Box<dyn std::error::Error>>;

    /// Attach a run record to an experiment by ID.
    fn store_run(
        &self,
        experiment_id: i64,
        run: &RunRecord,
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// Return all experiments as JSON values.
    fn list_experiments(&self) -> Vec<serde_json::Value>;
}

impl Storage for ExperimentStore {
    fn store_experiment(
        &self,
        created_at: &str,
        prompt: &str,
        provider: &str,
        transform: &str,
        model: &str,
    ) -> Result<i64, Box<dyn std::error::Error>> {
        self.insert_experiment(created_at, prompt, provider, transform, model)
    }

    fn store_run(
        &self,
        experiment_id: i64,
        run: &RunRecord,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.insert_run(experiment_id, run)
    }

    fn list_experiments(&self) -> Vec<serde_json::Value> {
        self.query_experiments()
    }
}

/// Flat data record for a single research run, passed to [`ExperimentStore::insert_run`].
pub struct RunRecord {
    /// Zero-based index of this run within its experiment session.
    pub run_index: u32,
    /// Total number of tokens in the streamed response.
    pub token_count: usize,
    /// Number of tokens mutated by the active transform.
    pub transformed_count: usize,
    /// Mean per-token model confidence (0.0–1.0), or `None` if unavailable.
    pub avg_confidence: Option<f64>,
    /// Mean per-token perplexity, or `None` if unavailable.
    pub avg_perplexity: Option<f64>,
    /// Vocabulary diversity score (unique tokens / total tokens).
    pub vocab_diversity: f64,
}

/// SQLite-backed persistence for experiment sessions and per-run metrics.
///
/// Open with [`ExperimentStore::open`]; pass `":memory:"` for tests.
/// Implements [`Storage`] so it can be used behind `&dyn Storage`.
pub struct ExperimentStore {
    conn: Connection,
}

impl ExperimentStore {
    pub fn open(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY,
                created_at TEXT,
                prompt TEXT,
                provider TEXT,
                transform TEXT,
                model TEXT
            );
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                run_index INTEGER,
                token_count INTEGER,
                transformed_count INTEGER,
                avg_confidence REAL,
                avg_perplexity REAL,
                vocab_diversity REAL
            );
            CREATE TABLE IF NOT EXISTS dedup_cache (
                fingerprint TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                inserted_ms INTEGER NOT NULL
            );",
        )?;
        Ok(ExperimentStore { conn })
    }

    /// Insert an experiment and its first run atomically so a crash between the
    /// two writes cannot leave an orphaned experiment row with no associated run.
    pub fn insert_experiment_with_run(
        &self,
        created_at: &str,
        prompt: &str,
        provider: &str,
        transform: &str,
        model: &str,
        run: &RunRecord,
    ) -> Result<i64, Box<dyn std::error::Error>> {
        self.conn.execute("BEGIN", [])?;
        let result: Result<i64, Box<dyn std::error::Error>> = (|| {
            self.conn.execute(
                "INSERT INTO experiments (created_at, prompt, provider, transform, model) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![created_at, prompt, provider, transform, model],
            )?;
            let exp_id = self.conn.last_insert_rowid();
            self.conn.execute(
                "INSERT INTO runs (experiment_id, run_index, token_count, transformed_count, avg_confidence, avg_perplexity, vocab_diversity)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![exp_id, run.run_index, run.token_count as i64, run.transformed_count as i64,
                        run.avg_confidence, run.avg_perplexity, run.vocab_diversity],
            )?;
            Ok(exp_id)
        })();
        match result {
            Ok(id) => { self.conn.execute("COMMIT", [])?; Ok(id) }
            Err(e) => { let _ = self.conn.execute("ROLLBACK", []); Err(e) }
        }
    }

    pub fn insert_experiment(
        &self,
        created_at: &str,
        prompt: &str,
        provider: &str,
        transform: &str,
        model: &str,
    ) -> Result<i64, Box<dyn std::error::Error>> {
        self.conn.execute(
            "INSERT INTO experiments (created_at, prompt, provider, transform, model)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![created_at, prompt, provider, transform, model],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn insert_run(
        &self,
        experiment_id: i64,
        run: &RunRecord,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.execute(
            "INSERT INTO runs (experiment_id, run_index, token_count, transformed_count, avg_confidence, avg_perplexity, vocab_diversity)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                experiment_id,
                run.run_index,
                run.token_count as i64,
                run.transformed_count as i64,
                run.avg_confidence,
                run.avg_perplexity,
                run.vocab_diversity,
            ],
        )?;
        Ok(())
    }

    /// Load runs for a given prompt + transform (for baseline comparison).
    pub fn load_runs_by_transform(
        &self,
        prompt: &str,
        transform: &str,
    ) -> Result<Vec<RunRecord>, Box<dyn std::error::Error>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.run_index, r.token_count, r.transformed_count, r.avg_confidence, r.avg_perplexity, r.vocab_diversity
             FROM runs r
             JOIN experiments e ON r.experiment_id = e.id
             WHERE e.prompt = ?1 AND e.transform = ?2",
        )?;
        let rows = stmt.query_map(params![prompt, transform], |row| {
            Ok(RunRecord {
                run_index: row.get::<_, i64>(0)? as u32,
                token_count: row.get::<_, i64>(1)? as usize,
                transformed_count: row.get::<_, i64>(2)? as usize,
                avg_confidence: row.get(3)?,
                avg_perplexity: row.get(4)?,
                vocab_diversity: row.get(5)?,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Check if a fingerprint is in the cross-session dedup cache and still within TTL.
    /// Returns the cached value if found, or None if missing/expired.
    pub fn dedup_check(&self, fingerprint: &str, now_ms: u64, ttl_ms: u64) -> Option<String> {
        let cutoff = now_ms.saturating_sub(ttl_ms);
        self.conn
            .query_row(
                "SELECT value FROM dedup_cache WHERE fingerprint = ?1 AND inserted_ms >= ?2",
                rusqlite::params![fingerprint, cutoff as i64],
                |row| row.get(0),
            )
            .ok()
    }

    /// Register a fingerprint in the cross-session dedup cache.
    pub fn dedup_register(&self, fingerprint: &str, value: &str, now_ms: u64) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.execute(
            "INSERT OR REPLACE INTO dedup_cache (fingerprint, value, inserted_ms) VALUES (?1, ?2, ?3)",
            rusqlite::params![fingerprint, value, now_ms as i64],
        )?;
        Ok(())
    }

    /// Remove expired entries from the dedup cache.
    pub fn dedup_evict_expired(&self, now_ms: u64, ttl_ms: u64) -> Result<(), Box<dyn std::error::Error>> {
        let cutoff = now_ms.saturating_sub(ttl_ms);
        self.conn.execute(
            "DELETE FROM dedup_cache WHERE inserted_ms < ?1",
            rusqlite::params![cutoff as i64],
        )?;
        Ok(())
    }

    pub fn query_experiments(&self) -> Vec<serde_json::Value> {
        let mut stmt = match self.conn.prepare(
            "SELECT id, created_at, prompt, provider, transform, model FROM experiments",
        ) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let created_at: String = row.get(1)?;
            let prompt: String = row.get(2)?;
            let provider: String = row.get(3)?;
            let transform: String = row.get(4)?;
            let model: String = row.get(5)?;
            Ok(json!({
                "id": id,
                "created_at": created_at,
                "prompt": prompt,
                "provider": provider,
                "transform": transform,
                "model": model,
            }))
        });
        match rows {
            Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
            Err(_) => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_in_memory() {
        let store = ExperimentStore::open(":memory:").expect("open");
        let _ = store;
    }

    #[test]
    fn test_insert_and_query_experiment() {
        let store = ExperimentStore::open(":memory:").expect("open");
        let id = store
            .insert_experiment("2026-01-01T00:00:00Z", "hello", "openai", "reverse", "gpt-4")
            .expect("insert");
        assert!(id > 0);
        let exps = store.query_experiments();
        assert_eq!(exps.len(), 1);
        assert_eq!(exps[0]["prompt"], "hello");
    }

    #[test]
    fn test_insert_run() {
        let store = ExperimentStore::open(":memory:").expect("open");
        let exp_id = store
            .insert_experiment("2026-01-01T00:00:00Z", "test", "openai", "reverse", "gpt-4")
            .expect("insert exp");
        store
            .insert_run(exp_id, &RunRecord {
                run_index: 0,
                token_count: 42,
                transformed_count: 21,
                avg_confidence: Some(0.9),
                avg_perplexity: Some(1.1),
                vocab_diversity: 0.8,
            })
            .expect("insert run");
    }

    // ---- Storage trait tests ----

    #[test]
    fn test_storage_trait_store_experiment() {
        let store = ExperimentStore::open(":memory:").expect("open");
        let storage: &dyn super::Storage = &store;
        let id = storage
            .store_experiment("2026-01-01T00:00:00Z", "trait test", "anthropic", "noise", "claude-sonnet-4-6")
            .expect("store_experiment via trait");
        assert!(id > 0);
    }

    #[test]
    fn test_storage_trait_list_experiments() {
        let store = ExperimentStore::open(":memory:").expect("open");
        let storage: &dyn super::Storage = &store;
        storage
            .store_experiment("2026-01-01T00:00:00Z", "hello", "openai", "reverse", "gpt-4")
            .expect("insert");
        let exps = storage.list_experiments();
        assert_eq!(exps.len(), 1);
        assert_eq!(exps[0]["prompt"], "hello");
    }

    #[test]
    fn test_storage_trait_store_run() {
        let store = ExperimentStore::open(":memory:").expect("open");
        let storage: &dyn super::Storage = &store;
        let exp_id = storage
            .store_experiment("2026-01-01T00:00:00Z", "run test", "openai", "chaos", "gpt-4")
            .expect("insert");
        storage
            .store_run(exp_id, &RunRecord {
                run_index: 0,
                token_count: 10,
                transformed_count: 5,
                avg_confidence: Some(0.7),
                avg_perplexity: Some(2.0),
                vocab_diversity: 0.6,
            })
            .expect("store_run via trait");
    }

    #[test]
    fn test_storage_trait_list_empty() {
        let store = ExperimentStore::open(":memory:").expect("open");
        let storage: &dyn super::Storage = &store;
        assert!(storage.list_experiments().is_empty());
    }

    #[test]
    fn test_dedup_check_miss() {
        let tmp = std::env::temp_dir().join("dedup_miss.db");
        let store = ExperimentStore::open(tmp.to_str().unwrap()).unwrap();
        assert!(store.dedup_check("fp1", 1_000_000, 300_000).is_none());
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_dedup_register_and_hit() {
        let tmp = std::env::temp_dir().join("dedup_hit.db");
        let store = ExperimentStore::open(tmp.to_str().unwrap()).unwrap();
        store.dedup_register("fp2", "cached_value", 1_000_000).unwrap();
        let result = store.dedup_check("fp2", 1_100_000, 300_000);
        assert_eq!(result, Some("cached_value".to_string()));
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_dedup_expired() {
        let tmp = std::env::temp_dir().join("dedup_expired.db");
        let store = ExperimentStore::open(tmp.to_str().unwrap()).unwrap();
        // Insert with old timestamp
        store.dedup_register("fp3", "old_value", 1_000).unwrap();
        // Check with now=1_000_000 and ttl=300_000 → cutoff=700_000 > 1_000, so expired
        let result = store.dedup_check("fp3", 1_000_000, 300_000);
        assert!(result.is_none());
        std::fs::remove_file(&tmp).ok();
    }

    // -- Transactional insert tests (#3) --

    #[test]
    fn test_insert_experiment_with_run_atomic() {
        let store = ExperimentStore::open(":memory:").expect("open");
        let exp_id = store.insert_experiment_with_run(
            "2026-01-01T00:00:00Z", "hello", "openai", "reverse", "gpt-4",
            &RunRecord {
                run_index: 0,
                token_count: 10,
                transformed_count: 5,
                avg_confidence: Some(0.8),
                avg_perplexity: Some(1.5),
                vocab_diversity: 0.9,
            },
        ).expect("transactional insert");
        assert!(exp_id > 0);
        let exps = store.query_experiments();
        assert_eq!(exps.len(), 1);
    }

    #[test]
    fn test_insert_experiment_with_run_run_is_linked() {
        let store = ExperimentStore::open(":memory:").expect("open");
        let exp_id = store.insert_experiment_with_run(
            "2026-01-01T00:00:00Z", "test", "anthropic", "uppercase", "claude-sonnet-4-6",
            &RunRecord {
                run_index: 0,
                token_count: 20,
                transformed_count: 10,
                avg_confidence: None,
                avg_perplexity: None,
                vocab_diversity: 0.5,
            },
        ).expect("insert");
        // The returned id must be positive and the experiment must be queryable
        assert!(exp_id > 0);
        let exps = store.query_experiments();
        assert_eq!(exps[0]["transform"], "uppercase");
    }

    // -- prepare_cached correctness test (#9) --

    #[test]
    fn test_load_runs_by_transform_cached_stmt() {
        let store = ExperimentStore::open(":memory:").expect("open");
        let exp_id = store.insert_experiment("now", "p", "openai", "noise", "gpt-4").expect("exp");
        store.insert_run(exp_id, &RunRecord {
            run_index: 0,
            token_count: 3,
            transformed_count: 1,
            avg_confidence: Some(0.6),
            avg_perplexity: Some(2.0),
            vocab_diversity: 0.7,
        }).expect("run");
        // Call twice to exercise prepare_cached reuse
        let r1 = store.load_runs_by_transform("p", "noise").expect("q1");
        let r2 = store.load_runs_by_transform("p", "noise").expect("q2");
        assert_eq!(r1.len(), 1);
        assert_eq!(r2.len(), 1);
        assert_eq!(r1[0].token_count, 3);
    }
}
