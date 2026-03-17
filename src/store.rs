use rusqlite::{Connection, params};
use serde_json::json;

/// Flat data record for a single research run, passed to `insert_run`.
pub struct RunRecord {
    pub run_index: u32,
    pub token_count: usize,
    pub transformed_count: usize,
    pub avg_confidence: Option<f64>,
    pub avg_perplexity: Option<f64>,
    pub vocab_diversity: f64,
}

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
            );",
        )?;
        Ok(ExperimentStore { conn })
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
}
