//! Persistent SQLite experiment log for research runs.
//!
//! Enabled with the `sqlite-log` feature. When active, every `--research` run
//! is appended to a local SQLite database so results can be queried across
//! sessions without manually diffing JSON files.
//!
//! Schema:
//!   experiments(id, run_ts, prompt_hash, prompt, provider, transform, model,
//!               runs_count, mean_token_count, mean_confidence, mean_perplexity,
//!               mean_vocab_diversity, output_path)

#[cfg(feature = "sqlite-log")]
mod inner {
    use rusqlite::{params, Connection, Result};
    use std::path::Path;

    pub struct ExperimentLog {
        conn: Connection,
    }

    impl ExperimentLog {
        /// Open (or create) the experiment database at `db_path`.
        pub fn open(db_path: &Path) -> Result<Self> {
            let conn = Connection::open(db_path)?;
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS experiments (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_ts              TEXT    NOT NULL,
                    prompt_hash         TEXT    NOT NULL,
                    prompt              TEXT    NOT NULL,
                    provider            TEXT    NOT NULL,
                    transform           TEXT    NOT NULL,
                    model               TEXT    NOT NULL,
                    runs_count          INTEGER NOT NULL,
                    mean_token_count    REAL    NOT NULL,
                    mean_confidence     REAL,
                    mean_perplexity     REAL,
                    mean_vocab_diversity REAL   NOT NULL,
                    output_path         TEXT    NOT NULL
                );",
            )?;
            Ok(ExperimentLog { conn })
        }

        /// Append one experiment summary row.
        #[allow(clippy::too_many_arguments)]
        pub fn append(
            &self,
            prompt: &str,
            provider: &str,
            transform: &str,
            model: &str,
            runs_count: u32,
            mean_token_count: f64,
            mean_confidence: Option<f64>,
            mean_perplexity: Option<f64>,
            mean_vocab_diversity: f64,
            output_path: &str,
        ) -> Result<()> {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            use std::time::{SystemTime, UNIX_EPOCH};

            let mut h = DefaultHasher::new();
            prompt.hash(&mut h);
            let prompt_hash = format!("{:016x}", h.finish());

            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let run_ts = chrono_or_epoch(ts);

            self.conn.execute(
                "INSERT INTO experiments
                    (run_ts, prompt_hash, prompt, provider, transform, model,
                     runs_count, mean_token_count, mean_confidence, mean_perplexity,
                     mean_vocab_diversity, output_path)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
                params![
                    run_ts,
                    prompt_hash,
                    prompt,
                    provider,
                    transform,
                    model,
                    runs_count as i64,
                    mean_token_count,
                    mean_confidence,
                    mean_perplexity,
                    mean_vocab_diversity,
                    output_path,
                ],
            )?;
            Ok(())
        }

        /// Return all stored experiment rows as a JSON array string.
        ///
        /// Each element is an object with the same columns as the `experiments`
        /// table.  Returns `"[]"` if the table is empty.
        pub fn list(&self) -> Result<String> {
            let mut stmt = self.conn.prepare(
                "SELECT id, run_ts, prompt_hash, prompt, provider, transform, model,
                        runs_count, mean_token_count, mean_confidence, mean_perplexity,
                        mean_vocab_diversity, output_path
                 FROM experiments ORDER BY id DESC",
            )?;

            let rows = stmt.query_map([], |row| {
                Ok(format!(
                    r#"{{"id":{},"run_ts":{},"prompt_hash":{},"prompt":{},"provider":{},"transform":{},"model":{},"runs_count":{},"mean_token_count":{},"mean_confidence":{},"mean_perplexity":{},"mean_vocab_diversity":{},"output_path":{}}}"#,
                    row.get::<_, i64>(0)?,
                    serde_json_str(row.get::<_, String>(1)?),
                    serde_json_str(row.get::<_, String>(2)?),
                    serde_json_str(row.get::<_, String>(3)?),
                    serde_json_str(row.get::<_, String>(4)?),
                    serde_json_str(row.get::<_, String>(5)?),
                    serde_json_str(row.get::<_, String>(6)?),
                    row.get::<_, i64>(7)?,
                    row.get::<_, f64>(8)?,
                    row.get::<_, Option<f64>>(9)?.map_or("null".to_string(), |v| v.to_string()),
                    row.get::<_, Option<f64>>(10)?.map_or("null".to_string(), |v| v.to_string()),
                    row.get::<_, f64>(11)?,
                    serde_json_str(row.get::<_, String>(12)?),
                ))
            })?;

            let items: Vec<String> = rows.filter_map(|r| r.ok()).collect();
            Ok(format!("[{}]", items.join(",")))
        }
    }

    /// Escape a string for embedding in a JSON literal (no external dep).
    fn serde_json_str(s: String) -> String {
        let escaped = s
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        format!("\"{}\"", escaped)
    }

    /// Format unix timestamp as ISO-8601 date-time string (no chrono dep needed).
    fn chrono_or_epoch(secs: u64) -> String {
        // Simple manual formatting: YYYY-MM-DDTHH:MM:SSZ
        let s = secs;
        let sec = s % 60;
        let min = (s / 60) % 60;
        let hour = (s / 3600) % 24;
        let days = s / 86400;
        // Days since 1970-01-01
        let (year, month, day) = days_to_ymd(days);
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            year, month, day, hour, min, sec
        )
    }

    fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
        let mut year = 1970u64;
        loop {
            let leap = is_leap(year);
            let days_in_year = if leap { 366 } else { 365 };
            if days < days_in_year {
                break;
            }
            days -= days_in_year;
            year += 1;
        }
        let leap = is_leap(year);
        let month_days: &[u64] = if leap {
            &[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        } else {
            &[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        };
        let mut month = 1u64;
        for &md in month_days {
            if days < md {
                break;
            }
            days -= md;
            month += 1;
        }
        (year, month, days + 1)
    }

    fn is_leap(y: u64) -> bool {
        (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_open_in_memory() {
            // rusqlite supports ":memory:" path
            let log = ExperimentLog::open(Path::new(":memory:")).expect("open");
            log.append(
                "What is consciousness?",
                "mock",
                "reverse",
                "mock-fixture-v1",
                10,
                42.5,
                Some(0.85),
                Some(1.18),
                0.72,
                "results.json",
            )
            .expect("append");
        }

        #[test]
        fn test_prompt_hash_is_hex() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            "hello".hash(&mut h);
            let hash = format!("{:016x}", h.finish());
            assert_eq!(hash.len(), 16);
            assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        }

        #[test]
        fn test_days_to_ymd_epoch() {
            let (y, m, d) = days_to_ymd(0);
            assert_eq!((y, m, d), (1970, 1, 1));
        }

        #[test]
        fn test_days_to_ymd_known_date() {
            // 1970-01-01 + 20530 days = 2026-03-18
            let (y, m, d) = days_to_ymd(20530);
            assert_eq!((y, m, d), (2026, 3, 18));
        }

        #[test]
        fn test_is_leap() {
            assert!(is_leap(2000));
            assert!(is_leap(2024));
            assert!(!is_leap(1900));
            assert!(!is_leap(2023));
        }

        #[test]
        fn test_chrono_or_epoch_format() {
            let s = chrono_or_epoch(0);
            assert_eq!(s, "1970-01-01T00:00:00Z");
        }
    }
}

#[cfg(feature = "sqlite-log")]
pub use inner::ExperimentLog;
