use crate::TokenEvent;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;

/// A single captured token event with a wall-clock timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRecord {
    /// Unix timestamp in milliseconds when the event was captured.
    pub timestamp_ms: u64,
    /// The captured token event.
    pub event: TokenEvent,
}

/// Collects [`TokenEvent`]s with timestamps for later serialisation.
pub struct Recorder {
    records: Vec<ReplayRecord>,
}

impl Recorder {
    /// Create an empty recorder.
    pub fn new() -> Self {
        Recorder {
            records: Vec::new(),
        }
    }

    /// Capture a token event with the current wall-clock timestamp.
    pub fn record(&mut self, event: &TokenEvent) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.records.push(ReplayRecord {
            timestamp_ms,
            event: event.clone(),
        });
    }

    /// Serialise all captured events to a pretty-printed JSON file at `path`.
    ///
    /// # Errors
    /// Returns an error if serialisation or file I/O fails.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.records)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Loads and replays previously recorded [`ReplayRecord`] streams.
pub struct Replayer;

impl Replayer {
    /// Deserialise a replay file from `path` into a list of [`ReplayRecord`]s.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or JSON parsing fails.
    pub fn load(path: &str) -> Result<Vec<ReplayRecord>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let records: Vec<ReplayRecord> = serde_json::from_str(&content)?;
        Ok(records)
    }

    /// Send all records into `tx` in order, as fast as the receiver can consume them.
    ///
    /// # Errors
    /// Returns an error if the channel is closed before all records are sent.
    pub fn replay_to_channel(
        records: Vec<ReplayRecord>,
        tx: UnboundedSender<TokenEvent>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for record in records {
            tx.send(record.event)
                .map_err(|e| format!("send error: {}", e))?;
        }
        Ok(())
    }

    /// Send records into `tx` with timing simulation based on recorded timestamps.
    ///
    /// `speed` controls playback rate: `1.0` = real-time, `2.0` = double speed,
    /// `0.0` (or any non-positive value) = instant (no delays).
    ///
    /// # Errors
    /// Returns an error if the channel is closed before all records are sent.
    pub async fn replay_to_channel_timed(
        records: Vec<ReplayRecord>,
        tx: UnboundedSender<TokenEvent>,
        speed: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start_wall = std::time::Instant::now();
        let first_ts = records.first().map(|r| r.timestamp_ms).unwrap_or(0);
        for record in records {
            if speed > 0.0 {
                let offset_ms = record.timestamp_ms.saturating_sub(first_ts);
                let target_wall_ms = (offset_ms as f64 / speed) as u64;
                let elapsed_ms = start_wall.elapsed().as_millis() as u64;
                if target_wall_ms > elapsed_ms {
                    tokio::time::sleep(std::time::Duration::from_millis(
                        target_wall_ms - elapsed_ms,
                    ))
                    .await;
                }
            }
            tx.send(record.event)
                .map_err(|e| format!("send error: {}", e))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenEvent;

    fn make_event(idx: usize) -> TokenEvent {
        TokenEvent {
            text: "hello".to_string(),
            original: "hello".to_string(),
            index: idx,
            transformed: false,
            importance: 0.0,
            chaos_label: None,
            provider: None,
            confidence: Some(0.9),
            perplexity: None,
            alternatives: vec![],
            is_error: false,
        }
    }

    #[test]
    fn test_recorder_save_load() {
        let mut rec = Recorder::new();
        rec.record(&make_event(0));
        rec.record(&make_event(1));

        let tmp = std::env::temp_dir().join("replay_test.json");
        rec.save(tmp.to_str().unwrap()).expect("save");

        let records = Replayer::load(tmp.to_str().unwrap()).expect("load");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].event.index, 0);
        assert_eq!(records[1].event.index, 1);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_replay_to_channel() {
        let event = make_event(0);
        let records = vec![ReplayRecord {
            timestamp_ms: 12345,
            event: event.clone(),
        }];
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        Replayer::replay_to_channel(records, tx).expect("replay");
        let received = rx.try_recv().expect("recv");
        assert_eq!(received.index, 0);
    }

    #[test]
    fn test_recorder_timestamps_non_decreasing() {
        let mut rec = Recorder::new();
        rec.record(&make_event(0));
        rec.record(&make_event(1));
        rec.record(&make_event(2));
        let times: Vec<u64> = rec.records.iter().map(|r| r.timestamp_ms).collect();
        for i in 1..times.len() {
            assert!(
                times[i] >= times[i - 1],
                "timestamps must be non-decreasing: {} < {}",
                times[i],
                times[i - 1]
            );
        }
    }

    #[test]
    fn test_replayer_load_empty_array() {
        let tmp = std::env::temp_dir().join("replay_empty.json");
        std::fs::write(&tmp, "[]").expect("write");
        let records = Replayer::load(tmp.to_str().unwrap()).expect("load empty");
        assert!(records.is_empty());
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_replayer_load_invalid_json_returns_err() {
        let tmp = std::env::temp_dir().join("replay_bad.json");
        std::fs::write(&tmp, "not json at all").expect("write");
        assert!(Replayer::load(tmp.to_str().unwrap()).is_err());
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_replay_to_channel_preserves_order() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let records: Vec<ReplayRecord> = (0..5)
            .map(|i| ReplayRecord {
                timestamp_ms: i as u64 * 100,
                event: make_event(i),
            })
            .collect();
        Replayer::replay_to_channel(records, tx).expect("replay");
        for expected_idx in 0..5 {
            let ev = rx.try_recv().expect("recv");
            assert_eq!(ev.index, expected_idx);
        }
    }

    #[tokio::test]
    async fn test_replay_to_channel_timed_instant_speed() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let records: Vec<ReplayRecord> = (0..3)
            .map(|i| ReplayRecord {
                timestamp_ms: i as u64 * 1_000, // 1s apart
                event: make_event(i),
            })
            .collect();
        let start = std::time::Instant::now();
        Replayer::replay_to_channel_timed(records, tx, 0.0)
            .await
            .expect("timed replay");
        // speed=0.0 should be instant — well under 100ms
        assert!(start.elapsed().as_millis() < 100);
        for expected in 0..3 {
            let ev = rx.try_recv().expect("recv");
            assert_eq!(ev.index, expected);
        }
    }
}
