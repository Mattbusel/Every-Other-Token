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
            tx.send(record.event).map_err(|e| format!("send error: {}", e))?;
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
}
