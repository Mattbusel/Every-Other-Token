//! External integration tests for the `store`, `heatmap`, and `replay` modules.

use every_other_token::heatmap::HeatmapExporter;
use every_other_token::replay::{Recorder, ReplayRecord, Replayer};
use every_other_token::store::{ExperimentStore, RunRecord, Storage};
use every_other_token::TokenEvent;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn make_event(idx: usize, confidence: Option<f32>) -> TokenEvent {
    TokenEvent {
        text: format!("tok{}", idx),
        original: format!("tok{}", idx),
        index: idx,
        transformed: idx % 2 == 0,
        importance: 0.5,
        chaos_label: None,
        provider: None,
        confidence,
        perplexity: confidence.map(|c| 1.0 / c.max(0.01)),
        alternatives: vec![],
        is_error: false,
    }
}

// ---------------------------------------------------------------------------
// ExperimentStore (public API surface)
// ---------------------------------------------------------------------------

#[test]
fn store_open_in_memory() {
    ExperimentStore::open(":memory:").expect("open in-memory store");
}

#[test]
fn store_insert_and_query_roundtrip() {
    let store = ExperimentStore::open(":memory:").expect("open");
    let id = store
        .insert_experiment(
            "2026-01-01T00:00:00Z",
            "test prompt",
            "openai",
            "reverse",
            "gpt-4",
        )
        .expect("insert");
    assert!(id > 0);
    let exps = store.query_experiments();
    assert_eq!(exps.len(), 1);
    assert_eq!(exps[0]["prompt"], "test prompt");
    assert_eq!(exps[0]["provider"], "openai");
    assert_eq!(exps[0]["transform"], "reverse");
}

#[test]
fn store_insert_run_and_load() {
    let store = ExperimentStore::open(":memory:").expect("open");
    let exp_id = store
        .insert_experiment("2026-01-01T00:00:00Z", "p", "openai", "noise", "gpt-4")
        .expect("insert exp");
    store
        .insert_run(
            exp_id,
            &RunRecord {
                run_index: 0,
                token_count: 50,
                transformed_count: 25,
                avg_confidence: Some(0.75),
                avg_perplexity: Some(1.5),
                vocab_diversity: 0.8,
            },
        )
        .expect("insert run");
    let runs = store.load_runs_by_transform("p", "noise").expect("query");
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].token_count, 50);
    assert_eq!(runs[0].transformed_count, 25);
    assert!((runs[0].avg_confidence.unwrap() - 0.75).abs() < 1e-9);
}

#[test]
fn store_insert_experiment_with_run_atomic() {
    let store = ExperimentStore::open(":memory:").expect("open");
    let id = store
        .insert_experiment_with_run(
            "2026-01-01T00:00:00Z",
            "atomic test",
            "anthropic",
            "uppercase",
            "claude-sonnet-4-6",
            &RunRecord {
                run_index: 0,
                token_count: 10,
                transformed_count: 5,
                avg_confidence: None,
                avg_perplexity: None,
                vocab_diversity: 0.5,
            },
        )
        .expect("transactional insert");
    assert!(id > 0);
    let exps = store.query_experiments();
    assert_eq!(exps.len(), 1);
    assert_eq!(exps[0]["transform"], "uppercase");
}

#[test]
fn store_dedup_miss_then_register_then_hit() {
    let store = ExperimentStore::open(":memory:").expect("open");
    // Miss: fingerprint not yet registered
    assert!(store.dedup_check("fp-ext", 1_000_000, 300_000).is_none());
    // Register
    store
        .dedup_register("fp-ext", "value123", 1_000_000)
        .expect("register");
    // Hit: same fingerprint within TTL
    let result = store.dedup_check("fp-ext", 1_100_000, 300_000);
    assert_eq!(result, Some("value123".to_string()));
}

#[test]
fn store_dedup_expired_returns_none() {
    let store = ExperimentStore::open(":memory:").expect("open");
    store
        .dedup_register("fp-old", "v", 1_000)
        .expect("register");
    // TTL = 300_000 ms, now = 1_000_000 ms → cutoff = 700_000 > 1_000 → expired
    assert!(store.dedup_check("fp-old", 1_000_000, 300_000).is_none());
}

#[test]
fn store_dedup_evict_expired() {
    let store = ExperimentStore::open(":memory:").expect("open");
    store
        .dedup_register("fp-evict", "v", 1_000)
        .expect("register");
    store
        .dedup_evict_expired(1_000_000, 300_000)
        .expect("evict");
    assert!(store.dedup_check("fp-evict", 1_000_000, 300_000).is_none());
}

#[test]
fn store_storage_trait_roundtrip() {
    let store = ExperimentStore::open(":memory:").expect("open");
    let storage: &dyn Storage = &store;
    let id = storage
        .store_experiment(
            "2026-03-18T00:00:00Z",
            "trait test",
            "openai",
            "delete",
            "gpt-4o",
        )
        .expect("store via trait");
    assert!(id > 0);
    storage
        .store_run(
            id,
            &RunRecord {
                run_index: 0,
                token_count: 8,
                transformed_count: 4,
                avg_confidence: Some(0.6),
                avg_perplexity: Some(2.0),
                vocab_diversity: 0.9,
            },
        )
        .expect("store run via trait");
    let exps = storage.list_experiments();
    assert_eq!(exps.len(), 1);
    assert_eq!(exps[0]["transform"], "delete");
}

// ---------------------------------------------------------------------------
// HeatmapExporter public API
// ---------------------------------------------------------------------------

#[test]
fn heatmap_single_run_exports_csv() {
    let mut exp = HeatmapExporter::new();
    let events: Vec<TokenEvent> = (0..3)
        .map(|i| make_event(i, Some(0.1 * (i + 1) as f32)))
        .collect();
    exp.record_run(&events);
    let tmp = std::env::temp_dir().join("heatmap_ext_test.csv");
    exp.export_csv(tmp.to_str().unwrap(), 0.0, "position")
        .expect("export");
    let content = std::fs::read_to_string(&tmp).expect("read");
    assert!(content.contains("position,run_0"));
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn heatmap_min_confidence_filters_rows() {
    let mut exp = HeatmapExporter::new();
    let events: Vec<TokenEvent> = vec![make_event(0, Some(0.9)), make_event(1, Some(0.1))];
    exp.record_run(&events);
    let tmp = std::env::temp_dir().join("heatmap_filter_test.csv");
    exp.export_csv(tmp.to_str().unwrap(), 0.5, "position")
        .expect("export");
    let content = std::fs::read_to_string(&tmp).expect("read");
    // Row for position 1 (mean=0.1) should be filtered out; position 0 (mean=0.9) should remain
    assert!(content.contains("0,"), "position 0 should appear");
    assert!(!content.contains("1,"), "position 1 should be filtered");
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn heatmap_sort_by_confidence() {
    let mut exp = HeatmapExporter::new();
    let events: Vec<TokenEvent> = vec![
        make_event(0, Some(0.2)),
        make_event(1, Some(0.9)),
        make_event(2, Some(0.5)),
    ];
    exp.record_run(&events);
    let tmp = std::env::temp_dir().join("heatmap_sort_test.csv");
    exp.export_csv(tmp.to_str().unwrap(), 0.0, "confidence")
        .expect("export");
    let content = std::fs::read_to_string(&tmp).expect("read");
    // Find the positions of each row index; higher-confidence row should come first
    let pos_of_1 = content.find("\n1,").unwrap_or(usize::MAX);
    let pos_of_0 = content.find("\n0,").unwrap_or(usize::MAX);
    assert!(
        pos_of_1 < pos_of_0,
        "row with highest confidence (pos=1, mean=0.9) should precede pos=0"
    );
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn heatmap_default_construction() {
    let exp = HeatmapExporter::default();
    let tmp = std::env::temp_dir().join("heatmap_default_test.csv");
    exp.export_csv(tmp.to_str().unwrap(), 0.0, "position")
        .expect("empty export");
    std::fs::remove_file(&tmp).ok();
}

// ---------------------------------------------------------------------------
// Recorder / Replayer public API
// ---------------------------------------------------------------------------

#[test]
fn recorder_save_and_replayer_load_roundtrip() {
    let mut rec = Recorder::new();
    rec.record(&make_event(0, Some(0.9)));
    rec.record(&make_event(1, None));
    rec.record(&make_event(2, Some(0.4)));

    let tmp = std::env::temp_dir().join("replay_ext_roundtrip.json");
    rec.save(tmp.to_str().unwrap()).expect("save");

    let records = Replayer::load(tmp.to_str().unwrap()).expect("load");
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].event.index, 0);
    assert_eq!(records[2].event.index, 2);
    assert!(records[0].event.confidence.is_some());
    assert!(records[1].event.confidence.is_none());
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn replayer_replay_to_channel_delivers_all_events() {
    let records: Vec<ReplayRecord> = (0..5)
        .map(|i| ReplayRecord {
            timestamp_ms: i * 100,
            event: make_event(i as usize, Some(0.5)),
        })
        .collect();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    Replayer::replay_to_channel(records, tx).expect("replay");

    let mut received = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        received.push(ev);
    }
    assert_eq!(received.len(), 5);
    for (i, ev) in received.iter().enumerate() {
        assert_eq!(ev.index, i);
    }
}

#[test]
fn recorder_default_is_empty() {
    let rec = Recorder::default();
    let tmp = std::env::temp_dir().join("recorder_default_empty.json");
    rec.save(tmp.to_str().unwrap()).expect("save");
    let records = Replayer::load(tmp.to_str().unwrap()).expect("load");
    assert!(records.is_empty());
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn replay_record_timestamp_is_set() {
    let mut rec = Recorder::new();
    rec.record(&make_event(0, None));
    let tmp = std::env::temp_dir().join("replay_timestamp.json");
    rec.save(tmp.to_str().unwrap()).expect("save");
    let records = Replayer::load(tmp.to_str().unwrap()).expect("load");
    assert!(
        records[0].timestamp_ms > 0,
        "timestamp should be a real epoch ms value"
    );
    std::fs::remove_file(&tmp).ok();
}
