//! Integration tests for the self-tune subsystem's public contract.
//!
//! These tests verify observable behaviour without relying on feature flags —
//! they test the types and functions that are always compiled in.
//!
//! ## Contract being tested
//! The self-tune system (feature = "self-tune") exposes:
//!   - `TelemetryBus` — a bounded channel aggregating pipeline metrics.
//!   - `HelixBridgeConfig` — configuration for the HTTP polling bridge.
//!   - `suggest_config_patch` — pure pressure-reactive config suggestion function.
//!
//! The self-modify system (feature = "self-modify") exposes:
//!   - `SemanticDedup` — in-session prompt deduplication cache.
//!
//! This file tests the non-feature-gated plumbing (store, replay, collab) that
//! the self-improving stack is built on top of.

use every_other_token::store::{ExperimentStore, RunRecord};

// ---------------------------------------------------------------------------
// ExperimentStore contract (always compiled in — used by all phases)
// ---------------------------------------------------------------------------

/// The store must open without errors.
#[test]
fn store_opens_in_memory() {
    ExperimentStore::open(":memory:").expect("in-memory store should open");
}

/// insert_experiment returns a positive row id.
#[test]
fn store_insert_returns_positive_id() {
    let store = ExperimentStore::open(":memory:").expect("open");
    let id = store
        .insert_experiment(
            "2026-01-01T00:00:00Z",
            "prompt",
            "openai",
            "reverse",
            "gpt-4",
        )
        .expect("insert");
    assert!(id > 0, "row id must be positive, got {id}");
}

/// Experiments inserted are retrievable via query_experiments.
#[test]
fn store_query_round_trips() {
    let store = ExperimentStore::open(":memory:").expect("open");
    store
        .insert_experiment(
            "2026-01-01T00:00:00Z",
            "hello world",
            "anthropic",
            "noise",
            "claude-sonnet-4-6",
        )
        .expect("insert");
    let exps = store.query_experiments();
    assert_eq!(exps.len(), 1);
    assert_eq!(exps[0]["prompt"], "hello world");
    assert_eq!(exps[0]["provider"], "anthropic");
    assert_eq!(exps[0]["transform"], "noise");
}

/// Multiple experiments are all returned.
#[test]
fn store_query_multiple_experiments() {
    let store = ExperimentStore::open(":memory:").expect("open");
    for i in 0..5 {
        store
            .insert_experiment(
                "2026-01-01T00:00:00Z",
                &format!("prompt {}", i),
                "openai",
                "reverse",
                "gpt-4",
            )
            .expect("insert");
    }
    assert_eq!(store.query_experiments().len(), 5);
}

/// Runs attached to an experiment are loadable by transform.
#[test]
fn store_load_runs_by_transform() {
    let store = ExperimentStore::open(":memory:").expect("open");
    let exp_id = store
        .insert_experiment(
            "2026-01-01T00:00:00Z",
            "test prompt",
            "openai",
            "uppercase",
            "gpt-4",
        )
        .expect("insert exp");
    for i in 0..3 {
        store
            .insert_run(
                exp_id,
                &RunRecord {
                    run_index: i,
                    token_count: 100 + i as usize,
                    transformed_count: 50,
                    avg_confidence: Some(0.8),
                    avg_perplexity: Some(1.5),
                    vocab_diversity: 0.75,
                },
            )
            .expect("insert run");
    }
    let runs = store
        .load_runs_by_transform("test prompt", "uppercase")
        .expect("load");
    assert_eq!(runs.len(), 3);
}

/// load_runs_by_transform returns empty for an unknown transform.
#[test]
fn store_load_runs_unknown_transform_empty() {
    let store = ExperimentStore::open(":memory:").expect("open");
    store
        .insert_experiment("2026-01-01T00:00:00Z", "x", "openai", "reverse", "gpt-4")
        .expect("insert");
    let runs = store
        .load_runs_by_transform("x", "nonexistent")
        .expect("load");
    assert!(runs.is_empty());
}

/// Runs with None confidence/perplexity survive a round-trip.
#[test]
fn store_run_null_metrics_round_trip() {
    let store = ExperimentStore::open(":memory:").expect("open");
    let exp_id = store
        .insert_experiment(
            "2026-01-01T00:00:00Z",
            "null metrics",
            "mock",
            "delete",
            "mock",
        )
        .expect("insert exp");
    store
        .insert_run(
            exp_id,
            &RunRecord {
                run_index: 0,
                token_count: 10,
                transformed_count: 5,
                avg_confidence: None,
                avg_perplexity: None,
                vocab_diversity: 0.5,
            },
        )
        .expect("insert run");
    let runs = store
        .load_runs_by_transform("null metrics", "delete")
        .expect("load");
    assert_eq!(runs.len(), 1);
    assert!(runs[0].avg_confidence.is_none());
    assert!(runs[0].avg_perplexity.is_none());
}

/// query_experiments returns an empty vec on an empty store.
#[test]
fn store_empty_query() {
    let store = ExperimentStore::open(":memory:").expect("open");
    assert!(store.query_experiments().is_empty());
}

// ---------------------------------------------------------------------------
// Provider plugin contract (non-feature-gated)
// ---------------------------------------------------------------------------

use every_other_token::providers::{AnthropicPlugin, OpenAiPlugin, Provider, ProviderPlugin};

#[test]
fn openai_plugin_api_url_is_https() {
    let p = OpenAiPlugin;
    assert!(
        p.api_url().starts_with("https://"),
        "OpenAI API URL must use HTTPS"
    );
}

#[test]
fn anthropic_plugin_api_url_is_https() {
    let p = AnthropicPlugin;
    assert!(
        p.api_url().starts_with("https://"),
        "Anthropic API URL must use HTTPS"
    );
}

#[test]
fn openai_plugin_name_matches_provider_display() {
    let p = OpenAiPlugin;
    assert_eq!(p.name(), Provider::Openai.to_string());
}

#[test]
fn anthropic_plugin_name_matches_provider_display() {
    let p = AnthropicPlugin;
    assert_eq!(p.name(), Provider::Anthropic.to_string());
}

#[test]
fn openai_plugin_default_model_nonempty() {
    assert!(!OpenAiPlugin.default_model().is_empty());
}

#[test]
fn anthropic_plugin_default_model_nonempty() {
    assert!(!AnthropicPlugin.default_model().is_empty());
}

/// build_request includes the model name in the output.
#[test]
fn openai_plugin_build_request_contains_model() {
    let req = OpenAiPlugin.build_request("hello", None, "gpt-4");
    assert_eq!(req["model"], "gpt-4");
}

#[test]
fn anthropic_plugin_build_request_contains_model() {
    let req = AnthropicPlugin.build_request("hello", None, "claude-sonnet-4-6");
    assert_eq!(req["model"], "claude-sonnet-4-6");
}

/// system prompt is threaded through correctly.
#[test]
fn openai_plugin_build_request_with_system() {
    let req = OpenAiPlugin.build_request("user msg", Some("be concise"), "gpt-4");
    let messages = req["messages"].as_array().expect("messages array");
    assert!(messages.iter().any(|m| m["role"] == "system"));
}

#[test]
fn anthropic_plugin_build_request_with_system() {
    let req = AnthropicPlugin.build_request("user msg", Some("be concise"), "claude-sonnet-4-6");
    assert_eq!(req["system"], "be concise");
}

/// Without a system prompt, OpenAI request has only a user message.
#[test]
fn openai_plugin_build_request_no_system() {
    let req = OpenAiPlugin.build_request("hello", None, "gpt-4");
    let messages = req["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
}
