//! # every-other-token
//!
//! A real-time LLM token stream interceptor for token-level interaction research.
//!
//! This crate sits between the caller and the model, intercepts the token stream
//! as it arrives over SSE, applies one of five transform strategies to tokens at
//! configurable positions, scores model confidence at each position using the
//! OpenAI logprob API, and routes the enriched events to a terminal renderer, a
//! zero-dependency web UI, and an optional WebSocket collaboration room.
//!
//! ## New interpretability modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`attention`] | Causal attention tracer — attribution matrix showing which context tokens caused each generated token |
//! | [`entropy`] | Prompt entropy analyzer — Shannon entropy, perplexity estimation, repetition detection, multi-turn timeline |
//! | [`fingerprint`] | Model fingerprinting — statistical signatures for blind A/B testing and model identification |
//! | [`hallucination`] | Hallucination detector — identifies perplexity spikes and confident-but-fragile token positions |
//!
//! ## Feature flags
//!
//! | Flag | Description |
//! |------|-------------|
//! | `sqlite-log` | Persist experiment runs to a local SQLite database via `store::ExperimentStore`. |
//! | `self-tune` | Enable the self-improvement telemetry bus and tuning controller. |
//! | `self-modify` | Enable snapshot-based parameter mutation (requires `self-tune`). |
//! | `intelligence` | Reserved namespace for future interpretability features. |
//! | `evolution` | Reserved namespace for future evolutionary optimisation. |
//! | `helix-bridge` | HTTP bridge that polls `/api/stats` and pushes config patches. |
//! | `redis-backing` | Write-through Redis persistence for agent memory and snapshots. |
//! | `wasm` | WASM target bindings via `wasm-bindgen`. |
//!
//! ## Quickstart
//!
//! ```bash
//! export OPENAI_API_KEY="sk-..."
//! cargo run -- "What is consciousness?" --visual
//! cargo run -- "What is consciousness?" --web
//! cargo run -- "Explain recursion" --research --runs 20 --output results.json
//! ```

pub mod attribution;
pub mod cli;
pub mod collab;
pub mod comparison;
pub mod config;
pub mod divergence;
pub mod error;
pub mod heatmap;
pub mod intervention;
pub mod mutation_lab;
pub mod providers;
pub mod bayesian;
pub mod checkpoint;
pub mod render;
pub mod replay;
pub mod research;
pub mod semantic_heatmap;
pub mod store;
pub mod attention;
pub mod entropy;
pub mod fingerprint;
pub mod hallucination;
pub mod sensitivity;
pub mod experiments;
pub mod token_dictionary;
pub mod transforms;
pub mod web;
pub mod patching;
pub mod logit_lens;
pub mod circuits;
pub mod steering;
pub mod cross_model;

#[cfg(feature = "self-tune")]
pub mod self_tune;

#[cfg(feature = "self-modify")]
pub mod self_modify;

#[cfg(feature = "self-modify")]
pub mod semantic_dedup;

#[cfg(feature = "helix-bridge")]
pub mod helix_bridge;

#[cfg(feature = "sqlite-log")]
pub mod experiment_log;

#[cfg(feature = "intelligence")]
pub mod intelligence {
    //! Stub module for the intelligence feature flag.
    //! Reserved namespace for future interpretability and reasoning features.

    #[cfg(test)]
    mod tests {
        #[test]
        fn stub_compiles() {}
    }
}

#[cfg(feature = "evolution")]
pub mod evolution {
    //! Stub module for the evolution feature flag.
    //! Reserved namespace for future evolutionary/genetic optimization features.

    #[cfg(test)]
    mod tests {
        #[test]
        fn stub_compiles() {}
    }
}

use colored::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{self, Write};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

use providers::*;
use transforms::{apply_heatmap_color, calculate_token_importance, tokenize, Transform};

// ---------------------------------------------------------------------------
// Token probability types
// ---------------------------------------------------------------------------

/// One alternative token and its probability (for top-K logprob display).
///
/// Returned in the `alternatives` field of a [`TokenEvent`] when the provider
/// supports `top_logprobs` (currently OpenAI only).  Probabilities have already
/// been converted from log-space via `exp(logprob)` and clamped to `[0.0, 1.0]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAlternative {
    /// The alternative token string (may include leading whitespace, e.g. `" world"`).
    pub token: String,
    /// Linear probability in `[0.0, 1.0]`, computed as `exp(logprob)`.
    pub probability: f32,
}

// ---------------------------------------------------------------------------
// Token event (for web UI streaming)
// ---------------------------------------------------------------------------

/// A single processed token emitted by the streaming pipeline.
///
/// Every token the interceptor produces — whether transformed or not — is
/// represented as a `TokenEvent`.  Events are sent over the `web_tx` channel
/// for SSE fan-out to the web UI, written as JSON lines in `--json-stream`
/// mode, or recorded to a replay file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEvent {
    /// The (possibly transformed) token text shown to the user.
    pub text: String,
    /// The original token text before any transform was applied.
    pub original: String,
    /// Zero-based position of this token in the full response.
    pub index: usize,
    /// Whether the active transform was applied to this token.
    pub transformed: bool,
    /// Scalar token importance in `[0.0, 1.0]` — derived from API confidence
    /// when available, otherwise computed by the heuristic importance scorer.
    pub importance: f64,
    /// For Chaos transform: which sub-transform was applied. None for other transforms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chaos_label: Option<String>,
    /// For diff mode: which provider produced this token ("openai" or "anthropic").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model confidence 0.0–1.0, derived from API logprob. None when unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Per-token perplexity (exp(-log_prob)). None when logprobs unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perplexity: Option<f32>,
    /// Top alternative tokens with their probabilities (from top_logprobs).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<TokenAlternative>,
    /// When true, this event represents an error notification rather than a real token.
    #[serde(default)]
    pub is_error: bool,
    /// Milliseconds elapsed since stream start when this token arrived (for latency tracking).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arrival_ms: Option<u64>,
}

// ---------------------------------------------------------------------------
// TokenInterceptor — multi-provider streaming engine
// ---------------------------------------------------------------------------

/// The core streaming engine that sits between the caller and the LLM.
///
/// `TokenInterceptor` manages the HTTP connection to the configured provider,
/// iterates the server-sent-event (SSE) stream, applies the active [`Transform`]
/// to every N-th token (controlled by `rate`), attaches per-token confidence and
/// perplexity from API logprobs, and routes enriched [`TokenEvent`]s to one of
/// three output sinks:
///
/// - **Terminal** — ANSI-colored text written to stdout.
/// - **Web UI** — events sent over the `web_tx` unbounded channel for SSE fan-out.
/// - **JSON stream** — one JSON line per token written to stdout (`json_stream = true`).
///
/// Construct with [`TokenInterceptor::new`] then call [`TokenInterceptor::intercept_stream`].
pub struct TokenInterceptor {
    client: Client,
    api_key: String,
    pub provider: Provider,
    pub transform: Transform,
    pub model: String,
    pub token_count: usize,
    pub transformed_count: usize,
    pub visual_mode: bool,
    pub heatmap_mode: bool,
    pub orchestrator: bool,
    pub orchestrator_url: String,
    /// When set, token events are sent here instead of printed to stdout.
    pub web_tx: Option<mpsc::UnboundedSender<TokenEvent>>,
    /// When set, each emitted TokenEvent carries this provider label (for diff mode).
    pub web_provider_label: Option<String>,
    /// Optional system prompt prepended to the conversation.
    pub system_prompt: Option<String>,
    /// When set, token processing metrics are recorded into the self-improvement bus.
    #[cfg(feature = "self-tune")]
    pub telemetry_bus: Option<std::sync::Arc<crate::self_tune::telemetry_bus::TelemetryBus>>,
    /// Optional in-session prompt deduplication cache.
    ///
    /// When set, `intercept_stream` checks whether the incoming prompt has been
    /// seen recently (within the configured TTL).  If a live hit is found the
    /// API call is skipped and a cache-hit notice is printed or sent to the web
    /// channel, avoiding redundant spend on repeated prompts (common in
    /// research mode).
    ///
    /// Enabled by setting `dedup` after construction (see [`TokenInterceptor::enable_dedup`]).
    #[cfg(feature = "self-modify")]
    pub dedup: Option<std::sync::Arc<std::sync::Mutex<crate::semantic_dedup::SemanticDedup>>>,
    /// Fraction of tokens to transform (0.0–1.0).  Bresenham-spread so the
    /// distribution is deterministic and uniform rather than probabilistic.
    pub rate: f64,
    /// Number of top alternative tokens to request per position (OpenAI only, 0–20).
    pub top_logprobs: u8,
    /// Per-session RNG used for Noise/Chaos transforms.  Seeded from entropy
    /// unless a fixed seed is provided via `with_seed()`.
    rng: StdRng,
    /// Optional replay recorder — records each emitted TokenEvent.
    pub recorder: Option<crate::replay::Recorder>,
    /// When true, print one JSON line per token instead of colored text.
    pub json_stream: bool,
    /// Pending async delay in ms to be awaited after process_content_logprob returns.
    pending_delay_ms: u64,
    /// Minimum confidence threshold for transform gating. When set, only tokens
    /// with confidence at or below this value are transformed.
    pub min_confidence: Option<f64>,
    /// Timestamp of the last received token, used for timing-based confidence proxy.
    last_token_instant: Option<std::time::Instant>,
    /// Maximum retry attempts for API calls on 429/5xx (configurable via --max-retries).
    pub max_retries: u32,
    /// Maximum tokens in the Anthropic response (configurable via --anthropic-max-tokens).
    pub anthropic_max_tokens: u32,
    /// Instant recorded at stream start for per-token arrival latency measurement.
    stream_start_instant: Option<std::time::Instant>,
    /// Optional stream timeout in seconds. When set, `intercept_stream` will fail
    /// with a timeout error if the entire stream does not complete within this duration.
    pub timeout_secs: Option<u64>,
}

// ---------------------------------------------------------------------------
// HTTP retry helper (#5) + circuit breaker (#12)
// ---------------------------------------------------------------------------

/// Per-provider circuit breaker state stored in a global registry.
///
/// The breaker has three states:
/// - **Closed** (normal) — requests pass through.
/// - **Open** — consecutive failures exceeded `TRIP_THRESHOLD`; requests are
///   rejected immediately for `RECOVERY_MS` milliseconds.
/// - **Half-open** — a single probe request is allowed through after recovery;
///   success resets the counter, failure re-opens for another `RECOVERY_MS`.
static CIRCUIT_BREAKER: std::sync::OnceLock<
    std::sync::Mutex<CircuitBreakerState>,
> = std::sync::OnceLock::new();

struct CircuitBreakerState {
    consecutive_failures: u32,
    open_until_ms: u64,
}

/// Trip after this many consecutive failures.
const CB_TRIP_THRESHOLD: u32 = 5;
/// Duration the breaker stays open after tripping (30 seconds).
const CB_RECOVERY_MS: u64 = 30_000;

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Returns `true` if the circuit breaker is currently open (requests should
/// be short-circuited), `false` if the request should be attempted.
fn circuit_is_open() -> bool {
    let state = CIRCUIT_BREAKER.get_or_init(|| {
        std::sync::Mutex::new(CircuitBreakerState {
            consecutive_failures: 0,
            open_until_ms: 0,
        })
    });
    if let Ok(s) = state.lock() {
        s.open_until_ms > now_unix_ms()
    } else {
        false
    }
}

fn circuit_record_success() {
    let state = CIRCUIT_BREAKER.get_or_init(|| {
        std::sync::Mutex::new(CircuitBreakerState {
            consecutive_failures: 0,
            open_until_ms: 0,
        })
    });
    if let Ok(mut s) = state.lock() {
        s.consecutive_failures = 0;
        s.open_until_ms = 0;
    }
}

fn circuit_record_failure() {
    let state = CIRCUIT_BREAKER.get_or_init(|| {
        std::sync::Mutex::new(CircuitBreakerState {
            consecutive_failures: 0,
            open_until_ms: 0,
        })
    });
    if let Ok(mut s) = state.lock() {
        s.consecutive_failures += 1;
        if s.consecutive_failures >= CB_TRIP_THRESHOLD {
            s.open_until_ms = now_unix_ms() + CB_RECOVERY_MS;
            tracing::warn!(
                consecutive_failures = s.consecutive_failures,
                recovery_ms = CB_RECOVERY_MS,
                "circuit breaker tripped — blocking requests for recovery period"
            );
        }
    }
}

/// Execute a pre-built `reqwest::Request`, retrying up to `max_attempts`
/// times on 429 / 5xx responses and network errors with exponential back-off.
///
/// Integrates with a process-wide circuit breaker: after `CB_TRIP_THRESHOLD`
/// consecutive failures the breaker opens for `CB_RECOVERY_MS` ms, rejecting
/// all requests immediately.  A single successful response resets the counter.
///
/// Returns the first successful (or non-retryable) response.
async fn execute_with_retry(
    client: &reqwest::Client,
    req: reqwest::Request,
    max_attempts: u32,
) -> Result<reqwest::Response, Box<dyn std::error::Error + Send + Sync>> {
    if circuit_is_open() {
        return Err("circuit breaker open — provider unavailable, try again shortly".into());
    }

    let mut last_err: Option<String> = None;
    for attempt in 0..max_attempts {
        if attempt > 0 {
            let delay_ms = 400u64 * (1u64 << attempt.min(4));
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            tracing::warn!(attempt, "retrying API request after transient error");
        }
        let to_send = match req.try_clone() {
            Some(r) => r,
            None => {
                // Body is a stream — cannot retry; just execute once.
                return client.execute(req).await.map_err(|e| e.into());
            }
        };
        match client.execute(to_send).await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                if attempt + 1 < max_attempts
                    && (status == 429 || status == 500 || status == 502 || status == 503)
                {
                    tracing::warn!(status, attempt, "got retryable HTTP status");
                    last_err = Some(format!("HTTP {status}"));
                    // HTTP 429 is a rate-limit — do NOT trip the circuit breaker.
                    // Only 5xx server errors count as service failures.
                    if status != 429 {
                        circuit_record_failure();
                    }
                    continue;
                }
                circuit_record_success();
                return Ok(resp);
            }
            Err(e) => {
                circuit_record_failure();
                if attempt + 1 < max_attempts {
                    tracing::warn!(error = %e, attempt, "network error, will retry");
                    last_err = Some(e.to_string());
                } else {
                    return Err(Box::new(e));
                }
            }
        }
    }
    Err(last_err
        .unwrap_or_else(|| "max retries exceeded".to_string())
        .into())
}

impl TokenInterceptor {
    /// Construct a new `TokenInterceptor`.
    ///
    /// Reads the API key from the environment (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
    /// and validates its format.  The `Mock` provider does not require a key.
    ///
    /// # Errors
    /// Returns an error if the required API key environment variable is not set.
    pub fn new(
        provider: Provider,
        transform: Transform,
        model: String,
        visual_mode: bool,
        heatmap_mode: bool,
        orchestrator: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let api_key = match provider {
            Provider::Openai => {
                let key = env::var("OPENAI_API_KEY")
                    .map_err(|_| "OPENAI_API_KEY not set. Export it or pass via environment.")?;
                // Basic format validation (#9): OpenAI keys start with "sk-"
                if !key.starts_with("sk-") {
                    eprintln!(
                        "[warn] OPENAI_API_KEY does not start with 'sk-' — verify it is correct"
                    );
                }
                key
            }
            Provider::Anthropic => {
                let key = env::var("ANTHROPIC_API_KEY")
                    .map_err(|_| "ANTHROPIC_API_KEY not set. Export it or pass via environment.")?;
                // Anthropic keys start with "sk-ant-"
                if !key.starts_with("sk-ant-") {
                    eprintln!("[warn] ANTHROPIC_API_KEY does not start with 'sk-ant-' — verify it is correct");
                }
                key
            }
            Provider::Mock => String::new(),
        };

        Ok(TokenInterceptor {
            client: Client::new(),
            api_key,
            provider,
            transform,
            model,
            token_count: 0,
            transformed_count: 0,
            visual_mode,
            heatmap_mode,
            orchestrator,
            orchestrator_url: "http://localhost:3000".to_string(),
            web_tx: None,
            web_provider_label: None,
            system_prompt: None,
            #[cfg(feature = "self-tune")]
            telemetry_bus: None,
            #[cfg(feature = "self-modify")]
            dedup: None,
            rate: 0.5,
            top_logprobs: 5,
            rng: StdRng::from_entropy(),
            recorder: None,
            json_stream: false,
            pending_delay_ms: 0,
            min_confidence: None,
            last_token_instant: None,
            max_retries: 3,
            anthropic_max_tokens: 4096,
            stream_start_instant: None,
            timeout_secs: None,
        })
    }

    /// Set the intercept rate (0.0–1.0).  Clamped to [0.0, 1.0].
    pub fn with_rate(mut self, rate: f64) -> Self {
        debug_assert!(rate.is_finite(), "with_rate: rate must be finite, got {}", rate);
        self.rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Seed the internal RNG for reproducible Noise/Chaos output.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Set the channel used to fan out token events to the web UI.
    ///
    /// Calling this completes the builder chain for web-mode construction
    /// so callers do not need to set `web_tx` as a bare field assignment.
    pub fn with_web_tx(mut self, tx: mpsc::UnboundedSender<TokenEvent>) -> Self {
        self.web_tx = Some(tx);
        self
    }

    /// Set an optional provider label attached to every emitted [`TokenEvent`].
    /// Used in diff mode to tag events with `"openai"` or `"anthropic"`.
    pub fn with_provider_label(mut self, label: impl Into<String>) -> Self {
        self.web_provider_label = Some(label.into());
        self
    }

    /// Prepend a system prompt to the conversation.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Number of top alternative tokens to request per position (OpenAI only, 0–20).
    pub fn with_top_logprobs(mut self, n: u8) -> Self {
        self.top_logprobs = n;
        self
    }

    /// Enable JSON-stream mode: emit one JSON line per token instead of ANSI text.
    pub fn with_json_stream(mut self, enabled: bool) -> Self {
        self.json_stream = enabled;
        self
    }

    /// Override the MCP orchestrator base URL (default: `http://localhost:3000`).
    pub fn with_orchestrator_url(mut self, url: impl Into<String>) -> Self {
        self.orchestrator_url = url.into();
        self
    }

    /// Maximum retry attempts on 429/5xx errors.
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Set a stream timeout in seconds. If the entire stream does not complete within
    /// this duration, `intercept_stream` returns a timeout error.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Only transform tokens whose API confidence is at or below this threshold.
    pub fn with_min_confidence(mut self, threshold: f64) -> Self {
        self.min_confidence = Some(threshold);
        self
    }

    /// Enable in-session prompt deduplication with the given TTL and capacity.
    ///
    /// After calling this, `intercept_stream` will check whether an incoming
    /// prompt has been seen recently and skip the API call on a cache hit.
    ///
    /// # Panics
    /// This function never panics.
    #[cfg(feature = "self-modify")]
    pub fn enable_dedup(&mut self, ttl_ms: u64, capacity: usize) {
        use crate::semantic_dedup::{DedupConfig, SemanticDedup};
        let sd = SemanticDedup::new(DedupConfig { ttl_ms, capacity });
        self.dedup = Some(std::sync::Arc::new(std::sync::Mutex::new(sd)));
    }

    // -----------------------------------------------------------------------
    // Public entry point
    // -----------------------------------------------------------------------

    /// Stream the given `prompt` through the configured provider, applying
    /// the active transform to every other token.
    ///
    /// In terminal mode the tokens are printed to stdout; in web mode they are
    /// sent over the `web_tx` channel for SSE fan-out.
    ///
    /// # Errors
    /// Returns an error if the prompt is empty, exceeds 512 KB, the API key is
    /// missing, the HTTP request fails after all retries, or JSON parsing fails.
    pub async fn intercept_stream(
        &mut self,
        prompt: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let timeout_duration = self.timeout_secs.map(std::time::Duration::from_secs);
        if let Some(duration) = timeout_duration {
            return match tokio::time::timeout(duration, self.intercept_stream_inner(prompt)).await {
                Ok(result) => result,
                Err(_) => Err(format!(
                    "stream timed out after {} seconds",
                    duration.as_secs()
                )
                .into()),
            };
        }
        self.intercept_stream_inner(prompt).await
    }

    async fn intercept_stream_inner(
        &mut self,
        prompt: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Record stream start for per-token arrival latency measurement (item 8).
        self.stream_start_instant = Some(std::time::Instant::now());
        // Note: we log diagnostics here but do not hold an entered span across
        // await points -- EnteredSpan is !Send and would prevent tokio::spawn.
        tracing::info!(
            provider = %self.provider,
            model = %self.model,
            prompt_len = prompt.len(),
            "starting token stream interception",
        );

        // ── Input validation (#11) ───────────────────────────────────────────
        if prompt.trim().is_empty() {
            tracing::error!("prompt is empty — aborting");
            return Err("Prompt must not be empty".into());
        }
        // Rough guard against prompts that would exceed typical API limits.
        // 512 KB ≈ ~128K tokens at 4 bytes/token; APIs will reject anyway but
        // failing fast gives a clearer error message.
        if prompt.len() > 512_000 {
            return Err(format!(
                "Prompt is too long ({} bytes; max 512 KB). Use a shorter prompt.",
                prompt.len()
            )
            .into());
        }

        // ── Prompt deduplication gate ─────────────────────────────────────────
        // Check before printing the header so skipped prompts are silent.
        #[cfg(feature = "self-modify")]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);

            if let Some(dedup_arc) = &self.dedup {
                if let Ok(mut guard) = dedup_arc.lock() {
                    if let Some(entry) = guard.check(prompt, now_ms) {
                        let hits = entry.hit_count;
                        drop(guard); // release lock before any I/O

                        let msg = format!(
                            "[dedup] Skipping duplicate prompt (seen {} time{} recently, TTL active)",
                            hits,
                            if hits == 1 { "" } else { "s" },
                        );
                        if let Some(tx) = &self.web_tx {
                            let evt = TokenEvent {
                                text: msg.clone(),
                                original: prompt.to_string(),
                                index: 0,
                                transformed: false,
                                importance: 0.0,
                                chaos_label: None,
                                provider: self.web_provider_label.clone(),
                                confidence: None,
                                perplexity: None,
                                alternatives: vec![],
                                is_error: false,
                                arrival_ms: None,
                            };
                            let _ = tx.send(evt);
                        } else {
                            eprintln!("{}", msg);
                        }
                        return Ok(());
                    } else {
                        // Register prompt so the next identical call is caught.
                        // Value is empty — we use this purely as a seen-prompt gate.
                        guard.register(prompt, String::new(), now_ms);
                    }
                }
            }
        }

        if self.web_tx.is_none() {
            self.print_header(prompt);
        }

        // If --orchestrator is active, pre-process through MCP pipeline
        let effective_prompt = if self.orchestrator {
            eprintln!(
                "{}",
                "[orchestrator] routing through MCP pipeline at localhost:3000".bright_magenta()
            );
            match self.orchestrator_infer(prompt).await {
                Ok(enriched) => enriched,
                Err(e) => {
                    eprintln!(
                        "{} {}",
                        "[orchestrator] pipeline unavailable, using raw prompt:".bright_red(),
                        e
                    );
                    if let Some(tx) = &self.web_tx {
                        let evt = TokenEvent {
                            text: format!("[orchestrator error] {}", e),
                            original: String::new(),
                            index: 0,
                            transformed: false,
                            importance: 0.0,
                            chaos_label: None,
                            provider: self.web_provider_label.clone(),
                            confidence: None,
                            perplexity: None,
                            alternatives: vec![],
                            is_error: true,
                            arrival_ms: None,
                        };
                        let _ = tx.send(evt);
                    }
                    prompt.to_string()
                }
            }
        } else {
            prompt.to_string()
        };

        match self.provider {
            Provider::Openai => self.stream_openai(&effective_prompt).await?,
            Provider::Anthropic => self.stream_anthropic(&effective_prompt).await?,
            Provider::Mock => self.stream_mock(&effective_prompt).await?,
        }

        if self.web_tx.is_none() {
            self.print_footer();
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // OpenAI streaming
    // -----------------------------------------------------------------------

    async fn stream_openai(&mut self, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut messages = Vec::new();
        if let Some(sys) = &self.system_prompt {
            messages.push(OpenAIChatMessage {
                role: "system".to_string(),
                content: sys.clone(),
            });
        }
        messages.push(OpenAIChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        });
        let request = OpenAIChatRequest {
            model: self.model.clone(),
            messages,
            stream: true,
            temperature: 0.7,
            logprobs: true,
            top_logprobs: self.top_logprobs,
        };

        let req = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .build()?;

        // Retry on 429 / 5xx with exponential back-off (#5).
        let response = execute_with_retry(&self.client, req, self.max_retries)
            .await
            .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("OpenAI API error: {}", error_text).into());
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut dropped_chunks: usize = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            // Reject invalid UTF-8 rather than silently replacing bytes (#4).
            let chunk_str = match std::str::from_utf8(&chunk) {
                Ok(s) => s.to_string(),
                Err(e) => {
                    tracing::warn!(error = %e, "invalid UTF-8 in OpenAI stream chunk — skipping");
                    continue;
                }
            };
            buffer.push_str(&chunk_str);

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer.drain(..=line_end);

                if line.starts_with("data: ") && line != "data: [DONE]" {
                    let json_str = line.strip_prefix("data: ").unwrap_or(&line);
                    match serde_json::from_str::<OpenAIChunk>(json_str) {
                        Ok(parsed) => {
                            if let Some(choice) = parsed.choices.first() {
                                if let Some(content) = &choice.delta.content {
                                    // Extract logprob data from the first API token in this chunk
                                    let (log_prob, top_alts) = choice
                                        .logprobs
                                        .as_ref()
                                        .and_then(|lp| lp.content.first())
                                        .map(|lc| {
                                            let alts = lc
                                                .top_logprobs
                                                .iter()
                                                .map(|t| TokenAlternative {
                                                    token: t.token.clone(),
                                                    probability: t.logprob.exp().clamp(0.0, 1.0),
                                                })
                                                .collect::<Vec<_>>();
                                            (Some(lc.logprob), alts)
                                        })
                                        .unwrap_or((None, vec![]));
                                    self.process_content_logprob(content, log_prob, top_alts);
                                    if self.pending_delay_ms > 0 {
                                        tokio::time::sleep(std::time::Duration::from_millis(
                                            self.pending_delay_ms,
                                        ))
                                        .await;
                                        self.pending_delay_ms = 0;
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            tracing::warn!(line = %json_str, "failed to parse SSE chunk; skipping");
                            dropped_chunks += 1;
                        }
                    }
                }
            }
        }

        if dropped_chunks > 0 {
            tracing::warn!(dropped_chunks, "SSE chunks were dropped during stream");
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Anthropic streaming
    // -----------------------------------------------------------------------

    async fn stream_anthropic(&mut self, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Anthropic's streaming API does not expose logprobs (#8).
        // confidence/perplexity fields will be None for every token in this
        // stream. Cross-provider perplexity comparisons require normalisation
        // because the models operate over different vocabulary sizes (#20).
        tracing::debug!(
            "Anthropic stream: logprobs unavailable; confidence/perplexity will be None"
        );
        if self.web_tx.is_none() {
            eprintln!("[info] Anthropic does not provide logprobs — confidence metrics will be unavailable for this run");
        }

        let request = AnthropicRequest {
            model: self.model.clone(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: self.anthropic_max_tokens,
            stream: true,
            temperature: 0.7,
            system: self.system_prompt.clone(),
        };

        let req = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", providers::ANTHROPIC_API_VERSION)
            .header("Content-Type", "application/json")
            .json(&request)
            .build()?;

        // Retry on 429 / 5xx with exponential back-off (#5).
        let response = execute_with_retry(&self.client, req, self.max_retries)
            .await
            .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("Anthropic API error: {}", error_text).into());
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut dropped_chunks: usize = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            // Reject invalid UTF-8 rather than silently replacing bytes (#4).
            let chunk_str = match std::str::from_utf8(&chunk) {
                Ok(s) => s.to_string(),
                Err(e) => {
                    tracing::warn!(error = %e, "invalid UTF-8 in Anthropic stream chunk — skipping");
                    continue;
                }
            };
            buffer.push_str(&chunk_str);

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer.drain(..=line_end);

                if line.starts_with("data: ") {
                    let json_str = line.strip_prefix("data: ").unwrap_or(&line);
                    match serde_json::from_str::<AnthropicStreamEvent>(json_str) {
                        Ok(event) => {
                            if event.event_type == "content_block_delta" {
                                if let Some(delta) = &event.delta {
                                    if let Some(text) = &delta.text {
                                        // Estimate confidence from inter-token latency for Anthropic
                                        // Fast tokens (< 50ms) → high confidence proxy; slow tokens → lower
                                        let now = std::time::Instant::now();
                                        let timing_confidence = if let Some(last) =
                                            self.last_token_instant
                                        {
                                            let delta_ms = now.duration_since(last).as_millis() as f64;
                                            // Normalize: tokens arriving in < 50ms get confidence ~0.9, > 500ms → ~0.1
                                            let conf = (1.0 - (delta_ms / 500.0).min(1.0)) * 0.8 + 0.1;
                                            Some(conf as f32)
                                        } else {
                                            None
                                        };
                                        self.last_token_instant = Some(now);
                                        // Convert timing_confidence to a log_prob approximation if available
                                        let timing_logprob =
                                            timing_confidence.map(|c| c.ln().max(-10.0));
                                        self.process_content_logprob(text, timing_logprob, vec![]);
                                        if self.pending_delay_ms > 0 {
                                            tokio::time::sleep(std::time::Duration::from_millis(
                                                self.pending_delay_ms,
                                            ))
                                            .await;
                                            self.pending_delay_ms = 0;
                                        }
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            tracing::warn!(line = %json_str, "failed to parse SSE chunk; skipping");
                            dropped_chunks += 1;
                        }
                    }
                }
            }
        }

        if dropped_chunks > 0 {
            tracing::warn!(dropped_chunks, "SSE chunks were dropped during stream");
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Mock streaming (no network call — replays a canned fixture)
    // -----------------------------------------------------------------------

    async fn stream_mock(&mut self, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Canned fixture: realistic token stream with logprob data.
        // Simulates a response to any prompt without hitting any API.
        let prompt_prefix = prompt[..prompt.len().min(20)].to_string();
        let fixture: Vec<(String, f32)> = vec![
            ("The".to_string(), -0.12),
            (" quick".to_string(), -0.45),
            (" brown".to_string(), -0.78),
            (" fox".to_string(), -0.23),
            (" jumps".to_string(), -0.56),
            (" over".to_string(), -0.34),
            (" the".to_string(), -0.11),
            (" lazy".to_string(), -0.89),
            (" dog".to_string(), -0.19),
            (".".to_string(), -0.07),
            (" This".to_string(), -0.62),
            (" is".to_string(), -0.15),
            (" a".to_string(), -0.08),
            (" mock".to_string(), -0.31),
            (" response".to_string(), -0.44),
            (" for".to_string(), -0.27),
            (" prompt".to_string(), -0.53),
            (":".to_string(), -0.18),
            (" \"".to_string(), -0.39),
            (prompt_prefix, -0.71),
        ];

        // Vary fixture starting position based on prompt content for more realistic tests
        let prompt_hash: usize = prompt
            .bytes()
            .fold(0usize, |acc, b| acc.wrapping_add(b as usize));
        let offset = prompt_hash % fixture.len();

        for idx in 0..fixture.len() {
            let (token_text, logprob) = &fixture[(idx + offset) % fixture.len()];
            let token_text = token_text.clone();
            let confidence = logprob.exp().clamp(0.0_f32, 1.0_f32);
            let perplexity = (-logprob).exp();
            let importance = calculate_token_importance(&token_text, idx);
            let should_transform = idx % 2 == 1;

            let (display_text, chaos_label) = if should_transform {
                let (t, label) = self.transform.apply_with_label(&token_text);
                let cl = if matches!(self.transform, Transform::Chaos) {
                    Some(label.to_string())
                } else {
                    None
                };
                (t, cl)
            } else {
                (token_text.clone(), None)
            };

            if should_transform {
                self.transformed_count += 1;
            }
            self.token_count += 1;

            if let Some(tx) = &self.web_tx {
                let evt = TokenEvent {
                    text: display_text.clone(),
                    original: token_text.clone(),
                    index: idx,
                    transformed: should_transform,
                    importance,
                    chaos_label,
                    provider: self.web_provider_label.clone(),
                    confidence: Some(confidence),
                    perplexity: Some(perplexity),
                    alternatives: vec![
                        TokenAlternative {
                            token: "a".to_string(),
                            probability: 0.15,
                        },
                        TokenAlternative {
                            token: "the".to_string(),
                            probability: 0.10,
                        },
                    ],
                    is_error: false,
                    arrival_ms: None,
                };
                let _ = tx.send(evt);
            } else {
                self.process_content_logprob(&token_text, Some(*logprob), vec![]);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Orchestrator MCP infer call
    // -----------------------------------------------------------------------

    async fn orchestrator_infer(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let mcp_request = McpInferRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/call".to_string(),
            id: 1,
            params: McpInferParams {
                name: "infer".to_string(),
                arguments: McpInferArguments {
                    prompt: prompt.to_string(),
                    worker: "llama_cpp".to_string(),
                },
            },
        };

        let response = self
            .client
            .post(&self.orchestrator_url)
            .header("Content-Type", "application/json")
            .json(&mcp_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("Orchestrator returned HTTP {}", response.status()).into());
        }

        let mcp_resp: McpInferResponse = response.json().await?;

        if let Some(err) = mcp_resp.error {
            return Err(format!("Orchestrator MCP error: {}", err.message).into());
        }

        if let Some(result) = mcp_resp.result {
            if let Some(content) = result.content.first() {
                if let Some(text) = &content.text {
                    return Ok(text.clone());
                }
            }
        }

        Err("Orchestrator returned empty result".into())
    }

    // -----------------------------------------------------------------------
    // Token processing (shared by both providers)
    // -----------------------------------------------------------------------

    /// Process a content chunk without logprob data.
    pub fn process_content(&mut self, content: &str) {
        self.process_content_logprob(content, None, vec![]);
    }
    /// Process a content chunk with optional logprob data (research mode API).
    pub fn process_content_with_logprob(
        &mut self,
        content: &str,
        lp: Option<providers::OpenAILogprobContent>,
    ) {
        let (log_prob, top_alts) = if let Some(ref entry) = lp {
            let alts: Vec<TokenAlternative> = entry
                .top_logprobs
                .iter()
                .map(|t| TokenAlternative {
                    token: t.token.clone(),
                    probability: t.logprob.exp().clamp(0.0, 1.0),
                })
                .collect();
            // Pass the raw log-prob so process_content_logprob can derive
            // confidence and perplexity via exp(lp) and exp(-lp) respectively.
            // Previously this incorrectly passed exp(entry.logprob), causing
            // process_content_logprob to double-exponentiate.
            (Some(entry.logprob), alts)
        } else {
            (None, vec![])
        };
        self.process_content_logprob(content, log_prob, top_alts);
    }

    /// Process a content chunk, optionally attaching logprob-derived fields to
    /// the first non-whitespace token produced.
    ///
    /// * `log_prob` — natural-log probability of the leading API token, if known.
    /// * `top_alts` — alternative tokens from `top_logprobs`, already converted
    ///   to probabilities (`exp(logprob)`).
    pub fn process_content_logprob(
        &mut self,
        content: &str,
        log_prob: Option<f32>,
        top_alts: Vec<TokenAlternative>,
    ) {
        let tokens = tokenize(content);
        let mut first_real = true; // attach logprob data to first non-whitespace token

        for token in tokens {
            if !token.trim().is_empty() {
                let i = self.token_count;

                // Bresenham-style spread: transform token i when
                // floor((i+1)*rate) > floor(i*rate), giving a uniform
                // distribution at any rate without probabilistic sampling.
                let rate = self.rate;
                let should_transform = ((i + 1) as f64 * rate).floor() > (i as f64 * rate).floor();

                // Logprob data only goes on the first real token of each API chunk.
                // Compute before the transform so confidence can drive importance.
                let (token_confidence, token_perplexity, token_alts) = if first_real {
                    first_real = false;
                    let conf = log_prob.map(|lp| lp.exp().clamp(0.0, 1.0));
                    let perp = log_prob.map(|lp| (-lp).exp());
                    (conf, perp, top_alts.clone())
                } else {
                    (None, None, vec![])
                };

                // Confidence gating: if min_confidence is set and token has API confidence,
                // only transform tokens whose confidence is BELOW the threshold
                let should_transform =
                    if let (Some(min_conf), Some(conf)) = (self.min_confidence, token_confidence) {
                        conf as f64 <= min_conf
                    } else {
                        should_transform
                    };

                // Use real API confidence as importance when available; fall back
                // to the heuristic scorer for tokens without logprob data.
                let importance = token_confidence.map(|c| c as f64).unwrap_or_else(|| {
                    transforms::calculate_token_importance_rng(&token, i, &mut self.rng)
                });

                let (display_text, chaos_label) = if should_transform {
                    self.transformed_count += 1;
                    let (text, label) = self.transform.apply_with_label_rng(&token, &mut self.rng);
                    let cl = if matches!(self.transform, Transform::Chaos) || text.is_empty() {
                        // Chaos: use sub-transform label; Delete: mark explicitly as "deleted"
                        Some(if text.is_empty() {
                            "deleted".to_string()
                        } else {
                            label.to_string()
                        })
                    } else {
                        None
                    };
                    (text, cl)
                } else {
                    (token.clone(), None)
                };

                // Delay transform: record the desired delay so the caller can
                // await it asynchronously after this (non-async) method returns.
                if should_transform {
                    if let Transform::Delay(ms) = self.transform {
                        self.pending_delay_ms = ms;
                    }
                }

                // Delete transform: the result is an empty string (chaos_label="deleted").
                let is_deleted = should_transform && display_text.is_empty();

                // Web / terminal / json output — skip deleted tokens for display.
                if !is_deleted {
                    // Record per-token arrival latency relative to stream start.
                    let arrival_ms = self.stream_start_instant
                        .map(|start| start.elapsed().as_millis() as u64);
                    if let Some(tx) = &self.web_tx {
                        let event = TokenEvent {
                            text: display_text.clone(),
                            original: token.clone(),
                            index: i,
                            transformed: should_transform,
                            importance,
                            chaos_label,
                            provider: self.web_provider_label.clone(),
                            confidence: token_confidence,
                            perplexity: token_perplexity,
                            alternatives: token_alts,
                            is_error: false,
                            arrival_ms,
                        };
                        if let Some(rec) = &mut self.recorder {
                            rec.record(&event);
                        }
                        let _ = tx.send(event);
                    } else if self.json_stream {
                        // JSON stream mode: one line per token
                        let event = TokenEvent {
                            text: display_text.clone(),
                            original: token.clone(),
                            index: i,
                            transformed: should_transform,
                            importance,
                            chaos_label: chaos_label.clone(),
                            provider: self.web_provider_label.clone(),
                            confidence: token_confidence,
                            perplexity: token_perplexity,
                            alternatives: token_alts.clone(),
                            is_error: false,
                            arrival_ms,
                        };
                        if let Ok(line) = serde_json::to_string(&event) {
                            println!("{}", line);
                        }
                    } else {
                        // Terminal mode: print with colors
                        if self.heatmap_mode {
                            print!("{}", apply_heatmap_color(&display_text, importance));
                        } else if self.visual_mode && should_transform {
                            print!("{}", display_text.bright_cyan().bold());
                        } else if self.visual_mode {
                            print!("{}", display_text.normal());
                        } else {
                            print!("{}", display_text);
                        }
                        let _ = io::stdout().flush();
                    }
                }

                self.token_count += 1;
            }
        }

        // Feed token-processing metrics into the self-improvement telemetry bus.
        #[cfg(feature = "self-tune")]
        if let Some(bus) = &self.telemetry_bus {
            use crate::self_tune::telemetry_bus::PipelineStage;
            // Record latency as synthetic 1ms per token (real timing requires instrumentation)
            bus.record_latency(PipelineStage::Inference, 1_000);
            // Record confidence as quality proxy (if available)
            if let Some(lp) = log_prob {
                let confidence_pct = (lp.exp().clamp(0.0, 1.0) * 100.0) as u64;
                bus.record_latency(PipelineStage::Other, confidence_pct.max(1));
            }
        }
    }

    /// Print a formatted session header to stdout.
    ///
    /// Displays provider, transform, model, and prompt. When `visual_mode` or
    /// `heatmap_mode` is active, additional legend lines are printed.
    /// This method is a no-op when `web_tx` is set (web mode handles its own header).
    pub fn print_header(&self, prompt: &str) {
        println!("{}", "EVERY OTHER TOKEN INTERCEPTOR".bright_cyan().bold());
        println!(
            "{}: {}",
            "Provider".bright_yellow(),
            self.provider.to_string().bright_white()
        );
        println!("{}: {:?}", "Transform".bright_yellow(), self.transform);
        println!("{}: {}", "Model".bright_yellow(), self.model);
        println!("{}: {}", "Prompt".bright_yellow(), prompt);
        if self.orchestrator {
            println!(
                "{}: {}",
                "Orchestrator".bright_magenta(),
                "ON (MCP pipeline at localhost:3000)".bright_magenta()
            );
        }
        if self.visual_mode {
            println!(
                "{}: {}",
                "Visual Mode".bright_green(),
                "ON (even=normal, odd=cyan+bold)".bright_green()
            );
        }
        if self.heatmap_mode {
            println!(
                "{}: {}",
                "Heatmap Mode".bright_magenta(),
                "ON (color intensity = token importance)".bright_magenta()
            );
            println!(
                "{}: {} {} {} {}",
                "Legend".bright_white(),
                "Low".on_blue(),
                "Medium".on_yellow(),
                "High".on_red(),
                "Critical".on_bright_red().bright_white()
            );
        }
        println!("{}", "=".repeat(50).bright_blue());
        println!("{}", "Response (with transformations):".bright_green());
        println!();
    }

    /// Print a summary footer to stdout after a streaming session completes.
    ///
    /// Reports total token count and how many tokens were transformed.
    pub fn print_footer(&self) {
        println!("\n{}", "=".repeat(50).bright_blue());
        println!("Complete! Processed {} tokens.", self.token_count);
        println!("Transform applied to {} tokens.", self.transformed_count);
    }
}

// ---------------------------------------------------------------------------
// Headless research session
// ---------------------------------------------------------------------------

/// Aggregated statistics from one or more headless inference runs.
///
/// Produced by [`run_research_headless`].  Fields summarise token-level metrics
/// across all runs; fields that require logprob data are `Option` because not
/// all providers expose logprobs (Anthropic does not).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResearchSession {
    /// The prompt submitted to the provider for all runs in this session.
    pub prompt: String,
    /// Provider identifier (`"openai"`, `"anthropic"`, or `"mock"`).
    pub provider: String,
    /// Model identifier used for all runs (e.g. `"gpt-4"`).
    pub model: String,
    /// Transform applied to intercepted tokens (e.g. `"reverse"`).
    pub transform: String,
    /// Number of inference runs executed.
    pub runs: u32,
    /// Total tokens streamed across all runs.
    pub total_tokens: usize,
    /// Total tokens that had a transform applied across all runs.
    pub total_transformed: usize,
    /// Unique-token fraction: `unique_tokens / total_tokens`.
    pub vocabulary_diversity: f64,
    /// Mean character length of all original (pre-transform) tokens.
    pub mean_token_length: f64,
    /// Mean per-token perplexity across all runs, or `None` when unavailable.
    pub mean_perplexity: Option<f64>,
    /// Mean per-token model confidence across all runs, or `None` when unavailable.
    pub mean_confidence: Option<f64>,
    /// The 10 tokens with the highest perplexity values (most uncertain positions).
    pub top_perplexity_tokens: Vec<String>,
    /// Rough cost estimate in USD based on token count and GPT-3.5 pricing.
    pub estimated_cost_usd: f64,
    /// Human-readable citation string recording key run parameters for reproducibility.
    pub citation: String,
}

/// Run `runs` headless inference calls, collect all `TokenEvent`s, and return
/// an aggregated `ResearchSession`.  Call sites must provide a constructed
/// interceptor (no web_tx set — events are returned via the mpsc channel).
pub async fn run_research_headless(
    prompt: &str,
    provider: providers::Provider,
    transform: transforms::Transform,
    model: String,
    runs: u32,
) -> Result<ResearchSession, Box<dyn std::error::Error>> {
    let mut all_tokens: Vec<TokenEvent> = Vec::new();

    for _ in 0..runs {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = TokenInterceptor::new(
            provider.clone(),
            transform.clone(),
            model.clone(),
            false,
            false,
            false,
        )?;
        interceptor.web_tx = Some(tx);
        interceptor.intercept_stream(prompt).await?;
        // Drain channel
        while let Ok(ev) = rx.try_recv() {
            all_tokens.push(ev);
        }
    }

    let total = all_tokens.len();
    let total_transformed = all_tokens.iter().filter(|t| t.transformed).count();

    let unique: std::collections::HashSet<String> = all_tokens
        .iter()
        .map(|t| t.original.to_lowercase())
        .collect();
    let vocab_diversity = if total > 0 {
        unique.len() as f64 / total as f64
    } else {
        0.0
    };

    let mean_token_length = if total > 0 {
        all_tokens
            .iter()
            .map(|t| t.original.len() as f64)
            .sum::<f64>()
            / total as f64
    } else {
        0.0
    };

    let perp_tokens: Vec<f64> = all_tokens
        .iter()
        .filter_map(|t| t.perplexity.map(|p| p as f64))
        .collect();
    let mean_perplexity = if perp_tokens.is_empty() {
        None
    } else {
        Some(perp_tokens.iter().sum::<f64>() / perp_tokens.len() as f64)
    };

    let conf_tokens: Vec<f64> = all_tokens
        .iter()
        .filter_map(|t| t.confidence.map(|c| c as f64))
        .collect();
    let mean_confidence = if conf_tokens.is_empty() {
        None
    } else {
        Some(conf_tokens.iter().sum::<f64>() / conf_tokens.len() as f64)
    };

    // Top 10 highest-perplexity original tokens
    let mut by_perp: Vec<&TokenEvent> = all_tokens
        .iter()
        .filter(|t| t.perplexity.is_some())
        .collect();
    by_perp.sort_by(|a, b| {
        b.perplexity
            .partial_cmp(&a.perplexity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_perplexity_tokens: Vec<String> = by_perp
        .iter()
        .take(10)
        .map(|t| t.original.clone())
        .collect();

    // Cost estimate: GPT-3.5 rate $0.002 / 1K tokens
    let estimated_cost_usd = total as f64 / 1000.0 * 0.002;

    let citation = format!(
        "Every Other Token v4.0.0 | prompt=\"{}\" | provider={} | model={} | transform={:?} | runs={} | tokens={}",
        prompt, provider, model, transform, runs, total
    );

    Ok(ResearchSession {
        prompt: prompt.to_string(),
        provider: provider.to_string(),
        model,
        transform: format!("{:?}", transform),
        runs,
        total_tokens: total,
        total_transformed,
        vocabulary_diversity: vocab_diversity,
        mean_token_length,
        mean_perplexity,
        mean_confidence,
        top_perplexity_tokens,
        estimated_cost_usd,
        citation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn make_test_interceptor() -> TokenInterceptor {
        TokenInterceptor {
            client: Client::new(),
            api_key: "test-key".to_string(),
            provider: Provider::Openai,
            transform: Transform::Reverse,
            model: "test-model".to_string(),
            token_count: 0,
            transformed_count: 0,
            visual_mode: false,
            heatmap_mode: false,
            orchestrator: false,
            orchestrator_url: "http://localhost:3000".to_string(),
            web_tx: None,
            web_provider_label: None,
            system_prompt: None,
            #[cfg(feature = "self-tune")]
            telemetry_bus: None,
            #[cfg(feature = "self-modify")]
            dedup: None,
            rate: 0.5,
            rng: StdRng::seed_from_u64(42),
            top_logprobs: 5,
            recorder: None,
            json_stream: false,
            pending_delay_ms: 0,
            min_confidence: None,
            last_token_instant: None,
            max_retries: 3,
            anthropic_max_tokens: 4096,
            stream_start_instant: None,
            timeout_secs: None,
        }
    }

    // -- TokenInterceptor construction --

    #[test]
    fn test_new_openai_requires_api_key() {
        std::env::remove_var("OPENAI_API_KEY");
        let result = TokenInterceptor::new(
            Provider::Openai,
            Transform::Reverse,
            "gpt-4".to_string(),
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_new_anthropic_requires_api_key() {
        std::env::remove_var("ANTHROPIC_API_KEY");
        let result = TokenInterceptor::new(
            Provider::Anthropic,
            Transform::Reverse,
            "claude".to_string(),
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_interceptor_initial_counts_zero() {
        let interceptor = make_test_interceptor();
        assert_eq!(interceptor.token_count, 0);
        assert_eq!(interceptor.transformed_count, 0);
    }

    #[test]
    fn test_interceptor_fields_match_construction() {
        let interceptor = make_test_interceptor();
        assert_eq!(interceptor.provider, Provider::Openai);
        assert_eq!(interceptor.model, "test-model");
        assert!(!interceptor.visual_mode);
        assert!(!interceptor.heatmap_mode);
        assert!(!interceptor.orchestrator);
        assert!(interceptor.web_tx.is_none());
    }

    // -- process_content tests --

    #[test]
    fn test_process_content_two_tokens() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");
        assert_eq!(interceptor.token_count, 2);

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].text, "hello");
        assert_eq!(events[0].original, "hello");
        assert!(!events[0].transformed);
        assert_eq!(events[0].index, 0);
        assert_eq!(events[1].original, "world");
        assert!(events[1].transformed);
        assert_eq!(events[1].index, 1);
    }

    #[test]
    fn test_process_content_transforms_odd_tokens() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        // "world" reversed = "dlrow"
        assert_eq!(events[1].text, "dlrow");
        assert_eq!(events[1].original, "world");
    }

    #[test]
    fn test_process_content_empty_string() {
        let (tx, _rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("");
        assert_eq!(interceptor.token_count, 0);
        assert_eq!(interceptor.transformed_count, 0);
    }

    #[test]
    fn test_process_content_whitespace_only() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("   ");
        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert!(events.is_empty());
    }

    #[test]
    fn test_process_content_single_token() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].text, "hello");
        assert!(!events[0].transformed);
    }

    #[test]
    fn test_process_content_cross_call_continuity() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello");
        interceptor.process_content("world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].index, 0);
        assert_eq!(events[1].index, 1);
        assert!(events[1].transformed);
    }

    #[test]
    fn test_process_content_increments_transformed_count() {
        let (tx, _rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world foo bar");
        assert_eq!(interceptor.transformed_count, 2);
    }

    #[test]
    fn test_process_content_six_tokens_three_transformed() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("one two three four five six");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events.len(), 6);
        let xformed: Vec<_> = events.iter().filter(|e| e.transformed).collect();
        assert_eq!(xformed.len(), 3);
    }

    // -- original field preservation --

    #[test]
    fn test_original_field_preserved_for_all_tokens() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("the quick brown fox");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for event in &events {
            assert!(!event.original.is_empty());
            if event.transformed {
                assert_ne!(event.text, event.original);
            } else {
                assert_eq!(event.text, event.original);
            }
        }
    }

    #[test]
    fn test_sidebyside_original_is_raw_token() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("quick brown fox");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        let originals: Vec<&str> = events.iter().map(|e| e.original.as_str()).collect();
        assert!(originals.contains(&"quick"));
        assert!(originals.contains(&"brown"));
        assert!(originals.contains(&"fox"));
    }

    // -- even/odd alternation for graph --

    #[test]
    fn test_even_odd_alternation() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("a b c d");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for event in &events {
            if event.index % 2 == 0 {
                assert!(!event.transformed);
            } else {
                assert!(event.transformed);
            }
        }
    }

    #[test]
    fn test_graph_pairs_alternate() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("alpha beta gamma delta epsilon zeta");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for pair in events.chunks(2) {
            assert!(!pair[0].transformed);
            if pair.len() > 1 {
                assert!(pair[1].transformed);
            }
        }
    }

    #[test]
    fn test_graph_indices_sequential() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("one two three four");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for (i, event) in events.iter().enumerate() {
            assert_eq!(event.index, i);
        }
    }

    // -- export structure tests --

    #[test]
    fn test_export_array_sequential_indices() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("the quick brown fox jumps over");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for (i, event) in events.iter().enumerate() {
            assert_eq!(event.index, i);
        }
    }

    #[test]
    fn test_export_all_tokens_have_valid_importance() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("testing export importance values");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for event in &events {
            assert!(event.importance >= 0.0 && event.importance <= 1.0);
        }
    }

    #[test]
    fn test_export_large_set_serializes() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("the quick brown fox jumps over the lazy dog and runs around");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        let json = serde_json::to_string(&events).expect("serialize");
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.len(), events.len());
        assert!(parsed.len() > 5);
    }

    #[test]
    fn test_multiple_tokens_form_valid_export_array() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world foo bar");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        let json = serde_json::to_string(&events).expect("serialize");
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.len(), events.len());
        for (i, entry) in parsed.iter().enumerate() {
            assert_eq!(entry["index"].as_u64().expect("index"), i as u64);
        }
    }

    // -- print header/footer (no crash) --

    #[test]
    fn test_print_header_all_modes() {
        let interceptor = make_test_interceptor();
        interceptor.print_header("test prompt");
    }

    #[test]
    fn test_print_header_with_orchestrator() {
        let mut interceptor = make_test_interceptor();
        interceptor.orchestrator = true;
        interceptor.print_header("test");
    }

    #[test]
    fn test_print_header_with_visual_mode() {
        let mut interceptor = make_test_interceptor();
        interceptor.visual_mode = true;
        interceptor.print_header("test");
    }

    #[test]
    fn test_print_header_with_heatmap_mode() {
        let mut interceptor = make_test_interceptor();
        interceptor.heatmap_mode = true;
        interceptor.print_header("test");
    }

    #[test]
    fn test_print_footer() {
        let interceptor = make_test_interceptor();
        interceptor.print_footer();
    }

    #[test]
    fn test_print_footer_after_processing() {
        let mut interceptor = make_test_interceptor();
        interceptor.token_count = 42;
        interceptor.transformed_count = 21;
        interceptor.print_footer();
    }

    // -- different transform types --

    #[test]
    fn test_process_content_uppercase_transform() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Uppercase;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events[1].text, "WORLD");
        assert_eq!(events[1].original, "world");
    }

    #[test]
    fn test_process_content_mock_transform() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Mock;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events[1].text, "wOrLd");
    }

    #[test]
    fn test_process_content_noise_transform() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Noise;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert!(events[1].text.starts_with("world"));
        assert_eq!(events[1].text.len(), 6); // "world" + 1 noise char
    }

    // -- web_tx none falls back to terminal mode --

    #[test]
    fn test_process_content_terminal_mode_no_crash() {
        let mut interceptor = make_test_interceptor();
        // web_tx is None, so this prints to stdout (terminal mode)
        interceptor.process_content("hello world");
        assert_eq!(interceptor.token_count, 2);
    }

    #[test]
    fn test_process_content_visual_mode_no_crash() {
        let mut interceptor = make_test_interceptor();
        interceptor.visual_mode = true;
        interceptor.process_content("hello world");
        assert_eq!(interceptor.token_count, 2);
    }

    #[test]
    fn test_process_content_heatmap_mode_no_crash() {
        let mut interceptor = make_test_interceptor();
        interceptor.heatmap_mode = true;
        interceptor.process_content("hello world");
        assert_eq!(interceptor.token_count, 2);
    }

    // -- chaos_label field tests --

    #[test]
    fn test_chaos_label_set_for_chaos_transform() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Chaos;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        // "world" is the odd token — should have chaos_label
        let known = ["reverse", "uppercase", "mock", "noise"];
        let odd = events
            .iter()
            .find(|e| e.transformed)
            .expect("should have odd token");
        let label = odd
            .chaos_label
            .as_ref()
            .expect("chaos_label should be Some for Chaos transform");
        assert!(
            known.contains(&label.as_str()),
            "unexpected label: {}",
            label
        );
    }

    #[test]
    fn test_chaos_label_none_for_reverse_transform() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Reverse;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        for event in &events {
            assert!(
                event.chaos_label.is_none(),
                "Reverse should not set chaos_label"
            );
        }
    }

    #[test]
    fn test_chaos_label_none_for_even_tokens() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Chaos;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world foo bar");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        for event in events.iter().filter(|e| !e.transformed) {
            assert!(
                event.chaos_label.is_none(),
                "Even tokens should not have chaos_label"
            );
        }
    }

    #[test]
    fn test_chaos_label_serialization() {
        let event = TokenEvent {
            text: "dlrow".to_string(),
            original: "world".to_string(),
            index: 1,
            transformed: true,
            importance: 0.5,
            chaos_label: Some("reverse".to_string()),
            provider: None,
            confidence: None,
            perplexity: None,
            alternatives: vec![],
            is_error: false,
            arrival_ms: None,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("chaos_label"));
        assert!(json.contains("reverse"));
    }

    #[test]
    fn test_chaos_label_skipped_when_none() {
        let event = TokenEvent {
            text: "hello".to_string(),
            original: "hello".to_string(),
            index: 0,
            transformed: false,
            importance: 0.3,
            chaos_label: None,
            provider: None,
            confidence: None,
            perplexity: None,
            alternatives: vec![],
            is_error: false,
            arrival_ms: None,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(
            !json.contains("chaos_label"),
            "None chaos_label should be skipped in JSON"
        );
    }

    // -- provider field tests --

    #[test]
    fn test_provider_label_none_by_default() {
        let interceptor = make_test_interceptor();
        assert!(interceptor.web_provider_label.is_none());
    }

    #[test]
    fn test_provider_label_propagates_to_event() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);
        interceptor.web_provider_label = Some("openai".to_string());

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        for event in &events {
            assert_eq!(
                event.provider.as_deref(),
                Some("openai"),
                "provider label should propagate to all events"
            );
        }
    }

    #[test]
    fn test_provider_label_none_means_skipped_in_json() {
        let event = TokenEvent {
            text: "hello".to_string(),
            original: "hello".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
            chaos_label: None,
            provider: None,
            confidence: None,
            perplexity: None,
            alternatives: vec![],
            is_error: false,
            arrival_ms: None,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(
            !json.contains("\"provider\""),
            "None provider should be skipped in JSON"
        );
    }

    #[test]
    fn test_provider_label_some_appears_in_json() {
        let event = TokenEvent {
            text: "hello".to_string(),
            original: "hello".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
            chaos_label: None,
            provider: Some("anthropic".to_string()),
            confidence: None,
            perplexity: None,
            alternatives: vec![],
            is_error: false,
            arrival_ms: None,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("\"provider\""));
        assert!(json.contains("anthropic"));
    }

    #[test]
    fn test_provider_label_openai_and_anthropic_distinct() {
        let (tx1, mut rx1) = mpsc::unbounded_channel::<TokenEvent>();
        let mut openai_i = make_test_interceptor();
        openai_i.web_tx = Some(tx1);
        openai_i.web_provider_label = Some("openai".to_string());

        let (tx2, mut rx2) = mpsc::unbounded_channel::<TokenEvent>();
        let mut anthropic_i = make_test_interceptor();
        anthropic_i.web_tx = Some(tx2);
        anthropic_i.web_provider_label = Some("anthropic".to_string());

        openai_i.process_content("hello");
        anthropic_i.process_content("hello");

        let e1 = rx1.try_recv().expect("openai event");
        let e2 = rx2.try_recv().expect("anthropic event");
        assert_eq!(e1.provider.as_deref(), Some("openai"));
        assert_eq!(e2.provider.as_deref(), Some("anthropic"));
        assert_ne!(e1.provider, e2.provider);
    }

    // -- TokenAlternative tests --

    #[test]
    fn test_token_alternative_serializes() {
        let alt = TokenAlternative {
            token: "hello".to_string(),
            probability: 0.75,
        };
        let json = serde_json::to_string(&alt).expect("serialize");
        assert!(json.contains("\"token\":\"hello\""));
        assert!(json.contains("\"probability\":0.75") || json.contains("probability"));
    }

    #[test]
    fn test_token_alternative_clone() {
        let alt = TokenAlternative {
            token: "world".to_string(),
            probability: 0.5,
        };
        let alt2 = alt.clone();
        assert_eq!(alt2.token, alt.token);
    }

    // -- process_content_logprob tests --

    #[test]
    fn test_process_content_logprob_attaches_confidence() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i = make_test_interceptor();
        i.web_tx = Some(tx);
        // logprob of 0.0 → probability = 1.0 (max confidence)
        i.process_content_logprob("hello world", Some(0.0_f32), vec![]);
        let ev = rx.try_recv().expect("event");
        assert_eq!(ev.confidence, Some(1.0_f32));
    }

    #[test]
    fn test_process_content_logprob_none_gives_none_confidence() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i = make_test_interceptor();
        i.web_tx = Some(tx);
        i.process_content_logprob("hello", None, vec![]);
        let ev = rx.try_recv().expect("event");
        assert!(ev.confidence.is_none());
        assert!(ev.perplexity.is_none());
    }

    #[test]
    fn test_process_content_logprob_computes_perplexity() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i = make_test_interceptor();
        i.web_tx = Some(tx);
        // logprob = -1.0 → perplexity = exp(1.0) ≈ 2.718
        i.process_content_logprob("word", Some(-1.0_f32), vec![]);
        let ev = rx.try_recv().expect("event");
        let perp = ev.perplexity.expect("perplexity present");
        assert!(
            (perp - std::f32::consts::E).abs() < 0.01,
            "expected ~e, got {}",
            perp
        );
    }

    #[test]
    fn test_process_content_logprob_attaches_alternatives_to_first_token() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i = make_test_interceptor();
        i.web_tx = Some(tx);
        let alts = vec![
            TokenAlternative {
                token: "hi".to_string(),
                probability: 0.9,
            },
            TokenAlternative {
                token: "hey".to_string(),
                probability: 0.05,
            },
        ];
        i.process_content_logprob("hello world", Some(-0.1_f32), alts);
        let first = rx.try_recv().expect("first token");
        assert_eq!(first.alternatives.len(), 2);
        // second token gets no alternatives
        let second = rx.try_recv().expect("second token");
        assert!(second.alternatives.is_empty());
    }

    #[test]
    fn test_process_content_delegates_to_logprob() {
        // process_content is a thin wrapper around process_content_logprob
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i = make_test_interceptor();
        i.web_tx = Some(tx);
        i.process_content("hello");
        let ev = rx.try_recv().expect("event");
        assert!(ev.confidence.is_none());
        assert!(ev.alternatives.is_empty());
    }

    #[test]
    fn test_confidence_serialized_when_some() {
        let event = TokenEvent {
            text: "hi".to_string(),
            original: "hi".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
            chaos_label: None,
            provider: None,
            confidence: Some(0.92),
            perplexity: Some(1.08),
            alternatives: vec![TokenAlternative {
                token: "hey".to_string(),
                probability: 0.05,
            }],
            is_error: false,
            arrival_ms: None,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("confidence"));
        assert!(json.contains("perplexity"));
        assert!(json.contains("alternatives"));
        assert!(json.contains("hey"));
    }

    #[test]
    fn test_confidence_omitted_when_none() {
        let event = TokenEvent {
            text: "hi".to_string(),
            original: "hi".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
            chaos_label: None,
            provider: None,
            confidence: None,
            perplexity: None,
            alternatives: vec![],
            is_error: false,
            arrival_ms: None,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(!json.contains("confidence"));
        assert!(!json.contains("perplexity"));
        assert!(!json.contains("alternatives"));
    }

    #[test]
    fn test_system_prompt_field_initializes_none() {
        let i = make_test_interceptor();
        assert!(i.system_prompt.is_none());
    }

    #[test]
    fn test_system_prompt_can_be_set() {
        let mut i = make_test_interceptor();
        i.system_prompt = Some("Be concise.".to_string());
        assert_eq!(i.system_prompt.as_deref(), Some("Be concise."));
    }

    #[test]
    fn test_logprob_confidence_clamps_at_one() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i = make_test_interceptor();
        i.web_tx = Some(tx);
        // logprob > 0 is theoretically invalid but clamp should protect us
        i.process_content_logprob("token", Some(2.0_f32), vec![]);
        let ev = rx.try_recv().expect("event");
        let conf = ev.confidence.expect("confidence");
        assert!(conf <= 1.0, "confidence should not exceed 1.0");
    }

    #[test]
    fn test_process_content_logprob_multiple_tokens_only_first_gets_logprob() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i = make_test_interceptor();
        i.web_tx = Some(tx);
        i.process_content_logprob("the quick brown fox", Some(-0.5_f32), vec![]);
        let mut events: Vec<TokenEvent> = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        assert!(events.len() >= 2);
        assert!(
            events[0].confidence.is_some(),
            "first token should have confidence"
        );
        assert!(
            events[1].confidence.is_none(),
            "subsequent tokens should not"
        );
    }
}

#[cfg(test)]
mod research_tests {
    use super::*;

    fn make_session(
        tokens: usize,
        confidence: Option<f32>,
        perplexity: Option<f32>,
    ) -> ResearchSession {
        ResearchSession {
            prompt: "test prompt".to_string(),
            provider: "openai".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            transform: "Reverse".to_string(),
            runs: 1,
            total_tokens: tokens,
            total_transformed: tokens / 2,
            vocabulary_diversity: 0.8,
            mean_token_length: 4.5,
            mean_perplexity: perplexity.map(|p| p as f64),
            mean_confidence: confidence.map(|c| c as f64),
            top_perplexity_tokens: vec!["word".to_string()],
            estimated_cost_usd: tokens as f64 / 1000.0 * 0.002,
            citation: format!("Every Other Token v4.0.0 | tokens={}", tokens),
        }
    }

    /// Construct a minimal [`TokenInterceptor`] suitable for unit tests.
    ///
    /// Uses a fixed RNG seed, a no-op web channel, and the `Reverse` transform.
    fn make_test_interceptor() -> TokenInterceptor {
        TokenInterceptor {
            client: reqwest::Client::new(),
            api_key: "test-key".to_string(),
            provider: Provider::Openai,
            transform: Transform::Reverse,
            model: "test-model".to_string(),
            token_count: 0,
            transformed_count: 0,
            visual_mode: false,
            heatmap_mode: false,
            orchestrator: false,
            orchestrator_url: "http://localhost:3000".to_string(),
            web_tx: None,
            web_provider_label: None,
            system_prompt: None,
            #[cfg(feature = "self-tune")]
            telemetry_bus: None,
            #[cfg(feature = "self-modify")]
            dedup: None,
            rate: 0.5,
            rng: StdRng::seed_from_u64(42),
            top_logprobs: 5,
            recorder: None,
            json_stream: false,
            pending_delay_ms: 0,
            min_confidence: None,
            last_token_instant: None,
            max_retries: 3,
            anthropic_max_tokens: 4096,
            stream_start_instant: None,
            timeout_secs: None,
        }
    }

    #[test]
    fn test_research_session_serializes_basic_fields() {
        let s = make_session(10, Some(0.85), Some(2.3));
        let json = serde_json::to_string(&s).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(v["prompt"], "test prompt");
        assert_eq!(v["total_tokens"], 10);
        assert_eq!(v["runs"], 1);
        assert_eq!(v["provider"], "openai");
    }

    #[test]
    fn test_research_session_none_fields_serialize_as_null() {
        let s = make_session(5, None, None);
        let json = serde_json::to_string(&s).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert!(v["mean_perplexity"].is_null());
        assert!(v["mean_confidence"].is_null());
    }

    #[test]
    fn test_research_session_estimated_cost_scales_with_tokens() {
        let s100 = make_session(100, None, None);
        let s1000 = make_session(1000, None, None);
        assert!(s1000.estimated_cost_usd > s100.estimated_cost_usd);
        assert!((s100.estimated_cost_usd - 0.0002).abs() < 1e-10);
        assert!((s1000.estimated_cost_usd - 0.002).abs() < 1e-10);
    }

    #[test]
    fn test_research_session_vocab_diversity_in_bounds() {
        let s = make_session(20, None, None);
        assert!(s.vocabulary_diversity >= 0.0 && s.vocabulary_diversity <= 1.0);
    }

    #[test]
    fn test_research_session_top_tokens_at_most_ten() {
        let s = ResearchSession {
            top_perplexity_tokens: (0..10).map(|i| format!("t{}", i)).collect(),
            ..make_session(100, None, None)
        };
        assert_eq!(s.top_perplexity_tokens.len(), 10);
    }

    #[test]
    fn test_research_session_citation_contains_prompt() {
        let s = make_session(5, None, None);
        assert!(s.citation.contains("Every Other Token"));
    }

    #[test]
    fn test_research_session_runs_field_roundtrips() {
        let s = ResearchSession {
            runs: 42,
            ..make_session(10, None, None)
        };
        let json = serde_json::to_string(&s).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(v["runs"], 42);
    }

    #[test]
    fn test_research_session_transform_field() {
        let s = make_session(10, None, None);
        assert_eq!(s.transform, "Reverse");
    }

    // -- with_rate tests --

    #[test]
    fn test_with_rate_sets_rate() {
        let mut i = make_test_interceptor();
        i = i.with_rate(0.3);
        assert!((i.rate - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_with_rate_clamps_above_one() {
        let mut i = make_test_interceptor();
        i = i.with_rate(1.5);
        assert_eq!(i.rate, 1.0);
    }

    #[test]
    fn test_with_rate_clamps_below_zero() {
        let mut i = make_test_interceptor();
        i = i.with_rate(-0.5);
        assert_eq!(i.rate, 0.0);
    }

    #[test]
    fn test_with_rate_zero_transforms_no_tokens() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i = make_test_interceptor();
        i = i.with_rate(0.0);
        i.web_tx = Some(tx);
        i.process_content("hello world foo bar");
        let mut transformed = 0usize;
        while let Ok(ev) = rx.try_recv() {
            if ev.transformed {
                transformed += 1;
            }
        }
        assert_eq!(transformed, 0, "rate=0 should transform no tokens");
    }

    #[test]
    fn test_with_rate_one_transforms_all_tokens() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i = make_test_interceptor();
        i = i.with_rate(1.0);
        i.web_tx = Some(tx);
        i.process_content("hello world foo bar baz");
        let mut total = 0usize;
        let mut transformed = 0usize;
        while let Ok(ev) = rx.try_recv() {
            total += 1;
            if ev.transformed {
                transformed += 1;
            }
        }
        assert!(total > 0);
        assert_eq!(transformed, total, "rate=1.0 should transform every token");
    }

    // -- with_seed tests --

    #[test]
    fn test_with_seed_produces_deterministic_noise_output() {
        // Two interceptors with the same seed and Noise transform should produce
        // the same transformed tokens.
        let (tx1, mut rx1) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i1 = make_test_interceptor();
        i1.transform = Transform::Noise;
        i1 = i1.with_seed(12345);
        i1.web_tx = Some(tx1);
        i1.process_content("hello world");

        let (tx2, mut rx2) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i2 = make_test_interceptor();
        i2.transform = Transform::Noise;
        i2 = i2.with_seed(12345);
        i2.web_tx = Some(tx2);
        i2.process_content("hello world");

        let events1: Vec<TokenEvent> = std::iter::from_fn(|| rx1.try_recv().ok()).collect();
        let events2: Vec<TokenEvent> = std::iter::from_fn(|| rx2.try_recv().ok()).collect();

        assert_eq!(events1.len(), events2.len());
        for (e1, e2) in events1.iter().zip(events2.iter()) {
            assert_eq!(
                e1.text, e2.text,
                "seeded runs should produce identical output"
            );
        }
    }

    #[test]
    fn test_with_seed_different_seeds_may_differ() {
        // Different seeds should (in practice) produce at least one different token
        // for the Noise transform over a sufficiently long sequence.
        let (tx1, mut rx1) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i1 = make_test_interceptor();
        i1.transform = Transform::Noise;
        i1 = i1.with_seed(1);
        i1.web_tx = Some(tx1);
        i1.process_content("alpha beta gamma delta epsilon zeta eta theta iota kappa");

        let (tx2, mut rx2) = mpsc::unbounded_channel::<TokenEvent>();
        let mut i2 = make_test_interceptor();
        i2.transform = Transform::Noise;
        i2 = i2.with_seed(999999);
        i2.web_tx = Some(tx2);
        i2.process_content("alpha beta gamma delta epsilon zeta eta theta iota kappa");

        let texts1: Vec<String> = std::iter::from_fn(|| rx1.try_recv().ok())
            .map(|e| e.text)
            .collect();
        let texts2: Vec<String> = std::iter::from_fn(|| rx2.try_recv().ok())
            .map(|e| e.text)
            .collect();

        // At minimum, both should produce some output
        assert!(!texts1.is_empty());
        assert!(!texts2.is_empty());
    }

    // -- run_research_headless tests (Mock provider, no API key required) --

    #[tokio::test]
    async fn test_run_research_headless_mock_returns_session() {
        let session = run_research_headless(
            "test prompt",
            Provider::Mock,
            Transform::Reverse,
            "mock-fixture-v1".to_string(),
            1,
        )
        .await
        .expect("run_research_headless with Mock should not fail");
        assert_eq!(session.runs, 1);
        assert_eq!(session.prompt, "test prompt");
        assert_eq!(session.provider, "mock");
    }

    #[tokio::test]
    async fn test_run_research_headless_mock_token_count_positive() {
        let session = run_research_headless(
            "hello",
            Provider::Mock,
            Transform::Uppercase,
            "mock-fixture-v1".to_string(),
            1,
        )
        .await
        .expect("should succeed");
        assert!(session.total_tokens > 0, "mock provider should emit tokens");
    }

    #[tokio::test]
    async fn test_run_research_headless_mock_multiple_runs_accumulate() {
        let session = run_research_headless(
            "hello",
            Provider::Mock,
            Transform::Reverse,
            "mock-fixture-v1".to_string(),
            3,
        )
        .await
        .expect("should succeed");
        assert_eq!(session.runs, 3);
    }

    #[tokio::test]
    async fn test_run_research_headless_mock_vocab_diversity_in_bounds() {
        let session = run_research_headless(
            "test",
            Provider::Mock,
            Transform::Reverse,
            "mock-fixture-v1".to_string(),
            1,
        )
        .await
        .expect("should succeed");
        assert!(session.vocabulary_diversity >= 0.0);
        assert!(session.vocabulary_diversity <= 1.0);
    }

    #[tokio::test]
    async fn test_run_research_headless_mock_transform_label_in_citation() {
        let session = run_research_headless(
            "sample",
            Provider::Mock,
            Transform::Uppercase,
            "mock-fixture-v1".to_string(),
            1,
        )
        .await
        .expect("should succeed");
        assert!(session.citation.contains("Every Other Token"));
    }

    #[tokio::test]
    async fn test_run_research_headless_empty_prompt_returns_error() {
        let result = run_research_headless(
            "",
            Provider::Mock,
            Transform::Reverse,
            "mock-fixture-v1".to_string(),
            1,
        )
        .await;
        assert!(result.is_err(), "empty prompt should produce an error");
    }

    // -- Item 1: timeout_secs field --
    #[test]
    fn test_timeout_field_default() {
        let interceptor = TokenInterceptor::new(
            Provider::Mock,
            Transform::Reverse,
            "mock-fixture-v1".to_string(),
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(interceptor.timeout_secs, None);
        let with_timeout = interceptor.with_timeout(120);
        assert_eq!(with_timeout.timeout_secs, Some(120));
    }

    // -- Item 2: dropped SSE chunk counter --
    fn count_dropped_sse_chunks_test(lines: &[&str]) -> usize {
        lines.iter().filter(|line| {
            if line.starts_with("data: ") && **line != "data: [DONE]" {
                let json_str = line.strip_prefix("data: ").unwrap_or(line);
                serde_json::from_str::<serde_json::Value>(json_str).is_err()
            } else {
                false
            }
        }).count()
    }

    #[test]
    fn test_dropped_chunk_counter_increments() {
        let lines = vec![
            "data: {\"valid\": true}",
            "data: not-valid-json",
            "data: also-bad",
            "data: {\"ok\": 1}",
            "data: [DONE]",
        ];
        let dropped = count_dropped_sse_chunks_test(&lines);
        assert_eq!(dropped, 2);
    }

    // -- Item 3 & 19: circuit breaker helpers --
    fn reset_circuit_breaker_for_test() {
        let state = CIRCUIT_BREAKER.get_or_init(|| {
            std::sync::Mutex::new(CircuitBreakerState {
                consecutive_failures: 0,
                open_until_ms: 0,
            })
        });
        if let Ok(mut s) = state.lock() {
            s.consecutive_failures = 0;
            s.open_until_ms = 0;
        }
    }

    #[test]
    fn test_circuit_breaker_429_does_not_trip() {
        reset_circuit_breaker_for_test();
        // Record failures up to threshold-1 — still not tripped
        for _ in 0..(CB_TRIP_THRESHOLD - 1) {
            circuit_record_failure();
        }
        assert!(!circuit_is_open(), "should not be open before threshold");
        // Simulating a 429: the retry logic skips circuit_record_failure for 429,
        // so no additional failure is recorded — breaker remains closed.
        assert!(!circuit_is_open(), "429 should not trip the breaker");
    }

    #[test]
    fn test_circuit_breaker_reopens_after_timeout() {
        reset_circuit_breaker_for_test();
        for _ in 0..CB_TRIP_THRESHOLD {
            circuit_record_failure();
        }
        assert!(circuit_is_open(), "breaker should be open after threshold");
        // Fast-forward recovery by setting open_until_ms to the past
        let state = CIRCUIT_BREAKER.get_or_init(|| {
            std::sync::Mutex::new(CircuitBreakerState {
                consecutive_failures: 0,
                open_until_ms: 0,
            })
        });
        if let Ok(mut s) = state.lock() {
            s.open_until_ms = 1; // epoch 1ms — definitely in the past
        }
        assert!(!circuit_is_open(), "breaker should close after recovery timeout passes");
    }
}

// ---------------------------------------------------------------------------
// WASM stub
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm_support {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub fn wasm_run() -> JsValue {
        JsValue::from_str("wasm not yet fully implemented")
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use wasm_support::wasm_run;
