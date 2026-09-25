//! Run the interceptor against the built-in mock provider. No API key needed.
//!
//! ```bash
//! cargo run --example mock_stream
//! cargo run --example mock_stream -- "Your prompt" uppercase
//! ```
//!
//! Every other token is rewritten by the transform, and each token carries the
//! confidence (exp(logprob)) and perplexity (exp(-logprob)) the stream reported.
//! The mock provider replays a fixed fixture, so the text itself is canned; swap
//! `Provider::Mock` for `Provider::Openai` or `Provider::Anthropic` (with the
//! matching API key set) to intercept a real model.

use every_other_token::providers::Provider;
use every_other_token::transforms::Transform;
use every_other_token::TokenInterceptor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let prompt = args
        .next()
        .unwrap_or_else(|| "What is consciousness?".to_string());
    let transform = Transform::from_str_loose(&args.next().unwrap_or_else(|| "reverse".into()))?;

    // Route token events into a channel instead of printing them, so we can
    // show the per-token data the interceptor attaches.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut interceptor = TokenInterceptor::new(
        Provider::Mock,
        transform.clone(),
        "mock-fixture-v1".to_string(),
        false,
        false,
        false,
    )?
    .with_web_tx(tx);
    interceptor.intercept_stream(&prompt).await?;

    println!("prompt:    {prompt}");
    println!("transform: {transform:?} (applied to every other token)\n");
    println!("{:>3}  {:<24} {:<24} {:>6} {:>6}", "#", "original", "shown", "conf", "ppl");
    println!("{}", "-".repeat(68));

    let mut original = String::new();
    let mut shown = String::new();
    while let Ok(ev) = rx.try_recv() {
        println!(
            "{:>3}  {:<24} {:<24} {:>6} {:>6}{}",
            ev.index,
            format!("{:?}", ev.original),
            format!("{:?}", ev.text),
            ev.confidence.map(|c| format!("{c:.2}")).unwrap_or_default(),
            ev.perplexity.map(|p| format!("{p:.2}")).unwrap_or_default(),
            if ev.transformed { "  *" } else { "" },
        );
        original.push_str(&ev.original);
        shown.push_str(&ev.text);
    }

    println!("\noriginal stream:    {original}");
    println!("intercepted stream: {shown}");
    println!(
        "\n{} tokens, {} transformed (* above)",
        interceptor.token_count, interceptor.transformed_count
    );
    Ok(())
}
