/// Integration test: spin up the web server and verify it serves the HTML UI.
///
/// This test binds to a random OS-assigned port, spawns `serve()` in a background
/// task, then makes a real TCP-level HTTP GET / request and asserts that the
/// response is 200 OK with a text/html Content-Type.
use every_other_token::cli::Args;
use every_other_token::providers::Provider;
use every_other_token::web::serve;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Build a minimal [`Args`] value that is safe to pass to `serve()` in tests.
fn test_args(port: u16) -> Args {
    Args {
        prompt: "test".to_string(),
        transform: "reverse".to_string(),
        model: "mock-fixture-v1".to_string(),
        provider: Provider::Mock,
        visual: false,
        heatmap: false,
        orchestrator: false,
        web: true,
        port,
        research: false,
        runs: 1,
        output: "research_output.json".to_string(),
        system_a: None,
        top_logprobs: 0,
        system_b: None,
        db: None,
        significance: false,
        heatmap_export: None,
        heatmap_min_confidence: 0.0,
        heatmap_sort_by: "position".to_string(),
        record: None,
        replay: None,
        rate: None,
        seed: None,
        log_db: None,
        baseline: false,
        prompt_file: None,
        diff_terminal: false,
        json_stream: false,
        completions: None,
        rate_range: None,
        dry_run: false,
        template: None,
        min_confidence: None,
        format: "json".to_string(),
        collapse_window: 5,
        orchestrator_url: "http://localhost:3000".to_string(),
        max_retries: 3,
        anthropic_max_tokens: 4096,
        synonym_file: None,
        api_key: None,
        replay_speed: 1.0,
        timeout: 120,
    }
}

#[tokio::test]
async fn test_web_server_returns_html() {
    // Bind port 0 to get a free ephemeral port from the OS, then drop the
    // listener so `serve()` can re-bind to the same port.
    let tmp = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let port = tmp.local_addr().unwrap().port();
    drop(tmp);

    let args = test_args(port);

    // Spawn the server in the background; it runs until the task is dropped.
    tokio::spawn(async move {
        let _ = serve(port, &args).await;
    });

    // Give the server a moment to bind and start accepting connections.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Open a raw TCP connection and send a minimal HTTP/1.0 GET request so we
    // don't need the `reqwest` crate in dev-dependencies.
    let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{}", port))
        .await
        .expect("connect to test server");

    let request = format!(
        "GET / HTTP/1.0\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\r\n",
        port
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .await
        .expect("read response");

    // Assert HTTP 200 OK
    assert!(
        response.starts_with("HTTP/1.0 200") || response.starts_with("HTTP/1.1 200"),
        "expected 200 OK, got: {}",
        &response[..response.find('\n').unwrap_or(response.len()).min(80)]
    );

    // Assert Content-Type contains text/html
    let lower = response.to_lowercase();
    assert!(
        lower.contains("content-type: text/html"),
        "expected text/html Content-Type in response headers"
    );
}
