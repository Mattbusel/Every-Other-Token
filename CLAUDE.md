# Every-Other-Token

## Project
Rust CLI + web UI for real-time LLM token stream mutation and
LLM interpretability research.

## Architecture
- src/main.rs — CLI entry point, token stream logic, transformations
- src/web/ — raw Tokio HTTP server, SSE streaming
- Web UI — single embedded HTML file, no build step, vanilla JS only

## Rules
- Keep web UI as single embedded HTML (no separate CSS/JS files)
- No npm, no webpack, no external JS dependencies
- cargo test must pass before every commit
- No panics in production paths
- git push to both main and master after every commit

## Workers
- OpenAI: streaming via openai crate
- Anthropic: direct HTTP to api.anthropic.com/v1/messages

## Testing
- Maintain 1:1 test-to-production line ratio minimum
- Every new module needs a corresponding test file
- Run ./scripts/ratio_check.sh before committing
- cargo test must show 0 failures
