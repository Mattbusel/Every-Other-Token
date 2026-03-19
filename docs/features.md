# Feature Flags

Enable optional features with `cargo build --features <flag>` or in `Cargo.toml`.

| Feature | Enables | Dependencies added |
|---------|---------|-------------------|
| `sqlite-log` | Persist research runs to SQLite; `/api/experiments` endpoint | `rusqlite` |
| `self-tune` | Telemetry bus + self-improvement orchestrator | `tokio`, `serde` |
| `self-modify` | Snapshot-based parameter mutation | requires `self-tune` |
| `intelligence` | Reserved namespace for interpretability features | -- |
| `evolution` | Reserved namespace for evolutionary optimisation | -- |
| `self-improving` | All of the above combined | all above |
| `helix-bridge` | HTTP bridge polling a HelixRouter `/api/stats` endpoint | `reqwest` |
| `redis-backing` | Write-through Redis persistence for snapshots | `redis` |
| `wasm` | WASM target bindings | `wasm-bindgen` |

## Compatibility matrix

| Feature combo | Tested | Notes |
|---------------|--------|-------|
| (none) | Yes | Default build |
| `sqlite-log` | Yes | |
| `self-improving` | Yes | |
| `helix-bridge` | Yes | Requires running HelixRouter |
| `wasm` | Yes | Build with `wasm-pack` |
| `self-improving` + `helix-bridge` | Yes | |
| `redis-backing` | Partial | Requires running Redis |
