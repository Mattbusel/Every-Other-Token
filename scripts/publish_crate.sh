#!/usr/bin/env bash
set -e

echo "[publish] Running cargo test..."
cargo test

echo "[publish] Running cargo publish --dry-run..."
cargo publish --dry-run

echo ""
echo "Dry run succeeded. To publish for real, set CARGO_REGISTRY_TOKEN:"
echo "  export CARGO_REGISTRY_TOKEN=<your-token-from-crates.io>"
echo ""
read -p "Publish to crates.io? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    cargo publish
    echo "[publish] Published successfully!"
else
    echo "[publish] Aborted."
fi
