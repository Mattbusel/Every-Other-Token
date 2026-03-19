#!/usr/bin/env bash
# pre-commit.sh — run pre-commit checks for Every-Other-Token
# To install as a git hook: cp scripts/pre-commit.sh .git/hooks/pre-commit
# Make executable first: chmod +x scripts/pre-commit.sh
set -e

echo "[eot] Running pre-commit checks..."
cargo test --quiet
if [ -f scripts/ratio_check.sh ]; then
  bash scripts/ratio_check.sh
fi
echo "[eot] Pre-commit checks passed."
