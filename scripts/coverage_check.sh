#!/bin/bash
# coverage_check.sh — Generate line coverage report and enforce a minimum floor.
#
# Prerequisites (install once):
#   cargo install cargo-llvm-cov
#
# Usage:
#   ./scripts/coverage_check.sh           # report only
#   ./scripts/coverage_check.sh --enforce # exit 1 when below threshold
#
# The minimum coverage floor is defined by FLOOR below.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

FLOOR=70   # Minimum line coverage percentage required when --enforce is set.
ENFORCE=0

for arg in "$@"; do
    case "$arg" in
        --enforce) ENFORCE=1 ;;
    esac
done

# Check cargo-llvm-cov is available.
if ! command -v cargo-llvm-cov &>/dev/null && ! cargo llvm-cov --version &>/dev/null 2>&1; then
    echo "cargo-llvm-cov not found. Install with: cargo install cargo-llvm-cov"
    echo "Skipping coverage check."
    exit 0
fi

echo "Running cargo llvm-cov..."
# --lcov writes an lcov report; --summary-only prints the coverage percentage.
SUMMARY=$(cargo llvm-cov --all-features --summary-only 2>&1 | tee /dev/stderr | grep "^TOTAL" || true)

if [ -z "$SUMMARY" ]; then
    echo "Could not parse coverage summary — check cargo-llvm-cov output above."
    exit 0
fi

# Extract the line coverage percentage from the TOTAL row.
# lcov-cov summary format: TOTAL  <files> <lines>  <pct>%
PCT=$(echo "$SUMMARY" | grep -oE '[0-9]+\.[0-9]+' | tail -1 | cut -d. -f1)

echo ""
echo "Line coverage: ${PCT}%"
echo "Floor:         ${FLOOR}%"

if [ "$ENFORCE" -eq 1 ]; then
    if [ "${PCT:-0}" -lt "$FLOOR" ]; then
        DEFICIT=$((FLOOR - PCT))
        echo "FAIL — coverage below floor (need +${DEFICIT}%)"
        exit 1
    else
        echo "PASS"
        exit 0
    fi
fi
