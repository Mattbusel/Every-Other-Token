#!/bin/bash
# ratio_check.sh — Verify test-to-production line ratio meets 1:1 minimum
# Supports multi-file src/ layout and external tests/ directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Count production lines across all src/ files (excluding #[cfg(test)] blocks)
PROD=0
TEST_INLINE=0
for f in $(find src -name '*.rs' -type f); do
    # Lines before #[cfg(test)] = production, lines after = test
    p=$(sed -n '/#\[cfg(test)\]/q;p' "$f" | grep -cv '^\s*$\|^\s*//' || true)
    t=$(sed -n '/#\[cfg(test)\]/,$ p' "$f" | grep -cv '^\s*$\|^\s*//' || true)
    PROD=$((PROD + p))
    TEST_INLINE=$((TEST_INLINE + t))
done

# Count external test files
TEST_EXT=0
if [ -d tests ]; then
    for f in $(find tests -name '*.rs' -type f); do
        t=$(grep -cv '^\s*$\|^\s*//' "$f" || true)
        TEST_EXT=$((TEST_EXT + t))
    done
fi

# Count bench files
BENCH=0
if [ -d benches ]; then
    for f in $(find benches -name '*.rs' -type f); do
        b=$(grep -cv '^\s*$\|^\s*//' "$f" || true)
        BENCH=$((BENCH + b))
    done
fi

TEST=$((TEST_INLINE + TEST_EXT + BENCH))

# Calculate ratio
RATIO=$(awk "BEGIN {printf \"%.2f\", $TEST / $PROD}")
PASS=$(awk "BEGIN {print ($TEST >= $PROD) ? 1 : 0}")

echo "Production LOC:  $PROD"
echo "Test LOC:        $TEST (inline: $TEST_INLINE, external: $TEST_EXT, bench: $BENCH)"
echo "Ratio:           ${RATIO}:1"

if [ "$PASS" -eq 1 ]; then
    echo "PASS (minimum 1:1)"
    exit 0
else
    DEFICIT=$((PROD - TEST))
    echo "FAIL — test ratio below 1:1 (need $DEFICIT more test lines)"
    exit 1
fi
