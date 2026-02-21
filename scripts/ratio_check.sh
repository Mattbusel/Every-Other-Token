#!/bin/bash
# ratio_check.sh — Verify test-to-production line ratio (minimum 1:1)

set -euo pipefail

PROD=$(find src -name "*.rs" -exec grep -v '^\s*$' {} + | grep -v '^\s*//' | grep -v '#\[cfg(test)\]' | wc -l)
TEST_INLINE=$(find src -name "*.rs" -exec sed -n '/#\[cfg(test)\]/,/^}/p' {} + | grep -v '^\s*$' | grep -v '^\s*//' | wc -l)
TEST_DIR=$(find tests -name "*.rs" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
TEST_DIR=${TEST_DIR:-0}
TEST=$((TEST_INLINE + TEST_DIR))

if [ "$PROD" -eq 0 ]; then
    echo "No production lines found"
    exit 1
fi

RATIO=$(echo "scale=2; $TEST / $PROD" | bc)

echo "Production LOC: $PROD"
echo "Test LOC:       $TEST"
echo "Ratio:          ${RATIO}:1"

if (( $(echo "$RATIO >= 1.0" | bc -l) )); then
    echo "Status: PASS (>= 1:1)"
    exit 0
else
    echo "Status: FAIL (< 1:1)"
    exit 1
fi
