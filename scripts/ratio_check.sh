#!/bin/bash
# ratio_check.sh — Verify test-to-production line ratio meets 1:1 minimum

set -e

# Count production lines (src/, excluding test blocks and blank/comment lines)
PROD=$(sed -n '/#\[cfg(test)\]/q;p' src/main.rs | grep -v '^\s*$' | grep -v '^\s*//' | wc -l | tr -d ' ')

# Count test lines (everything inside #[cfg(test)] to end of file)
TEST=$(sed -n '/#\[cfg(test)\]/,$ p' src/main.rs | grep -v '^\s*$' | grep -v '^\s*//' | wc -l | tr -d ' ')

# Calculate ratio (using awk for float division)
RATIO=$(awk "BEGIN {printf \"%.2f\", $TEST / $PROD}")
PASS=$(awk "BEGIN {print ($TEST >= $PROD) ? 1 : 0}")

echo "Production LOC:  $PROD"
echo "Test LOC:        $TEST"
echo "Ratio:           ${RATIO}:1"

if [ "$PASS" -eq 1 ]; then
    echo "PASS (minimum 1:1)"
    exit 0
else
    echo "FAIL — test ratio below 1:1, write more tests"
    exit 1
fi
