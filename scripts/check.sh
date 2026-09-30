#!/bin/bash
# Full quality gate: formatting/lint checks followed by the test suite.
set -e
cd "$(dirname "$0")/.."

./scripts/lint.sh
echo "==> pytest"
uv run pytest
echo "All checks passed."
