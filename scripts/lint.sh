#!/bin/bash
# Check formatting and lint without modifying files (suitable for CI).
set -e
cd "$(dirname "$0")/.."

echo "==> isort (check)"
uv run isort --check-only --diff backend main.py
echo "==> black (check)"
uv run black --check --diff backend main.py
echo "==> flake8"
uv run flake8 backend main.py
echo "All quality checks passed."
