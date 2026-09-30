#!/bin/bash
# Auto-format the codebase: sort imports, then apply black.
set -e
cd "$(dirname "$0")/.."

echo "==> isort"
uv run isort backend main.py
echo "==> black"
uv run black backend main.py
echo "Formatting complete."
