#!/bin/bash

EXCLUDE_LIST="venv/|__pycache__|.pytest_cache|dist|build|html|.benchmarks|.vscode"
DEPTH_VALUE=5

echo "Running Command: tree -I '$EXCLUDE_LIST' -L $DEPTH_VALUE"

tree -I "$EXCLUDE_LIST" -L "$DEPTH_VALUE"