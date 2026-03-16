#!/bin/bash

set -e

echo -e "\e[36m--- Starting Daedalus Deep Clean ---\e[0m"

# Remove Build Artifacts
echo "Removing build directories..."
rm -rf build/ dist/ .pytest_cache/ *.egg-info/

# Remove Compiled Binaries
echo "Cleaning compiled binaries and caches..."
# Specifically targets .so files (Linux) and .pyd (if run in WSL/Mingw)
find . -type d -name "__pycache__" -exec rm -rf {} +
find ./daedalus -type f -name "*.so" -delete
find ./daedalus -type f -name "*.pyd" -delete

# Reinstall in Editable Mode
echo -e "\e[36m--- Rebuilding Project ---\e[0m"
pip install -e ".[test]"

echo -e "\e[32mBuild Complete!\e[0m"