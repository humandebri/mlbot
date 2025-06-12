#!/bin/bash
# Helper script to run commands in venv with proper environment

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate venv
source "$SCRIPT_DIR/.venv/bin/activate"

# Set Python path
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run command
"$@"