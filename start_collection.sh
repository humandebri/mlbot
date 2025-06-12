#!/bin/bash
# Script to start data collection with proper environment

# Activate virtual environment
source .venv/bin/activate

# Check Python version
echo "Python version:"
python --version

# Start data collection
echo "Starting data collection..."
python scripts/collect_data.py "$@"