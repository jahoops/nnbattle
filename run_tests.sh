#!/bin/bash

# Ensure the script runs from the project root directory
cd "$(dirname "$0")"

# Add the project root to PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Clear the cache
find . -type d -name "__pycache__" -exec rm -r {} +

if [ "$#" -eq 0 ]; then
    # Run all tests
    pytest nnbattle/tests
elif [ "$#" -eq 1 ]; then
    # Run specific test file or folder
    pytest "$1"
elif [ "$#" -eq 3 ]; then
    # Run specific test case
    pytest "$1::$2::$3"
else
    echo "Usage:"
    echo "  Run all tests: $0"
    echo "  Run specific test file: $0 nnbattle/tests/agents/alphazero/test_agent_code.py"
    echo "  Run specific test case: $0 nnbattle/tests/agents/alphazero/test_agent_code.py TestAgentCode test_load_agent_model_failure"
    exit 1
fi