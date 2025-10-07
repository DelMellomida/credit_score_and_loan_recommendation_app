#!/usr/bin/env bash
# set -euo pipefail

# Ensure logs directory exists
mkdir -p logs

# Run the main Python script, logging to both console and file
uvicorn main:app --host 0.0.0.0 --port 8000 >> logs/app.log 2>&1