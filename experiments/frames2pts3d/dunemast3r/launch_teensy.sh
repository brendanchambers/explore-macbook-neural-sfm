#!/bin/bash
#
# Launch frames2pts3d workflow on teensy image dataset
#
# This script extracts dense 3D points and camera poses from the small
# test image set using DuneMast3r and sparse global alignment.
#
# Configuration: config/frames2pts3d/config.yaml
# Script: scripts/frames2pts3d.py
# Input: data/incoming/images_teensy/
# Output: data/intermediates/frames2pts3d/
#
# Usage:
#   ./experiments/frames2pts3d/launch_teensy.sh
#   ./experiments/frames2pts3d/launch_teensy.sh experiment_name=my_experiment
#

set -e  # Exit on error

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================="
echo "frames2pts3d Workflow Launcher"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Input images: data/incoming/images_teensy/"
echo "Config: config/frames2pts3d/config.yaml"
echo "=========================================="
echo ""

# Change to project root
cd "${PROJECT_ROOT}"

# Run the workflow
echo "Starting frames2pts3d workflow..."
echo ""

uv run python scripts/frames2pts3d.py \
  --config-path="../config/frames2pts3d" \
  --config-name="teensy" \
  "$@"

echo ""
echo "=========================================="
echo "✅ Workflow complete!"
echo "=========================================="
