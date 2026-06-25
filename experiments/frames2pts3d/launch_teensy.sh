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
# Output: data/intermediates/frames2pts3d/{experiment_name}/
#
# Usage:
#   ./experiments/frames2pts3d/launch_teensy.sh
#   ./experiments/frames2pts3d/launch_teensy.sh my_experiment_name
#

set -e  # Exit on error

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Experiment name (default to current_experiment, or use first argument)
EXPERIMENT_NAME="${1:-current_experiment}"

echo "=========================================="
echo "frames2pts3d Workflow Launcher"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Input images: data/incoming/images_teensy/"
echo "Output directory: data/intermediates/frames2pts3d/${EXPERIMENT_NAME}/"
echo "Config: config/frames2pts3d/config.yaml"
echo "=========================================="
echo ""

# Change to project root
cd "${PROJECT_ROOT}"

# Run the workflow
echo "Starting frames2pts3d workflow..."
echo ""

uv run python scripts/frames2pts3d.py \
  --config-path config/frames2pts3d \
  --config-name config \
  hydra.run.dir="logs/hydra/frames2pts3d/${EXPERIMENT_NAME}"

echo ""
echo "=========================================="
echo "✅ Workflow complete!"
echo "Results saved to: data/intermediates/frames2pts3d/${EXPERIMENT_NAME}/"
echo "=========================================="
