#!/bin/bash
#
# Launch frames2pts3d workflow on teensy image dataset using VGGT-Omega encoder
#
# This script extracts dense 3D points and camera poses from the small
# test image set using VGGT-Omega features and sparse global alignment.
#
# Configuration: config/frames2pts3d/vggtomega.yaml
# Script: scripts/frames2pts3d_vggtomega.py
# Input: data/incoming/images_teensy/
# Output: data/intermediates/frames2pts3d/
#
# Usage:
#   ./experiments/frames2pts3d/vggtomega/launch_teensy.sh
#   ./experiments/frames2pts3d/vggtomega/launch_teensy.sh experiment_name=my_experiment
#

set -e  # Exit on error

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "frames2pts3d VGGT-Omega Launcher"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Input images: data/incoming/images_teensy/"
echo "Config: config/frames2pts3d/vggtomega.yaml"
echo "Model: VGGT-Omega (1B, 512-dim)"
echo "=========================================="
echo ""

# Change to project root
cd "${PROJECT_ROOT}"

# Run the workflow
echo "Starting frames2pts3d workflow with VGGT-Omega..."
echo ""

uv run python scripts/frames2pts3d_vggtomega.py \
  --config-path="../config/frames2pts3d" \
  --config-name="vggtomega_teensy" \
  "$@"

echo ""
echo "=========================================="
echo "✅ Workflow complete!"
echo "=========================================="
