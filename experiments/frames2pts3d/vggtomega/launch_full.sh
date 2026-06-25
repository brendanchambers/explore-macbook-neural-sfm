#!/bin/bash
#
# Launch frames2pts3d workflow on full image dataset using VGGT-Omega encoder
#
# This script extracts dense 3D points and camera poses from the full
# image set using VGGT-Omega features and sparse global alignment.
#
# Configuration: config/frames2pts3d/vggtomega.yaml (modified for full dataset)
# Script: scripts/frames2pts3d_vggtomega.py
# Input: data/incoming/images/
# Output: data/intermediates/frames2pts3d/
#
# Usage:
#   ./experiments/frames2pts3d/vggtomega/launch_full.sh
#   ./experiments/frames2pts3d/vggtomega/launch_full.sh experiment_name=my_experiment
#   ./experiments/frames2pts3d/vggtomega/launch_full.sh paths.image_dir=data/incoming/images
#

set -e  # Exit on error

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "frames2pts3d VGGT-Omega Launcher (Full Dataset)"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Input images: data/incoming/images/"
echo "Config: config/frames2pts3d/vggtomega.yaml"
echo "Model: VGGT-Omega (1B, 512-dim)"
echo "=========================================="
echo ""

# Change to project root
cd "${PROJECT_ROOT}"

# Run the workflow with full dataset
echo "Starting frames2pts3d workflow with VGGT-Omega (full dataset)..."
echo ""

uv run python scripts/frames2pts3d_vggtomega.py \
  --config-path="../config/frames2pts3d" \
  --config-name="vggtomega_full" \
  paths.image_dir=data/incoming/images \
  experiment.name=vggtomega_full \
  "$@"

echo ""
echo "=========================================="
echo "✅ Workflow complete!"
echo "=========================================="
