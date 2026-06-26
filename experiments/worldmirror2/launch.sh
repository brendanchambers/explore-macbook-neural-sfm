#!/bin/bash

# Launch script for WorldMirror 2.0 inference on full movie frames
# This script generates a 3D pointcloud (.ply) from all video frames
# using HunyuanWorld-Mirror 2.0 with macOS-compatible workarounds

set -e

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
SCENE_DIR="data/intermediates/frames2pts3d/worldmirror2"
SUBSAMPLE=${1:-3}          # Default: use every 3rd frame (subsample=3, ~42 frames from 126)
TARGET_SIZE=${2:-518}      # Default: 518px (lower memory usage on M4)
EXPERIMENT_NAME="worldmirror2"
EXPERIMENT_GROUP="current_experiment"

# Source frames
INPUT_FRAMES_DIR="data/incoming/images"

echo "=========================================="
echo "WorldMirror 2.0 - Full Movie Pipeline"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Scene Directory: $SCENE_DIR"
echo "  Input Frames: $INPUT_FRAMES_DIR"
echo "  Subsample: $SUBSAMPLE"
echo "  Target Size: ${TARGET_SIZE}x${TARGET_SIZE}"
echo "  Experiment: $EXPERIMENT_NAME"
echo ""

# Step 0: Prepare frames
echo "Step 0: Preparing frames..."
mkdir -p "$SCENE_DIR/frames"

# Count total frames
TOTAL_FRAMES=$(ls -1 "$INPUT_FRAMES_DIR"/frame_*.jpg 2>/dev/null | wc -l)
echo "  Found $TOTAL_FRAMES frames in $INPUT_FRAMES_DIR"

# Copy and rename frames
frame_count=0
for f in $(ls -1 "$INPUT_FRAMES_DIR"/frame_*.jpg | sort -V); do
  # Extract just the number from the filename (e.g., "frame_0001.jpg" -> "0001")
  base_name=$(basename "$f")
  frame_num_str="${base_name#frame_}"  # Remove "frame_" prefix
  frame_num_str="${frame_num_str%.jpg}"  # Remove ".jpg" suffix
  frame_num=$((10#$frame_num_str - 1))  # Convert to 0-indexed (0001 -> 0)
  new_name=$(printf "%06d.jpg" $frame_num)
  cp "$f" "$SCENE_DIR/frames/$new_name"
  ((frame_count++))
  if [ $((frame_count % 20)) -eq 0 ]; then
    echo "  Copied $frame_count/$TOTAL_FRAMES frames..."
  fi
done

echo "  Copied all $TOTAL_FRAMES frames to $SCENE_DIR/frames"

# Create frames.json metadata
export SCENE_DIR="$SCENE_DIR"
python3 << 'PYMETA'
import json
from pathlib import Path
from PIL import Image
import os

scene_dir = os.environ.get("SCENE_DIR")
frames_dir = Path(scene_dir) / "frames"
frames = sorted([f for f in frames_dir.glob("*.jpg")])

if frames:
    img = Image.open(frames[0])
    width, height = img.size

    metadata = {
        "width": width,
        "height": height,
        "fps": 30,
        "num_frames": len(frames),
        "source": "full_movie_frames"
    }

    with open(Path(scene_dir) / "frames.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created frames.json: {len(frames)} frames, {width}x{height}")
PYMETA

echo ""

# Step 1: Run WorldMirror 2.0 inference
echo "Step 1: Running WorldMirror 2.0 inference..."
echo "  Command: uv run python scripts/run_worldmirror2.py $SCENE_DIR --subsample $SUBSAMPLE --target-size $TARGET_SIZE"
echo ""

uv run python scripts/run_worldmirror2.py "$SCENE_DIR" \
  --subsample "$SUBSAMPLE" \
  --target-size "$TARGET_SIZE"

echo ""
echo "Step 1 Complete: WorldMirror 2.0 inference finished"
echo ""

# Step 2: Convert pointcloud to PLY
echo "Step 2: Converting pointcloud to PLY..."
PLY_FILE="$SCENE_DIR/worldmirror2/scene_pointmap.ply"
echo "  Output: $PLY_FILE"
echo "  Command: uv run python scripts/worldmirror2_to_ply.py $SCENE_DIR"
echo ""

uv run python scripts/worldmirror2_to_ply.py "$SCENE_DIR"

echo ""
echo "Step 2 Complete: PLY conversion finished"
echo ""

# Step 3: Log results
RESULTS_DIR="reports/experiments/$EXPERIMENT_GROUP"
mkdir -p "$RESULTS_DIR"

LOG_FILE="$RESULTS_DIR/worldmirror2.jsonl"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Create summary
cat > "$RESULTS_DIR/worldmirror2_summary.txt" << EOF
WorldMirror 2.0 - Full Movie Generation
========================================
Timestamp: $TIMESTAMP
Experiment: $EXPERIMENT_NAME
Group: $EXPERIMENT_GROUP

Configuration:
  Scene Dir: $SCENE_DIR
  Subsample: $SUBSAMPLE
  Target Size: $TARGET_SIZE

Outputs:
  Cameras: $SCENE_DIR/worldmirror2/cameras.json
  Depth Maps: $SCENE_DIR/worldmirror2/depth/
  Pointmaps: $SCENE_DIR/worldmirror2/pointmap/
  PLY File: $PLY_FILE

Pointcloud Statistics:
$(if [ -f "$PLY_FILE" ]; then
  BYTE_SIZE=$(stat -f%z "$PLY_FILE" 2>/dev/null || stat -c%s "$PLY_FILE" 2>/dev/null)
  echo "  PLY File Size: $(numfmt --to=iec-i --suffix=B $BYTE_SIZE 2>/dev/null || echo "$BYTE_SIZE bytes")"
  echo "  Status: ✓ Generated successfully"
else
  echo "  Status: ✗ PLY file not found"
fi)

End Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Generated Files:"
echo "  PLY Pointcloud: $PLY_FILE"
echo "  Camera Poses:   $SCENE_DIR/worldmirror2/cameras.json"
echo "  Depth Maps:     $SCENE_DIR/worldmirror2/depth/"
echo "  Per-Frame 3D:   $SCENE_DIR/worldmirror2/pointmap/"
echo ""
echo "Summary saved to: $RESULTS_DIR/worldmirror2_summary.txt"
echo ""
