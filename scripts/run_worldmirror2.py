#!/usr/bin/env python
"""Run HunyuanWorld-Mirror 2.0 for camera poses + depth + global pointmap.

This wrapper script sets up the environment and calls the inner video-vision
WorldMirror 2.0 runner with workarounds for macOS compatibility.

Usage:
    uv run python scripts/run_worldmirror2.py <scene_dir> [--subsample N] [--target-size 518]

Reads <scene_dir>/frames/NNNNNN.jpg (produced by extract_frames.py).
Runs WorldMirror 2.0 feed-forward inference in image-only mode (no conditioning)
to produce camera poses, depth, per-frame pointmaps, and a global scene-level
point cloud.

HunyuanWorld-Mirror 2.0 is the 1.2B-parameter successor to WorldMirror 1.0,
released inside the HY-World-2.0 umbrella repo (HF subfolder
`HY-WorldMirror-2.0`). It exposes a diffusers-style Pipeline API; we use its
`_run_inference` directly to get raw predictions without their file-saving
layer. The gaussian-splat head is disabled via `disable_heads=['gs']` to save
~33 M params and avoid needing a working `gsplat` build.

Outputs:
    <scene_dir>/worldmirror2/cameras.json         — same schema as other plugins
    <scene_dir>/worldmirror2/depth/NNNNNN.npz     — per-frame depth
    <scene_dir>/worldmirror2/pointmap/NNNNNN.npz  — per-frame camera-space pointmap + conf
    <scene_dir>/worldmirror2/scene_pointmap_chunks.json — global cloud manifest
    <scene_dir>/worldmirror2/scene_pointmap_NNN.npz     — chunked global cloud
"""

import sys
import os
from pathlib import Path

# Add the inner repo scripts directory to path
project_root = Path(__file__).resolve().parent.parent
inner_scripts_dir = project_root / "scripts" / "inner_repo_scripts"
sys.path.insert(0, str(inner_scripts_dir))

# Fix the path in the inner script: make HYWORLD2_ROOT absolute
# by patching sys.argv before importing
os.chdir(str(project_root))

# Import and run the WorldMirror 2.0 runner
from run_worldmirror2 import main

if __name__ == "__main__":
    main()
