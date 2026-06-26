# WorldMirror 2.0 Setup and Usage

## Summary
Integrated HunyuanWorld-Mirror 2.0 from the video-vision repository to generate 3D pointclouds from video frames on macOS M4 MacBook Air.

## Changes Made

### 1. Environment Updates
- **Deleted**: `mlx-mast3r` inner repository (no longer needed)
- **Updated** `pyproject.toml`: Removed mlx-mast3r dependency, added:
  - `torch>=2.1.0`
  - `torchvision>=0.16.0`
  - `opencv-python>=4.8.0`
  - `diffusers>=0.24.0`
  - `transformers>=4.36.0`

### 2. Inner Repository Setup
- Fetched video-vision repository from https://github.com/rms80/video_vision/
- Removed .git directory from inner repo
- Copied WorldMirror 2.0 runner scripts to `scripts/inner_repo_scripts/`:
  - `run_worldmirror2.py` - Main inference runner
  - `_pointcloud_io.py` - Pointcloud I/O utilities
  - `_progress.py` - Progress reporting utilities

### 3. Model Repository
- Cloned HY-World-2.0 to `models/external/hy-world-2.0/`
- Checked out specific pinned commit: `ee5d5bc02c92671486ab6dd81f0cba577d0478c8`

### 4. Project Scripts
- **`scripts/run_worldmirror2.py`**: Wrapper script that invokes the WorldMirror 2.0 runner
- **`scripts/worldmirror2_to_ply.py`**: Converts WorldMirror 2.0 pointcloud outputs to PLY format

## Workflow

### Step 1: Extract Video Frames
```bash
# From video_vision repository docs - extract frames from video
# Frame directory: data/intermediates/<experiment_name>/frames/
# Frame naming: NNNNNN.jpg (zero-padded 6-digit indices)
```

### Step 2: Run WorldMirror 2.0 Inference
```bash
uv run python scripts/run_worldmirror2.py <scene_dir> [--subsample N] [--target-size 518]
```

**Arguments:**
- `<scene_dir>`: Directory containing frames at `<scene_dir>/frames/NNNNNN.jpg`
- `--subsample N`: Use every Nth frame (default: 3)
- `--target-size`: Target image size for inference (default: 518, lower for less VRAM)

**Outputs** (written to `<scene_dir>/worldmirror2/`):
- `cameras.json` - Camera intrinsics and extrinsics (pose estimates)
- `depth/NNNNNN.npz` - Per-frame depth maps
- `pointmap/NNNNNN.npz` - Per-frame camera-space pointmaps with confidence
- `scene_pointmap_chunks.json` - Manifest of point cloud chunks
- `scene_pointmap_NNN.npz` - Chunked scene-level pointcloud data

### Step 3: Convert to PLY
```bash
uv run python scripts/worldmirror2_to_ply.py <scene_dir> [--output <ply_file>]
```

**Outputs:**
- `<scene_dir>/worldmirror2/scene_pointmap.ply` (or custom output path)

## Technical Details

### macOS Compatibility Workarounds
The WorldMirror 2.0 script includes workarounds for macOS compatibility:
1. **gsplat stubbing**: The gaussian-splat head is disabled because gsplat is CUDA-only. Disabled via `disable_heads=['gs']` to save ~33M params
2. **flash_attn replacement**: Replaced with PyTorch's native scaled_dot_product_attention (SDPA) shim

### Model Info
- **Model**: HunyuanWorld-Mirror 2.0 (1.2B parameters)
- **HuggingFace ID**: `tencent/HY-World-2.0`
- **Subfolder**: `HY-WorldMirror-2.0/`
- **Weights Size**: ~2.4 GB (downloaded on first run)
- **Framework**: Diffusers pipeline API

### Output Schema
**cameras.json:**
```json
{
  "model": "HUNYUAN_WORLD_MIRROR_2",
  "checkpoint": "tencent/HY-World-2.0",
  "width": 518,    // Working resolution
  "height": 518,
  "source_width": 1920,   // Original video resolution
  "source_height": 1080,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],  // Camera intrinsics
  "frames": [
    {"idx": 0, "name": "000000.jpg", "R": [...], "t": [...], ...}
  ]
}
```

## HuggingFace Weight Caching
The first run will download weights (~2.4 GB) from HuggingFace Hub. These are cached at:
```
~/.cache/huggingface/hub/
```

No manual pre-caching is required, but ensure you have internet access on first run.

## Performance Notes
- **Target Size**: Lower target-size (default 518) reduces VRAM usage
- **Subsample**: Higher subsample values (e.g., --subsample 5) use fewer frames
- **M4 MacBook Air (24GB)**: Should handle default settings (target_size=518, subsample=3)
- **Inference Speed**: ~0.5-1.0s per frame depending on settings

## Next Steps
Once PLY is generated, the pointcloud can be:
1. Used with 3DGS training pipelines (e.g., OpenSplat)
2. Visualized in 3D viewers (CloudCompare, MeshLab, etc.)
3. Further refined with depth refinement models (e.g., InfiniDepth from video-vision)

## References
- HunyuanWorld-Mirror 2.0: https://github.com/Tencent-Hunyuan/HY-World-2.0
- Video-Vision Repository: https://github.com/rms80/video_vision
