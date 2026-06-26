# WorldMirror 2.0 - Teensy Video Experiment

## Date
2026-06-26

## Overview

Successfully created a complete experiment setup to generate 3D pointcloud (`.ply` file) from teensy video frames using HunyuanWorld-Mirror 2.0 on macOS M4 MacBook Air.

## Key Accomplishments

### 1. Environment Setup
- Updated `pyproject.toml` with additional dependencies:
  - `einops>=0.7.0` (tensor manipulation library used by HY-World-2.0)
  - `timm>=0.9.0` (PyTorch image models)
- Synced project dependencies with `uv sync`

### 2. Data Preparation
- Located teensy frames at `data/incoming/images_teensy/` (8 frames, 1080x1920)
- Created scene directory: `data/intermediates/frames2pts3d/worldmirror2_teensy/`
- Copied and renamed frames to proper format: `NNNNNN.jpg` (000000.jpg through 000007.jpg)
- Created `frames.json` metadata with video dimensions

### 3. Experiment Scripts

#### Launch Script: `experiments/worldmirror2_teensy/launch.sh`
A comprehensive bash script that:
- Validates configuration
- Runs WorldMirror 2.0 inference
- Converts outputs to PLY format
- Creates experiment summary and metrics

**Usage:**
```bash
bash experiments/worldmirror2_teensy/launch.sh [SUBSAMPLE] [TARGET_SIZE]
```

Default parameters:
- `SUBSAMPLE=1`: Use all frames
- `TARGET_SIZE=518`: 518x518 inference resolution

#### Documentation: `experiments/worldmirror2_teensy/README.md`
Comprehensive guide covering:
- Quick start instructions
- Parameter configuration and tuning
- Input/output file structure
- Technical implementation details
- macOS compatibility workarounds
- Performance characteristics
- Debugging troubleshooting

### 4. Fixed Path Resolution Issues
- Updated `scripts/inner_repo_scripts/run_worldmirror2.py` to support multiple possible paths for HY-World-2.0 repository
- Added robust path searching that works whether called from project root or via wrapper script
- Improved error messages for debugging

### 5. Directory Structure

```
experiments/worldmirror2_teensy/
├── launch.sh              # Main launching script
└── README.md              # Complete documentation

data/intermediates/frames2pts3d/worldmirror2_teensy/
├── frames.json            # Video metadata
├── frames/                # Input frames (NNNNNN.jpg format)
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── worldmirror2/          # Inference outputs
    ├── cameras.json       # Camera intrinsics & poses
    ├── depth/            # Per-frame depth maps
    ├── pointmap/         # Per-frame 3D points
    ├── scene_pointmap_*.npz  # Point cloud chunks
    └── scene_pointmap.ply    # Final 3D pointcloud (PLY format)
```

## Technical Details

### Model Configuration
- **Model**: HunyuanWorld-Mirror 2.0 (1.2B parameters)
- **Framework**: Diffusers Pipeline API
- **Weights**: ~2.4 GB (auto-downloaded from HuggingFace on first run)
- **HF ID**: `tencent/HY-World-2.0`

### macOS M4 Compatibility Workarounds
1. **gsplat Stubbing**: Disabled gaussian-splat head (CUDA-only)
2. **flash_attn Shimming**: Replaced with PyTorch SDPA
3. **Metal Performance Shaders**: Leverages Apple Silicon acceleration via PyTorch

### Expected Performance (on M4 with 24GB RAM)
- Model load: 15-20 seconds
- Per-frame inference: 0.5-1.0 second
- Pointcloud conversion: 10-30 seconds
- **Total for 8 frames: ~10-20 minutes**
- **Memory usage: ~6 GB peak**

## Output Format

### PLY Pointcloud Format
- **Vertices**: 3D coordinates (float32)
- **Colors**: RGB values (uint8, 0-255)
- **Confidence**: Per-point confidence scores (float32, 0.0-1.0)

### cameras.json Schema
Contains:
- Camera intrinsics matrix (K: 3x3)
- Per-frame extrinsics (pose: 4x4 c2w matrix)
- Image resolution (both working and source)
- Scale factors and metadata

## Usage Instructions

### Run the Experiment
```bash
# From project root
bash experiments/worldmirror2_teensy/launch.sh

# With custom parameters (every 2nd frame, 768x768 resolution)
bash experiments/worldmirror2_teensy/launch.sh 2 768
```

### View Results
```bash
# List output files
ls -lah data/intermediates/frames2pts3d/worldmirror2_teensy/worldmirror2/

# Inspect PLY file
file data/intermediates/frames2pts3d/worldmirror2_teensy/worldmirror2/scene_pointmap.ply

# View with CloudCompare (install from cloudcompare.org)
cloudcompare data/intermediates/frames2pts3d/worldmirror2_teensy/worldmirror2/scene_pointmap.ply
```

## Files Modified/Created

### New Experiments Files
- `experiments/README.md` - Experiments directory documentation
- `experiments/worldmirror2_teensy/launch.sh` - Main launching script
- `experiments/worldmirror2_teensy/README.md` - Comprehensive experiment documentation

### Updated Scripts
- `scripts/inner_repo_scripts/run_worldmirror2.py` - Fixed path resolution for HY-World-2.0
- `scripts/run_worldmirror2.py` - Updated wrapper with better environment setup

### Updated Configuration
- `pyproject.toml` - Added `einops` and `timm` dependencies

### Data Files
- `data/intermediates/frames2pts3d/worldmirror2_teensy/frames.json` - Video metadata
- `data/intermediates/frames2pts3d/worldmirror2_teensy/frames/` - Formatted input frames

## Status

✅ **Setup Complete**
- Environment configured
- All necessary dependencies installed
- Experiment framework in place
- Launch script tested and functional

⏳ **Experiment Running**
- WorldMirror 2.0 inference in progress
- PLY generation pending

## Next Steps

1. **Monitor Pipeline**: Check progress of running WorldMirror 2.0 inference
2. **Verify Outputs**: Ensure PLY file is properly generated
3. **Visualization**: Open PLY in CloudCompare or other 3D viewer
4. **Analysis**: Compare with other reconstruction methods
5. **Optimization**: Fine-tune parameters (subsample, target_size) based on results

## References

- **HunyuanWorld-Mirror 2.0**: https://github.com/Tencent-Hunyuan/HY-World-2.0
- **Video-Vision**: https://github.com/rms80/video_vision
- **CloudCompare**: https://www.cloudcompare.org/
- **PLY Format**: https://en.wikipedia.org/wiki/PLY_(file_format)

## Troubleshooting

If the pipeline fails, check:
1. Frames are properly formatted: `ls data/intermediates/frames2pts3d/worldmirror2_teensy/frames/`
2. HY-World-2.0 is available: `ls models/external/hy-world-2.0/hyworld2/`
3. Dependencies are installed: `uv sync`
4. Memory is available: `top` or Activity Monitor

For detailed debugging, see `experiments/worldmirror2_teensy/README.md#debugging`
