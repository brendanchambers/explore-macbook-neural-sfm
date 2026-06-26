# WorldMirror 2.0 - Full Movie Experiment

## Overview

This experiment generates a 3D pointcloud (`.ply` file) from the full movie frames using **HunyuanWorld-Mirror 2.0** on macOS M4 MacBook Air.

- **Model**: HunyuanWorld-Mirror 2.0 (1.2B parameters)
- **Input**: 126 frames from full movie (1080x1920 resolution)
- **Outputs**: Camera poses, depth maps, 3D pointcloud (PLY format)
- **Target Hardware**: macOS M4 24GB RAM

## Quick Start

### Run the Full Pipeline

```bash
# From project root
bash experiments/worldmirror2/launch.sh
```

### With Custom Parameters

```bash
# Subsample=3: use every 3rd frame (~42 frames total, faster)
# Target size=768: higher resolution inference
bash experiments/worldmirror2/launch.sh 3 768
```

## Launching Script

The `launch.sh` script automates the complete pipeline:

### What It Does

1. **Prepares Frames**: Copies and reformats frames from `data/incoming/images/`
2. **Validates Environment**: Ensures project structure and dependencies are in place
3. **Runs WorldMirror 2.0 Inference**:
   - Loads the 1.2B parameter model
   - Processes selected frames
   - Outputs camera poses, depth, and pointmaps
4. **Converts to PLY Format**: Combines all outputs into a single `.ply` pointcloud file
5. **Logs Results**: Creates experiment summary and metrics

### Script Parameters

```bash
bash experiments/worldmirror2/launch.sh [SUBSAMPLE] [TARGET_SIZE]
```

**Arguments:**
- `SUBSAMPLE` (default: 3): Use every Nth frame (1=all 126 frames, 3=every 3rd frame ~42 frames, etc.)
- `TARGET_SIZE` (default: 518): Inference resolution in pixels (518px = ~6GB VRAM, 768px = ~10GB VRAM)

### Example Configurations

| Config | Subsample | Target Size | Frames | Approx VRAM | Speed |
|--------|-----------|-------------|--------|------------|-------|
| Fast | 5 | 518 | ~25 | 6GB | ~2s/frame |
| Balanced | 3 | 518 | ~42 | 6GB | ~4s/frame |
| Thorough | 2 | 518 | ~63 | 6GB | ~4s/frame |
| High Quality | 3 | 768 | ~42 | 10GB | ~8s/frame |

**Note**: M4 MacBook Air has 24GB unified memory, so can handle full movie with subsample=2.

## Input Data Structure

```
data/incoming/images/
├── frame_0001.jpg
├── frame_0002.jpg
├── ...
└── frame_0126.jpg          (126 total frames)

data/intermediates/frames2pts3d/worldmirror2/
├── frames/                 (Auto-created and populated by launch.sh)
│   ├── 000000.jpg
│   ├── 000001.jpg
│   ├── ...
│   └── 000125.jpg
└── frames.json             (Auto-generated metadata)
```

**frames.json**: Metadata about the video (resolution, fps, frame count)

## Output Structure

After running the pipeline:

```
data/intermediates/frames2pts3d/worldmirror2/
├── frames.json
├── frames/
│   ├── 000000.jpg
│   └── ...
│   └── 000125.jpg
└── worldmirror2/
    ├── cameras.json                 # Camera intrinsics & poses
    ├── depth/
    │   ├── 000000.npz
    │   └── ...
    ├── pointmap/
    │   ├── 000000.npz
    │   └── ...
    ├── scene_pointmap_chunks.json   # Manifest of point cloud chunks
    ├── scene_pointmap_000.npz       # Chunk data
    ├── scene_pointmap_001.npz
    ├── scene_pointmap_002.npz
    └── scene_pointmap.ply            # ← Final 3D pointcloud (PLY format)
```

### Output Files Explained

| File | Format | Description |
|------|--------|-------------|
| `cameras.json` | JSON | Camera intrinsics (K matrix) and extrinsics (poses) for all frames |
| `depth/*.npz` | NumPy | Per-frame depth maps (float16) |
| `pointmap/*.npz` | NumPy | Per-frame 3D points + confidence scores in camera space |
| `scene_pointmap_chunks.json` | JSON | Manifest listing all point cloud chunks |
| `scene_pointmap_*.npz` | NumPy | Point cloud data split into chunks (for memory efficiency) |
| `scene_pointmap.ply` | PLY (binary) | **Complete 3D pointcloud** with RGB colors and confidence |

### PLY File Format

The generated `.ply` file contains:
- **Vertices**: 3D point coordinates (X, Y, Z as float32)
- **Colors**: RGB values (0-255 as uint8)
- **Confidence**: Per-point confidence scores (0.0-1.0 as float32)

**Viewable in:**
- CloudCompare (free, full-featured)
- MeshLab (free)
- Blender (free)
- Point Cloud Library viewers
- Custom Python scripts with libraries like `trimesh`, `plyfile`, `pyntcloud`

## Camera Model Details

### Intrinsic Matrix (K)

WorldMirror 2.0 outputs a median intrinsic matrix across all views:

```
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]

where:
  fx, fy = focal length (pixels)
  cx, cy = principal point offset (pixels)
```

The actual resolution used during inference is included in `cameras.json`:
- `width`, `height`: Working resolution during inference
- `source_width`, `source_height`: Original video resolution
- `scale_factor`: Scaling between source and working resolution

### Camera Poses

Each frame has:
- **R**: 3x3 rotation matrix (world-to-camera)
- **t**: 3x1 translation vector (world-to-camera)
- `idx`: Original frame index
- `name`: Frame filename
- `registered`: Whether camera pose is valid

## Technical Notes

### macOS Compatibility Workarounds

The WorldMirror 2.0 runner includes workarounds for macOS M4:

1. **gsplat Stubbing**: The gaussian-splat head is disabled since gsplat requires CUDA
   - Saves ~33M parameters
   - Avoids need for working gsplat build
   - Controlled via `disable_heads=['gs']`

2. **flash_attn Shim**: Replaced NVIDIA flash-attention with PyTorch's native SDPA
   - flash_attn is CUDA-only
   - Uses `torch.nn.functional.scaled_dot_product_attention` instead
   - Identical semantics, Apple Silicon compatible

### Performance Characteristics

On M4 MacBook Air (24GB unified memory) with 126 total frames:

```
Fast config (subsample=5, target_size=518, ~25 frames):
  Model load time: ~15-20s
  Per-frame inference: ~0.5-1.0s per frame × 25 = ~20-25 minutes
  Pointcloud conversion: ~30-60s
  Total: ~25-40 minutes

Balanced config (subsample=3, target_size=518, ~42 frames):
  Model load time: ~15-20s
  Per-frame inference: ~0.5-1.0s per frame × 42 = ~35-42 minutes
  Pointcloud conversion: ~1-2 minutes
  Total: ~55 minutes to 1.5 hours

High Quality (subsample=2, target_size=768, ~63 frames):
  Model load time: ~15-20s
  Per-frame inference: ~1.0-2.0s per frame × 63 = ~60-120 minutes
  Pointcloud conversion: ~1-2 minutes
  Total: ~1.5-2.5 hours
```

Memory usage peaks at ~6GB for target_size=518, ~10GB for target_size=768.

### Model Information

- **HuggingFace ID**: `tencent/HY-World-2.0`
- **Subfolder**: `HY-WorldMirror-2.0/`
- **Size**: ~2.4 GB weights
- **Framework**: Diffusers Pipeline API
- **First Run**: Downloads weights from HuggingFace Hub (~2.4 GB)
- **Cache Location**: `~/.cache/huggingface/hub/`

## Experiment Workflow

```
┌─────────────────────────────────────────┐
│  Data: data/incoming/images/     │
│  8 frames @ 1080x1920                   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Frames Setup                           │
│  Copy & rename to frames/ directory     │
│  Create frames.json metadata            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  WorldMirror 2.0 Inference              │
│  (run_worldmirror2.py)                  │
│  ├─ Load HF weights                     │
│  ├─ Process frames                      │
│  ├─ Output: cameras.json                │
│  ├─ Output: depth/*.npz                 │
│  └─ Output: pointmap/*.npz              │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Pointcloud Merging                     │
│  (implicit in run_worldmirror2.py)      │
│  Merge all frames into global scene     │
│  with edge/confidence filtering         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  PLY Conversion                         │
│  (worldmirror2_to_ply.py)               │
│  ├─ Load scene_pointmap chunks          │
│  ├─ Concatenate into single point set   │
│  └─ Write binary PLY with RGB + conf    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Output: scene_pointmap.ply             │
│  3D Pointcloud ready for visualization  │
│  or 3D gaussian splatting               │
└─────────────────────────────────────────┘
```

## Debugging

### Check frames are set up correctly
```bash
ls -la data/intermediates/frames2pts3d/worldmirror2/frames/
cat data/intermediates/frames2pts3d/worldmirror2/frames.json
```

### Run WorldMirror 2.0 only (skip PLY conversion)
```bash
uv run python scripts/run_worldmirror2.py data/intermediates/frames2pts3d/worldmirror2
```

### Convert existing outputs to PLY
```bash
uv run python scripts/worldmirror2_to_ply.py data/intermediates/frames2pts3d/worldmirror2
```

### Inspect PLY file
```bash
python3 << 'EOF'
import numpy as np
from pathlib import Path

ply_file = Path("data/intermediates/frames2pts3d/worldmirror2/worldmirror2/scene_pointmap.ply")
with open(ply_file, 'rb') as f:
    # Read header
    lines = []
    while True:
        line = f.readline().decode().strip()
        if line == 'end_header':
            break
        lines.append(line)
        if line.startswith('element vertex'):
            num_points = int(line.split()[-1])

    print(f"PLY Statistics:")
    print(f"  Points: {num_points:,}")
    print(f"  File size: {ply_file.stat().st_size / (1024**2):.1f} MB")
EOF
```

### View the PLY file

**Using CloudCompare:**
```bash
# Install: https://www.cloudcompare.org/
cloudcompare data/intermediates/frames2pts3d/worldmirror2/worldmirror2/scene_pointmap.ply
```

**Using Python with trimesh:**
```python
import trimesh
cloud = trimesh.load('data/intermediates/frames2pts3d/worldmirror2/worldmirror2/scene_pointmap.ply')
print(f"Points: {len(cloud.vertices)}")
cloud.show()
```

## Next Steps

Once the PLY is generated, you can:

1. **Visualize** in CloudCompare or MeshLab
2. **Refine depth** using InfiniDepth (available in video-vision repo)
3. **Train 3D Gaussian Splat** using OpenSplat with the pointcloud as initialization
4. **Compare** with other reconstruction methods (COLMAP, CUT3R, VGGT, etc.)

## References

- **HunyuanWorld-Mirror 2.0**: https://github.com/Tencent-Hunyuan/HY-World-2.0
- **Video-Vision Repository**: https://github.com/rms80/video_vision (source of the runner scripts)
- **CloudCompare**: https://www.cloudcompare.org/
- **PLY Format**: https://en.wikipedia.org/wiki/PLY_(file_format)

## Troubleshooting

### "ModuleNotFoundError: No module named 'hyworld2'"

The HY-World-2.0 repository is not properly set up. Check:
```bash
ls -la models/external/hy-world-2.0/
```

If missing, set it up manually:
```bash
git clone https://github.com/Tencent-Hunyuan/HY-World-2.0.git models/external/hy-world-2.0
cd models/external/hy-world-2.0
git checkout ee5d5bc02c92671486ab6dd81f0cba577d0478c8
```

### "AttributeError: module 'torch' has no attribute 'cuda'"

This is expected on macOS. The code detects this and skips CUDA cleanup. PyTorch uses Metal Performance Shaders (MPS) instead.

### Out of memory errors

Lower the `TARGET_SIZE` parameter:
```bash
bash experiments/worldmirror2/launch.sh 1 300  # 300x300 resolution
```

### PLY file not generated

Check that the WorldMirror 2.0 inference completed successfully:
```bash
ls -la data/intermediates/frames2pts3d/worldmirror2/worldmirror2/
```

If `scene_pointmap_chunks.json` is missing, the inference failed.
