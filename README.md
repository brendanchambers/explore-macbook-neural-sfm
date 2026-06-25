# Goal
Explore and learn about feedforward networks in the context of 3D Guassian Splatting of outdoor scenes on macbook M4 24GB.

## Workflows

### frames2pts3d
Extracts dense 3D points and camera poses from video frames using DuneMast3r and sparse global alignment.

**Core script:** `scripts/frames2pts3d.py`
**Configuration:** `config/frames2pts3d/config.yaml`

#### Quick Start with Preset Experiments

Reproducible launch scripts are provided in `experiments/frames2pts3d/`:

```bash
# Launch on teensy image set (default: current_experiment)
./experiments/frames2pts3d/launch_teensy.sh

# Launch with custom experiment name
./experiments/frames2pts3d/launch_teensy.sh my_experiment_name
```

#### Manual Usage

**View all configuration options:**
```bash
uv run python scripts/frames2pts3d.py --help
```

**Run with default config:**
```bash
uv run python scripts/frames2pts3d.py
```

**Run with custom configuration:**
```bash
uv run python scripts/frames2pts3d.py experiment.name=my_experiment \
  paths.image_dir=data/incoming/images_teensy \
  model.resolution=336
```

**Output:** Results are saved to `data/intermediates/frames2pts3d/{experiment_name}/`
- `poses.npy` - Camera poses [N, 4, 4]
- `focals.npy` - Focal lengths [N]
- `pts3d/` - Directory containing point clouds for each image
- `visualization.png` - 3D point cloud visualization
- `config.yaml` - Configuration used for this run

