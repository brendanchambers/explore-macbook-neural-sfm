# Experiments Directory

This directory contains experiment configurations and launching scripts for various 3D reconstruction pipelines.

## Structure

```
experiments/
├── README.md                          ← This file
└── worldmirror2_teensy/               ← WorldMirror 2.0 teensy video experiment
    ├── launch.sh                      ← Launching script (use this to run the experiment)
    └── README.md                      ← Detailed documentation
```

## Experiments

### worldmirror2_teensy

**Purpose**: Generate a 3D pointcloud (`.ply` file) from teensy video frames using HunyuanWorld-Mirror 2.0.

**Quick Start**:
```bash
bash experiments/worldmirror2_teensy/launch.sh
```

**Key Features**:
- Runs WorldMirror 2.0 inference on 8 frame teensy video
- Generates camera poses, depth maps, and global pointcloud
- Converts outputs to standard PLY format for visualization
- Includes macOS M4 compatibility workarounds
- Comprehensive logging and results documentation

**Output**:
- `data/intermediates/frames2pts3d/worldmirror2_teensy/worldmirror2/scene_pointmap.ply` (3D pointcloud)
- Camera intrinsics and poses (JSON format)
- Per-frame depth maps (NPZ format)
- Experiment summary and metrics

**Parameters**:
```bash
bash experiments/worldmirror2_teensy/launch.sh [SUBSAMPLE] [TARGET_SIZE]
```
- `SUBSAMPLE`: Use every Nth frame (default: 1)
- `TARGET_SIZE`: Inference resolution (default: 518px)

See `experiments/worldmirror2_teensy/README.md` for complete documentation.

## Running Experiments

### Standard Launch

```bash
cd /path/to/project
bash experiments/worldmirror2_teensy/launch.sh
```

### With Custom Parameters

```bash
# Use every other frame, higher resolution inference
bash experiments/worldmirror2_teensy/launch.sh 2 768
```

### Monitor Progress

```bash
# Check if pipeline is running
ps aux | grep run_worldmirror2

# Watch the output directory
watch ls -lh data/intermediates/frames2pts3d/worldmirror2_teensy/worldmirror2/
```

## Experiment Results

Results are logged to `reports/experiments/<experiment_group>/`:
- `worldmirror2_teensy_summary.txt` - Experiment summary and metrics
- `worldmirror2_teensy.jsonl` - Structured results (for comparison across runs)

## Adding New Experiments

To create a new experiment:

1. Create a new directory: `experiments/<experiment_name>/`
2. Create a `launch.sh` script that:
   - Takes parameters as arguments
   - Runs the reconstruction pipeline
   - Logs results to `reports/experiments/`
3. Create a `README.md` with documentation

Example structure:
```bash
experiments/my_experiment/
├── launch.sh                  # Executable launching script
└── README.md                  # Documentation
```

## Integration with CLAUDE.md Project Structure

These experiments follow the project conventions from `CLAUDE.md`:

- **Experiments**: Defined with `experiment_name` and `experiment_group`
- **Organization**: Results organized in `reports/experiments/<experiment_group>.jsonl`
- **Data**: Intermediate outputs in `data/intermediates/<experiment_name>`
- **Logging**: Hydra logs in `logs/hydra/`

## References

- Project Instructions: `CLAUDE.md`
- Setup Documentation: `reports/work_history/worldmirror2_setup.md`
- Main Scripts: `scripts/run_worldmirror2.py`, `scripts/worldmirror2_to_ply.py`
