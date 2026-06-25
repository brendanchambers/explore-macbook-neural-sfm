# VGGT-Omega Launcher Scripts

This directory contains launcher scripts for the VGGT-Omega 3D reconstruction workflow.

## Scripts

### launch_teensy.sh
Runs VGGT-Omega reconstruction on the teensy image dataset (8 test images).

```bash
./launch_teensy.sh
./launch_teensy.sh experiment_name=my_experiment
./launch_teensy.sh pairs.scene_graph=complete alignment.coarse_phase.iterations=500
```

**Default Configuration:**
- Input: `data/incoming/images_teensy/`
- Experiment name: `vggtomega_teensy`
- Scene graph: `sequential` (7 image pairs)
- Image resolution: 512×512

### launch_full.sh
Runs VGGT-Omega reconstruction on the full image dataset (125 images).

```bash
./launch_full.sh
./launch_full.sh experiment_name=my_full_experiment
./launch_full.sh pairs.scene_graph=complete
```

**Default Configuration:**
- Input: `data/incoming/images/`
- Experiment name: `vggtomega_full`
- Scene graph: `sequential`
- Image resolution: 512×512

## Configuration

Both scripts use the Hydra configuration system. Override any parameter by passing `key=value`:

```bash
./launch_teensy.sh experiment.name=test_run \
  alignment.coarse_phase.iterations=500 \
  visualization.confidence_threshold=0.2
```

### Common Parameters

| Parameter | Default | Options |
|-----------|---------|---------|
| `experiment.name` | vggtomega_teensy | Any string |
| `experiment.group` | frames2pts3d_vggtomega | Any string |
| `pairs.scene_graph` | sequential | sequential, complete, overlapping |
| `images.resize_resolution` | 512 | 256, 512, 1024 |
| `alignment.coarse_phase.iterations` | 300 | 1-1000 |
| `alignment.fine_phase.iterations` | 300 | 1-1000 |
| `visualization.enabled` | true | true, false |
| `visualization.confidence_threshold` | 0.1 | 0.0-1.0 |

## Output

Results are saved to `data/intermediates/frames2pts3d/{experiment_name}/`:

- `poses.npy` - Camera poses [N, 4, 4]
- `focals.npy` - Focal lengths [N]
- `pts3d/` - Directory with point cloud files (pts3d_000.npy, pts3d_001.npy, ...)
- `visualization.png` - 3D scatter plot visualization
- `config.yaml` - Configuration used for this run
- `visualizations/{timestamp}/` - Visualization subdirectory with copies of results

## Model

Uses VGGT-Omega checkpoint: `data/models/vggt_omega_1b_512.pt`
- Model: VGGt-Omega 1B parameters
- Feature dimension: 512
- Checkpoint size: ~4.5 GB

## Related

- Main workflow script: `scripts/frames2pts3d_vggtomega.py`
- Configuration file: `config/frames2pts3d/vggtomega.yaml`
- DuneMast3r launchers: `experiments/frames2pts3d/dunemast3r/`
