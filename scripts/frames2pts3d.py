"""
Frames to 3D Points (frames2pts3d) workflow.

This workflow extracts dense 3D points and camera poses from video frames using:
1. DuneMast3r model for feature extraction
2. Sparse global alignment for 3D reconstruction
3. Visualization of the resulting point clouds

Configuration is managed via Hydra and YAML files in config/frames2pts3d/
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import hydra
import mlx.core as mx
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mlx_mast3r import DuneMast3r
from mlx_mast3r.cloud_opt import sparse_global_alignment
from mlx_mast3r.image_pairs import make_pairs
from mlx_mast3r.utils import load_image

# Configure logging
log = logging.getLogger(__name__)


class Frames2Pts3D:
    """Core workflow for extracting 3D points from image frames."""

    def __init__(self, cfg: DictConfig):
        """Initialize the workflow with configuration."""
        self.cfg = cfg
        self.log_config()

        # Set up paths
        self.image_dir = Path(cfg.paths.image_dir)
        self.output_dir = Path(cfg.paths.output_base) / cfg.experiment.name
        self.cache_dir = Path(cfg.paths.cache_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Output directory: {self.output_dir}")

        self.model = None
        self.imgs_data = None
        self.pairs = None
        self.result = None
        self.poses = None
        self.focals = None
        self.pts3d = None
        self.depths = None
        self.confs = None

    def log_config(self):
        """Log the configuration for reproducibility."""
        log.info("=" * 80)
        log.info("CONFIGURATION")
        log.info("=" * 80)
        log.info(OmegaConf.to_yaml(self.cfg))
        log.info("=" * 80)

    def load_model(self):
        """Load the DuneMast3r model."""
        log.info(f"[1/5] Loading DuneMast3r model (variant={self.cfg.model.encoder_variant}, "
                 f"resolution={self.cfg.model.resolution})...")

        self.model = DuneMast3r.from_pretrained(
            encoder_variant=self.cfg.model.encoder_variant,
            resolution=self.cfg.model.resolution
        )
        log.info("✓ Model loaded successfully")

    def load_images(self) -> List[str]:
        """Load images from the configured directory."""
        log.info(f"\n[2/5] Loading images from {self.image_dir}/...")

        # Find image files
        image_paths = sorted(self.image_dir.glob(self.cfg.images.file_pattern))
        images = [str(p) for p in image_paths]

        if not images:
            log.error(f"No images found in {self.image_dir} matching pattern {self.cfg.images.file_pattern}")
            raise ValueError(f"No images found in {self.image_dir}")

        log.info(f"Found {len(images)} images")

        # Load images into MLX arrays
        self.imgs_data = []
        resolution = self.cfg.images.resize_resolution

        for idx, path in enumerate(images):
            img = load_image(path, resolution=resolution)
            self.imgs_data.append({
                "img": mx.array(img).transpose(2, 0, 1)[None],
                "true_shape": img.shape[:2],
                "idx": idx,
                "instance": path,
            })
            log.info(f"  ✓ Loaded {Path(path).name} ({idx + 1}/{len(images)})")

        log.info(f"✓ All {len(images)} images loaded successfully")
        return images

    def build_pairs(self, images: List[str]):
        """Build image pairs for matching."""
        log.info(f"\n[3/5] Building image pairs ({self.cfg.pairs.scene_graph} graph)...")

        self.pairs = make_pairs(
            self.imgs_data,
            scene_graph=self.cfg.pairs.scene_graph,
            symmetrize=self.cfg.pairs.symmetrize
        )
        log.info(f"✓ Built {len(self.pairs)} image pairs")

    def run_alignment(self, images: List[str]):
        """Run sparse global alignment."""
        log.info(f"\n[4/5] Running sparse global alignment...")
        log.info(f"  Phase 1 (coarse): {self.cfg.alignment.coarse_phase.iterations} iterations "
                f"at lr={self.cfg.alignment.coarse_phase.learning_rate}")
        log.info(f"  Phase 2 (fine): {self.cfg.alignment.fine_phase.iterations} iterations "
                f"at lr={self.cfg.alignment.fine_phase.learning_rate}")

        self.result = sparse_global_alignment(
            imgs=images,
            pairs_in=self.pairs,
            cache_path=str(self.cache_dir),
            model=self.model,
            lr1=self.cfg.alignment.coarse_phase.learning_rate,
            niter1=self.cfg.alignment.coarse_phase.iterations,
            lr2=self.cfg.alignment.fine_phase.learning_rate,
            niter2=self.cfg.alignment.fine_phase.iterations,
        )
        log.info("✓ Alignment complete")

    def extract_results(self):
        """Extract poses, focals, and 3D points from alignment result."""
        log.info(f"\n[5/5] Extracting results...")

        self.poses = self.result.get_im_poses()
        self.focals = self.result.get_focals()
        self.pts3d, self.depths, self.confs = self.result.get_dense_pts3d()

        log.info(f"✓ Extracted poses shape: {self.poses.shape}")
        log.info(f"✓ Extracted focals shape: {self.focals.shape}")

        if isinstance(self.pts3d, list):
            log.info(f"✓ Extracted {len(self.pts3d)} point clouds (list format)")
        else:
            log.info(f"✓ Extracted pts3d shape: {self.pts3d.shape}")

    def save_results(self, images: List[str]):
        """Save extracted results to disk."""
        if self.cfg.output.save_poses:
            poses_path = self.output_dir / "poses.npy"
            np.save(poses_path, self.poses)
            log.info(f"✓ Saved poses to {poses_path}")

        if self.cfg.output.save_focals:
            focals_path = self.output_dir / "focals.npy"
            np.save(focals_path, self.focals)
            log.info(f"✓ Saved focals to {focals_path}")

        if self.cfg.output.save_point_clouds:
            pts3d_path = self.output_dir / "pts3d"
            pts3d_path.mkdir(exist_ok=True)
            for idx, pts in enumerate(self.pts3d):
                pts_array = pts.numpy() if hasattr(pts, 'numpy') else np.array(pts)
                np.save(pts3d_path / f"pts3d_{idx:03d}.npy", pts_array)
            log.info(f"✓ Saved {len(self.pts3d)} point clouds to {pts3d_path}")

    def visualize(self, images: List[str]):
        """Create 3D visualization of point clouds."""
        if not self.cfg.visualization.enabled:
            log.info("\n[6/6] Visualization disabled in config")
            return

        log.info(f"\n[6/6] Creating 3D visualization...")

        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            log.warning("matplotlib not available, skipping visualization")
            return

        img_idx = self.cfg.visualization.image_idx

        # Validate index
        if img_idx >= len(self.pts3d):
            log.warning(f"Image index {img_idx} out of range (0-{len(self.pts3d)-1}). Using index 0.")
            img_idx = 0

        # Extract point cloud and confidence
        pts = self.pts3d[img_idx]
        conf = self.confs[img_idx]

        # Convert to numpy
        pts_np = pts.numpy() if hasattr(pts, 'numpy') else np.array(pts)
        conf_np = conf.numpy() if hasattr(conf, 'numpy') else np.array(conf)

        # Flatten if needed
        if len(pts_np.shape) > 2:
            pts_flat = pts_np.reshape(-1, 3)
            conf_flat = conf_np.flatten()
        else:
            pts_flat = pts_np
            conf_flat = conf_np.flatten() if len(conf_np.shape) > 0 else conf_np

        # Filter by confidence and valid values
        conf_threshold = self.cfg.visualization.confidence_threshold
        valid_mask = (conf_flat > conf_threshold) & np.isfinite(pts_flat).all(axis=1)
        pts_filtered = pts_flat[valid_mask]
        conf_filtered = conf_flat[valid_mask]

        log.info(f"✓ Filtered {len(pts_filtered)} valid points from {len(pts_flat)} total points "
                f"(confidence > {conf_threshold})")

        # Create 3D plot
        fig = plt.figure(figsize=tuple(self.cfg.visualization.figure_size))
        ax = fig.add_subplot(111, projection='3d')

        if len(pts_filtered) > 0:
            scatter = ax.scatter(
                pts_filtered[:, 0], pts_filtered[:, 1], pts_filtered[:, 2],
                c=conf_filtered,
                cmap=self.cfg.visualization.colormap,
                s=self.cfg.visualization.point_size,
                alpha=self.cfg.visualization.alpha
            )
            plt.colorbar(scatter, ax=ax, label='Confidence')

            # Set equal aspect ratio
            max_range = np.array([
                pts_flat[:, 0].max() - pts_flat[:, 0].min(),
                pts_flat[:, 1].max() - pts_flat[:, 1].min(),
                pts_flat[:, 2].max() - pts_flat[:, 2].min()
            ]).max() / 2.0
            mid_x = (pts_flat[:, 0].max() + pts_flat[:, 0].min()) * 0.5
            mid_y = (pts_flat[:, 1].max() + pts_flat[:, 1].min()) * 0.5
            mid_z = (pts_flat[:, 2].max() + pts_flat[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        focal = float(self.focals[img_idx])
        image_name = Path(images[img_idx]).name
        ax.set_title(f'3D Point Cloud: {image_name}\nPoints: {len(pts_filtered)} | Focal: {focal:.1f}px')

        plt.tight_layout()

        # Save visualization
        if self.cfg.output.save_visualization:
            output_path = self.output_dir / 'visualization.png'
            plt.savefig(output_path, dpi=self.cfg.visualization.dpi, bbox_inches='tight')
            log.info(f"✓ Saved visualization to {output_path}")

        plt.close()

    def save_config(self):
        """Save the configuration used for this run."""
        if self.cfg.output.save_config:
            config_path = self.output_dir / "config.yaml"
            with open(config_path, 'w') as f:
                OmegaConf.save(self.cfg, f)
            log.info(f"✓ Saved configuration to {config_path}")

    def run(self):
        """Execute the complete workflow."""
        try:
            images = self.load_images()
            self.load_model()
            self.build_pairs(images)
            self.run_alignment(images)
            self.extract_results()
            self.save_results(images)
            self.visualize(images)
            self.save_config()

            log.info("\n" + "=" * 80)
            log.info("✅ Processing complete!")
            log.info("=" * 80)
            log.info(f"Results saved to: {self.output_dir}")

        except Exception as e:
            log.error(f"❌ Error during processing: {e}", exc_info=True)
            raise


# Set up the config path relative to the project root
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CONFIG_DIR = _PROJECT_ROOT / "config" / "frames2pts3d"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    workflow = Frames2Pts3D(cfg)
    workflow.run()


if __name__ == "__main__":
    # Ensure we're in the project root directory for relative path resolution
    os.chdir(_PROJECT_ROOT)
    main()
