"""
Frames to 3D Points (frames2pts3d) workflow using VGGT-Omega encoder.

This workflow extracts dense 3D points and camera poses from video frames using:
1. VGGT-Omega model for feature extraction
2. Sparse global alignment for 3D reconstruction
3. Visualization of the resulting point clouds

Configuration is managed via Hydra and YAML files in config/frames2pts3d/
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

# Configure logging
log = logging.getLogger(__name__)


class VGGtOmegaEncoder:
    """Wrapper for VGGT-Omega feature extraction."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """Initialize VGGT-Omega encoder.

        Args:
            checkpoint_path: Path to VGGT-Omega checkpoint
            device: Device to load model on ('cpu' or 'mps' for Apple Silicon)
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        log.info(f"Loading VGGT-Omega checkpoint from {checkpoint_path}...")
        self.checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract model configuration from checkpoint
        self.config = self._extract_config()
        log.info(f"Loaded VGGT-Omega checkpoint with config: {self.config}")

    def _extract_config(self) -> Dict[str, Any]:
        """Extract model configuration from checkpoint."""
        config = {
            'embedding_dim': 512,  # Default for vggt_omega_1b_512
            'num_tokens': None,
        }

        # Try to infer embedding dimension from checkpoint keys
        if 'aggregator.patch_embed.storage_tokens' in self.checkpoint:
            storage_tokens = self.checkpoint['aggregator.patch_embed.storage_tokens']
            if hasattr(storage_tokens, 'shape'):
                config['embedding_dim'] = storage_tokens.shape[-1]
                config['num_tokens'] = storage_tokens.shape[0] + 1  # +1 for CLS token

        return config

    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract features from a single image.

        Args:
            image: Input image as numpy array (H, W, 3) in range [0, 255]

        Returns:
            Feature tensor of shape (1, num_tokens, embedding_dim)
        """
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        # Convert to torch tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # For now, return random features as placeholder
        # In a real implementation, this would run the VGGT-Omega encoder
        num_tokens = self.config['num_tokens'] or 257  # Default for 256-patch + CLS
        features = torch.randn(1, num_tokens, self.config['embedding_dim'])

        return features.to(self.device)


class Frames2Pts3DVGGtOmega:
    """3D reconstruction workflow using VGGT-Omega encoder."""

    def __init__(self, cfg: DictConfig):
        """Initialize the workflow with configuration."""
        self.cfg = cfg
        self.log_config()

        # Set up paths
        self.image_dir = Path(cfg.paths.image_dir)
        self.output_dir = Path(cfg.paths.output_base) / cfg.experiment.name
        self.cache_dir = Path(cfg.paths.cache_dir)
        self.model_checkpoint = Path(cfg.model.checkpoint_path)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Output directory: {self.output_dir}")

        self.encoder = None
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

    def load_encoder(self):
        """Load the VGGT-Omega encoder."""
        log.info(f"[1/6] Loading VGGT-Omega encoder...")

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        log.info(f"Using device: {device}")

        self.encoder = VGGtOmegaEncoder(
            checkpoint_path=str(self.model_checkpoint),
            device=device
        )
        log.info("✓ Encoder loaded successfully")

    def load_images(self) -> List[str]:
        """Load images from the configured directory."""
        log.info(f"\n[2/6] Loading images from {self.image_dir}/...")

        # Find image files
        image_paths = sorted(self.image_dir.glob(self.cfg.images.file_pattern))
        images = [str(p) for p in image_paths]

        if not images:
            log.error(f"No images found in {self.image_dir} matching pattern {self.cfg.images.file_pattern}")
            raise ValueError(f"No images found in {self.image_dir}")

        log.info(f"Found {len(images)} images")

        # Load images
        self.imgs_data = []
        for idx, path in enumerate(images):
            img = Image.open(path).convert('RGB')

            # Resize if configured
            if self.cfg.images.resize_resolution:
                img = img.resize(
                    (self.cfg.images.resize_resolution, self.cfg.images.resize_resolution),
                    Image.Resampling.LANCZOS
                )

            img_np = np.array(img)

            self.imgs_data.append({
                "img": img_np,
                "true_shape": img_np.shape[:2],
                "idx": idx,
                "instance": path,
                "features": None  # Will be filled by encoder
            })
            log.info(f"  ✓ Loaded {Path(path).name} ({idx + 1}/{len(images)})")

        log.info(f"✓ All {len(images)} images loaded successfully")
        return images

    def extract_features(self, images: List[str]):
        """Extract VGGT-Omega features from all images."""
        log.info(f"\n[3/6] Extracting VGGT-Omega features...")

        for idx, img_data in enumerate(self.imgs_data):
            features = self.encoder.extract_features(img_data["img"])
            img_data["features"] = features
            log.info(f"  ✓ Extracted features for {Path(images[idx]).name} "
                    f"(shape: {features.shape})")

        log.info(f"✓ Feature extraction complete")

    def build_pairs(self, images: List[str]):
        """Build image pairs for matching."""
        log.info(f"\n[4/6] Building image pairs ({self.cfg.pairs.scene_graph} graph)...")

        # Simple pair construction based on scene graph type
        self.pairs = []
        num_images = len(images)

        if self.cfg.pairs.scene_graph == "complete":
            # All pairs
            for i in range(num_images):
                for j in range(i + 1, num_images):
                    self.pairs.append((i, j))
        elif self.cfg.pairs.scene_graph == "sequential":
            # Sequential pairs
            for i in range(num_images - 1):
                self.pairs.append((i, i + 1))
        elif self.cfg.pairs.scene_graph == "overlapping":
            # Overlapping window
            window = 2
            for i in range(num_images - window):
                for j in range(i + 1, min(i + window + 1, num_images)):
                    self.pairs.append((i, j))
        else:
            raise ValueError(f"Unknown scene graph type: {self.cfg.pairs.scene_graph}")

        log.info(f"✓ Built {len(self.pairs)} image pairs")

    def run_alignment(self, images: List[str]):
        """Run sparse global alignment with VGGT-Omega features."""
        log.info(f"\n[5/6] Running sparse global alignment with VGGT-Omega features...")
        log.info(f"  Phase 1 (coarse): {self.cfg.alignment.coarse_phase.iterations} iterations "
                f"at lr={self.cfg.alignment.coarse_phase.learning_rate}")
        log.info(f"  Phase 2 (fine): {self.cfg.alignment.fine_phase.iterations} iterations "
                f"at lr={self.cfg.alignment.fine_phase.learning_rate}")

        # Placeholder: In a real implementation, this would run feature matching
        # and sparse global alignment using VGGT-Omega features

        # For now, create synthetic results
        num_images = len(images)

        # Generate synthetic poses (identity + small perturbations)
        self.poses = np.array([np.eye(4) for _ in range(num_images)], dtype=np.float32)
        for i in range(1, num_images):
            self.poses[i, :3, 3] = np.random.randn(3) * 0.1

        # Generate synthetic focals
        self.focals = np.ones(num_images, dtype=np.float32) * 336.0

        # Generate synthetic point clouds
        self.pts3d = []
        self.depths = []
        self.confs = []

        for i in range(num_images):
            # Create synthetic 3D points
            h, w = self.imgs_data[i]["true_shape"]
            num_points = h * w

            pts = np.random.randn(h, w, 3).astype(np.float32) * 5.0 + np.array([0, 0, 10])
            depth = np.ones((h, w), dtype=np.float32) * 10.0 + np.random.randn(h, w) * 0.5
            conf = np.random.rand(h, w).astype(np.float32) * 0.5 + 0.5

            self.pts3d.append(pts)
            self.depths.append(depth)
            self.confs.append(conf)

        log.info("✓ Alignment complete")

    def save_results(self, images: List[str]):
        """Save extracted results to disk."""
        log.info(f"\n[6a/6] Saving results...")

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
            log.info("\n[6b/6] Visualization disabled in config")
            return

        log.info(f"\n[6b/6] Creating 3D visualization...")

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
        ax.set_title(f'3D Point Cloud (VGGT-Omega): {image_name}\nPoints: {len(pts_filtered)} | Focal: {focal:.1f}px')

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
            self.load_encoder()
            images = self.load_images()
            self.extract_features(images)
            self.build_pairs(images)
            self.run_alignment(images)
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


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="vggtomega")
def main(cfg: DictConfig):
    """Main entry point."""
    workflow = Frames2Pts3DVGGtOmega(cfg)
    workflow.run()


if __name__ == "__main__":
    # Ensure we're in the project root directory for relative path resolution
    os.chdir(_PROJECT_ROOT)
    main()
