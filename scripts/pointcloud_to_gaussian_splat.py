#!/usr/bin/env python
"""Convert point cloud data to 3D Gaussian Splat PLY format.

This script converts a point cloud (position, color, confidence) into
3D Gaussian Splat parameters suitable for viewing in Gaussian Splat viewers.

The GS initialization strategy:
  - means: Direct from point positions
  - scales: Initialized from local point density (log space)
  - rotations: Initialized to identity quaternion [0, 0, 0, 1]
  - opacity: Computed from confidence scores (logit space)
  - SH coefficients: DC term from RGB, higher bands initialized to zero

Usage:
    uv run python scripts/pointcloud_to_gaussian_splat.py <scene_dir> [options]

Options:
    --downsample N          Downsample by factor N (e.g., 10 for 10x lighter)
    --max-splats N          Maximum number of splats (alternative to downsample)
    --no-sh-rest            Omit higher-order SH coefficients (saves space)
    --output NAME           Output filename (default: scene_gaussian_splat.ply)

Reads:
    <scene_dir>/worldmirror2/scene_pointmap_chunks.json
    <scene_dir>/worldmirror2/scene_pointmap_*.npz

Outputs:
    <scene_dir>/worldmirror2/scene_gaussian_splat.ply (or custom name)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# Spherical Harmonics (SH) encoding utilities
# Based on: https://github.com/graphdeco-inria/gaussian-splatting

def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB colors to SH DC coefficient.

    Args:
        rgb: (N, 3) uint8 array with values [0, 255]

    Returns:
        sh_dc: (N, 1, 3) float32 array with SH DC coefficient
    """
    # Normalize RGB to [0, 1]
    rgb_norm = rgb.astype(np.float32) / 255.0

    # Convert linear RGB to SH DC term (0-th order spherical harmonic)
    # DC term for RGB: multiply by sqrt(4*pi) / sqrt(1) ≈ 3.545
    C0 = 0.28209479177387814
    sh_dc = (rgb_norm - 0.5) / C0

    return sh_dc[:, np.newaxis, :]  # (N, 1, 3)


def initialize_scales(points: np.ndarray, conf: np.ndarray, quantile: float = 0.1) -> np.ndarray:
    """Initialize splat scales from point density.

    Uses the confidence scores to estimate initial scale.
    Higher confidence points get smaller scales (more certain geometry).

    Args:
        points: (N, 3) point positions
        conf: (N,) confidence scores
        quantile: Percentile for scale computation (default 0.1)

    Returns:
        scales: (N, 3) log-space scale factors
    """
    # Normalize confidence to [0.1, 1.0] range
    conf_min = np.quantile(conf, quantile)
    conf_max = np.quantile(conf, 1.0 - quantile)
    conf_norm = np.clip((conf - conf_min) / (conf_max - conf_min + 1e-8), 0.0, 1.0)

    # Map confidence to scale: high confidence → small scale
    # Range: [0.01, 0.05] in linear space
    scale_linear = 0.01 + (1.0 - conf_norm) * 0.04

    # Convert to log space (GS uses log-space scales)
    # log(scale) is the actual representation
    scale_log = np.log(scale_linear)

    # Replicate across XYZ
    scales = np.tile(scale_log[:, np.newaxis], (1, 3))

    return scales.astype(np.float32)


def initialize_rotations(n: int) -> np.ndarray:
    """Initialize rotations to identity quaternion.

    Args:
        n: Number of points

    Returns:
        quats: (N, 4) quaternions [qx, qy, qz, qw] with w=1 (identity)
    """
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 3] = 1.0  # qw = 1 (identity quaternion)
    return quats


def opacity_from_confidence(conf: np.ndarray) -> np.ndarray:
    """Convert confidence to opacity in logit space.

    Args:
        conf: (N,) confidence scores (typically normalized 0-1 or in some range)

    Returns:
        opacity: (N,) logit-space opacity values
    """
    # Normalize confidence to (0, 1) range, excluding extremes
    conf_min = np.percentile(conf, 5)
    conf_max = np.percentile(conf, 95)
    conf_norm = np.clip((conf - conf_min) / (conf_max - conf_min + 1e-8), 0.0, 1.0)

    # Ensure we're in (0.01, 0.99) to avoid numerical issues
    conf_norm = np.clip(conf_norm, 0.01, 0.99)

    # Convert to logit space: logit(x) = log(x / (1-x))
    # GS uses logit for opacity during optimization
    opacity = np.log(conf_norm / (1.0 - conf_norm))

    return opacity.astype(np.float32)


def write_gaussian_splat_ply(
    filename: str,
    means: np.ndarray,      # (N, 3) float32
    scales: np.ndarray,     # (N, 3) float32
    quats: np.ndarray,      # (N, 4) float32 [qx, qy, qz, qw]
    opacities: np.ndarray,  # (N,) float32
    sh_dc: np.ndarray,      # (N, 1, 3) float32
    include_sh_rest: bool = True,  # Include higher-order SH coefficients
):
    """Write a PLY file in 3DGS format.

    Args:
        filename: Output PLY file path
        means: Splat centers [N, 3]
        scales: Splat scales in log-space [N, 3]
        quats: Quaternion rotations [N, 4] as [qx, qy, qz, qw]
        opacities: Opacity in logit space [N]
        sh_dc: SH DC coefficient (0-th order) [N, 1, 3]
        include_sh_rest: Include higher-order SH coefficients (45 extra floats per splat)
    """
    assert means.shape[0] == len(opacities)
    assert means.dtype == np.float32
    assert scales.dtype == np.float32
    assert quats.dtype == np.float32
    assert opacities.dtype == np.float32
    assert sh_dc.dtype == np.float32

    N = len(means)

    # Prepare structured array for PLY
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]

    # Add higher-order SH coefficients (optional)
    # Standard format includes up to 3rd order SH = 16 coefficients total
    # DC is 1, so we need 15 more for a full set (4*4-1=15), each with RGB = 45 coeffs
    if include_sh_rest:
        num_rest_coeffs = 15
        for i in range(num_rest_coeffs * 3):
            dtype_list.append((f'f_rest_{i}', 'f4'))

    dtype_list.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])

    # Create structured array
    vertex = np.zeros(N, dtype=dtype_list)

    # Fill in data
    vertex['x'] = means[:, 0]
    vertex['y'] = means[:, 1]
    vertex['z'] = means[:, 2]

    vertex['f_dc_0'] = sh_dc[:, 0, 0]
    vertex['f_dc_1'] = sh_dc[:, 0, 1]
    vertex['f_dc_2'] = sh_dc[:, 0, 2]

    # Higher-order SH coefficients are zero (initialized)
    if include_sh_rest:
        for i in range(15 * 3):
            vertex[f'f_rest_{i}'] = 0.0

    vertex['opacity'] = opacities
    vertex['scale_0'] = scales[:, 0]
    vertex['scale_1'] = scales[:, 1]
    vertex['scale_2'] = scales[:, 2]
    vertex['rot_0'] = quats[:, 0]
    vertex['rot_1'] = quats[:, 1]
    vertex['rot_2'] = quats[:, 2]
    vertex['rot_3'] = quats[:, 3]

    # Write PLY file
    with open(filename, 'wb') as f:
        # Write header
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {N}\n'.encode())

        # Write property declarations
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        f.write(b'property float f_dc_0\n')
        f.write(b'property float f_dc_1\n')
        f.write(b'property float f_dc_2\n')

        if include_sh_rest:
            for i in range(15 * 3):
                f.write(f'property float f_rest_{i}\n'.encode())

        f.write(b'property float opacity\n')
        f.write(b'property float scale_0\n')
        f.write(b'property float scale_1\n')
        f.write(b'property float scale_2\n')
        f.write(b'property float rot_0\n')
        f.write(b'property float rot_1\n')
        f.write(b'property float rot_2\n')
        f.write(b'property float rot_3\n')
        f.write(b'end_header\n')

        # Write binary data
        f.write(vertex.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Convert point cloud to 3D Gaussian Splat PLY format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full quality
  uv run python scripts/pointcloud_to_gaussian_splat.py data/intermediates/frames2pts3d/worldmirror2

  # 10x lighter (every 10th point)
  uv run python scripts/pointcloud_to_gaussian_splat.py data/intermediates/frames2pts3d/worldmirror2 --downsample 10

  # Max 500K splats
  uv run python scripts/pointcloud_to_gaussian_splat.py data/intermediates/frames2pts3d/worldmirror2 --max-splats 500000

  # Lightweight version without higher-order SH
  uv run python scripts/pointcloud_to_gaussian_splat.py data/intermediates/frames2pts3d/worldmirror2 --downsample 10 --no-sh-rest
        """
    )
    parser.add_argument("scene_dir", help="Path to scene directory")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample by factor N (default: 1)")
    parser.add_argument("--max-splats", type=int, default=None,
                        help="Maximum number of splats (overrides --downsample)")
    parser.add_argument("--no-sh-rest", action="store_true",
                        help="Omit higher-order SH coefficients (saves ~40%% space)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: scene_gaussian_splat.ply)")

    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    wm2_dir = scene_dir / "worldmirror2"

    if not wm2_dir.exists():
        print(f"Error: {wm2_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load manifest
    manifest_path = wm2_dir / "scene_pointmap_chunks.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load all chunks
    print(f"Loading point cloud chunks from {wm2_dir}...")
    all_pts = []
    all_rgb = []
    all_conf = []

    for chunk_info in manifest["chunks"]:
        chunk_file = wm2_dir / chunk_info["file"]
        if not chunk_file.exists():
            print(f"Warning: {chunk_file} not found, skipping")
            continue

        data = np.load(chunk_file)
        all_pts.append(data["pts3d"].astype(np.float32))
        all_rgb.append(data["rgb"].astype(np.uint8))
        all_conf.append(data["conf"].astype(np.float32))

    if not all_pts:
        print("Error: No chunks loaded", file=sys.stderr)
        sys.exit(1)

    # Concatenate
    pts = np.concatenate(all_pts, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    conf = np.concatenate(all_conf, axis=0)

    original_count = len(pts)
    print(f"Loaded {original_count:,} points")

    # Apply downsampling
    downsample_factor = args.downsample
    if args.max_splats is not None:
        # Calculate downsample factor from max_splats
        downsample_factor = max(1, original_count // args.max_splats)

    if downsample_factor > 1:
        print(f"Downsampling by factor {downsample_factor}...")
        indices = np.arange(0, len(pts), downsample_factor)
        pts = pts[indices]
        rgb = rgb[indices]
        conf = conf[indices]
        print(f"Downsampled to {len(pts):,} points ({len(pts) / original_count * 100:.1f}%)")

    # Initialize Gaussian Splat parameters
    print("Initializing Gaussian Splat parameters...")
    means = pts
    scales = initialize_scales(pts, conf)
    quats = initialize_rotations(len(pts))
    opacities = opacity_from_confidence(conf)
    sh_dc = rgb_to_sh(rgb)

    # Determine output path
    if args.output:
        output_path = wm2_dir / args.output
    else:
        if downsample_factor > 1:
            output_path = wm2_dir / f"scene_gaussian_splat_{downsample_factor}x.ply"
        else:
            output_path = wm2_dir / "scene_gaussian_splat.ply"

    # Write PLY
    print(f"Writing Gaussian Splat to {output_path}...")
    write_gaussian_splat_ply(
        str(output_path), means, scales, quats, opacities, sh_dc,
        include_sh_rest=not args.no_sh_rest
    )

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ Wrote {output_path}")
    print(f"  Splats: {len(means):,} ({len(means) / original_count * 100:.1f}% of original)")
    print(f"  File size: {file_size_mb:.1f} MB")
    if args.no_sh_rest:
        print(f"  SH: DC only (no higher-order coefficients)")
    else:
        print(f"  SH: 0-3 order (full)")
    print(f"  Bounds: x [{means[:, 0].min():.3f}, {means[:, 0].max():.3f}]")
    print(f"          y [{means[:, 1].min():.3f}, {means[:, 1].max():.3f}]")
    print(f"          z [{means[:, 2].min():.3f}, {means[:, 2].max():.3f}]")
    print(f"  Opacity range: [{opacities.min():.3f}, {opacities.max():.3f}] (logit)")
    print(f"  Scale range: [{scales.min():.3f}, {scales.max():.3f}] (log)")


if __name__ == "__main__":
    main()
