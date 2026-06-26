#!/usr/bin/env python
"""Convert WorldMirror 2.0 pointcloud outputs to PLY format.

Usage:
    uv run python scripts/worldmirror2_to_ply.py <scene_dir> [--output <ply_file>]

Reads WorldMirror 2.0 outputs from <scene_dir>/worldmirror2/ and converts
the scene_pointmap to a standard PLY file format.

Inputs:
    <scene_dir>/worldmirror2/scene_pointmap_chunks.json
    <scene_dir>/worldmirror2/scene_pointmap_NNN.npz

Outputs:
    <output_file> or <scene_dir>/worldmirror2/scene_pointmap.ply
"""

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np


def write_ply(filename, points, colors=None, confidences=None):
    """Write a PLY file from points and optional colors/confidences.

    Args:
        filename: Output PLY file path
        points: (N, 3) array of point positions (float32)
        colors: (N, 3) array of RGB colors (uint8), optional
        confidences: (N,) array of confidence scores (float32), optional
    """
    assert points.shape[1] == 3, "Points must be Nx3"
    assert points.dtype == np.float32, "Points must be float32"

    N = points.shape[0]

    # Build vertex dtype
    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if colors is not None:
        assert colors.shape == (N, 3) and colors.dtype == np.uint8
        dtype_list.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    if confidences is not None:
        assert confidences.shape == (N,) and confidences.dtype == np.float32
        dtype_list.append(('confidence', 'f4'))

    # Build structured array
    vertex = np.zeros(N, dtype=dtype_list)
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    if colors is not None:
        vertex['red'] = colors[:, 0]
        vertex['green'] = colors[:, 1]
        vertex['blue'] = colors[:, 2]
    if confidences is not None:
        vertex['confidence'] = confidences

    # Write PLY header
    with open(filename, 'wb') as f:
        # Header
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {N}\n'.encode())
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        if colors is not None:
            f.write(b'property uchar red\n')
            f.write(b'property uchar green\n')
            f.write(b'property uchar blue\n')
        if confidences is not None:
            f.write(b'property float confidence\n')
        f.write(b'end_header\n')

        # Binary data
        f.write(vertex.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Convert WorldMirror 2.0 pointcloud to PLY"
    )
    parser.add_argument("scene_dir", help="Path to scene directory")
    parser.add_argument(
        "--output", "-o",
        help="Output PLY file (default: <scene_dir>/worldmirror2/scene_pointmap.ply)"
    )
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    wm2_dir = scene_dir / "worldmirror2"

    if not wm2_dir.exists():
        print(f"Error: {wm2_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load scene pointmap manifest
    manifest_path = wm2_dir / "scene_pointmap_chunks.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load and concatenate all chunks
    print(f"Loading pointmap chunks from {wm2_dir}...")
    all_pts = []
    all_rgb = []
    all_conf = []

    for chunk_info in manifest["chunks"]:
        chunk_file = wm2_dir / chunk_info["file"]
        if not chunk_file.exists():
            print(f"Warning: {chunk_file} not found, skipping", file=sys.stderr)
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

    print(f"Loaded {pts.shape[0]:,} points")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = wm2_dir / "scene_pointmap.ply"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write PLY
    print(f"Writing PLY to {output_path}...")
    write_ply(str(output_path), pts, rgb, conf)
    print(f"Done! Wrote {output_path}")


if __name__ == "__main__":
    main()
