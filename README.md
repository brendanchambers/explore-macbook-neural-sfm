# Goal
Explore and learn about feedforward networks in the context of 3D Guassian Splatting of outdoor scenes on macbook M4 24GB.




  Quick Usage

  # 10x lighter, ultra-compact (recommended)
  uv run python scripts/pointcloud_to_gaussian_splat.py <scene_dir> \
    --downsample 10 --no-sh-rest

  # 20x lighter for web
  uv run python scripts/pointcloud_to_gaussian_splat.py <scene_dir> \
    --downsample 20 --no-sh-rest

  All versions are ready for viewing in Supersplat, Nerfstudio, or any 3DGS-compatible viewer.