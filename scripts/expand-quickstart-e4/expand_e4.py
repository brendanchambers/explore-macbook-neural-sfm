from mlx_mast3r import DuneMast3r
from mlx_mast3r.cloud_opt import sparse_global_alignment
from mlx_mast3r.image_pairs import make_pairs
from mlx_mast3r.utils import load_image
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIG SECTION
# ============================================================================
experiment_name = "current_experiment"
IMAGE_DIR = "data/incoming/images_teensy"  # Directory containing images
OUTPUT_DIR = f"data/intermediates/expand-e4/{experiment_name}"  # Output directory
VISUALIZATION_IMAGE_IDX = 1
# ============================================================================

# Load model
print("[1/5] Loading DuneMast3r model (base variant, resolution=336)...")
model = DuneMast3r.from_pretrained(encoder_variant="base", resolution=336)
print("✓ Model loaded successfully")

# Load images
print(f"\n[2/5] Loading images from {IMAGE_DIR}/...")
image_paths = sorted(Path(IMAGE_DIR).glob("*.jpg"))
images = [str(p) for p in image_paths]
resolution = 336

if not images:
    print(f"⚠️  No images found in {IMAGE_DIR}")
    exit(1)

imgs_data = []
for idx, path in enumerate(images):
    img = load_image(path, resolution=resolution)
    imgs_data.append({
        "img": mx.array(img).transpose(2, 0, 1)[None],
        "true_shape": img.shape[:2],
        "idx": idx,
        "instance": path,
    })
    print(f"  ✓ Loaded {path} ({idx + 1}/{len(images)})")
print(f"✓ All {len(images)} images loaded successfully")

# Build pairs (complete graph for small sets)
print("\n[3/5] Building image pairs (complete graph)...")
pairs = make_pairs(imgs_data, scene_graph="complete", symmetrize=True)
print(f"✓ Built {len(pairs)} image pairs")

# Run global alignment
print("\n[4/5] Running sparse global alignment...")
print("  Phase 1 (coarse): 300 iterations at lr=0.07")
print("  Phase 2 (fine): 300 iterations at lr=0.01")
result = sparse_global_alignment(
    imgs=images,
    pairs_in=pairs,
    cache_path="/tmp/cache",
    model=model,
    lr1=0.07,      # Coarse phase learning rate
    niter1=300,    # Coarse phase iterations
    lr2=0.01,      # Fine phase learning rate
    niter2=300,    # Fine phase iterations
)
print("✓ Alignment complete")

# Access results
print("\n[5/5] Extracting results...")
poses = result.get_im_poses()      # [N, 4, 4] camera-to-world matrices
focals = result.get_focals()       # [N] focal lengths
pts3d, depths, confs = result.get_dense_pts3d()  # Dense reconstruction
print(f"✓ Extracted poses shape: {poses.shape}")
print(f"✓ Extracted focals shape: {focals.shape}")
if isinstance(pts3d, list):
    print(f"✓ Extracted {len(pts3d)} point clouds (list format)")
else:
    print(f"✓ Extracted pts3d shape: {pts3d.shape}")

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Visualization
print(f"\n[6/6] Creating 3D visualization for image index {VISUALIZATION_IMAGE_IDX}...")
print(f"Output directory: {OUTPUT_DIR}")
img_idx = VISUALIZATION_IMAGE_IDX
if img_idx >= len(pts3d):
    print(f"⚠️  Image index {img_idx} out of range (0-{len(pts3d)-1}). Using index 0.")
    img_idx = 0

# Extract point cloud and confidence
pts = pts3d[img_idx]
conf = confs[img_idx]

# Convert to numpy if needed
if hasattr(pts, 'numpy'):
    pts_np = pts.numpy()
    conf_np = conf.numpy()
else:
    pts_np = np.array(pts)
    conf_np = np.array(conf)

# Flatten if needed
if len(pts_np.shape) > 2:
    pts_flat = pts_np.reshape(-1, 3)
    conf_flat = conf_np.flatten()
else:
    pts_flat = pts_np
    conf_flat = conf_np.flatten() if len(conf_np.shape) > 0 else conf_np

# Filter out points with very low confidence or NaN values
valid_mask = (conf_flat > 0.1) & np.isfinite(pts_flat[:, 0]) & np.isfinite(pts_flat[:, 1]) & np.isfinite(pts_flat[:, 2])
pts_filtered = pts_flat[valid_mask]
conf_filtered = conf_flat[valid_mask]

print(f"✓ Filtered {len(pts_filtered)} valid points from {len(pts_flat)} total points")

# Get focal length
focal = float(focals[img_idx])

# Create 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot point cloud
if len(pts_filtered) > 0:
    scatter = ax.scatter(pts_filtered[:, 0], pts_filtered[:, 1], pts_filtered[:, 2],
                        c=conf_filtered, cmap='viridis', s=1, alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='Confidence')
    print(f"✓ Plotted point cloud")

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'3D Point Cloud: Image {img_idx} ({images[img_idx]})\nValid Points: {len(pts_filtered)} | Focal: {focal:.1f}px')

# Set equal aspect ratio
max_range = np.array([pts_flat[:, 0].max()-pts_flat[:, 0].min(),
                      pts_flat[:, 1].max()-pts_flat[:, 1].min(),
                      pts_flat[:, 2].max()-pts_flat[:, 2].min()]).max() / 2.0
mid_x = (pts_flat[:, 0].max()+pts_flat[:, 0].min()) * 0.5
mid_y = (pts_flat[:, 1].max()+pts_flat[:, 1].min()) * 0.5
mid_z = (pts_flat[:, 2].max()+pts_flat[:, 2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
output_path = Path(OUTPUT_DIR) / 'visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to {output_path}")
plt.show()

print("\n✅ Processing complete!")
