from mlx_mast3r import DUNE
from PIL import Image, ImageDraw
import numpy as np
import os
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
RESTRICT_TO_LOWER_HALF = True  # Toggle to restrict matching to lower half of images
# ============================================================================

# Load encoder
print("Loading encoder...")
encoder = DUNE.from_pretrained("base", resolution=336)

# Load images
image1_path = "data/incoming/images/frame_0001.jpg"
image2_path = "data/incoming/images/frame_0005.jpg"

print("Loading images...")
image1_array = np.array(Image.open(image1_path))
image2_array = np.array(Image.open(image2_path))
image1_pil = Image.open(image1_path)
image2_pil = Image.open(image2_path)

# Extract features from both images
print("Extracting features from Image 1...")
features1 = encoder.encode(image1_array)  # [N1, 768]
print("Extracting features from Image 2...")
features2 = encoder.encode(image2_array)  # [N2, 768]

print(f"Image 1 features shape: {features1.shape}")
print(f"Image 2 features shape: {features2.shape}")

# Normalize features for cosine similarity
print("Normalizing features...")
features1_normalized = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
features2_normalized = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)

# Compute cosine similarity between all pairs using matrix multiplication
print("Computing cosine similarity matrix...")
similarity_matrix = features1_normalized @ features2_normalized.T  # [N1, N2]

# Find K=25 most similar patch pairs (globally across all patches)
K = 3
print(f"Finding top {K} global matches...")

# Helper to determine if patch is in lower half
def is_in_lower_half(patch_idx, num_features, image_pil):
    """Check if patch is in the lower half of the image."""
    h = image_pil.size[1]
    stride = 8
    grid_h = h // stride
    grid_w = (num_features + (num_features // grid_h) - 1) // (num_features // grid_h)

    row = patch_idx // grid_w
    # Lower half starts at row >= grid_h / 2
    return row >= grid_h / 2

# Create a mask for valid matches if restricting to lower half
if RESTRICT_TO_LOWER_HALF:
    print("Restricting matches to lower half of images...")
    valid_mask = np.zeros_like(similarity_matrix, dtype=bool)

    for i in range(similarity_matrix.shape[0]):
        if is_in_lower_half(i, features1.shape[0], image1_pil):
            for j in range(similarity_matrix.shape[1]):
                if is_in_lower_half(j, features2.shape[0], image2_pil):
                    valid_mask[i, j] = True

    # Apply mask by setting invalid similarities to very low value
    masked_similarity = similarity_matrix.copy()
    masked_similarity[~valid_mask] = -np.inf
else:
    masked_similarity = similarity_matrix

# Flatten similarity matrix and find top K matches
similarity_flat = masked_similarity.flatten()
top_k_indices_flat = np.argsort(similarity_flat)[-K:][::-1]

matches = []
for flat_idx in top_k_indices_flat:
    patch_idx1 = flat_idx // masked_similarity.shape[1]
    patch_idx2 = flat_idx % masked_similarity.shape[1]

    # Skip if similarity is invalid (lower than -inf threshold)
    if masked_similarity[patch_idx1, patch_idx2] == -np.inf:
        continue

    matches.append({
        'patch1_idx': patch_idx1,
        'patch2_idx': patch_idx2,
        'similarity': similarity_matrix[patch_idx1, patch_idx2]
    })

print(f"\nTop {K} matches (by cosine similarity):")
if RESTRICT_TO_LOWER_HALF:
    print("(Restricted to lower half of images)")

if len(matches) == 0:
    print("WARNING: No matches found! Check restrictions and similarity matrix.")
else:
    for i, match in enumerate(matches):
        print(f"  Match {i+1}: Patch1[{match['patch1_idx']}] <-> Patch2[{match['patch2_idx']}] (similarity: {match['similarity']:.4f})")

# Debug: Check match distribution
patch1_distribution = {}
for match in matches:
    p1 = match['patch1_idx']
    patch1_distribution[p1] = patch1_distribution.get(p1, 0) + 1

print(f"\nDebug - Found {len(matches)} total matches")
if len(matches) > 0:
    print(f"Debug - Patch1 index distribution:")
    for p1_idx in sorted(patch1_distribution.keys()):
        print(f"  Patch1[{p1_idx}]: {patch1_distribution[p1_idx]} matches")

# Helper function to get patch coordinates
# NOTE: The DUNE encoder produces features at a specific stride/grid resolution.
# We estimate patch size from encoder resolution and distribute patches uniformly.
def get_patch_coordinates(image_pil, num_features, encoder_resolution=336):
    """
    Map feature indices to patch coordinates on the image.
    Assumes features are extracted on a regular grid at encoder resolution.
    """
    h, w = image_pil.size[::-1]  # Get height and width (H, W)

    # Estimate grid dimensions based on typical encoder output
    # DUNE typically uses patches at multiple scales, we'll use a fixed stride
    stride = 8  # Common stride for vision transformers
    grid_h = h // stride
    grid_w = w // stride

    # Clamp to actual number of features
    total_grid_cells = grid_h * grid_w
    if num_features < total_grid_cells:
        # Features are sparse, distribute proportionally
        grid_h = int(np.sqrt(num_features))
        grid_w = (num_features + grid_h - 1) // grid_h
        patch_h = h / grid_h
        patch_w = w / grid_w
    else:
        patch_h = stride
        patch_w = stride

    def idx_to_coords(idx):
        row = idx // grid_w
        col = idx % grid_w
        x_min = int(col * patch_w)
        y_min = int(row * patch_h)
        x_max = min(int((col + 1) * patch_w), w)
        y_max = min(int((row + 1) * patch_h), h)
        return (x_min, y_min, x_max, y_max), (x_min + x_max) / 2, (y_min + y_max) / 2

    return idx_to_coords

# Create coordinate functions for both images
get_coords1 = get_patch_coordinates(image1_pil, features1.shape[0])
get_coords2 = get_patch_coordinates(image2_pil, features2.shape[0])

# Create side-by-side visualization
combined_width = image1_pil.width + image2_pil.width
combined_height = max(image1_pil.height, image2_pil.height)
combined_image = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))

# Paste images side by side
combined_image.paste(image1_pil, (0, 0))
combined_image.paste(image2_pil, (image1_pil.width, 0))

# Create drawing context
draw = ImageDraw.Draw(combined_image, "RGBA")

# Color palette for K matches
colors = [
    (255, 0, 0, 200),    # Red
    (0, 255, 0, 200),    # Green
    (0, 0, 255, 200),    # Blue
]

# Draw grid lines for visualization (only for matched patches)
if len(matches) > 0:
    print("Drawing grid lines for matched patches...")

    # Collect unique patches that are matched
    matched_patch1_indices = set(match['patch1_idx'] for match in matches)
    matched_patch2_indices = set(match['patch2_idx'] for match in matches)

    print(f"Debug - Unique patches in image 1: {len(matched_patch1_indices)}")
    print(f"Debug - Unique patches in image 2: {len(matched_patch2_indices)}")

    # Draw grid lines for matched patches in image 1
    for patch_idx in tqdm(matched_patch1_indices, desc="Grid 1 (matches)"):
        coords, _, _ = get_coords1(patch_idx)
        x_min, y_min, x_max, y_max = coords
        print(f"Debug - Grid1 Patch {patch_idx}: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        draw.rectangle([x_min, y_min, x_max, y_max], outline=(200, 200, 200, 100), width=1)

    # Draw grid lines for matched patches in image 2
    for patch_idx in tqdm(matched_patch2_indices, desc="Grid 2 (matches)"):
        coords, _, _ = get_coords2(patch_idx)
        x_min, y_min, x_max, y_max = coords
        print(f"Debug - Grid2 Patch {patch_idx}: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        draw.rectangle(
            [x_min + image1_pil.width, y_min, x_max + image1_pil.width, y_max],
            outline=(200, 200, 200, 100),
            width=1
        )

    # Draw matches
    print("Drawing match annotations...")
    for match_idx, match in enumerate(tqdm(matches, desc="Drawing matches")):
        color = colors[match_idx % len(colors)]
        patch1_idx = match['patch1_idx']
        patch2_idx = match['patch2_idx']

        # Get patch coordinates
        coords1, cx1, cy1 = get_coords1(patch1_idx)
        coords2, cx2, cy2 = get_coords2(patch2_idx)

        # Draw highlighted patch boundaries
        x_min1, y_min1, x_max1, y_max1 = coords1
        x_min2, y_min2, x_max2, y_max2 = coords2

        print(f"Debug - Match {match_idx+1}:")
        print(f"  Image1 patch {patch1_idx}: ({x_min1}, {y_min1})-({x_max1}, {y_max1}) center:({cx1}, {cy1})")
        print(f"  Image2 patch {patch2_idx}: ({x_min2}, {y_min2})-({x_max2}, {y_max2}) center:({cx2}, {cy2})")
        print(f"  Color: {color}")

        draw.rectangle([x_min1, y_min1, x_max1, y_max1], outline=color, width=3)
        draw.rectangle(
            [x_min2 + image1_pil.width, y_min2, x_max2 + image1_pil.width, y_max2],
            outline=color,
            width=3
        )

        # Draw connecting line
        x2_offset = cx2 + image1_pil.width
        print(f"  Line from ({cx1}, {cy1}) to ({x2_offset}, {cy2})")
        draw.line(
            [(cx1, cy1), (x2_offset, cy2)],
            fill=color,
            width=2
        )
else:
    print("Skipping drawing - no matches to annotate")

# Save combined image
print("Saving visualization...")
output_dir = "data/intermediates/patch_matching"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "patch_matching_comparison.jpg")
combined_image.save(output_path, quality=95)

print(f"\n✓ Visualization saved to {output_path}")