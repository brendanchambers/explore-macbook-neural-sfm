from mlx_mast3r import DUNE
from PIL import Image, ImageDraw
import numpy as np
import os
from tqdm import tqdm
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
RESTRICT_TO_LOWER_HALF = False  # Toggle to restrict matching to lower half of images
K = 25  # Number of top matches to find

image1_path = "data/incoming/images/frame_0067.jpg"
image2_path = "data/incoming/images/frame_0075.jpg"

np.show_config()
# ============================================================================

# Timing tracking
timings = {}

def log_step(step_name):
    """Decorator to time a step"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            print(f"\n[STEP] {step_name}...", flush=True)
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            timings[step_name] = elapsed
            print(f"  ✓ Completed in {elapsed:.2f}s", flush=True)
            return result
        return wrapper
    return decorator

# ============================================================================
# STEP 1: Load encoder
# ============================================================================
@log_step("1. Load encoder")
def load_encoder():
    return DUNE.from_pretrained("base", resolution=336)

encoder = load_encoder()

# ============================================================================
# STEP 2: Load images
# ============================================================================
@log_step("2. Load images")
def load_images():
    image1_array = np.array(Image.open(image1_path))
    image2_array = np.array(Image.open(image2_path))
    image1_pil = Image.open(image1_path)
    image2_pil = Image.open(image2_path)
    print(f"  Image 1 shape: {image1_array.shape}")
    print(f"  Image 2 shape: {image2_array.shape}")
    return image1_array, image2_array, image1_pil, image2_pil

image1_array, image2_array, image1_pil, image2_pil = load_images()

# ============================================================================
# STEP 3: Extract features
# ============================================================================
@log_step("3. Extract features from both images")
def extract_features():
    print("  Encoding Image 1...")
    features1 = encoder.encode(image1_array).astype(np.float32)  # [N1, 768]
    print(f"    Features shape: {features1.shape}")
    print("  Encoding Image 2...")
    features2 = encoder.encode(image2_array).astype(np.float32)  # [N2, 768]
    print(f"    Features shape: {features2.shape}")
    return features1, features2

features1, features2 = extract_features()

# ============================================================================
# STEP 4: Normalize features
# ============================================================================
@log_step("4. Normalize features for cosine similarity")
def normalize_features():
    f1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
    f2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
    return f1, f2

features1_normalized, features2_normalized = normalize_features()

# ============================================================================
# STEP 5: Compute similarity matrix
# ============================================================================
@log_step("5. Compute cosine similarity matrix")
def compute_similarity():
    matrix = features1_normalized @ features2_normalized.T  # [N1, N2]
    print(f"  Similarity matrix shape: {matrix.shape}")
    return matrix

similarity_matrix = compute_similarity()

# ============================================================================
# STEP 6: Create patch coordinate system
# ============================================================================

# Helper function to get patch coordinates
def get_patch_coordinates(image_pil, num_features, encoder_resolution=336):
    """Map feature indices to patch coordinates on the image."""
    h, w = image_pil.size[::-1]  # Get height and width (H, W)

    # Estimate grid dimensions based on typical encoder output
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

def is_in_lower_half(patch_idx, get_coords_fn, image_pil):
    """Check if patch is in the lower half of the image."""
    _, _, cy = get_coords_fn(patch_idx)
    h = image_pil.size[1]
    return cy > h / 2

# ============================================================================
# STEP 7: Find top K matches
# ============================================================================
@log_step("6. Create patch coordinate system and find top K matches")
def find_matches():
    get_coords1 = get_patch_coordinates(image1_pil, features1.shape[0])
    get_coords2 = get_patch_coordinates(image2_pil, features2.shape[0])

    if RESTRICT_TO_LOWER_HALF:
        print("  Applying lower-half restriction...")
        valid_mask = np.zeros_like(similarity_matrix, dtype=bool)

        for i in range(similarity_matrix.shape[0]):
            if is_in_lower_half(i, get_coords1, image1_pil):
                for j in range(similarity_matrix.shape[1]):
                    if is_in_lower_half(j, get_coords2, image2_pil):
                        valid_mask[i, j] = True

        masked_similarity = similarity_matrix.copy()
        masked_similarity[~valid_mask] = -np.inf
        print(f"  Valid matches in restricted region: {np.sum(valid_mask)}")
    else:
        masked_similarity = similarity_matrix

    # Find top K matches using argpartition (O(n) instead of O(n log n))
    similarity_flat = masked_similarity.flatten()
    # Use argpartition to find the K largest elements efficiently
    partitioned_indices = np.argpartition(similarity_flat, -K)[-K:]
    # Sort the K largest elements to get them in descending order
    top_k_indices_flat = partitioned_indices[np.argsort(similarity_flat[partitioned_indices])[::-1]]

    matches = []
    for flat_idx in top_k_indices_flat:
        patch_idx1 = flat_idx // masked_similarity.shape[1]
        patch_idx2 = flat_idx % masked_similarity.shape[1]

        if masked_similarity[patch_idx1, patch_idx2] == -np.inf:
            continue

        matches.append({
            'patch1_idx': patch_idx1,
            'patch2_idx': patch_idx2,
            'similarity': similarity_matrix[patch_idx1, patch_idx2]
        })

    print(f"  Found {len(matches)} matches:")
    if len(matches) == 0:
        print("    ⚠ WARNING: No matches found!")
    else:
        for i, match in enumerate(matches):
            print(f"    Match {i+1}: Patch1[{match['patch1_idx']}] ↔ Patch2[{match['patch2_idx']}] (sim: {match['similarity']:.4f})")

    return matches, get_coords1, get_coords2

matches, get_coords1, get_coords2 = find_matches()

# ============================================================================
# STEP 8: Create visualization
# ============================================================================
@log_step("7. Create side-by-side visualization")
def create_visualization():
    # Setup canvas
    combined_width = image1_pil.width + image2_pil.width
    combined_height = max(image1_pil.height, image2_pil.height)
    combined_image = Image.new("RGBA", (combined_width, combined_height), (255, 255, 255, 255))

    # Convert images to RGBA with 60% opacity
    image1_rgba = image1_pil.convert("RGBA")
    image2_rgba = image2_pil.convert("RGBA")

    alpha1 = image1_rgba.split()[3]
    alpha1 = alpha1.point(lambda p: int(p * 0.6))
    image1_rgba.putalpha(alpha1)

    alpha2 = image2_rgba.split()[3]
    alpha2 = alpha2.point(lambda p: int(p * 0.6))
    image2_rgba.putalpha(alpha2)

    # Paste images side by side
    combined_image.paste(image1_rgba, (0, 0), image1_rgba)
    combined_image.paste(image2_rgba, (image1_pil.width, 0), image2_rgba)

    draw = ImageDraw.Draw(combined_image, "RGBA")

    # Color palette
    colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]

    if len(matches) > 0:
        # Draw matched patch boundaries
        matched1 = set(m['patch1_idx'] for m in matches)
        matched2 = set(m['patch2_idx'] for m in matches)

        for patch_idx in matched1:
            coords, _, _ = get_coords1(patch_idx)
            x_min, y_min, x_max, y_max = coords
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(200, 200, 200, 100), width=1)

        for patch_idx in matched2:
            coords, _, _ = get_coords2(patch_idx)
            x_min, y_min, x_max, y_max = coords
            draw.rectangle(
                [x_min + image1_pil.width, y_min, x_max + image1_pil.width, y_max],
                outline=(200, 200, 200, 100),
                width=1
            )

        # Draw match annotations
        for match_idx, match in enumerate(matches):
            color = colors[match_idx % len(colors)]
            patch1_idx = match['patch1_idx']
            patch2_idx = match['patch2_idx']

            coords1, cx1, cy1 = get_coords1(patch1_idx)
            coords2, cx2, cy2 = get_coords2(patch2_idx)

            x_min1, y_min1, x_max1, y_max1 = coords1
            x_min2, y_min2, x_max2, y_max2 = coords2

            # Draw circles and rectangles
            circle_radius = 40
            circle_color = (*color[:3], 10)

            draw.ellipse(
                [cx1 - circle_radius, cy1 - circle_radius, cx1 + circle_radius, cy1 + circle_radius],
                fill=circle_color,
                outline=color,
                width=2
            )

            draw.ellipse(
                [cx2 + image1_pil.width - circle_radius, cy2 - circle_radius,
                 cx2 + image1_pil.width + circle_radius, cy2 + circle_radius],
                fill=circle_color,
                outline=color,
                width=2
            )

            draw.rectangle([x_min1, y_min1, x_max1, y_max1], outline=color, width=4)
            draw.rectangle(
                [x_min2 + image1_pil.width, y_min2, x_max2 + image1_pil.width, y_max2],
                outline=color,
                width=4
            )

            # Connecting line
            draw.line(
                [(cx1, cy1), (cx2 + image1_pil.width, cy2)],
                fill=color,
                width=4
            )

    # Draw separator line
    separator_x = image1_pil.width
    dash_length = 15
    dash_spacing = 10
    y = 0
    while y < combined_height:
        draw.line(
            [(separator_x, y), (separator_x, y + dash_length)],
            fill=(0, 0, 0, 255),
            width=2
        )
        y += dash_length + dash_spacing

    return combined_image

combined_image = create_visualization()

# ============================================================================
# STEP 9: Save visualization
# ============================================================================
@log_step("8. Save visualization")
def save_visualization():
    output_dir = "data/intermediates/patch_matching"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "patch_matching_comparison.jpg")

    rgb_image = Image.new("RGB", combined_image.size, (255, 255, 255))
    rgb_image.paste(combined_image, mask=combined_image.split()[3])
    rgb_image.save(output_path, quality=95)

    return output_path

output_path = save_visualization()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TIMING SUMMARY")
print("="*70)

sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
total_time = sum(timings.values())

for step, elapsed in sorted_timings:
    percentage = (elapsed / total_time) * 100
    print(f"{step:<50} {elapsed:>8.2f}s ({percentage:>5.1f}%)")

print("-" * 70)
print(f"{'TOTAL':<50} {total_time:>8.2f}s (100.0%)")
print("="*70)

slowest = sorted_timings[0]
print(f"\n🐌 Slowest step: {slowest[0]} ({slowest[1]:.2f}s)")
print(f"✓ Visualization saved to {output_path}")