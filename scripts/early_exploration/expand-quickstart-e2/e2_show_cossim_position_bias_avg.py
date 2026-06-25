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

image_dir_for_position_debiasing = "data/incoming/images"  # Directory to load images from for feature preprocessing
N_image_pairs = 3  # Number of image pairs to sample for computing average cosine similarity
P_sample_image = 0.1  # Probability (10%) to encode features for each image for faster iteration

image1_path = "data/incoming/images/frame_0067.jpg"
image2_path = "data/incoming/images/frame_0075.jpg"

# Output directory for all generated files
output_dir = "data/intermediates/patch_matching/position_bias_average"

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
# STEP 2: Feature preprocessing - compute average cosine similarity map
# ============================================================================
@log_step("2. Feature preprocessing - load images and compute average cosine similarity")
def compute_position_debiasing_map():
    """
    Load all images from the position debiasing directory, encode their features,
    and compute average cosine similarity across sampled image pairs.
    """
    # Get list of images from directory
    image_files = sorted([f for f in os.listdir(image_dir_for_position_debiasing)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  Found {len(image_files)} images in {image_dir_for_position_debiasing}")

    if len(image_files) == 0:
        print("  ⚠ WARNING: No images found in directory, skipping preprocessing")
        return None

    # Load and encode features for sampled images (with probability P_sample_image)
    print(f"  Loading and encoding features for sampled images (P_sample_image={P_sample_image})...")
    image_features_dict = {}
    np.random.seed(42)  # For reproducibility
    for img_file in tqdm(image_files, desc="  Encoding images"):
        # Sample with probability P_sample_image
        if np.random.random() >= P_sample_image:
            continue

        img_path = os.path.join(image_dir_for_position_debiasing, img_file)
        try:
            img_array = np.array(Image.open(img_path))
            features = encoder.encode(img_array).astype(np.float32)
            image_features_dict[img_file] = features
            print(f"    {img_file}: features shape {features.shape}")
        except Exception as e:
            print(f"    ⚠ Failed to load {img_file}: {e}")

    if len(image_features_dict) == 0:
        print("  ⚠ WARNING: No images were sampled/loaded, skipping preprocessing")
        return None

    print(f"  Successfully encoded {len(image_features_dict)}/{len(image_files)} sampled images (expected ~{int(len(image_files)*P_sample_image)} at P_sample_image={P_sample_image})")

    # Sample image pairs
    image_names = list(image_features_dict.keys())
    n_pairs_to_sample = min(N_image_pairs, len(image_names) * (len(image_names) - 1) // 2)

    print(f"  Sampling {n_pairs_to_sample} image pairs...")
    np.random.seed(42)  # For reproducibility
    sampled_pairs = []
    for _ in range(n_pairs_to_sample):
        idx1, idx2 = np.random.choice(len(image_names), size=2, replace=False)
        sampled_pairs.append((image_names[idx1], image_names[idx2]))

    # Compute average cosine similarity map
    print(f"  Computing cosine similarity for sampled pairs...")
    accumulated_similarity = None

    for pair_idx, (img1_name, img2_name) in enumerate(sampled_pairs):
        features1 = image_features_dict[img1_name]
        features2 = image_features_dict[img2_name]

        # Normalize features
        f1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
        f2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity matrix [N1, N2]
        similarity_matrix_pair = f1 @ f2.T

        if accumulated_similarity is None:
            accumulated_similarity = similarity_matrix_pair
        else:
            # Accumulate - need to handle different shapes by averaging columns
            # Take mean along dimension that varies
            mean_sim_pair = np.mean(similarity_matrix_pair, axis=0, keepdims=True)
            if accumulated_similarity.shape != similarity_matrix_pair.shape:
                print(f"    Pair {pair_idx+1}: shapes {accumulated_similarity.shape} vs {similarity_matrix_pair.shape}")
                # Average similarity scores across features
                accumulated_similarity = accumulated_similarity + np.mean(similarity_matrix_pair)
            else:
                accumulated_similarity = accumulated_similarity + similarity_matrix_pair

        print(f"    Pair {pair_idx+1} ({img1_name} ↔ {img2_name}): similarity shape {similarity_matrix_pair.shape}")

    # Average across sampled pairs
    if accumulated_similarity is not None:
        avg_similarity_map = accumulated_similarity / n_pairs_to_sample
        print(f"  Average cosine similarity map shape: {avg_similarity_map.shape if isinstance(avg_similarity_map, np.ndarray) else 'scalar'}")
        if isinstance(avg_similarity_map, np.ndarray):
            print(f"    Mean: {np.mean(avg_similarity_map):.4f}")
            print(f"    Std: {np.std(avg_similarity_map):.4f}")
            print(f"    Min: {np.min(avg_similarity_map):.4f}")
            print(f"    Max: {np.max(avg_similarity_map):.4f}")
    else:
        avg_similarity_map = None

    return image_features_dict, avg_similarity_map

image_features_dict, avg_position_debiasing_map = compute_position_debiasing_map()

# ============================================================================
# STEP 4: Load images
# ============================================================================
@log_step("3. Load images")
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
# STEP 5: Extract features
# ============================================================================
@log_step("4. Extract features from both images")
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
# STEP 6: Normalize features
# ============================================================================
# @log_step("5. Normalize features for cosine similarity")
# def normalize_features():
#     f1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
#     f2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
#     return f1, f2

# features1_normalized, features2_normalized = normalize_features()

# ============================================================================
# STEP 7: Compute similarity matrix
# ============================================================================
@log_step("6. Compute cosine similarity matrix")
def compute_similarity():
    matrix = features1 @ features2.T  # [N1, N2]
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
# STEP 8: Find top K matches
# ============================================================================
@log_step("7. Create patch coordinate system and find top K matches")
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
# STEP 9: Create visualization
# ============================================================================
@log_step("8. Create side-by-side visualization")
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

    # Generate unique random colors for each match
    np.random.seed(42)  # For reproducibility
    colors = []
    for _ in range(len(matches)):
        # Generate bright, saturated colors
        color = (
            int(np.random.uniform(100, 255)),
            int(np.random.uniform(100, 255)),
            int(np.random.uniform(100, 255)),
            255
        )
        colors.append(color)

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
            color = colors[match_idx]
            patch1_idx = match['patch1_idx']
            patch2_idx = match['patch2_idx']

            coords1, cx1, cy1 = get_coords1(patch1_idx)
            coords2, cx2, cy2 = get_coords2(patch2_idx)

            x_min1, y_min1, x_max1, y_max1 = coords1
            x_min2, y_min2, x_max2, y_max2 = coords2

            # Draw rectangles around patches
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
# STEP 10: Save visualization
# ============================================================================
@log_step("9. Save visualization")
def save_visualization():
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "patch_matching_comparison.jpg")

    rgb_image = Image.new("RGB", combined_image.size, (255, 255, 255))
    rgb_image.paste(combined_image, mask=combined_image.split()[3])
    rgb_image.save(output_path, quality=95)

    return output_path

output_path = save_visualization()

# ============================================================================
# STEP 11: Create histogram of cosine similarity by patch index
# ============================================================================
@log_step("10. Create histogram of cosine similarity by patch index")
def create_similarity_histogram():
    import matplotlib.pyplot as plt

    # Check if avg map is available
    if avg_position_debiasing_map is None:
        print("  ⚠ Skipping: average cosine similarity map not available")
        return None

    # Flatten average similarity matrix and get indices
    similarity_flat = avg_position_debiasing_map.flatten()
    patch_indices = np.arange(len(similarity_flat))

    # Create figure with histogram
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create step histogram
    ax.hist(similarity_flat, bins=100, histtype='step', linewidth=2, color='steelblue', edgecolor='steelblue')

    ax.set_xlabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Cosine Similarity Across All Patch Pairs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add statistics text box
    mean_sim = np.mean(similarity_flat)
    median_sim = np.median(similarity_flat)
    std_sim = np.std(similarity_flat)
    max_sim = np.max(similarity_flat)
    min_sim = np.min(similarity_flat)

    stats_text = f'Mean: {mean_sim:.4f}\nMedian: {median_sim:.4f}\nStd: {std_sim:.4f}\nMax: {max_sim:.4f}\nMin: {min_sim:.4f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save histogram
    os.makedirs(output_dir, exist_ok=True)
    histogram_path = os.path.join(output_dir, "cosine_similarity_histogram.png")
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Statistics:")
    print(f"    Mean similarity: {mean_sim:.4f}")
    print(f"    Median similarity: {median_sim:.4f}")
    print(f"    Std deviation: {std_sim:.4f}")
    print(f"    Max similarity: {max_sim:.4f}")
    print(f"    Min similarity: {min_sim:.4f}")

    return histogram_path

histogram_path = create_similarity_histogram()

# ============================================================================
# STEP 12: Create merged scatter plot (top p, bottom p, random p)
# ============================================================================
@log_step("11. Create merged scatter plot of top/bottom/random p on single axis")
def create_merged_scatter_plot():
    import matplotlib.pyplot as plt

    # Check if avg map is available
    if avg_position_debiasing_map is None:
        print("  ⚠ Skipping: average cosine similarity map not available")
        return None

    # Flatten average similarity matrix and get indices
    similarity_flat = avg_position_debiasing_map.flatten()
    patch_indices = np.arange(len(similarity_flat))

    # Prepare datasets for top p (99.9th percentile)
    p_top_percentile = 99.9
    p_top_threshold = np.percentile(similarity_flat, p_top_percentile)
    top_p_mask = similarity_flat >= p_top_threshold
    top_indices = patch_indices[top_p_mask]
    top_similarities = similarity_flat[top_p_mask]

    # Prepare datasets for bottom p (0.01th percentile)
    p_bottom_percentile = 0.01
    p_bottom_threshold = np.percentile(similarity_flat, p_bottom_percentile)
    bottom_p_mask = similarity_flat <= p_bottom_threshold
    bottom_indices = patch_indices[bottom_p_mask]
    bottom_similarities = similarity_flat[bottom_p_mask]

    # Prepare datasets for random p (0.0001 fraction)
    sample_fraction = 0.0001
    sample_size = int(len(similarity_flat) * sample_fraction)
    np.random.seed(42)  # For reproducibility
    random_sample_indices = np.random.choice(len(similarity_flat), size=sample_size, replace=False)
    random_indices = patch_indices[random_sample_indices]
    random_similarities = similarity_flat[random_sample_indices]

    # Create single figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot all three groups on single axis with different colors
    ax.plot(top_indices, top_similarities, '.', markersize=2, alpha=0.6, color='darkred', label=f'Top 0.1% (n={len(top_similarities)})')
    ax.plot(bottom_indices, bottom_similarities, '.', markersize=2, alpha=0.6, color='darkblue', label=f'Bottom 0.01% (n={len(bottom_similarities)})')
    ax.plot(random_indices, random_similarities, '.', markersize=2, alpha=0.6, color='darkgreen', label=f'Random 0.01% (n={len(random_similarities)})')

    ax.set_xlabel('Flattened Patch Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Cosine Similarity: Top vs Bottom vs Random Samples', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # Save merged scatter plot
    os.makedirs(output_dir, exist_ok=True)
    merged_path = os.path.join(output_dir, "cosine_similarity_scatter_merged.png")
    plt.savefig(merged_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Merged scatter plot created on single axis")
    print(f"    Top p (red): {len(top_similarities)} points")
    print(f"    Bottom p (blue): {len(bottom_similarities)} points")
    print(f"    Random p (green): {len(random_similarities)} points")

    return merged_path

merged_scatter_path = create_merged_scatter_plot()

# ============================================================================
# STEP 13: Create merged plot showing cosine similarity vs y-axis patch position
# ============================================================================
@log_step("12. Create merged plot of cosine similarity vs y-axis patch position")
def create_merged_plot_y_position():
    import matplotlib.pyplot as plt

    # Check if avg map is available
    if avg_position_debiasing_map is None:
        print("  ⚠ Skipping: average cosine similarity map not available")
        return None

    # Flatten average similarity matrix and get indices
    similarity_flat = avg_position_debiasing_map.flatten()
    patch_indices = np.arange(len(similarity_flat))

    # Get patch coordinate functions for image1
    get_coords1 = get_patch_coordinates(image1_pil, features1.shape[0])

    # Prepare datasets for top p (99.9th percentile)
    p_top_percentile = 99.9
    p_top_threshold = np.percentile(similarity_flat, p_top_percentile)
    top_p_mask = similarity_flat >= p_top_threshold
    top_flat_indices = patch_indices[top_p_mask]
    top_similarities = similarity_flat[top_p_mask]

    # Prepare datasets for bottom p (0.01th percentile)
    p_bottom_percentile = 0.01
    p_bottom_threshold = np.percentile(similarity_flat, p_bottom_percentile)
    bottom_p_mask = similarity_flat <= p_bottom_threshold
    bottom_flat_indices = patch_indices[bottom_p_mask]
    bottom_similarities = similarity_flat[bottom_p_mask]

    # Prepare datasets for random p (0.0001 fraction)
    sample_fraction = 0.0001
    sample_size = int(len(similarity_flat) * sample_fraction)
    np.random.seed(42)  # For reproducibility
    random_sample_indices = np.random.choice(len(similarity_flat), size=sample_size, replace=False)
    random_flat_indices = patch_indices[random_sample_indices]
    random_similarities = similarity_flat[random_sample_indices]

    # Convert flattened indices to 2D patch indices and extract y-positions
    def get_y_positions(flat_indices):
        y_positions = []
        for flat_idx in flat_indices:
            patch_i = flat_idx // avg_position_debiasing_map.shape[1]
            _, _, cy = get_coords1(patch_i)
            y_positions.append(cy)
        return np.array(y_positions)

    top_y_positions = get_y_positions(top_flat_indices)
    bottom_y_positions = get_y_positions(bottom_flat_indices)
    random_y_positions = get_y_positions(random_flat_indices)

    # Create single figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot all three groups on single axis with different colors
    ax.plot(top_y_positions, top_similarities, '.', markersize=2, alpha=0.6, color='darkred', label=f'Top 0.1% (n={len(top_similarities)})')
    ax.plot(bottom_y_positions, bottom_similarities, '.', markersize=2, alpha=0.6, color='darkblue', label=f'Bottom 0.01% (n={len(bottom_similarities)})')
    ax.plot(random_y_positions, random_similarities, '.', markersize=2, alpha=0.6, color='darkgreen', label=f'Random 0.01% (n={len(random_similarities)})')

    ax.set_xlabel('Y-Axis Patch Position (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Cosine Similarity vs Y-Axis Patch Position (Image 1)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    y_pos_path = os.path.join(output_dir, "cosine_similarity_vs_y_position.png")
    plt.savefig(y_pos_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Y-position plot created")

    return y_pos_path

y_pos_path = create_merged_plot_y_position()

# ============================================================================
# STEP 14: Create merged plot showing cosine similarity vs x-axis patch position
# ============================================================================
@log_step("13. Create merged plot of cosine similarity vs x-axis patch position")
def create_merged_plot_x_position():
    import matplotlib.pyplot as plt

    # Check if avg map is available
    if avg_position_debiasing_map is None:
        print("  ⚠ Skipping: average cosine similarity map not available")
        return None

    # Flatten average similarity matrix and get indices
    similarity_flat = avg_position_debiasing_map.flatten()
    patch_indices = np.arange(len(similarity_flat))

    # Get patch coordinate functions for image1
    get_coords1 = get_patch_coordinates(image1_pil, features1.shape[0])

    # Prepare datasets for top p (99.9th percentile)
    p_top_percentile = 99.9
    p_top_threshold = np.percentile(similarity_flat, p_top_percentile)
    top_p_mask = similarity_flat >= p_top_threshold
    top_flat_indices = patch_indices[top_p_mask]
    top_similarities = similarity_flat[top_p_mask]

    # Prepare datasets for bottom p (0.01th percentile)
    p_bottom_percentile = 0.01
    p_bottom_threshold = np.percentile(similarity_flat, p_bottom_percentile)
    bottom_p_mask = similarity_flat <= p_bottom_threshold
    bottom_flat_indices = patch_indices[bottom_p_mask]
    bottom_similarities = similarity_flat[bottom_p_mask]

    # Prepare datasets for random p (0.0001 fraction)
    sample_fraction = 0.0001
    sample_size = int(len(similarity_flat) * sample_fraction)
    np.random.seed(42)  # For reproducibility
    random_sample_indices = np.random.choice(len(similarity_flat), size=sample_size, replace=False)
    random_flat_indices = patch_indices[random_sample_indices]
    random_similarities = similarity_flat[random_sample_indices]

    # Convert flattened indices to 2D patch indices and extract x-positions
    def get_x_positions(flat_indices):
        x_positions = []
        for flat_idx in flat_indices:
            patch_i = flat_idx // avg_position_debiasing_map.shape[1]
            _, cx, _ = get_coords1(patch_i)
            x_positions.append(cx)
        return np.array(x_positions)

    top_x_positions = get_x_positions(top_flat_indices)
    bottom_x_positions = get_x_positions(bottom_flat_indices)
    random_x_positions = get_x_positions(random_flat_indices)

    # Create single figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot all three groups on single axis with different colors
    ax.plot(top_x_positions, top_similarities, '.', markersize=2, alpha=0.6, color='darkred', label=f'Top 0.1% (n={len(top_similarities)})')
    ax.plot(bottom_x_positions, bottom_similarities, '.', markersize=2, alpha=0.6, color='darkblue', label=f'Bottom 0.01% (n={len(bottom_similarities)})')
    ax.plot(random_x_positions, random_similarities, '.', markersize=2, alpha=0.6, color='darkgreen', label=f'Random 0.01% (n={len(random_similarities)})')

    ax.set_xlabel('X-Axis Patch Position (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Cosine Similarity vs X-Axis Patch Position (Image 1)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    x_pos_path = os.path.join(output_dir, "cosine_similarity_vs_x_position.png")
    plt.savefig(x_pos_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  X-position plot created")

    return x_pos_path

x_pos_path = create_merged_plot_x_position()

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