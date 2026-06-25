#!/usr/bin/env python3
"""
Visualize C-RADIOv4 learned absolute positional embeddings.

1. Load C-RADIOv4 model using HuggingFace pretrained weights
2. Extract pos_embed from the model
3. Plot magnitude of each embedding vector
4. Compute outer product (cosine similarity) of patch embeddings
5. Visualize both as 2D images

Usage:
    uv run python scripts/visualize_cradio_pos_embeddings.py --output-dir data/intermediates/position_embeddings/explore_C_RADIO_V4/
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
import torch


def load_cradio_model(model_name: str = "nvidia/C-RADIOv4-H"):
    """Load C-RADIOv4 model using HuggingFace pretrained weights.

    Args:
        model_name: HuggingFace model identifier (e.g., "nvidia/C-RADIOv4-H" or "nvidia/C-RADIOv4-SO400M")

    Returns:
        (pos_embed array, model)
    """
    from transformers import AutoModel

    print(f"Loading {model_name}...")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # C-RADIO stores position embeddings in the patch generator
    # Structure: model.radio_model.model.patch_generator.pos_embed
    pos_embed = model.radio_model.model.patch_generator.pos_embed

    # Convert to numpy if needed
    if isinstance(pos_embed, torch.Tensor):
        pos_embed = pos_embed.detach().cpu().numpy()

    print(f"Successfully extracted position embeddings with shape: {pos_embed.shape}")
    return pos_embed, model


def visualize_magnitude(pos_embed: np.ndarray, output_path: str | Path = None, max_tokens_for_histogram: int = 10000):
    """Plot magnitude of each embedding vector.

    Args:
        pos_embed: [1, num_tokens, embed_dim] or [num_tokens, embed_dim]
        output_path: Optional path to save figure
        max_tokens_for_histogram: Max tokens to plot in histogram (for visualization clarity)
    """
    # Handle different shapes
    if len(pos_embed.shape) == 3:
        embeddings = pos_embed[0]  # Remove batch dimension [num_tokens, embed_dim]
    else:
        embeddings = pos_embed  # Already [num_tokens, embed_dim]

    # Compute magnitude (L2 norm) for each token
    magnitudes = np.linalg.norm(embeddings, axis=1)  # [num_tokens]

    # Infer grid structure
    num_tokens = len(magnitudes)
    # C-RADIO doesn't use CLS token in pos_embed (all tokens are patches)
    has_cls = False
    num_patches = num_tokens

    try:
        side = int(np.sqrt(num_patches))
        if side * side != num_patches:
            # Not a perfect square, might be a different structure
            side = None
    except:
        side = None

    print(f"Position embeddings shape: {pos_embed.shape}")
    print(f"Number of tokens: {num_tokens}")
    if side is not None:
        print(f"  Grid structure: {side}x{side} patches")
    print(f"Magnitude stats:")
    print(f"  Min: {magnitudes.min():.4f}, Max: {magnitudes.max():.4f}, Mean: {magnitudes.mean():.4f}")
    print(f"  Std: {magnitudes.std():.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram of magnitudes (instead of bar chart for large num_tokens)
    ax = axes[0]
    ax.hist(magnitudes, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel("Embedding Magnitude (L2 Norm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of C-RADIOv4 Position Embedding Magnitudes")
    ax.grid(axis='y', alpha=0.3)

    # Add statistics text
    stats_text = f"Mean: {magnitudes.mean():.4f}\nStd: {magnitudes.std():.4f}\nMin: {magnitudes.min():.4f}\nMax: {magnitudes.max():.4f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: 2D heatmap of patch magnitudes (if grid structure detected)
    ax = axes[1]
    if side is not None:
        patch_magnitudes = magnitudes.reshape(side, side)
        im = ax.imshow(patch_magnitudes, cmap='viridis', aspect='auto')
        ax.set_xlabel("Width Index")
        ax.set_ylabel("Height Index")
        ax.set_title(f"Magnitude Heatmap ({side}x{side} patches)")
        plt.colorbar(im, ax=ax, label="Magnitude")
    else:
        ax.text(0.5, 0.5, "No square grid structure detected",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved magnitude visualization to {output_path}")
    plt.show()

    return embeddings, magnitudes, side


def compute_cosine_similarity_matrix(embeddings: np.ndarray, side: int, sample_size: int = None) -> np.ndarray:
    """Compute cosine similarity between all patch embeddings.

    Args:
        embeddings: [num_tokens, embed_dim]
        side: Side length of patch grid
        sample_size: If specified, only compute similarity for a sampled subset of patches

    Returns:
        cosine_sim: [side, side, side, side] 4D similarity tensor (or sampled version)
    """
    # For C-RADIO with 16384 tokens, computing full 16384x16384 matrix is expensive
    # We'll sample a smaller subset for computational efficiency
    patch_embeddings = embeddings  # All embeddings are patches (no CLS token)

    if sample_size is None:
        # Default to a manageable subset if the full matrix would be too large
        num_patches = len(patch_embeddings)
        if num_patches > 4096:  # 64x64 grid
            sample_size = 4096
            print(f"Large embedding set ({num_patches} tokens). Sampling {sample_size} tokens for similarity computation.")
        else:
            sample_size = num_patches

    if sample_size < len(patch_embeddings):
        # Sample indices
        np.random.seed(42)
        sample_indices = np.random.choice(len(patch_embeddings), sample_size, replace=False)
        patch_embeddings = patch_embeddings[sample_indices]
        sample_side = int(np.sqrt(sample_size))
    else:
        sample_side = side

    # Normalize for cosine similarity
    patch_embeddings_norm = patch_embeddings / (np.linalg.norm(patch_embeddings, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity via outer product
    cosine_sim = patch_embeddings_norm @ patch_embeddings_norm.T  # [sample_size, sample_size]

    return cosine_sim.reshape(sample_side, sample_side, sample_side, sample_side), sample_side


def visualize_cosine_similarity(cosine_sim_4d: np.ndarray, side: int, output_path: str | Path = None):
    """Visualize cosine similarity matrix.

    Args:
        cosine_sim_4d: [side, side, side, side] 4D similarity tensor
        side: Side length of patch grid
        output_path: Optional path to save figure
    """
    # Reshape to 2D for visualization: each patch's similarity to all other patches
    cosine_sim_2d = cosine_sim_4d.reshape(side * side, side * side)

    print(f"\nCosine Similarity Matrix stats:")
    print(f"  Shape: {cosine_sim_2d.shape}")
    print(f"  Min: {cosine_sim_2d.min():.4f}")
    print(f"  Max: {cosine_sim_2d.max():.4f}")
    print(f"  Mean: {cosine_sim_2d.mean():.4f}")
    print(f"  Diagonal (self-similarity): {np.diag(cosine_sim_2d).mean():.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Full similarity matrix
    ax = axes[0]
    im = ax.imshow(cosine_sim_2d, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel(f"Patch Index (flattened)")
    ax.set_ylabel(f"Patch Index (flattened)")
    ax.set_title("Cosine Similarity Between All Patches (Outer Product)")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    # Plot 2: Magnitude of similarity (for better visualization of structure)
    ax = axes[1]
    im2 = ax.imshow(np.abs(cosine_sim_2d), cmap='hot', aspect='auto')
    ax.set_xlabel(f"Patch Index (flattened)")
    ax.set_ylabel(f"Patch Index (flattened)")
    ax.set_title("Absolute Cosine Similarity (Magnitude)")
    plt.colorbar(im2, ax=ax, label="|Cosine Similarity|")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved cosine similarity visualization to {output_path}")
    plt.show()

    # Plot 3: Similarity from selected patches (corners and center)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    select_indices = [
        (0, 0),      # Top-left
        (0, side-1), # Top-right
        (side//2, side//2),  # Center
        (side-1, side-1),    # Bottom-right
    ]

    titles = ["Top-left corner", "Top-right corner", "Center", "Bottom-right corner"]

    for (i, j), ax, title in zip(select_indices, axes.flat, titles):
        patch_idx = i * side + j
        similarity_row = cosine_sim_2d[patch_idx, :].reshape(side, side)
        im = ax.imshow(similarity_row, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(f"{title}\n(patch at [{i}, {j}])")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if output_path:
        output_path_str = str(output_path).replace('.png', '_per_patch_corners.png')
        plt.savefig(output_path_str, dpi=150, bbox_inches='tight')
        print(f"Saved per-patch (corners) visualization to {output_path_str}")
    plt.show()


def visualize_cosine_similarity_random(cosine_sim_4d: np.ndarray, side: int, output_path: str | Path = None, seed: int = 42):
    """Visualize cosine similarity from 4 randomly selected patches.

    Args:
        cosine_sim_4d: [side, side, side, side] 4D similarity tensor
        side: Side length of patch grid
        output_path: Optional path to save figure
        seed: Random seed for reproducibility
    """
    # Reshape to 2D for visualization
    cosine_sim_2d = cosine_sim_4d.reshape(side * side, side * side)

    # Randomly select 4 patches
    np.random.seed(seed)
    num_patches = side * side
    selected_flat_indices = np.random.choice(num_patches, 4, replace=False)
    select_indices = [(idx // side, idx % side) for idx in selected_flat_indices]

    print(f"\nVisualizing cosine similarity from 4 random patches (seed={seed}):")
    print(f"  Selected patches (flat indices): {selected_flat_indices}")
    print(f"  Selected patches (grid coords): {select_indices}")

    # Plot: Similarity from random patches
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for (i, j), ax, flat_idx in zip(select_indices, axes.flat, selected_flat_indices):
        similarity_row = cosine_sim_2d[flat_idx, :].reshape(side, side)
        im = ax.imshow(similarity_row, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(f"Random patch [{i}, {j}]\n(flat index {flat_idx})")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if output_path:
        output_path_str = str(output_path).replace('.png', '_per_patch_random.png')
        plt.savefig(output_path_str, dpi=150, bbox_inches='tight')
        print(f"Saved per-patch (random) visualization to {output_path_str}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize C-RADIOv4 position embeddings")
    parser.add_argument("--model", default="nvidia/C-RADIOv4-H", help="HuggingFace model identifier")
    parser.add_argument("--output-dir", default="data/intermediates/position_embeddings/explore_C_RADIO_V4", help="Directory to save visualizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load C-RADIOv4 model from HuggingFace
    pos_embed, model = load_cradio_model(model_name=args.model)

    # Visualize magnitude
    embeddings, magnitudes, side = visualize_magnitude(
        pos_embed,
        output_path=output_dir / "cradio_pos_embed_magnitude.png"
    )

    # Visualize cosine similarity only if we have a grid structure
    if side is not None:
        cosine_sim_4d, sample_side = compute_cosine_similarity_matrix(embeddings, side)
        visualize_cosine_similarity(
            cosine_sim_4d,
            sample_side,
            output_path=output_dir / "cradio_cosine_similarity.png"
        )

        # Visualize cosine similarity from random patches
        visualize_cosine_similarity_random(
            cosine_sim_4d,
            sample_side,
            output_path=output_dir / "cradio_cosine_similarity.png",
            seed=args.seed
        )
    else:
        print("\nSkipping cosine similarity analysis: no grid structure detected in embeddings")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
