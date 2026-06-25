#!/usr/bin/env python3
"""
Visualize DUNE learned absolute positional embeddings.

1. Load DUNE model using HuggingFace pretrained weights
2. Extract pos_embed from the model
3. Plot magnitude of each embedding vector
4. Compute outer product (cosine similarity) of patch embeddings
5. Visualize both as 2D images

Usage:
    uv run python scripts/visualize_dune_pos_embeddings.py --output-dir reports/
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
import mlx.core as mx

def load_dune_model(variant: str = "base", resolution: int = 336):
    """Load DUNE model using HuggingFace pretrained weights.

    Args:
        variant: "small" or "base"
        resolution: Image resolution (336 or 448)

    Returns:
        (pos_embed array, config)
    """
    from mlx_mast3r import DUNE

    print(f"Loading DUNE {variant} @ {resolution}...")
    model = DUNE.from_pretrained(variant=variant, resolution=resolution)

    # Extract position embeddings from the model
    pos_embed = model.engine.model.pos_embed

    # Convert MLX array to numpy if needed
    if isinstance(pos_embed, mx.array):
        pos_embed = np.array(pos_embed)

    return pos_embed, model.engine.config


def visualize_magnitude(pos_embed: np.ndarray, output_path: str | Path = None):
    """Plot magnitude of each embedding vector.

    Args:
        pos_embed: [1, num_tokens, embed_dim]
        output_path: Optional path to save figure
    """
    # Remove batch dimension
    embeddings = pos_embed[0]  # [num_tokens, embed_dim]

    # Compute magnitude (L2 norm) for each token
    magnitudes = np.linalg.norm(embeddings, axis=1)  # [num_tokens]

    # Infer grid structure (24x24 for standard DUNE)
    num_tokens = len(magnitudes)
    num_patches = num_tokens - 1  # Exclude CLS token
    side = int(np.sqrt(num_patches))

    print(f"Position embeddings shape: {pos_embed.shape}")
    print(f"Number of tokens: {num_tokens} (1 CLS + {num_patches} patches)")
    print(f"Inferred grid: {side}x{side}")
    print(f"Magnitude stats:")
    print(f"  CLS token magnitude: {magnitudes[0]:.4f}")
    print(f"  Patch magnitudes: min={magnitudes[1:].min():.4f}, max={magnitudes[1:].max():.4f}, mean={magnitudes[1:].mean():.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Bar chart of all magnitudes
    ax = axes[0]
    token_labels = ["CLS"] + [f"P{i}" for i in range(len(magnitudes) - 1)]
    ax.bar(range(len(magnitudes)), magnitudes, alpha=0.7, color='steelblue')
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Embedding Magnitude (L2 Norm)")
    ax.set_title("Magnitude of DUNE Position Embeddings")
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: 2D heatmap of patch magnitudes (excluding CLS)
    ax = axes[1]
    patch_magnitudes = magnitudes[1:].reshape(side, side)
    im = ax.imshow(patch_magnitudes, cmap='viridis', aspect='auto')
    ax.set_xlabel("Width Index")
    ax.set_ylabel("Height Index")
    ax.set_title(f"Magnitude Heatmap ({side}x{side} patches)")
    plt.colorbar(im, ax=ax, label="Magnitude")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved magnitude visualization to {output_path}")
    plt.show()

    return embeddings, magnitudes, side


def compute_cosine_similarity_matrix(embeddings: np.ndarray, side: int) -> np.ndarray:
    """Compute cosine similarity between all patch embeddings.

    Args:
        embeddings: [num_tokens, embed_dim] including CLS token
        side: Side length of patch grid

    Returns:
        cosine_sim: [num_patches, num_patches] similarity matrix
    """
    # Extract patch embeddings (exclude CLS)
    patch_embeddings = embeddings[1:]  # [num_patches, embed_dim]

    # Normalize for cosine similarity
    patch_embeddings_norm = patch_embeddings / (np.linalg.norm(patch_embeddings, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity via outer product
    cosine_sim = patch_embeddings_norm @ patch_embeddings_norm.T  # [num_patches, num_patches]

    return cosine_sim.reshape(side, side, side, side)


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

    print(f"\nVisualing cosine similarity from 4 random patches (seed={seed}):")
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
    parser = argparse.ArgumentParser(description="Visualize DUNE position embeddings")
    parser.add_argument("--variant", default="base", choices=["small", "base"], help="DUNE variant")
    parser.add_argument("--resolution", type=int, default=336, choices=[336, 448], help="Image resolution")
    parser.add_argument("--output-dir", default="data/intermediates/position_embeddings/explore_DUNE", help="Directory to save visualizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DUNE model from HuggingFace
    pos_embed, config = load_dune_model(variant=args.variant, resolution=args.resolution)
    print(f"Model config: embed_dim={config.embed_dim}, num_patches={config.num_patches}")

    # Visualize magnitude
    embeddings, magnitudes, side = visualize_magnitude(
        pos_embed,
        output_path=output_dir / "dune_pos_embed_magnitude.png"
    )

    # Compute and visualize cosine similarity
    cosine_sim_4d = compute_cosine_similarity_matrix(embeddings, side)
    visualize_cosine_similarity(
        cosine_sim_4d,
        side,
        output_path=output_dir / "dune_cosine_similarity.png"
    )

    # Visualize cosine similarity from random patches
    visualize_cosine_similarity_random(
        cosine_sim_4d,
        side,
        output_path=output_dir / "dune_cosine_similarity.png",
        seed=args.seed
    )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
