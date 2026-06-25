#!/usr/bin/env python3
"""
Visualize VGGt-Omega learned token embeddings.

VGGt-Omega uses Rotary Positional Embeddings (RoPE) for position encoding in attention.
This script extracts and visualizes learned token embeddings (CLS and storage tokens),
which are NOT position embeddings but rather trainable tokens used by the model.

1. Load VGGt-Omega model checkpoint
2. Extract learned token embeddings (CLS token and storage tokens)
3. Plot magnitude of each token embedding
4. Compute cosine similarity between token embeddings
5. Visualize results as 2D heatmaps

Usage:
    uv run python scripts/early_exploration/visualize_vggt_omega_pos_embeddings.py \
        --checkpoint data/models/vggt_omega_1b_512.pt \
        --output-dir data/intermediates/position_embeddings/explore_VGGt_Omega/
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
import torch


def load_vggt_omega_model(checkpoint_path: str = "data/models/vggt_omega_1b_512.pt"):
    """Load VGGt-Omega checkpoint and extract learned token embeddings.

    VGGt-Omega uses Rotary Positional Embeddings (RoPE) for position encoding during attention.
    This function extracts learned token embeddings (CLS token and storage tokens), which are
    trainable parameters used by the model, not position embeddings.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        (learned token embeddings array, checkpoint state dict)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading VGGt-Omega checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract learned token embeddings (CLS token and storage tokens)
    learned_embeds = []
    embed_keys = [
        'aggregator.patch_embed.cls_token',
        'aggregator.patch_embed.storage_tokens',
    ]

    print("\nExtracting learned token embeddings from VGGt-Omega checkpoint:")
    print("Note: VGGt-Omega uses Rotary Positional Embeddings (RoPE) for position encoding,")
    print("not learned absolute position embeddings.\n")

    for key in embed_keys:
        if key in checkpoint:
            embed = checkpoint[key]
            shape = embed.shape
            print(f"  {key}: shape {shape}")
            learned_embeds.append(embed)

    if not learned_embeds:
        print("Available embedding-related keys in checkpoint:")
        for key in sorted(checkpoint.keys()):
            if 'embed' in key.lower() or 'token' in key.lower():
                print(f"  {key}: shape {checkpoint[key].shape}")
        raise ValueError("Could not find learned token embeddings in checkpoint.")

    # Concatenate CLS token and storage tokens
    token_embeds = torch.cat(learned_embeds, dim=1)

    # Convert to numpy
    if isinstance(token_embeds, torch.Tensor):
        token_embeds = token_embeds.detach().cpu().numpy()

    print(f"\nSuccessfully extracted learned token embeddings with shape: {token_embeds.shape}")
    return token_embeds, checkpoint


def visualize_magnitude(token_embeds: np.ndarray, output_path: str | Path = None):
    """Plot magnitude of each learned token embedding.

    Args:
        token_embeds: [1, num_tokens, embed_dim] or [num_tokens, embed_dim]
        output_path: Optional path to save figure
    """
    # Handle different shapes
    if len(token_embeds.shape) == 3:
        embeddings = token_embeds[0]  # Remove batch dimension [num_tokens, embed_dim]
    else:
        embeddings = token_embeds  # Already [num_tokens, embed_dim]

    # Compute magnitude (L2 norm) for each token
    magnitudes = np.linalg.norm(embeddings, axis=1)

    # Infer token structure (CLS token + storage tokens)
    num_tokens = len(magnitudes)
    has_cls = num_tokens > 1
    num_storage_tokens = num_tokens - (1 if has_cls else 0)

    # Check if storage tokens have a grid structure (unlikely but check anyway)
    try:
        side = int(np.sqrt(num_storage_tokens))
        if side * side != num_storage_tokens:
            side = None
    except:
        side = None

    print(f"Token embeddings shape: {token_embeds.shape}")
    print(f"Number of tokens: {num_tokens}")
    if has_cls:
        print(f"  CLS token + {num_storage_tokens} storage tokens")
    if side is not None:
        print(f"  Storage tokens arranged as: {side}x{side} grid")
    print(f"Magnitude statistics:")
    print(f"  Min: {magnitudes.min():.4f}, Max: {magnitudes.max():.4f}, Mean: {magnitudes.mean():.4f}")
    print(f"  Std: {magnitudes.std():.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram of token embedding magnitudes
    ax = axes[0]
    ax.hist(magnitudes, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel("Embedding Magnitude (L2 Norm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of VGGt-Omega Token Embedding Magnitudes")
    ax.grid(axis='y', alpha=0.3)

    # Add statistics text
    stats_text = f"Mean: {magnitudes.mean():.4f}\nStd: {magnitudes.std():.4f}\nMin: {magnitudes.min():.4f}\nMax: {magnitudes.max():.4f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: 2D heatmap if grid structure is detected
    ax = axes[1]
    if side is not None:
        # Use only storage token embeddings (skip CLS token if present)
        storage_magnitudes = magnitudes[1:] if has_cls else magnitudes
        storage_magnitudes = storage_magnitudes.reshape(side, side)
        im = ax.imshow(storage_magnitudes, cmap='viridis', aspect='auto')
        ax.set_xlabel("Storage Token Index (horizontal)")
        ax.set_ylabel("Storage Token Index (vertical)")
        ax.set_title(f"Token Magnitude Heatmap ({side}x{side} grid)")
        plt.colorbar(im, ax=ax, label="Magnitude")
    else:
        ax.text(0.5, 0.5, "Storage tokens do not form a square grid",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved magnitude visualization to {output_path}")
    plt.show()

    return embeddings, magnitudes, side, has_cls


def compute_cosine_similarity_matrix(embeddings: np.ndarray, side: int, has_cls: bool = True, sample_size: int = None) -> np.ndarray:
    """Compute cosine similarity between token embeddings.

    Args:
        embeddings: [num_tokens, embed_dim]
        side: Side length of token grid
        has_cls: Whether the first token is a CLS token
        sample_size: If specified, only compute similarity for a sampled subset of tokens

    Returns:
        cosine_sim: [side, side, side, side] 4D similarity tensor (or sampled version)
    """
    # Extract storage token embeddings (skip CLS token if present)
    if has_cls:
        token_embeddings = embeddings[1:]  # All embeddings except CLS
    else:
        token_embeddings = embeddings  # All embeddings are tokens

    if sample_size is None:
        # Default to a manageable subset if the full matrix would be too large
        num_tokens = len(token_embeddings)
        if num_tokens > 4096:  # 64x64 grid
            sample_size = 4096
            print(f"Large token set ({num_tokens} tokens). Sampling {sample_size} tokens for similarity computation.")
        else:
            sample_size = num_tokens

    if sample_size < len(token_embeddings):
        # Sample indices
        np.random.seed(42)
        sample_indices = np.random.choice(len(token_embeddings), sample_size, replace=False)
        token_embeddings = token_embeddings[sample_indices]
        sample_side = int(np.sqrt(sample_size))
    else:
        sample_side = side

    # Normalize for cosine similarity
    token_embeddings_norm = token_embeddings / (np.linalg.norm(token_embeddings, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity via outer product
    cosine_sim = token_embeddings_norm @ token_embeddings_norm.T

    return cosine_sim.reshape(sample_side, sample_side, sample_side, sample_side), sample_side


def visualize_cosine_similarity(cosine_sim_4d: np.ndarray, side: int, output_path: str | Path = None):
    """Visualize cosine similarity between token embeddings.

    Args:
        cosine_sim_4d: [side, side, side, side] 4D similarity tensor
        side: Side length of token grid
        output_path: Optional path to save figure
    """
    # Reshape to 2D for visualization: each token's similarity to all other tokens
    cosine_sim_2d = cosine_sim_4d.reshape(side * side, side * side)

    print(f"\nCosine Similarity Matrix statistics:")
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
    ax.set_xlabel("Token Index (flattened)")
    ax.set_ylabel("Token Index (flattened)")
    ax.set_title("Cosine Similarity Between All Tokens")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    # Plot 2: Magnitude of similarity
    ax = axes[1]
    im2 = ax.imshow(np.abs(cosine_sim_2d), cmap='hot', aspect='auto')
    ax.set_xlabel("Token Index (flattened)")
    ax.set_ylabel("Token Index (flattened)")
    ax.set_title("Absolute Cosine Similarity Magnitude")
    plt.colorbar(im2, ax=ax, label="|Cosine Similarity|")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved cosine similarity visualization to {output_path}")
    plt.show()

    # Plot 3: Similarity from selected tokens (corners and center)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    select_indices = [
        (0, 0),      # Top-left
        (0, side-1), # Top-right
        (side//2, side//2),  # Center
        (side-1, side-1),    # Bottom-right
    ]

    titles = ["Top-left corner", "Top-right corner", "Center", "Bottom-right corner"]

    for (i, j), ax, title in zip(select_indices, axes.flat, titles):
        token_idx = i * side + j
        similarity_row = cosine_sim_2d[token_idx, :].reshape(side, side)
        im = ax.imshow(similarity_row, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(f"{title}\n(token at [{i}, {j}])")
        ax.set_xlabel("Grid Index (horizontal)")
        ax.set_ylabel("Grid Index (vertical)")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if output_path:
        output_path_str = str(output_path).replace('.png', '_per_token_corners.png')
        plt.savefig(output_path_str, dpi=150, bbox_inches='tight')
        print(f"Saved per-token corner visualization to {output_path_str}")
    plt.show()


def visualize_cosine_similarity_random(cosine_sim_4d: np.ndarray, side: int, output_path: str | Path = None, seed: int = 42):
    """Visualize cosine similarity from 4 randomly selected tokens.

    Args:
        cosine_sim_4d: [side, side, side, side] 4D similarity tensor
        side: Side length of token grid
        output_path: Optional path to save figure
        seed: Random seed for reproducibility
    """
    # Reshape to 2D for visualization
    cosine_sim_2d = cosine_sim_4d.reshape(side * side, side * side)

    # Randomly select 4 tokens
    np.random.seed(seed)
    num_tokens = side * side
    selected_flat_indices = np.random.choice(num_tokens, 4, replace=False)
    select_indices = [(idx // side, idx % side) for idx in selected_flat_indices]

    print(f"\nVisualizing cosine similarity from 4 random tokens (seed={seed}):")
    print(f"  Selected token indices (flat): {selected_flat_indices}")
    print(f"  Selected token indices (grid): {select_indices}")

    # Plot: Similarity from random tokens
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for (i, j), ax, flat_idx in zip(select_indices, axes.flat, selected_flat_indices):
        similarity_row = cosine_sim_2d[flat_idx, :].reshape(side, side)
        im = ax.imshow(similarity_row, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(f"Random token [{i}, {j}]\n(flat index {flat_idx})")
        ax.set_xlabel("Grid Index (horizontal)")
        ax.set_ylabel("Grid Index (vertical)")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if output_path:
        output_path_str = str(output_path).replace('.png', '_per_token_random.png')
        plt.savefig(output_path_str, dpi=150, bbox_inches='tight')
        print(f"Saved per-token random visualization to {output_path_str}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize VGGt-Omega learned token embeddings")
    parser.add_argument("--checkpoint", default="data/models/vggt_omega_1b_512.pt", help="Path to VGGt-Omega checkpoint")
    parser.add_argument("--output-dir", default="data/intermediates/position_embeddings/explore_VGGt_Omega", help="Directory to save visualizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VGGt-Omega checkpoint and extract learned token embeddings
    token_embeds, _ = load_vggt_omega_model(checkpoint_path=args.checkpoint)

    # Visualize token embedding magnitudes
    embeddings, magnitudes, side, has_cls = visualize_magnitude(
        token_embeds,
        output_path=output_dir / "vggt_omega_token_embed_magnitude.png"
    )

    # Visualize cosine similarity only if token embeddings have a grid structure
    if side is not None:
        cosine_sim_4d, sample_side = compute_cosine_similarity_matrix(embeddings, side, has_cls=has_cls)
        visualize_cosine_similarity(
            cosine_sim_4d,
            sample_side,
            output_path=output_dir / "vggt_omega_token_cosine_similarity.png"
        )

        # Visualize cosine similarity from random tokens
        visualize_cosine_similarity_random(
            cosine_sim_4d,
            sample_side,
            output_path=output_dir / "vggt_omega_token_cosine_similarity.png",
            seed=args.seed
        )
    else:
        print("\nSkipping cosine similarity analysis: token embeddings do not form a grid structure")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
