"""
Sample code for the Enhanced Synthetic Distribution class with 10 labels and 5 dimensions.
Each label has a distinct pattern through the cell (input space).

Information and Decision Systems Group - FCFM - Universidad de Chile
"""

import argparse
import itertools
import os
from pathlib import Path
from typing import List, Set

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from bm.examples.sample_plots import create_dimension_combination_plots
from bm.synthetic_distribution import FloatArray, SyntheticDistribution

ObjectArray = npt.NDArray[np.object_]


def get_label_probabilities(selected_labels: List[int] = None) -> FloatArray:
    """Get uniform probabilities for selected labels.

    Args:
        selected_labels: List of label indices to include (0-9). If None, uses all 10 labels.

    Returns:
        Array of uniform probabilities for the selected labels.
    """
    if selected_labels is None:
        selected_labels = list(range(10))  # All labels 0-9

    num_labels = len(selected_labels)
    uniform_prob = 1.0 / num_labels

    demo_s_prob: FloatArray = np.full(num_labels, uniform_prob, dtype=np.float64)

    return demo_s_prob


def get_interval_bounds() -> List[List[float]]:
    demo_i_bounds: List[List[float]] = [
        [-1.0, -0.33, 0.33, 1.0],  # Dimension 1: 3 intervals
        [-1.0, -0.33, 0.33, 1.0],  # Dimension 2: 3 intervals
        [-1.0, -0.33, 0.33, 1.0],  # Dimension 3: 3 intervals
        [-1.0, -0.33, 0.33, 1.0],  # Dimension 4: 3 intervals
        [-1.0, -0.33, 0.33, 1.0],  # Dimension 5: 3 intervals
    ]

    return demo_i_bounds


def data_patterns(
    selected_labels: List[int] = None,
    base_noise: float = 0.0003,
    save_dir: str = None,
    plot: bool = False,
) -> List[np.ndarray]:
    """Create distinct patterns for selected labels.

    Args:
        selected_labels: List of label indices to include (0-9). If None, uses all 10 labels.
        base_noise: Base noise level for the distribution.
        save_dir: Directory to save the patterns.
        plot: Whether to plot the patterns.

    Returns:
        List of patterns for the selected labels.
    """
    if selected_labels is None:
        selected_labels = list(range(10))  # All labels 0-9

    # Each pattern is a 3x3x3x3x3 = 243-cell probability distribution
    patterns = []

    # Base probability for controlled noise distribution to hit target accuracy range
    base_prob = base_noise

    for label in selected_labels:
        # Initialize with minimal uniform probability everywhere
        pattern = np.full((3, 3, 3, 3, 3), base_prob, dtype=np.float64)

        if label == 0:
            # Label 0: Low preference with blurred dim1-dim2 boundary
            pattern[0, 0, 0, 0, 0] = 0.2
            pattern[1, 0, 0, 0, 0] = 0.15  # Blur dim1 boundary
            pattern[0, 1, 0, 0, 0] = 0.15  # Blur dim2 boundary
            pattern[1, 1, 0, 0, 0] = 0.12  # Mixed dim1-dim2
            pattern[0, 0, 0, 0, 1] = 0.1
            pattern[0, 0, 0, 1, 0] = 0.1
            pattern[0, 0, 1, 0, 0] = 0.08
            pattern[1, 0, 0, 0, 1] = 0.05
            pattern[0, 1, 0, 0, 1] = 0.05

        elif label == 1:
            # Label 1: High preference with blurred dim1-dim2 boundary
            pattern[2, 2, 2, 2, 2] = 0.2
            pattern[1, 2, 2, 2, 2] = 0.15  # Blur dim1 boundary
            pattern[2, 1, 2, 2, 2] = 0.15  # Blur dim2 boundary
            pattern[1, 1, 2, 2, 2] = 0.12  # Mixed dim1-dim2
            pattern[2, 2, 2, 2, 1] = 0.1
            pattern[2, 2, 2, 1, 2] = 0.1
            pattern[2, 2, 1, 2, 2] = 0.08
            pattern[1, 2, 2, 2, 1] = 0.05
            pattern[2, 1, 2, 2, 1] = 0.05

        elif label == 2:
            # Label 2: Blurred dimension 1 preference with dim1-dim2 overlap
            pattern[2, 0, 1, 0, 1] = 0.2  # Main dim1 signal
            pattern[1, 0, 1, 0, 1] = 0.15  # Blur dim1 boundary down
            pattern[2, 1, 1, 0, 1] = 0.15  # Blur dim2 boundary up
            pattern[1, 1, 1, 0, 1] = 0.12  # Mixed dim1-dim2 overlap
            pattern[2, 0, 0, 0, 1] = 0.1
            pattern[2, 0, 1, 1, 1] = 0.1
            pattern[2, 0, 1, 0, 0] = 0.08
            pattern[1, 0, 1, 0, 1] = 0.05  # Additional dim1 blur
            pattern[2, 0, 2, 0, 1] = 0.05

        elif label == 3:
            # Label 3: Blurred dimension 2 preference with dim1-dim2 overlap
            pattern[0, 2, 1, 0, 1] = 0.2  # Main dim2 signal
            pattern[0, 1, 1, 0, 1] = 0.15  # Blur dim2 boundary down
            pattern[1, 2, 1, 0, 1] = 0.15  # Blur dim1 boundary up
            pattern[1, 1, 1, 0, 1] = 0.12  # Mixed dim1-dim2 overlap
            pattern[0, 2, 0, 0, 1] = 0.1
            pattern[0, 2, 1, 1, 1] = 0.1
            pattern[0, 2, 1, 0, 0] = 0.08
            pattern[0, 1, 1, 0, 1] = 0.05  # Additional dim2 blur
            pattern[0, 2, 2, 0, 1] = 0.05

        elif label == 4:
            # Label 4: Dimension 3 control with substantial noise
            pattern[1, 0, 2, 0, 1] = 0.25
            pattern[1, 1, 2, 0, 1] = 0.18
            pattern[1, 0, 2, 1, 1] = 0.15
            pattern[0, 0, 2, 0, 1] = 0.12
            pattern[1, 0, 2, 0, 0] = 0.1
            pattern[1, 2, 2, 0, 1] = 0.08
            pattern[1, 0, 1, 0, 1] = 0.06
            pattern[1, 0, 2, 0, 2] = 0.06

        elif label == 5:
            # Label 5: Dimension 4 control with substantial noise
            pattern[1, 0, 1, 2, 0] = 0.25
            pattern[1, 1, 1, 2, 0] = 0.18
            pattern[1, 0, 0, 2, 0] = 0.15
            pattern[0, 0, 1, 2, 0] = 0.12
            pattern[1, 0, 1, 2, 1] = 0.1
            pattern[1, 0, 2, 2, 0] = 0.08
            pattern[1, 0, 1, 1, 0] = 0.06
            pattern[1, 0, 1, 2, 2] = 0.06

        elif label == 6:
            # Label 6: Dimension 5 control with substantial noise
            pattern[1, 0, 1, 0, 2] = 0.25
            pattern[1, 1, 1, 0, 2] = 0.18
            pattern[1, 0, 0, 0, 2] = 0.15
            pattern[0, 0, 1, 0, 2] = 0.12
            pattern[1, 0, 1, 1, 2] = 0.1
            pattern[1, 0, 1, 0, 1] = 0.08
            pattern[1, 0, 1, 0, 0] = 0.06
            pattern[1, 0, 2, 0, 2] = 0.06

        elif label == 7:
            # Label 7: Alternating pattern with substantial noise
            pattern[0, 2, 0, 2, 0] = 0.2
            pattern[2, 0, 2, 0, 2] = 0.2
            pattern[0, 2, 0, 2, 1] = 0.15
            pattern[2, 0, 2, 0, 1] = 0.15
            pattern[0, 1, 0, 2, 0] = 0.08
            pattern[1, 0, 2, 0, 2] = 0.08
            pattern[0, 2, 1, 2, 0] = 0.06
            pattern[1, 2, 0, 2, 0] = 0.04
            pattern[2, 1, 2, 0, 2] = 0.04

        elif label == 8:
            # Label 8: Center with substantial distributed noise
            pattern[1, 1, 1, 1, 1] = 0.2
            pattern[0, 1, 1, 1, 1] = 0.15
            pattern[2, 1, 1, 1, 1] = 0.15
            pattern[1, 0, 1, 1, 1] = 0.12
            pattern[1, 2, 1, 1, 1] = 0.12
            pattern[1, 1, 0, 1, 1] = 0.08
            pattern[1, 1, 2, 1, 1] = 0.08
            pattern[1, 1, 1, 0, 1] = 0.05
            pattern[1, 1, 1, 2, 1] = 0.05

        elif label == 9:
            # Label 9: Edge pattern with highly distributed probability
            pattern[0, 0, 0, 2, 2] = 0.10
            pattern[2, 2, 2, 0, 0] = 0.10
            pattern[0, 2, 2, 0, 2] = 0.11
            pattern[2, 0, 0, 2, 0] = 0.11
            pattern[0, 0, 2, 2, 0] = 0.1
            pattern[2, 2, 0, 0, 2] = 0.1
            pattern[0, 2, 0, 2, 2] = 0.09
            pattern[2, 0, 2, 0, 0] = 0.09
            pattern[1, 1, 1, 2, 2] = 0.08
            pattern[1, 1, 2, 1, 0] = 0.06
            pattern[1, 2, 1, 1, 0] = 0.06

        # Normalize to ensure probabilities sum to 1
        pattern_sum = np.sum(pattern)
        if pattern_sum > 0:
            pattern = pattern / pattern_sum

        patterns.append(pattern)
    patterns = np.asarray(patterns, dtype=np.float64)
    # Print the sum of probabilities for each label to verify normalization
    for i, pattern in enumerate(patterns):
        print(f"Label {i} probability sum: {pattern.sum():.3f}")
    print(f"Total probability sum across all labels: {patterns.sum():.3f}")

    # Create a visualization of all patterns with comprehensive analysis including all dimension pairs and PCA

    # Visualize the patterns
    if plot:
        visualize_patterns(patterns, save_dir=save_dir)

    return patterns


def get_cell_probabilities(
    patterns: List[np.ndarray], cell_bounds: List[List[float]]
) -> ObjectArray:
    demo_c_prob: List[np.ndarray] = [
        np.stack(patterns, axis=0)  # Shape: (10, 3, 3, 3, 3, 3)
    ]

    demo_i_bound_array: ObjectArray = np.empty(shape=len(cell_bounds), dtype=object)
    demo_i_bound_array[:] = [
        np.asarray(a=bounds, dtype=np.float64) for bounds in cell_bounds
    ]

    demo_c_prob_array: ObjectArray = np.empty(shape=len(demo_c_prob), dtype=object)
    demo_c_prob_array[:] = [
        np.asarray(a=cell_cond_probs, dtype=np.float64)
        for cell_cond_probs in demo_c_prob
    ]

    return demo_c_prob_array


def generate_all_dimension_combinations(n_dims: int) -> List[Set[int]]:
    """Generate all possible combinations of dimensions from 1 to n_dims."""
    all_combinations = []

    # Generate all combinations from size 1 to n_dims
    for r in range(1, n_dims + 1):
        for combo in itertools.combinations(range(1, n_dims + 1), r):
            all_combinations.append(set(combo))

    return all_combinations


def compute_comprehensive_mi(distribution: SyntheticDistribution, n_dims: int) -> dict:
    """Compute mutual information for all possible dimension combinations."""
    mi_dict = {}

    # Generate all dimension combinations
    combinations = generate_all_dimension_combinations(n_dims)

    print(f"\nComputing MI for {len(combinations)} dimension combinations...")

    for coords in combinations:
        mi_value = distribution.get_mutual_information(coordinates=coords)

        # Create string key for the combination
        if len(coords) == 1:
            key = str(list(coords)[0])
        else:
            key = "-".join(map(str, sorted(coords)))

        mi_dict[key] = mi_value

        # Also store the joint MI (all dimensions)
        if len(coords) == n_dims:
            mi_dict["joint"] = mi_value

    return mi_dict


def compute_max_accuracy(entropy_y: float, joint_mi: float) -> float:
    """Compute theoretical maximum accuracy based on mutual information."""
    # Theoretical max accuracy is related to how much of the entropy is captured by MI
    # This is an approximation - actual accuracy depends on the classifier
    information_ratio = joint_mi / entropy_y if entropy_y > 0 else 0

    # Apply sigmoid-like transformation to map to reasonable accuracy range
    # This gives values typically between 0.1 (random) and 0.95 (very good)
    max_acc = 0.1 + 0.85 * information_ratio

    # Cap at reasonable maximum
    # max_acc = min(max_acc, 0.95)

    return max_acc


def generate_metadata(
    distribution: SyntheticDistribution, n_dims: int, num_classes: int
) -> dict:
    """Generate comprehensive metadata for the distribution."""
    print("Generating comprehensive metadata...")

    # Basic metrics
    entropy_y = distribution.get_symbol_entropy()

    # Compute MI for all dimension combinations
    mi_dict = compute_comprehensive_mi(distribution, n_dims)

    # Compute theoretical max accuracy
    joint_mi = mi_dict.get("joint", 0.0)
    max_acc1 = compute_max_accuracy(entropy_y, joint_mi)

    metadata = {
        "entropy_y": entropy_y,
        "in_shape": [n_dims],
        "mi": mi_dict,
        "num_classes": num_classes,
        "max_acc1": max_acc1,
    }

    return metadata


def generate_data(
    distribution: SyntheticDistribution,
    num_samples: int,
    seed: int,
    num_classes: int,
):
    # Sample data and display information
    print(f"Sampling {num_samples:,} data points...")
    x_samples, y_samples = distribution.sample_data(num_samples=num_samples, seed=seed)
    print(x_samples.shape)
    print(y_samples.shape)
    print(f"Symbol entropy: {distribution.get_symbol_entropy():.4f}")
    print(f"Max possible entropy for {num_classes} labels: {np.log2(num_classes):.4f}")

    # Display label distribution
    unique_labels, counts = np.unique(y_samples, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique_labels, counts):
        print(
            f"  Label {label}: {count:5d} samples ({count / len(y_samples) * 100:.1f}%)"
        )

    # Generate comprehensive metadata
    metadata = generate_metadata(distribution, n_dims=5, num_classes=num_classes)

    # Display some key MI results for verification
    print("\nKey Mutual Information results:")
    key_coords = ["1", "2", "3", "4", "5", "1-2", "1-2-3", "1-2-3-4", "joint"]
    for key in key_coords:
        if key in metadata["mi"]:
            mi_value = metadata["mi"][key]
            optimal_loss = distribution.compute_analytical_loss(
                mi_value, metadata["entropy_y"]
            )
            print(f"  MI(X{key}; Y): {mi_value:.4f} | Optimal loss: {optimal_loss:.4f}")

    analytical_loss = distribution.compute_analytical_loss(
        metadata["mi"]["joint"], metadata["entropy_y"]
    )
    print(f"\nAnalytical loss (H(Y) - MI(X; Y)): {analytical_loss:.4f}")
    print(f"Theoretical max accuracy: {metadata['max_acc1']:.4f}")

    return x_samples, y_samples, metadata


def main(args: argparse.Namespace):
    # Parse selected labels
    if args.labels is not None:
        selected_labels = [int(x) for x in args.labels.split(",")]
        # Validate label indices
        for label in selected_labels:
            if label < 0 or label > 9:
                raise ValueError(f"Label {label} is out of range [0-9]")
    else:
        selected_labels = None  # Use all labels

    print(f"Using labels: {selected_labels if selected_labels else list(range(10))}")

    # Step A.1 and A.2: k symbols with equal probabilities
    label_probs = get_label_probabilities(selected_labels=selected_labels)
    num_classes = len(label_probs)

    # Steps A.5 and A.6: 5 dimensions, each with 3 intervals (4 bounds)
    cell_bounds = get_interval_bounds()

    # Generate distinct patterns for selected labels
    patterns = data_patterns(
        selected_labels=selected_labels,
        base_noise=args.base_noise,
        save_dir=args.save_dir,
        plot=True,
    )

    # Steps A.3, A.4, and A.10: Create cell probabilities for one coordinate group with all 5 dimensions
    cell_probabilities = get_cell_probabilities(
        patterns=patterns, cell_bounds=cell_bounds
    )

    demo_i_bound_array: ObjectArray = np.empty(shape=len(cell_bounds), dtype=object)
    demo_i_bound_array[:] = [
        np.asarray(a=bounds, dtype=np.float64) for bounds in cell_bounds
    ]

    distribution: SyntheticDistribution = SyntheticDistribution(
        symbol_probabilities=label_probs,
        interval_bounds=demo_i_bound_array,
        cell_probabilities=cell_probabilities,
    )

    x_samples, y_samples, metadata = generate_data(
        distribution=distribution,
        num_samples=args.num_samples,
        seed=args.seed,
        num_classes=num_classes,
    )

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            file=save_dir / "data.npz",
            x_samples=x_samples,
            y_samples=y_samples,
        )

        # Save metadata as JSON
        import json

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nData and metadata saved to: {save_dir}")

        # Clean per-dimension combination plots with proper styling
        if args.plot:
            create_dimension_combination_plots(
                x_samples=x_samples,
                y_samples=y_samples,
                metadata=metadata,
                save_dir=save_dir,
            )


def visualize_patterns(patterns: List[np.ndarray], save_dir: str = "./") -> None:
    """Visualize per-label patterns across all dimension pairs and perform PCA analysis."""
    from itertools import combinations

    from matplotlib.colors import LogNorm
    from sklearn.decomposition import PCA

    n_labels = len(patterns)
    n_dims = 5
    dim_pairs = list(combinations(range(n_dims), 2))
    n_pairs = len(dim_pairs)

    print(
        f"Creating visualizations for {n_labels} labels and {n_pairs} dimension pairs..."
    )

    # 1. Comprehensive grid: Labels (rows) × Dimension pairs (columns)
    fig = plt.figure(figsize=(30, 3 * n_labels))

    # Find global min/max for consistent color scaling across all subplots
    all_values = []
    for label_idx, pattern in enumerate(patterns):
        for dim1, dim2 in dim_pairs:
            # Get dimensions to average over (all except dim1 and dim2)
            avg_dims = tuple(i for i in range(n_dims) if i not in [dim1, dim2])
            heatmap_data = np.mean(pattern, axis=avg_dims)
            all_values.extend(heatmap_data.flatten())

    vmin = max(1e-6, min(all_values))
    vmax = max(all_values)

    for label_idx, pattern in enumerate(patterns):
        for pair_idx, (dim1, dim2) in enumerate(dim_pairs):
            ax = plt.subplot(n_labels, n_pairs, label_idx * n_pairs + pair_idx + 1)

            # Get dimensions to average over (all except dim1 and dim2)
            avg_dims = tuple(i for i in range(n_dims) if i not in [dim1, dim2])
            heatmap_data = np.mean(pattern, axis=avg_dims)

            # Ensure consistent orientation (dim1 as rows, dim2 as columns)
            if dim1 > dim2:
                heatmap_data = heatmap_data.T
                plot_dim1, plot_dim2 = dim2, dim1
            else:
                plot_dim1, plot_dim2 = dim1, dim2

            im = ax.imshow(
                heatmap_data,
                cmap="viridis",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                interpolation="nearest",
                aspect="equal",
            )

            # Labels and titles
            if label_idx == 0:  # Top row - dimension pair labels
                ax.set_title(f"D{plot_dim1 + 1} vs D{plot_dim2 + 1}", fontsize=10)
            if pair_idx == 0:  # Left column - label numbers
                ax.set_ylabel(
                    f"Label {label_idx}",
                    fontsize=10,
                    rotation=0,
                    ha="right",
                    va="center",
                )

            # Clean up axes - set ticks based on actual pattern size
            n_ticks = min(pattern.shape[0], 10)  # Limit to 10 ticks max for readability
            ax.set_xticks(np.linspace(0, pattern.shape[1] - 1, n_ticks, dtype=int))
            ax.set_yticks(np.linspace(0, pattern.shape[0] - 1, n_ticks, dtype=int))
            if label_idx < n_labels - 1:  # Not bottom row
                ax.set_xticklabels([])
            if pair_idx > 0:  # Not left column
                ax.set_yticklabels([])

            # Add grid for clarity
            ax.grid(True, color="white", linewidth=0.5, alpha=0.3)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Pattern Intensity (log scale)")

    plt.suptitle("Per-Label Patterns Across All Dimension Pairs", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])

    if save_dir != "./":
        os.makedirs(save_dir, exist_ok=True)

    print(f"Saving per-label pattern analysis to {save_dir}")
    plt.savefig(
        os.path.join(save_dir, "pattern_per_label_all_pairs.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # 2. Alternative view: One figure per dimension pair showing all labels
    print("Creating per-dimension-pair comparisons...")
    fig2, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    for pair_idx, (dim1, dim2) in enumerate(dim_pairs):
        ax = axes[pair_idx]

        # Create a grid showing all labels for this dimension pair
        label_grid_size = int(np.ceil(np.sqrt(n_labels)))

        # Get the actual size of each pattern dimension
        pattern_dim_size = patterns[0].shape[
            0
        ]  # Assuming all dimensions have same size

        combined_heatmap = np.zeros(
            (label_grid_size * pattern_dim_size, label_grid_size * pattern_dim_size)
        )

        for label_idx, pattern in enumerate(patterns):
            if label_idx >= label_grid_size * label_grid_size:
                break

            # Calculate position in grid
            grid_row = label_idx // label_grid_size
            grid_col = label_idx % label_grid_size

            # Get heatmap data for this label and dimension pair
            avg_dims = tuple(i for i in range(n_dims) if i not in [dim1, dim2])
            heatmap_data = np.mean(pattern, axis=avg_dims)

            # Place in combined heatmap
            start_row = grid_row * pattern_dim_size
            start_col = grid_col * pattern_dim_size
            combined_heatmap[
                start_row : start_row + pattern_dim_size,
                start_col : start_col + pattern_dim_size,
            ] = heatmap_data

        im = ax.imshow(
            combined_heatmap,
            cmap="plasma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation="nearest",
        )

        ax.set_title(f"Dims {dim1 + 1} vs {dim2 + 1}\n(All Labels)", fontsize=12)

        # Add grid lines to separate labels
        for i in range(1, label_grid_size):
            ax.axhline(y=i * pattern_dim_size - 0.5, color="white", linewidth=2)
            ax.axvline(x=i * pattern_dim_size - 0.5, color="white", linewidth=2)

        # Add label numbers
        for label_idx in range(min(n_labels, label_grid_size * label_grid_size)):
            grid_row = label_idx // label_grid_size
            grid_col = label_idx % label_grid_size
            center_row = grid_row * pattern_dim_size + pattern_dim_size // 2
            center_col = grid_col * pattern_dim_size + pattern_dim_size // 2
            ax.text(
                center_col,
                center_row,
                str(label_idx),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("All Labels per Dimension Pair", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_dir != "./":
        os.makedirs(save_dir, exist_ok=True)

    print(f"Saving all-labels-per-pair analysis to {save_dir}")
    plt.savefig(
        os.path.join(save_dir, "pattern_all_labels_per_pair.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # 3. PCA Analysis (keeping patterns separate)
    print("Performing PCA analysis on individual label patterns...")
    fig_pca, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Flatten each pattern separately (no averaging across labels)
    flattened_patterns = np.array([pattern.flatten() for pattern in patterns])

    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(flattened_patterns)

    # Plot 1: PCA components scatter plot
    colors = plt.cm.tab10(np.linspace(0, 1, n_labels))
    scatter = ax1.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=colors[:n_labels],
        s=100,
        alpha=0.8,
        edgecolors="black",
    )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax1.set_title("PCA: Label Separation in First Two PCs")
    ax1.grid(True, alpha=0.3)

    # Add labels for each point
    for i, (x, y) in enumerate(pca_result[:, :2]):
        ax1.annotate(
            f"L{i}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 2: Explained variance ratio
    ax2.bar(
        range(1, min(len(pca.explained_variance_ratio_), 10) + 1),
        pca.explained_variance_ratio_[:10],
        alpha=0.7,
        color="steelblue",
    )
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_title("PCA: Explained Variance by Component")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative explained variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax3.plot(
        range(1, min(len(cumvar), 20) + 1),
        cumvar[:20],
        "o-",
        color="red",
        alpha=0.7,
    )
    ax3.axhline(y=0.95, color="gray", linestyle="--", alpha=0.7, label="95% variance")
    ax3.axhline(
        y=0.99, color="lightgray", linestyle="--", alpha=0.7, label="99% variance"
    )
    ax3.set_xlabel("Number of Components")
    ax3.set_ylabel("Cumulative Explained Variance")
    ax3.set_title("PCA: Cumulative Explained Variance")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: 3D PCA if we have at least 3 components
    if pca_result.shape[1] >= 3:
        ax4 = fig_pca.add_subplot(2, 2, 4, projection="3d")
        scatter_3d = ax4.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            pca_result[:, 2],
            c=colors[:n_labels],
            s=100,
            alpha=0.8,
            edgecolors="black",
        )
        ax4.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        ax4.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        ax4.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%})")
        ax4.set_title("PCA: First Three Principal Components")

        # Add labels for 3D plot
        for i, (x, y, z) in enumerate(pca_result[:, :3]):
            ax4.text(x, y, z, f"L{i}", fontsize=9)
    else:
        ax4.text(
            0.5,
            0.5,
            "Not enough components\nfor 3D visualization",
            transform=ax4.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax4.set_title("3D PCA (Not Available)")

    plt.suptitle("PCA Analysis of Individual Label Patterns", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir != "./":
        os.makedirs(save_dir, exist_ok=True)

    print(f"Saving PCA analysis to {save_dir}")
    plt.savefig(
        os.path.join(save_dir, "pattern_pca_analysis.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # Print PCA summary
    print("\nPCA Summary:")
    print(f"Total components: {len(pca.explained_variance_ratio_)}")
    print(f"First 2 components explain {cumvar[1]:.2%} of variance")
    if len(cumvar) >= 3:
        print(f"First 3 components explain {cumvar[2]:.2%} of variance")
    components_95 = np.argmax(cumvar >= 0.95) + 1
    print(f"Components needed for 95% variance: {components_95}")

    # Calculate pairwise distances in PCA space
    from scipy.spatial.distance import pdist, squareform

    pca_distances = squareform(pdist(pca_result[:, :3]))  # Use first 3 PCs
    print("\nLabel separation analysis (using first 3 PCs):")
    print(f"Mean pairwise distance: {np.mean(pca_distances[pca_distances > 0]):.4f}")
    print(f"Min pairwise distance: {np.min(pca_distances[pca_distances > 0]):.4f}")
    print(f"Max pairwise distance: {np.max(pca_distances):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic data with distinct patterns for selected labels"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--num_samples", type=int, default=50_000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save data and metadata"
    )
    parser.add_argument(
        "--plot",
        type=bool,
        default=False,
        help="Whether to do plots",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated list of labels to include (0-9). Example: '0,1,2' for first 3 labels. If not specified, uses all 10 labels.",
    )
    parser.add_argument(
        "--base_noise", type=float, default=0.0, help="Base noise level"
    )
    args = parser.parse_args()

    main(args=args)
