from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_dimension_combination_plots(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    metadata: dict,
    save_dir: Path,
) -> None:
    """Create clean 3D dimension combination plots with proper styling."""

    # Apply consistent styling
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "lines.linewidth": 1.5,
            "axes.linewidth": 1.5,
            "grid.alpha": 0.3,
        }
    )

    # Sample data if too large (max 10k samples for plotting)
    max_plot_samples = 100_000
    if len(x_samples) > max_plot_samples:
        print(
            f"Sampling {max_plot_samples:,} out of {len(x_samples):,} samples for plotting..."
        )
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        sample_indices = rng.choice(
            len(x_samples), size=max_plot_samples, replace=False
        )
        x_plot = x_samples[sample_indices]
        y_plot = y_samples[sample_indices]
        plot_info = f"(showing {max_plot_samples:,} of {len(x_samples):,} samples)"
    else:
        x_plot = x_samples
        y_plot = y_samples
        plot_info = f"(all {len(x_samples):,} samples)"

    n_dims = x_plot.shape[1]
    unique_labels = np.unique(y_plot)
    n_classes = len(unique_labels)

    print(
        f"Creating clean 3D dimension combination plots for {n_dims} dimensions {plot_info}..."
    )

    # Generate all 3D combinations
    dim_triplets = list(combinations(range(n_dims), 3))

    # Create colormap for labels with proper mapping
    if n_classes == 10:
        cmap = plt.cm.tab10
    elif n_classes == 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis

    # Create label-to-color mapping
    colors = [cmap(i) for i in np.linspace(0, 1, n_classes)]
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Individual 3D plots for each dimension triplet
    print(f"Generating {len(dim_triplets)} individual 3D scatter plots...")
    for i, (dim1, dim2, dim3) in enumerate(dim_triplets):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Create 3D scatter plot for each label separately for better control
        for label in unique_labels:
            mask = y_plot == label
            ax.scatter(
                x_plot[mask, dim1],
                x_plot[mask, dim2],
                x_plot[mask, dim3],
                c=[label_to_color[label]],
                label=f"Label {label}",
                s=10,
                edgecolors="white",
            )

        # Styling
        ax.set_xlabel(f"Dimension {dim1 + 1}", fontweight="bold")
        ax.set_ylabel(f"Dimension {dim2 + 1}", fontweight="bold")
        ax.set_zlabel(f"Dimension {dim3 + 1}", fontweight="bold")
        ax.set_title(
            f"Dimensions {dim1 + 1} vs {dim2 + 1} vs {dim3 + 1}",
            fontweight="bold",
            pad=20,
        )

        # Add grid
        ax.grid(True, alpha=0.3)

        # Legend positioned to avoid cutoff
        legend = ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=1 if n_classes <= 8 else 2,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.9)

        # Add metadata info box
        if "mi" in metadata:
            # Get MI for this dimension triplet
            dim_key = f"{dim1 + 1}-{dim2 + 1}-{dim3 + 1}"
            mi_value = metadata["mi"].get(dim_key, 0.0)

            info_text = (
                f"MI(X_{{{dim1 + 1}}}, X_{{{dim2 + 1}}}, X_{{{dim3 + 1}}}; Y) = {mi_value:.4f}\n"
                f"Plot samples: {len(y_plot):,}\n"
                f"Total samples: {len(y_samples):,}\n"
                f"Classes: {n_classes}"
            )

            # Position text box in 3D space
            ax.text2D(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="gray",
                ),
                verticalalignment="top",
                fontsize=10,
            )

        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)

        # Adjust layout to prevent cutoff - use subplots_adjust for better control
        plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

        # Save individual 3D plot
        filename = f"x_samples_{dim1 + 1}_vs_{dim2 + 1}_vs_{dim3 + 1}.png"
        plt.savefig(
            save_dir / filename,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.3,
        )
        plt.close()

        print(f"  Saved: {filename}")

    # Create a comprehensive grid of 3D plots
    print("Creating comprehensive 3D grid plot...")
    n_triplets = len(dim_triplets)

    # Calculate grid size (prefer more columns for 3D plots)
    grid_cols = min(5, n_triplets)
    grid_rows = (n_triplets + grid_cols - 1) // grid_cols

    fig = plt.figure(figsize=(6 * grid_cols, 5 * grid_rows))

    # Create a consistent colormap for the scatter plots
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    y_indices = np.array([label_to_index[label] for label in y_plot])

    for idx, (dim1, dim2, dim3) in enumerate(dim_triplets):
        ax = fig.add_subplot(grid_rows, grid_cols, idx + 1, projection="3d")

        # Create 3D scatter plot using indices for consistent coloring
        scatter = ax.scatter(
            x_plot[:max_plot_samples, dim1],
            x_plot[:max_plot_samples, dim2],
            x_plot[:max_plot_samples, dim3],
            c=y_indices,
            cmap=cmap,
            s=2,
            alpha=0.6,
            vmin=0,
            vmax=n_classes - 1,
        )

        ax.set_xlabel(f"D{dim1 + 1}", fontsize=10)
        ax.set_ylabel(f"D{dim2 + 1}", fontsize=10)
        ax.set_zlabel(f"D{dim3 + 1}", fontsize=10)
        ax.set_title(f"D{dim1 + 1}-D{dim2 + 1}-D{dim3 + 1}", fontsize=11)

        # Set smaller viewing angle and remove ticks for cleaner look
        ax.view_init(elev=20, azim=45)
        ax.tick_params(labelsize=8)

        # Add MI info if available
        if "mi" in metadata:
            dim_key = f"{dim1 + 1}-{dim2 + 1}-{dim3 + 1}"
            mi_value = metadata["mi"].get(dim_key, 0.0)
            ax.text2D(
                0.02,
                0.98,
                f"MI: {mi_value:.3f}",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment="top",
                fontsize=8,
            )

    # Add overall title and colorbar
    plt.suptitle(
        f"All 3D Dimension Combinations {plot_info}\n"
        f"Total: {len(y_samples):,} samples, {n_classes} classes",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Adjust layout to prevent cutoff in grid plot BEFORE adding colorbar
    plt.subplots_adjust(
        left=0.05, right=0.75, top=0.92, bottom=0.08, wspace=0.3, hspace=0.4
    )

    # Add colorbar in the reserved space
    if n_triplets > 0:
        # Create a mappable for the colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_classes - 1)
        )
        sm.set_array([])

        # Create colorbar with more padding to avoid overlap
        cbar_ax = fig.add_axes([0.78, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Class Label", fontweight="bold", fontsize=12)
        cbar.set_ticks(range(n_classes))
        cbar.set_ticklabels([f"Label {label}" for label in unique_labels])
        cbar.ax.tick_params(labelsize=10)

    # Save comprehensive 3D grid plot
    plt.savefig(
        save_dir / "dimension_combinations_3d_grid.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.2,
    )
    plt.close()

    print("  Saved: dimension_combinations_3d_grid.png")
    print("✅ All 3D dimension combination plots completed!")
