"""Plotting utilities for the bm package."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import bm.estimate


def set_plot_style():
    """Sets a consistent style for all plots with bold fonts and larger text."""
    plt.style.use("seaborn-v0_8")

    # Set font sizes
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 20,
        },
    )

    # Make fonts bold
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    # Increase line widths
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["grid.linewidth"] = 1.5

    # Improve grid appearance
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["grid.linestyle"] = "--"

    # Enhance tick marks
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6

    # Set black borders
    # plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.bottom"] = True
    plt.rcParams["axes.spines.left"] = True


# Call the style function when module is imported
set_plot_style()


def plot_dataset_distribution(
    x: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    hp_config: tuple[str, dict[str, Any]],
    plot_idx: int = 0,
    save_fig: bool | None = None,
    return_fig: bool | None = None,
    add_info: bool | None = None,
    no_borders: bool | None = None,
    transparent: bool | None = None,
    top_k_features: int | None = None,
    skip_joint: bool | None = None,
) -> None | plt.Figure:
    """Plots a 2D visualization of the dataset distribution."""
    joint_metadata, marginal_metadata, analytical_loss = (
        bm.estimate.estimate_synthetic_mi(
            x,
            y,
            grid_size=10,
            top_k_features=top_k_features,
            skip_joint=skip_joint,
        )
    )
    config_name = hp_config[0]
    if x.shape[1] >= 2:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        scatter = ax.scatter(
            x[:, 0],
            x[:, 1],
            c=y,
            cmap="turbo",
            s=1,
            alpha=0.7,
        )

        if add_info is not None and add_info:
            ax.set_title(f"{config_name.replace('_', ' ').title()}")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.text(
                0.1,
                -0.3,
                f"Symbol entropy: {joint_metadata['symbol_entropy']:>8.3f} bits\n"
                f"Joint MI:      {joint_metadata['mi']['joint']:>16.3f} bits\n"
                f"Marginal MI:   {marginal_metadata['mi']['joint']:>12.3f} bits\n"
                f"Analytical Loss:  {analytical_loss['joint']:>11.3f} bits",
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "black",
                    "boxstyle": "round,pad=0.5",
                    "linewidth": 1.5,
                },
                fontsize=14,
            )
            cbar = plt.colorbar(scatter, ax=ax, label="Class")
            cbar.set_ticks(np.arange(len(np.unique(y))))
            cbar.set_ticklabels([f"Class {i}" for i in range(len(np.unique(y)))])
            plt.tight_layout()

        if save_fig is not None and save_fig:
            plt.savefig(
                f"experiment1_plot_{plot_idx}_{dataset_name}_{config_name}.png",
                dpi=300,
                transparent=transparent,
            )

        if no_borders is not None and no_borders:
            plt.xticks([])
            plt.yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

    else:
        print(
            f"Skipping 2D plot for {dataset_name} due to insufficient features ({x.shape[1]})",
        )
        return None

    if return_fig is not None and return_fig:
        return fig
    plt.close(fig)
    return None


def plot_history(
    history: dict[str, Any],
    analytical_loss: float,
    dataset_name: str,
    config_name: str,
    output_filename: str = "suite2_history.png",
    transparent: bool | None = None,
):
    """Plots the history of the MLP training."""
    fig, axs = plt.subplots(ncols=2, figsize=(10, 6))
    axs[0].plot(
        history["epochs"],
        np.array(history["train_loss"]),
        label="Train Loss",
    )
    axs[0].plot(
        history["epochs"],
        np.array(history["test_loss"]),
        label="Test Loss",
    )
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss (nats)")
    axs[0].axhline(
        analytical_loss,
        color="red",
        linestyle="--",
        label="Analytical Loss",
    )
    axs[0].legend()
    axs[1].plot(history["epochs"], history["train_accuracy"], label="Train Accuracy")
    axs[1].plot(history["epochs"], history["test_accuracy"], label="Test Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    fig.suptitle(f"MLP Training History for {dataset_name} - {config_name}")

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, transparent=transparent)
    plt.close(fig)


def plot_all_results(
    results: list[dict[str, Any]],
    dataset_name: str,
    config_names: list[str],
    output_filename: str = "suite2_all_results.png",
    transparent: bool | None = None,
):
    """Plots all results for Suite 2."""
    fig, axs = plt.subplots(
        ncols=len(results),
        figsize=(4 * len(results), 5),
        sharey=True,
    )

    for i, result in enumerate(results):
        axs[i].plot(
            result["mlp_history"]["epochs"],
            result["mlp_history"]["train_loss"],
            label="Train Loss" if i == len(results) - 1 else None,
        )
        axs[i].plot(
            result["mlp_history"]["epochs"],
            result["mlp_history"]["test_loss"],
            label="Test Loss" if i == len(results) - 1 else None,
        )
        axs[i].axhline(
            result["analytical_loss"],
            color="red",
            linestyle="--",
            label="Analytical Loss" if i == len(results) - 1 else None,
        )
        axs[i].set_title(f"{config_names[i]}")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Loss (nats)")

    # Add single legend for the whole plot
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.15, 0.5),
        fontsize=12,
    )

    fig.suptitle(f"MLP Training History for {dataset_name}")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, transparent=transparent)
    plt.close(fig)


def plot_loss_comparison(
    results: list[dict[str, Any]],
    dataset_name: str,
    config_names: list[str],
    output_filename: str = "suite2_loss_comparison.png",
    transparent: bool | None = None,
):
    """Plots MLP test loss against analytical loss estimate for Suite 2."""
    # Sort results by analytical_loss_estimate to clearly show the "forced increasing order"
    results.sort(key=lambda x: x["analytical_loss"])

    losses = [res["mlp_history"]["final_test_loss"] for res in results]
    analytical_losses = [res["analytical_loss"] for res in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(losses)), losses, marker="o", label="Empirical Test Loss")
    ax.plot(
        range(len(analytical_losses)),
        analytical_losses,
        marker="x",
        linestyle="--",
        label="Analytical Loss (Ours)",
    )

    # Use generic labels if HP configs are too long for x-axis ticks
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(
        config_names,
        rotation=45,
        ha="right",
    )

    ax.set_title(f"MLP Test Loss vs. Analytical Loss for {dataset_name}")
    ax.set_xlabel("Hyperparameter Configuration")
    ax.set_ylabel("Loss (nats)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, transparent=transparent)
    plt.close(fig)


def plot_loss_interpolation(
    results: list[dict[str, Any]],
    dataset_name: str,
    output_filename: str = "suite3_loss_interpolation.png",
    transparent: bool | None = None,
):
    """Plots loss interpolation results for Suite 3."""
    # Ensure results are sorted by interpolation step
    results.sort(key=lambda x: x["step"])

    steps = [res["alpha"] for res in results]
    mlp_losses = [res["mlp_history"]["final_test_loss"] for res in results]
    analytical_losses = [res["analytical_loss"] for res in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, mlp_losses, marker="o", label="PyTorch MLP Test Loss")
    ax.plot(
        steps,
        analytical_losses,
        marker="x",
        linestyle="--",
        label="Analytical Loss (Framework)",
    )
    ax.set_title(f"Loss Interpolation for {dataset_name} (PyTorch)")
    ax.set_xlabel("Interpolation Factor (0=Start, 1=End)")
    ax.set_ylabel("Loss (nats)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, transparent=transparent)
    plt.close(fig)
