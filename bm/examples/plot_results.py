#!/usr/bin/env python3
"""
Plot MLP training results and compare with analytical bounds.

Information and Decision Systems Group - FCFM - Universidad de Chile
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results(results_file):
    """Plot training results with analytical comparisons."""

    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)

    # Extract data
    train_losses = results["training_history"]["train_losses"]
    val_losses = results["training_history"]["val_losses"]
    val_accuracies = results["training_history"]["val_accuracies"]

    analytical_loss_nats = results["analytical_loss_nats"]
    analytical_loss_bits = results["analytical_loss_bits"]
    theoretical_max_accuracy = results["theoretical_max_accuracy"]

    epochs = range(1, len(train_losses) + 1)

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Training and Validation Loss (nats)
    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    ax1.axhline(
        y=analytical_loss_nats,
        color="green",
        linestyle="--",
        label=f"Analytical Loss ({analytical_loss_nats:.4f} nats)",
        linewidth=2,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (nats)")
    ax1.set_title("Training Progress: Loss in Nats")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training and Validation Loss (bits)
    train_losses_bits = [loss / np.log(2) for loss in train_losses]
    val_losses_bits = [loss / np.log(2) for loss in val_losses]

    ax2.plot(epochs, train_losses_bits, "b-", label="Train Loss", linewidth=2)
    ax2.plot(epochs, val_losses_bits, "r-", label="Validation Loss", linewidth=2)
    ax2.axhline(
        y=analytical_loss_bits,
        color="green",
        linestyle="--",
        label=f"Analytical Loss ({analytical_loss_bits:.4f} bits)",
        linewidth=2,
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (bits)")
    ax2.set_title("Training Progress: Loss in Bits")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Validation Accuracy
    ax3.plot(epochs, val_accuracies, "purple", linewidth=2, label="Validation Accuracy")
    ax3.axhline(
        y=theoretical_max_accuracy,
        color="orange",
        linestyle="--",
        label=f"Theoretical Max ({theoretical_max_accuracy:.4f})",
        linewidth=2,
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Validation Accuracy Progress")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.5, 1.0)

    # Plot 4: Loss Ratio over Training
    loss_ratios = [val_loss / analytical_loss_nats for val_loss in val_losses]
    ax4.plot(epochs, loss_ratios, "red", linewidth=2, label="MLP/Analytical Loss Ratio")
    ax4.axhline(
        y=1.0, color="green", linestyle="-", label="Optimal (Ratio = 1.0)", linewidth=2
    )
    ax4.axhline(
        y=1.1,
        color="orange",
        linestyle="--",
        label="Near-Optimal (Ratio = 1.1)",
        linewidth=1,
    )
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss Ratio (MLP/Analytical)")
    ax4.set_title("Loss Ratio: How Close to Analytical Bound")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path(results_file).parent
    plot_path = output_dir / "mlp_training_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")

    plt.show()

    # Print summary
    print("=" * 60)
    print("MLP vs ANALYTICAL LOSS COMPARISON")
    print("=" * 60)
    print("Architecture: 3 hidden layers of 1024 units each")
    print(
        f"Training: {results['args']['epochs']} epochs with SGD momentum on {results['args']['num_samples']:,} samples"
    )
    print()
    print(
        f"Analytical loss:    {analytical_loss_bits:.4f} bits = {analytical_loss_nats:.4f} nats"
    )
    print(
        f"MLP final loss:     {results['mlp_final_loss_bits']:.4f} bits = {results['mlp_final_loss_nats']:.4f} nats"
    )
    print(
        f"Loss difference:    {abs(results['mlp_final_loss_nats'] - analytical_loss_nats):.4f} nats"
    )
    print(
        f"Loss ratio:         {results['mlp_final_loss_nats'] / analytical_loss_nats:.3f}"
    )
    print()
    print(f"Theoretical max accuracy: {theoretical_max_accuracy:.4f}")
    print(f"MLP final accuracy:       {results['mlp_final_accuracy']:.4f}")
    print(
        f"Accuracy difference:      {abs(results['mlp_final_accuracy'] - theoretical_max_accuracy):.4f}"
    )
    print()

    # Success metrics
    loss_within_10_percent = (
        results["mlp_final_loss_nats"] <= analytical_loss_nats * 1.1
    )
    accuracy_within_10_percent = (
        results["mlp_final_accuracy"] >= theoretical_max_accuracy * 0.9
    )

    print("SUCCESS METRICS:")
    if loss_within_10_percent:
        print("✅ MLP reached near-optimal loss (within 10% of analytical)")
    else:
        print("❌ MLP did not reach near-optimal loss")

    if accuracy_within_10_percent:
        print("✅ MLP reached near-optimal accuracy (within 10% of theoretical)")
    else:
        print("❌ MLP did not reach near-optimal accuracy")

    # Additional insights
    print("\nINSIGHTS:")
    print(
        f"• The MLP achieved {results['mlp_final_loss_nats'] / analytical_loss_nats:.1f}x the analytical loss"
    )
    print(
        f"• This suggests the MLP is {(1 - analytical_loss_nats / results['mlp_final_loss_nats']) * 100:.1f}% away from optimal"
    )
    print(
        f"• The MLP accuracy exceeded theoretical max by {(results['mlp_final_accuracy'] - theoretical_max_accuracy) * 100:.2f}%"
    )

    return results


def create_comparison_plot(results_files):
    """Create comparison plot for multiple experiment results."""

    all_results = []
    for file in results_files:
        with open(file, "r") as f:
            results = json.load(f)
        all_results.append((Path(file).stem, results))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    colors = ["blue", "red", "green", "purple", "orange", "brown"]

    for i, (name, results) in enumerate(all_results):
        color = colors[i % len(colors)]

        train_losses = results["training_history"]["train_losses"]
        val_losses = results["training_history"]["val_losses"]
        val_accuracies = results["training_history"]["val_accuracies"]
        epochs = range(1, len(train_losses) + 1)

        # Loss plots
        ax1.plot(epochs, val_losses, color=color, label=f"{name} Val Loss", linewidth=2)
        ax2.plot(
            epochs, val_accuracies, color=color, label=f"{name} Accuracy", linewidth=2
        )

        # Final metrics
        final_loss_ratio = (
            results["mlp_final_loss_nats"] / results["analytical_loss_nats"]
        )
        ax3.bar(i, final_loss_ratio, color=color, alpha=0.7, label=name)

        accuracy_vs_theoretical = (
            results["mlp_final_accuracy"] / results["theoretical_max_accuracy"]
        )
        ax4.bar(i, accuracy_vs_theoretical, color=color, alpha=0.7, label=name)

    # Formatting
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss (nats)")
    ax1.set_title("Validation Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Validation Accuracy Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_ylabel("Final Loss Ratio (MLP/Analytical)")
    ax3.set_title("Final Loss Ratio Comparison")
    ax3.axhline(y=1.0, color="green", linestyle="--", alpha=0.7, label="Optimal")
    ax3.set_xticks(range(len(all_results)))
    ax3.set_xticklabels([name for name, _ in all_results], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.set_ylabel("Accuracy Ratio (MLP/Theoretical)")
    ax4.set_title("Accuracy vs Theoretical Comparison")
    ax4.axhline(y=1.0, color="green", linestyle="--", alpha=0.7, label="Theoretical")
    ax4.set_xticks(range(len(all_results)))
    ax4.set_xticklabels([name for name, _ in all_results], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mlp_comparison_analysis.png", dpi=150, bbox_inches="tight")
    print("Comparison plot saved to: mlp_comparison_analysis.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot MLP training results")
    parser.add_argument("results_file", type=str, help="Path to JSON results file")
    parser.add_argument(
        "--compare", nargs="+", help="Additional result files for comparison"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="single",
        choices=["single", "comparison"],
        help="Type of plot to create",
    )

    args = parser.parse_args()

    # Single file analysis
    if Path(args.results_file).exists():
        if args.plot_type == "single":
            plot_results(args.results_file)
        elif args.plot_type == "comparison":
            create_comparison_plot([args.results_file])
    else:
        print(f"Error: Results file {args.results_file} not found")
        return

    # Multi-file comparison
    if args.compare:
        all_files = [args.results_file] + args.compare
        all_files = set(all_files)
        valid_files = [f for f in all_files if Path(f).exists()]
        if len(valid_files) > 1:
            print(f"\nCreating comparison plot for {len(valid_files)} experiments...")
            create_comparison_plot(valid_files)
        else:
            print("Not enough valid files for comparison")


if __name__ == "__main__":
    main()
