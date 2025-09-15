#!/usr/bin/env python3
"""
Train an MLP on synthetic data and compare with analytical loss.

Information and Decision Systems Group - FCFM - Universidad de Chile
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from is_demo_10_labels_5_dims import (
    data_patterns,
    generate_metadata,
    get_cell_probabilities,
    get_interval_bounds,
    get_label_probabilities,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Import the synthetic data generation functions
from bm.examples.plot_results import plot_results
from bm.synthetic_distribution import SyntheticDistribution


class MLP(nn.Module):
    """Multi-layer perceptron with 3 hidden layers of 1024 units each."""

    def __init__(self, input_dim: int, num_classes: int):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def create_distribution(selected_labels=None, base_noise=0.0003):
    """Create the synthetic distribution."""
    print(
        f"Creating distribution with labels: "
        f"{selected_labels if selected_labels else list(range(10))} "
        f"and base noise: {base_noise}"
    )

    # Get components
    label_probs = get_label_probabilities(selected_labels=selected_labels)
    num_classes = len(label_probs)
    cell_bounds = get_interval_bounds()
    patterns = data_patterns(selected_labels=selected_labels, base_noise=base_noise)
    cell_probabilities = get_cell_probabilities(
        patterns=patterns, cell_bounds=cell_bounds
    )

    # Create interval bounds array
    demo_i_bound_array = np.empty(shape=len(cell_bounds), dtype=object)
    demo_i_bound_array[:] = [
        np.asarray(a=bounds, dtype=np.float64) for bounds in cell_bounds
    ]

    # Create distribution
    distribution = SyntheticDistribution(
        symbol_probabilities=label_probs,
        interval_bounds=demo_i_bound_array,
        cell_probabilities=cell_probabilities,
    )

    return distribution, num_classes


def train_mlp(x_train, y_train, x_val, y_val, num_classes, device, epochs=15):
    """Train the MLP model."""
    print(f"Training MLP on device: {device}")
    print(f"Input shape: {x_train.shape}, Output classes: {num_classes}")

    # Create model
    model = MLP(input_dim=x_train.shape[1], num_classes=num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(x_train).to(device), torch.LongTensor(y_train).to(device)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(x_val).to(device), torch.LongTensor(y_val).to(device)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512 * 4, shuffle=False)

    # Training loop
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Create data loaders
    train_losses = []
    val_losses = []
    val_accuracies = []
    early_stop_patience = 10
    best_val_loss = float("inf")
    early_stop_counter = 0

    print("\nTraining:")
    print("Epoch | Train Loss | Val Loss | Val Acc |    LR    | Time")
    print("------|------------|----------|---------|----------|------")

    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        lr_scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        epoch_time = time.time() - start_time
        print(
            f"{epoch + 1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {val_accuracy:7.3f} | {lr_scheduler.get_last_lr()[0]:6.6f} | {epoch_time:5.1f}s"
        )
        if val_loss < best_val_loss * 0.99:  # 1% relative improvement
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_val_accuracy": val_accuracies[-1],
    }


def bits_to_nats(bits_value):
    """Convert loss from bits to nats."""
    return bits_value * np.log(2)


def nats_to_bits(nats_value):
    """Convert loss from nats to bits."""
    return nats_value / np.log(2)


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP and compare with analytical loss"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--num_samples", type=int, default=50_000, help="Number of samples"
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument(
        "--labels", type=str, default=None, help="Comma-separated labels (0-9)"
    )
    parser.add_argument(
        "--save_results", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--base_noise", type=float, default=0.0, help="Base noise level"
    )
    parser.add_argument(
        "--plot_results",
        type=bool,
        default=True,
        help="Plot results",
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Parse selected labels
    if args.labels is not None:
        selected_labels = [int(x) for x in args.labels.split(",")]
        for label in selected_labels:
            if label < 0 or label > 9:
                raise ValueError(f"Label {label} is out of range [0-9]")
    else:
        selected_labels = None

    # Create distribution and generate data
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 60)

    distribution, num_classes = create_distribution(
        selected_labels, base_noise=args.base_noise
    )

    # Generate data
    print(f"\nSampling {args.num_samples:,} data points...")
    x_samples, y_samples = distribution.sample_data(
        num_samples=args.num_samples, seed=args.seed
    )
    print(f"Data shape: X={x_samples.shape}, Y={y_samples.shape}")

    # Fix labels to be 0-based (synthetic distribution uses 1-based indexing)
    print(f"Original label range: {y_samples.min()} to {y_samples.max()}")
    y_samples = y_samples - 1  # Convert from 1-based to 0-based indexing

    # Remap labels to consecutive indices [0, num_classes-1] if needed
    if selected_labels is not None:
        print(f"Remapping labels {selected_labels} to [0, {num_classes - 1}]")
        # Create mapping from original labels (now 0-based) to new consecutive labels
        label_mapping = {
            orig_label: new_label
            for new_label, orig_label in enumerate(selected_labels)
        }
        y_samples_remapped = np.array(
            [label_mapping.get(label, label) for label in y_samples]
        )
        y_samples = y_samples_remapped

    # Verify label range
    print(f"Final label range: {y_samples.min()} to {y_samples.max()}")
    print(f"Expected range: 0 to {num_classes - 1}")
    assert y_samples.min() >= 0 and y_samples.max() < num_classes, (
        f"Labels out of range! Got {y_samples.min()}-{y_samples.max()}, expected 0-{num_classes - 1}"
    )

    # Generate metadata with analytical loss
    metadata = generate_metadata(distribution, n_dims=5, num_classes=num_classes)

    # Get analytical loss in bits and convert to nats
    analytical_loss_bits = distribution.compute_analytical_loss(
        metadata["mi"]["joint"], metadata["entropy_y"]
    )
    analytical_loss_nats = bits_to_nats(analytical_loss_bits)

    print("\nANALYTICAL RESULTS:")
    print(f"Joint MI: {metadata['mi']['joint']:.4f} bits")
    print(f"Entropy Y: {metadata['entropy_y']:.4f} bits")
    print(
        f"Analytical loss: {analytical_loss_bits:.4f} bits = {analytical_loss_nats:.4f} nats"
    )
    print(f"Theoretical max accuracy: {metadata['max_acc1']:.4f}")
    if metadata["max_acc1"] < 0.6:
        print("❌ Theoretical max accuracy is less than 0.6")
        print("❌ This is not a good distribution")
        print("❌ Please try again with a different base noise")
        return

    # Split data
    x_train, x_val, y_train, y_val = train_test_split(
        x_samples, y_samples, test_size=0.2, random_state=args.seed, stratify=y_samples
    )

    print(f"\nTrain set: {x_train.shape[0]:,} samples")
    print(f"Val set: {x_val.shape[0]:,} samples")

    # Train MLP
    print("\n" + "=" * 60)
    print("MLP TRAINING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, training_results = train_mlp(
        x_train, y_train, x_val, y_val, num_classes, device, epochs=args.epochs
    )

    # Results comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    final_loss_nats = training_results["final_val_loss"]
    final_loss_bits = nats_to_bits(final_loss_nats)
    final_accuracy = training_results["final_val_accuracy"]

    print(
        f"Analytical loss:    {analytical_loss_bits:.4f} bits = {analytical_loss_nats:.4f} nats"
    )
    print(
        f"MLP final loss:     {final_loss_bits:.4f} bits = {final_loss_nats:.4f} nats"
    )
    print(f"Loss difference:    {abs(final_loss_nats - analytical_loss_nats):.4f} nats")
    print(f"Loss ratio (MLP/Analytical): {final_loss_nats / analytical_loss_nats:.3f}")
    print()
    print(f"Theoretical max accuracy: {metadata['max_acc1']:.4f}")
    print(f"MLP final accuracy:       {final_accuracy:.4f}")
    print(f"Accuracy difference:      {abs(final_accuracy - metadata['max_acc1']):.4f}")
    print()

    if final_loss_nats <= analytical_loss_nats * 1.1:
        print("✅ MLP reached near-optimal loss (within 10% of analytical)")
    else:
        print("❌ MLP did not reach analytical loss")

    if final_accuracy >= metadata["max_acc1"] * 0.9:
        print("✅ MLP reached near-optimal accuracy (within 10% of theoretical)")
    else:
        print("❌ MLP did not reach theoretical accuracy")

    # Save results if requested
    if args.save_results:
        save_dir = Path(args.save_results)
        save_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "analytical_loss_bits": analytical_loss_bits,
            "analytical_loss_nats": analytical_loss_nats,
            "theoretical_max_accuracy": metadata["max_acc1"],
            "mlp_final_loss_nats": final_loss_nats,
            "mlp_final_loss_bits": final_loss_bits,
            "mlp_final_accuracy": final_accuracy,
            "training_history": training_results,
            "metadata": metadata,
            "args": vars(args),
        }

        with open(save_dir / "mlp_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {save_dir}/mlp_results.json")

    if args.plot_results:
        plot_results(save_dir / "mlp_results.json")
        print("Comparison plot saved to: mlp_comparison_analysis.png")


if __name__ == "__main__":
    main()
