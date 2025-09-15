"""Training module for the MLP model.

Information and Decision Systems Group - FCFM - Universidad de Chile
"""

from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) for classification."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_sizes: tuple[int, ...],
    ) -> None:
        """Initialize the MLP."""
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.network(x)


def train_mlp_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    metadata: dict[str, Any],
    num_epochs: int = 200,
    learning_rate: float = 0.01,
    device: str = "cpu",
    verbose: bool | None = None,
) -> dict[str, Any]:
    """Trains a PyTorch MLP model and returns its final test loss and training metadata."""
    is_binary = len(metadata["output_shape"]) == 1 and metadata["output_shape"][0] == 2
    output_dim = metadata["output_shape"][0] if not is_binary else 1

    if len(metadata["input_shape"]) != 1:
        raise ValueError(
            f"Input shape must be 1D, but got {metadata['input_shape']}",
        )

    mlp_hidden_layer_sizes = (32, 16) if metadata["input_shape"][0] < 3 else (128, 64)
    model = SimpleMLP(
        metadata["input_shape"][0],
        output_dim,
        mlp_hidden_layer_sizes,
    ).to(
        device,
    )
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.7,
        # nesterov=True,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "epochs": [],
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            if is_binary:
                y_batch = y_batch.float().unsqueeze(1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        # Testing
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch)
                if is_binary:
                    y_batch = y_batch.float().unsqueeze(1)
                loss = criterion(outputs, y_batch)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += y_batch.size(0)
                test_correct += (predicted == y_batch).sum().item()

        # Record metrics
        history["train_loss"].append(train_loss / len(train_loader))
        history["test_loss"].append(test_loss / len(test_loader))
        history["train_accuracy"].append(100 * train_correct / train_total)
        history["test_accuracy"].append(100 * test_correct / test_total)
        history["epochs"].append(epoch + 1)

        if (epoch + 1) % 10 == 0 and verbose:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {history['train_loss'][-1]:.4f}, "
                f"Test Loss: {history['test_loss'][-1]:.4f}, "
                f"Train Acc: {history['train_accuracy'][-1]:.2f}%, "
                f"Test Acc: {history['test_accuracy'][-1]:.2f}%",
            )
        # Simple early stopping
        if (
            epoch > 50
            and history["test_loss"][-1] > history["test_loss"][-2]
            and history["test_loss"][-2] > history["test_loss"][-3]
        ):
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    final_test_loss = history["test_loss"][-1]
    final_test_accuracy = history["test_accuracy"][-1]
    history["n_epochs"] = len(history["epochs"])
    history["final_test_loss"] = final_test_loss
    history["final_test_accuracy"] = final_test_accuracy
    return history
