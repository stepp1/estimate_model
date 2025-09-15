"""Dataset utilities for the benchmark.

Information and Decision Systems Group - FCFM - Universidad de Chile
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.datasets import load_iris, make_blobs, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def generate_data_sklearn(
    dataset_name: str,
    hp_config: dict[str, Any],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic datasets using scikit-learn based on provided hyperparameters."""
    rng = np.random.default_rng(seed)
    if dataset_name == "blobs":
        hp_config.pop("return_centers", None)
        x, y, _ = make_blobs(random_state=seed, **hp_config, return_centers=True)
    elif dataset_name == "moons":
        x, y = make_moons(random_state=seed, **hp_config)
    elif dataset_name == "classification":
        x, y = make_classification(random_state=seed, **hp_config)
    elif dataset_name == "iris":
        x, y = load_iris(return_X_y=True)
        # Simulate HP variation for fixed datasets
        if "n_samples" in hp_config and hp_config["n_samples"] < x.shape[0]:
            idx = rng.choice(x.shape[0], hp_config["n_samples"], replace=False)
            x, y = x[idx], y[idx]
        if "noise" in hp_config and hp_config["noise"] > 0:
            x = x + rng.normal(0, hp_config["noise"], x.shape)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return x, y


@dataclass
class Dataset:
    """A dataset with a dataloaders and metadata.

    Parameters
    ----------
    x: np.ndarray
        The features of the dataset.
    y: np.ndarray
        The labels of the dataset.
    train_loader: DataLoader
        The dataloader for the training set.
    test_loader: DataLoader
        The dataloader for the test set.
    metadata: dict[str, Any]
        The metadata of the dataset.
            Should include:
            - dataset_name: str
            - hp_config: dict[str, Any]
            - batch_size: int
            - test_size: float
            - random_state: int
            - device: str
    """

    x: np.ndarray
    y: np.ndarray
    train_loader: DataLoader
    test_loader: DataLoader
    metadata: dict[str, Any]


def prepare_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    batch_size: int = 64,
    random_state: int = 42,
    device: str = "cpu",
) -> tuple[DataLoader, DataLoader]:
    """Splits data, scales features, and creates PyTorch DataLoaders."""
    # Scale features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False)

    return train_loader, test_loader


def prepare_dataset(
    dataset_name: str,
    hp_config: dict[str, Any],
    test_size: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu",
) -> Dataset:
    """Prepares a dataset for training and testing."""
    x, y = generate_data_sklearn(dataset_name, hp_config, seed=seed)

    train_loader, test_loader = prepare_dataloaders(
        x,
        y,
        test_size,
        batch_size,
        seed,
        device,
    )
    metadata = {
        "dataset_name": dataset_name,
        "hp_config": hp_config,
        "batch_size": batch_size,
        "test_size": test_size,
        "seed": seed,
        "device": str(device),
        "input_shape": x.shape[1:],
        "output_shape": (len(np.unique(y)),),
    }

    return Dataset(x, y, train_loader, test_loader, metadata)
