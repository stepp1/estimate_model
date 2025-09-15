"""Estimate functionality for the Synthetic Distribution."""

from typing import Any

import numpy as np
from absl import logging

from bm.synthetic_distribution import SyntheticDistribution


def get_marginal_dist(
    x: np.ndarray, y: np.ndarray, grid_size: int = 10
) -> SyntheticDistribution:
    """Get the marginal distribution of the dataset."""
    n_classes = len(np.unique(y))
    n_features = x.shape[1]
    
    probs = SyntheticDistribution.compute_empirical_probs(y, n_classes)
    logging.debug("Empirical probs")
    bounds = SyntheticDistribution.make_bounds(x, n_features, grid_size)
    logging.debug("Make probs")
    
    marginal_cond_probs = SyntheticDistribution.make_marginal_cond_probs(
        x,
        y,
        bounds,
        grid_size,
    )
    logging.debug("Make marginal probs")
    logging.debug(
        f"Marginal cond probs shape: {marginal_cond_probs.shape} "
        f"mean: {marginal_cond_probs[0].mean()}, {marginal_cond_probs[1].mean()}",
    )
    return SyntheticDistribution(
        symbol_probabilities=probs,
        interval_bounds=bounds,
        cell_probabilities=marginal_cond_probs,
    )


def get_joint_dist(
    x: np.ndarray, y: np.ndarray, grid_size: int = 10
) -> SyntheticDistribution:
    """Get the joint distribution of the dataset."""
    n_classes = len(np.unique(y))
    n_features = x.shape[1]
    probs = SyntheticDistribution.compute_empirical_probs(y, n_classes)
    bounds = SyntheticDistribution.make_bounds(x, n_features, grid_size)
    joint_cond_probs = SyntheticDistribution.make_joint_cond_probs(
        x,
        y,
        bounds,
        grid_size,
    )
    logging.debug(
        f"Joint cond probs shape: {joint_cond_probs.shape} "
        f"mean: {joint_cond_probs[0].mean()}",
    )
    return SyntheticDistribution(
        symbol_probabilities=probs,
        interval_bounds=bounds,
        cell_probabilities=joint_cond_probs,
    )


def estimate_synthetic_mi(
    x: np.ndarray,
    y: np.ndarray,
    grid_size: int = 10,
    skip_joint: bool = False,
    top_k_features: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Estimate the mutual information between using the Synthetic Distribution."""
    n_features = x.shape[1]

    analytical_losses = {
        "joint": 0.0,
        "marginal": 0.0,
    }
    analytical_acc1 = {
        "joint": 0.0,
        "marginal": 0.0,
    }
    logging.info("Computing marginal distribution")
    marginal_dist = get_marginal_dist(x, y, grid_size)
    marginal_metadata = {
        "symbol_entropy": marginal_dist.get_symbol_entropy(in_nats=True),
        "mi": {
            "0": marginal_dist.get_mutual_information(coordinates=set(), in_nats=True),
            **{
                f"{i}": marginal_dist.get_mutual_information(
                    coordinates={i},
                    in_nats=True,
                )
                for i in range(1, n_features + 1)
            },
        },
    }

    if top_k_features is not None and top_k_features < n_features:
        mi_values = list(marginal_metadata["mi"].values())
        top_k = np.argsort(mi_values)[-top_k_features:]
        logging.info(
            f"Top {top_k_features} features: {top_k} with mi: {[mi_values[i] for i in top_k]}"
        )
        feature_set = set(map(lambda x: int(x), top_k))
    else:
        feature_set = set(range(1, n_features + 1))

    marginal_metadata["mi"]["joint"] = marginal_dist.get_mutual_information(
        coordinates=feature_set,
        in_nats=True,
    )

    if not skip_joint:
        logging.info("Computing joint distribution")
        joint_dist = get_joint_dist(x, y, grid_size)
        joint_metadata = {
            "symbol_entropy": joint_dist.get_symbol_entropy(in_nats=True),
            "mi": {
                "0": joint_dist.get_mutual_information(coordinates=set(), in_nats=True),
                **{
                    f"{i}": joint_dist.get_mutual_information(
                        coordinates={i}, in_nats=True
                    )
                    for i in range(1, n_features + 1)
                },
            },
        }
        if top_k_features is not None and top_k_features < n_features:
            mi_values = list(joint_metadata["mi"].values())
            top_k = np.argsort(mi_values)[-top_k_features:]
            logging.info(
                f"Top {top_k_features} features: {top_k} with mi: {[mi_values[i] for i in top_k]}"
            )
            feature_set = set(map(lambda x: int(x), top_k))
        else:
            feature_set = set(range(1, n_features + 1))

        joint_metadata["mi"]["joint"] = joint_dist.get_mutual_information(
            coordinates=feature_set,
            in_nats=True,
        )

    else:
        joint_metadata = marginal_metadata

    # Calculate analytical losses
    analytical_losses["joint"] = SyntheticDistribution.compute_analytical_loss(
        mi=joint_metadata["mi"]["joint"],
        entropy_y=joint_metadata["symbol_entropy"],
    )

    analytical_losses["marginal"] = SyntheticDistribution.compute_analytical_loss(
        mi=marginal_metadata["mi"]["joint"],
        entropy_y=marginal_metadata["symbol_entropy"],
    )
    joint_metadata["loss"] = analytical_losses["joint"]
    marginal_metadata["loss"] = analytical_losses["marginal"]

    try:
        analytical_acc1["marginal"] = marginal_dist.get_optimal_acc1(x, y)
        analytical_acc1["joint"] = joint_dist.get_optimal_acc1(x, y)
    except NotImplementedError as e:
        logging.warning(
            "Optimal accuracy not implemented. Setting to 0.0. Error: %s",
            e,
        )
        analytical_acc1["marginal"] = 0.0
        analytical_acc1["joint"] = 0.0

    joint_metadata["acc1"] = analytical_acc1["joint"]
    marginal_metadata["acc1"] = analytical_acc1["marginal"]

    return joint_metadata, marginal_metadata, analytical_losses
