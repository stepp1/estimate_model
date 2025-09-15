"""
Sample code for the Enhanced Synthetic Distribution class.

Information and Decision Systems Group - FCFM - Universidad de Chile
"""

import argparse
import itertools
from pathlib import Path
from typing import List, Set

import numpy as np

from bm.examples.sample_plots import create_dimension_combination_plots
from bm.synthetic_distribution import FloatArray, ObjectArray, SyntheticDistribution


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


def main(args: argparse.Namespace):
    demo_s_prob: FloatArray = np.asarray(
        a=[0.4, 0.6], dtype=np.float64
    )  # Steps A.1 and A.2
    demo_i_bounds: List[
        List[float]
    ] = [  # Steps A.5 and A.6 (step A.3 and A.4 are performed later)
        [-0.5, 0.0, 0.5],
        [-1.0, 0.0, 1.0, 2.0],
        [-2.0, -1.0, -0.5, 0.0, 2.0],
    ]
    # Steps A.3, A.4, and A.10 (Steps A.7 and A.9 are implicit, and step A.8 is performed at methods gamma and
    # inverse_gamma of class EnhancedSyntheticDistribution)
    demo_c_prob: List[List] = [
        [[0.5, 0.5], [0.2, 0.8]],
        [
            [[0.2, 0.05, 0.2, 0.04], [0.0, 0.01, 0.0, 0.15], [0.1, 0.1, 0.0, 0.15]],
            [[0.0, 0.1, 0.5, 0.03], [0.02, 0.05, 0.01, 0.24], [0.03, 0.0, 0.02, 0.0]],
        ],
    ]

    demo_i_bound_array: ObjectArray = np.empty(shape=len(demo_i_bounds), dtype=object)
    demo_i_bound_array[:] = [
        np.asarray(a=bounds, dtype=np.float64) for bounds in demo_i_bounds
    ]

    demo_c_prob_array: ObjectArray = np.empty(shape=len(demo_c_prob), dtype=object)
    demo_c_prob_array[:] = [
        np.asarray(a=cell_cond_probs, dtype=np.float64)
        for cell_cond_probs in demo_c_prob
    ]

    distribution: SyntheticDistribution = SyntheticDistribution(
        symbol_probabilities=demo_s_prob,
        interval_bounds=demo_i_bound_array,
        cell_probabilities=demo_c_prob_array,
    )
    x_samples, y_samples = distribution.sample_data(
        num_samples=args.num_samples, seed=args.seed
    )
    metadata = generate_metadata(distribution, n_dims=3, num_classes=2)
    print("Symbol entropy:", distribution.get_symbol_entropy())
    coordinates_list: List[Set[int]] = [
        {1, 2, 3},
        {1, 2},
        {1, 3},
        {2, 3},
        {1},
        {2},
        {3},
        set(),
    ]
    for coords in coordinates_list:
        print(
            f"MI with coordinates {coords}: {distribution.get_mutual_information(coordinates=coords)}"
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.plot:
        create_dimension_combination_plots(
            x_samples=x_samples,
            y_samples=y_samples,
            metadata=metadata,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the demo")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--num_samples", type=int, default=50_000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save data and metadata"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to do plots",
    )
    args = parser.parse_args()
    main(args=args)
