#!/usr/bin/env python
"""Verification script to ensure all implementations produce the same results."""

import numpy as np
from absl.testing import absltest
from sklearn.datasets import make_blobs

# Use the base EnhancedSyntheticDistribution class for testing both sampling strategies
from bm.synthetic_distribution import SyntheticDistribution


def verify_same_results(
    n_samples: int = 1000,
    n_features: int = 2,
    n_classes: int = 4,
    sample_size: int = 5000,
    seed: int = 42,
):
    """Verify that all implementations produce identical results."""
    # Create test data
    centers = []
    rng = np.random.default_rng(seed)
    for i in range(n_classes):
        center = rng.normal(size=n_features) * 3
        centers.append(center)

    x, y, centers = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=1.0,
        random_state=seed,
        return_centers=True,
    )

    # Compute empirical probabilities
    probs = np.count_nonzero(y == np.arange(n_classes)[:, None], axis=1) / y.size

    # Define grid bounds
    grid_size = 50 if n_features > 2 else 100
    bounds = np.empty(shape=(n_features,), dtype=object)
    for i in range(n_features):
        bounds[i] = np.linspace(x[:, i].min(), x[:, i].max(), grid_size + 1)

    # Compute joint conditional probabilities
    joint_cond_probs = np.empty(shape=(1,), dtype=object)
    joint_cond_probs[0] = np.empty(shape=(n_classes,) + (grid_size,) * n_features)

    for class_idx in range(n_classes):
        # Ensure there are samples for the current class before computing histogram
        if np.count_nonzero(y == class_idx) > 0:
            joint_cond_probs[0][class_idx] = np.histogramdd(
                sample=x[y == class_idx, :],
                bins=tuple(bounds[i] for i in range(n_features)),
                density=False,
            )[0] / np.count_nonzero(y == class_idx)
        else:
            # Handle case where a class has no samples
            joint_cond_probs[0][class_idx] = np.zeros((grid_size,) * n_features)

    # Create distribution instance
    dist = SyntheticDistribution(
        symbol_probabilities=probs,
        interval_bounds=bounds,
        cell_probabilities=joint_cond_probs,
    )

    # Sample using original sampling method
    x_orig_sampling, y_orig_sampling = dist.sample_data(
        num_samples=sample_size,
        seed=seed,
        use_direct_sampling=False,
        use_numba=False,
    )

    # Sample using direct sampling method (NumPy implementation)
    x_direct_sampling_numpy, y_direct_sampling_numpy = dist.sample_data(
        num_samples=sample_size,
        seed=seed,
        use_direct_sampling=True,
        use_numba=False,
    )

    # Sample using direct sampling method (Numba implementation)
    x_direct_sampling_numba, y_direct_sampling_numba = dist.sample_data(
        num_samples=sample_size,
        seed=seed,
        use_direct_sampling=True,  # Numba path requires direct sampling
        use_numba=True,
    )

    # Verify results
    # Check Y arrays (symbols) - All implementations should produce the same sequence of symbols
    y_match_direct_numpy = np.array_equal(
        y_orig_sampling,
        y_direct_sampling_numpy,
    )

    y_match_numba = np.array_equal(
        y_orig_sampling,  # Compare Numba with original for stricter check
        y_direct_sampling_numba,
    )

    # Check X arrays (continuous values) - Continuous values should be close
    x_match_direct_numpy = np.allclose(
        x_orig_sampling,
        x_direct_sampling_numpy,
        rtol=1e-10,
    )

    x_match_numba = np.allclose(
        x_orig_sampling,  # Compare Numba with original for stricter check
        x_direct_sampling_numba,
        rtol=1e-10,
    )

    # Check mutual information computation
    coords = set(range(1, n_features + 1))
    # get_mutual_information and get_symbol_entropy are deterministic and do not depend on the sampling strategy used for sampling data.
    # We call them once on the created distribution instance.
    mi = dist.get_mutual_information(
        coordinates=coords,
        show_tqdm=False,
    )  # Disable tqdm in tests

    # Check entropy computation
    entropy = dist.get_symbol_entropy()

    # Return detailed comparison
    return {
        "y_match_direct_numpy": y_match_direct_numpy,
        "y_match_numba": y_match_numba,
        "x_match_direct_numpy": x_match_direct_numpy,
        "x_match_numba": x_match_numba,
        "mi": mi,  # Not directly tested for matching between sampling methods, but returned
        "entropy": entropy,  # Not directly tested for matching between sampling methods, but returned
    }


class EnhancedSyntheticDistributionTest(absltest.TestCase):
    """Test the EnhancedSyntheticDistribution class."""

    def test_same_results_more_features_classes(self) -> None:
        """Verify with more features and classes."""
        results = verify_same_results(
            n_samples=500,
            n_features=3,
            n_classes=6,
            sample_size=2000,
        )
        self.assertTrue(
            results["y_match_direct_numpy"],
            "More features/classes Y array (direct numpy vs original) verification failed.",
        )
        self.assertTrue(
            results["x_match_direct_numpy"],
            "More features/classes X array (direct numpy vs original) verification failed.",
        )
        self.assertTrue(
            results["y_match_numba"],
            "More features/classes Y array (numba vs original) verification failed.",
        )
        self.assertTrue(
            results["x_match_numba"],
            "More features/classes X array (numba vs original) verification failed.",
        )

    def test_same_results_various_configs(self) -> None:
        """Verify with various sample sizes, dimensions, and classes."""
        # Test 1: Large sample size
        results_large_sample = verify_same_results(
            n_samples=2000,
            n_features=2,
            n_classes=8,
            sample_size=5000,  # Increased sample size for better coverage
        )
        self.assertTrue(
            results_large_sample["y_match_direct_numpy"],
            "Large sample size Y array (direct numpy vs original) verification failed.",
        )
        self.assertTrue(
            results_large_sample["x_match_direct_numpy"],
            "Large sample size X array (direct numpy vs original) verification failed.",
        )
        self.assertTrue(
            results_large_sample["y_match_numba"],
            "Large sample size Y array (numba vs original) verification failed.",
        )
        self.assertTrue(
            results_large_sample["x_match_numba"],
            "Large sample size X array (numba vs original) verification failed.",
        )

        # Test 2: Very small sample size
        results_small_sample = verify_same_results(
            n_samples=50,
            n_features=2,
            n_classes=2,
            sample_size=10,
        )
        self.assertTrue(
            results_small_sample["y_match_direct_numpy"],
            "Very small sample size Y array (direct numpy vs original) verification failed.",
        )
        self.assertTrue(
            results_small_sample["x_match_direct_numpy"],
            "Very small sample size X array (direct numpy vs original) verification failed.",
        )
        self.assertTrue(
            results_small_sample["y_match_numba"],
            "Very small sample size Y array (numba vs original) verification failed.",
        )
        self.assertTrue(
            results_small_sample["x_match_numba"],
            "Very small sample size X array (numba vs original) verification failed.",
        )

        # Test 3: High dimensions
        results_high_dim = verify_same_results(
            n_samples=1000,
            n_features=5,
            n_classes=3,
            sample_size=1000,  # Increased sample size
        )
        self.assertTrue(
            results_high_dim["y_match_direct_numpy"],
            "Higher dimensions Y array (direct numpy vs original) verification failed.",
        )
        self.assertTrue(
            results_high_dim["x_match_direct_numpy"],
            "Higher dimensions X array (direct numpy vs original) verification failed.",
        )
        self.assertTrue(
            results_high_dim["y_match_numba"],
            "Higher dimensions Y array (numba vs original) verification failed.",
        )
        self.assertTrue(
            results_high_dim["x_match_numba"],
            "Higher dimensions X array (numba vs original) verification failed.",
        )

        # Test 4: Many classes
        results_many_classes = verify_same_results(
            n_samples=2000,
            n_features=2,
            n_classes=10,
            sample_size=1000,
        )
        self.assertTrue(
            results_many_classes["y_match_direct_numpy"],
            "Many classes Y array (direct numpy vs original) verification failed.",
        )
        self.assertTrue(
            results_many_classes["x_match_direct_numpy"],
            "Many classes X array (direct numpy vs original) verification failed.",
        )
        self.assertTrue(
            results_many_classes["y_match_numba"],
            "Many classes Y array (numba vs original) verification failed.",
        )
        self.assertTrue(
            results_many_classes["x_match_numba"],
            "Many classes X array (numba vs original) verification failed.",
        )

    def test_determinism(self) -> None:
        """Test that same seed produces same results across multiple runs for both strategies."""
        # Create test data
        x, y, centers = make_blobs(
            n_samples=500,
            n_features=2,
            centers=4,
            cluster_std=1.0,
            random_state=1234,
            return_centers=True,
        )

        probs = np.count_nonzero(y == np.arange(4)[:, None], axis=1) / y.size
        bounds = np.empty(shape=(2,), dtype=object)
        for i in range(2):
            bounds[i] = np.linspace(x[:, i].min(), x[:, i].max(), 51)

        joint_cond_probs = np.empty(shape=(1,), dtype=object)
        joint_cond_probs[0] = np.empty(shape=(4, 50, 50))

        for class_idx in range(4):
            if np.count_nonzero(y == class_idx) > 0:
                joint_cond_probs[0][class_idx] = np.histogramdd(
                    sample=x[y == class_idx, :],
                    bins=(bounds[0], bounds[1]),
                    density=False,
                )[0] / np.count_nonzero(y == class_idx)
            else:
                joint_cond_probs[0][class_idx] = np.zeros((50, 50))

        dist = SyntheticDistribution(
            symbol_probabilities=probs,
            interval_bounds=bounds,
            cell_probabilities=joint_cond_probs,
        )

        # Test determinism for original sampling method
        results_orig_sampling = []
        for i in range(3):
            x_sample, y_sample = dist.sample_data(
                num_samples=1000,
                seed=123,
                use_direct_sampling=False,
                use_numba=False,
            )
            results_orig_sampling.append((x_sample.copy(), y_sample.copy()))

        # Check all runs are identical for original sampling
        all_match_original = True
        for i in range(1, 3):
            if not np.array_equal(
                results_orig_sampling[0][1],
                results_orig_sampling[i][1],
            ):
                all_match_original = False
                break
            if not np.allclose(
                results_orig_sampling[0][0],
                results_orig_sampling[i][0],
            ):
                all_match_original = False
                break

        self.assertTrue(all_match_original, "Original Sampling determinism failed.")

        # Test determinism for direct sampling method (NumPy)
        results_direct_sampling_numpy = []
        for i in range(3):
            x_sample, y_sample = dist.sample_data(
                num_samples=1000,
                seed=456,  # Use a different seed for this strategy
                use_direct_sampling=True,
                use_numba=False,
            )
            results_direct_sampling_numpy.append((x_sample.copy(), y_sample.copy()))

        # Check all runs are identical for direct sampling (NumPy)
        all_match_direct_numpy = True
        for i in range(1, 3):
            if not np.array_equal(
                results_direct_sampling_numpy[0][1],
                results_direct_sampling_numpy[i][1],
            ):
                all_match_direct_numpy = False
                break
            if not np.allclose(
                results_direct_sampling_numpy[0][0],
                results_direct_sampling_numpy[i][0],
            ):
                all_match_direct_numpy = False
                break

        self.assertTrue(
            all_match_direct_numpy,
            "Direct Sampling (NumPy) determinism failed.",
        )

        # Test determinism for direct sampling method (Numba)
        results_direct_sampling_numba = []
        for i in range(3):
            x_sample, y_sample = dist.sample_data(
                num_samples=1000,
                seed=789,  # Use a different seed for this strategy
                use_direct_sampling=True,
                use_numba=True,
            )
            results_direct_sampling_numba.append((x_sample.copy(), y_sample.copy()))

        # Check all runs are identical for direct sampling (Numba)
        all_match_direct_numba = True
        for i in range(1, 3):
            if not np.array_equal(
                results_direct_sampling_numba[0][1],
                results_direct_sampling_numba[i][1],
            ):
                all_match_direct_numba = False
                break
            if not np.allclose(
                results_direct_sampling_numba[0][0],
                results_direct_sampling_numba[i][0],
            ):
                all_match_direct_numba = False
                break

        self.assertTrue(
            all_match_direct_numba,
            "Direct Sampling (Numba) determinism failed.",
        )


if __name__ == "__main__":
    # The tests are automatically discovered and run by absl.testing.main()
    absltest.main()
