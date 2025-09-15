import numpy as np

_CONFIGS = {}


def config_name(name: str):
    def decorator(func):
        _CONFIGS[name] = func
        return func

    return decorator


@config_name("blobs")
def get_blobs_hps(n_samples: int, seed: int):
    rng = np.random.default_rng(seed)

    return [
        # Basic dispersion spectrum
        (
            "low_dispersion",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "centers": 10,
                "cluster_std": 0.2,
            },
        ),
        (
            "moderate_dispersion",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "centers": 10,
                "cluster_std": 0.5,
            },
        ),
        (
            "high_dispersion",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "centers": 10,
                "cluster_std": 0.8,
            },
        ),
        (
            "extreme_dispersion",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "centers": 10,
                "cluster_std": 1.2,
            },
        ),
        (
            "overlapping_dispersion",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "centers": 10,
                "cluster_std": 2.0,
            },
        ),
        # Geometric patterns
        (
            "grid_pattern",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "centers": [
                    [x, y] for x in np.linspace(-5, 5, 5) for y in np.linspace(-3, 3, 2)
                ][:10],
                "cluster_std": 0.4,
            },
        ),
        (
            "spiral_pattern",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "centers": [
                    [
                        (r + 0.5 * rng.random())
                        * np.cos(2 * np.pi * r / 2),  # Add spiral winding
                        (r + 0.5 * rng.random()) * np.sin(2 * np.pi * r / 2),
                    ]
                    for r in np.linspace(1, 3, 10)  # 10 points from radius 1 to 5
                ],
                "cluster_std": 0.3,  # Tighter clusters for clearer spiral pattern
            },
        ),
        (
            "cross_pattern",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "centers": [[float(x), 0] for x in np.linspace(-8, 8, 5)]
                + [[0, float(y)] for y in np.linspace(-8, 8, 5)][:5],
                "cluster_std": 0.7,
            },
        ),
        # Special configurations
        (
            "asymmetric_clusters",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "centers": [
                    [-4, -4],
                    [4, 4],
                    [-4, 4],
                    [4, -4],
                    [0, 0],
                    [6, 0],
                    [0, 6],
                    [-6, 0],
                    [0, -6],
                    [6, 6],
                ],
                "cluster_std": 1.0,
            },
        ),
        (
            "variable_stddev",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "centers": 10,
                "cluster_std": [0.2 + 0.3 * i for i in range(10)],
            },
        ),
    ]


@config_name("classification")
def get_make_classification_hps(n_samples: int, seed: int):
    return [
        (
            "low_dim_no_noise",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "n_classes": 2,
                "n_informative": 2,
                "n_redundant": 0,
                "n_repeated": 0,
                "n_clusters_per_class": 1,
                "flip_y": 0.0,
            },
        ),
        (
            "low_dim_medium_noise",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "n_classes": 2,
                "n_informative": 2,
                "n_redundant": 0,
                "n_repeated": 0,
                "n_clusters_per_class": 1,
                "flip_y": 0.01,
            },
        ),
        (
            "low_dim_high_noise",
            {
                "n_samples": n_samples,
                "n_features": 2,
                "n_classes": 2,
                "n_informative": 2,
                "n_redundant": 0,
                "n_repeated": 0,
                "n_clusters_per_class": 1,
                "flip_y": 0.2,
            },
        ),
        (
            "high_dim_no_noise",
            {
                "n_samples": n_samples,
                "n_features": 10,
                "n_classes": 2,
                "n_informative": 10,
                "n_redundant": 0,
                "n_repeated": 0,
                "n_clusters_per_class": 1,
                "flip_y": 0.0,
            },
        ),
        (
            "high_dim_medium_noise",
            {
                "n_samples": n_samples,
                "n_features": 10,
                "n_classes": 2,
                "n_informative": 10,
                "n_redundant": 0,
                "n_repeated": 0,
                "n_clusters_per_class": 1,
                "flip_y": 0.01,
            },
        ),
        (
            "high_dim_high_noise",
            {
                "n_samples": n_samples,
                "n_features": 10,
                "n_classes": 2,
                "n_informative": 10,
                "n_redundant": 0,
                "n_repeated": 0,
                "n_clusters_per_class": 1,
                "flip_y": 0.3,
            },
        ),
        (
            "high_dim_redundant",
            {
                "n_samples": n_samples,
                "n_features": 10,
                "n_classes": 2,
                "n_informative": 5,
                "n_redundant": 2,
                "n_repeated": 0,
                "n_clusters_per_class": 1,
                "flip_y": 0.0,
                "class_sep": 1.0,
            },
        ),
        (
            "high_dim_redundant_high_noise",
            {
                "n_samples": n_samples,
                "n_features": 10,
                "n_classes": 2,
                "n_informative": 5,
                "n_redundant": 2,
                "n_repeated": 0,
                "n_clusters_per_class": 1,
                "flip_y": 0.3,
                "class_sep": 1.0,
            },
        ),
        (
            "high_dim_redundant_high_noise_medium_class_sep",
            {
                "n_samples": n_samples,
                "n_features": 10,
                "n_classes": 2,
                "n_informative": 5,
                "n_redundant": 2,
                "n_repeated": 0,
                "n_clusters_per_class": 1,
                "flip_y": 0.3,
                "class_sep": 2.0,
            },
        ),
        (
            "high_dim_redundant_high_noise_high_class_sep",
            {
                "n_samples": n_samples,
                "n_features": 10,
                "n_classes": 2,
                "n_informative": 5,
                "n_redundant": 2,
                "n_repeated": 0,
                "n_clusters_per_class": 1,
                "flip_y": 0.3,
                "class_sep": 3.0,
            },
        ),
    ]

    # make_moons_s1_hps = [
    #     {"n_samples": 500, "noise": 0.05},
    #     {"n_samples": 500, "noise": 0.3},  # More noise
    # ]
    # load_iris_s1_hps = [
    #     {"n_samples": 150},  # Full dataset
    #     {"n_samples": 75, "noise": 0.5},  # Subsampled with added noise
    # ]

    # Experiment 2 HPs (selected for varying difficulty)
    # make_moons_s2_hps = [
    #     {"n_samples": 500, "noise": 0.05},  # Easy
    #     {"n_samples": 500, "noise": 0.2},  # Medium
    #     {"n_samples": 500, "noise": 0.4},  # Hard
    # ]

    # Experiment 3 HPs (start and end points for interpolation)
    # hp_start_classif = {
    #     "n_samples": 500,
    #     "n_features": 2,
    #     "n_classes": 2,
    #     "n_informative": 2,
    #     "flip_y": 0.01,
    # }
    # hp_end_classif = {
    #     "n_samples": 500,
    #     "n_features": 2,
    #     "n_classes": 2,
    #     "n_informative": 2,
    #     "flip_y": 0.4,
    # }

    # hp_start_blobs = {
    #     "n_samples": n_samples,
    #     "n_features": 2,
    #     "centers": 3,
    #     "cluster_std": 0.2,
    # }  # Tightly separated
    # hp_end_blobs = {
    #     "n_samples": n_samples,
    #     "n_features": 2,
    #     "centers": 3,
    #     "cluster_std": 1.5,
    # }  # Highly overlapping


def get_config(config_name: str, n_samples: int, seed: int):
    if config_name not in _CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")
    return _CONFIGS[config_name](n_samples, seed)


if __name__ == "__main__":
    from bm.experiments.dataset_utils import prepare_dataset

    for cfg_name, cfg_hps in get_config("blobs", 1000, 42):
        dataset = prepare_dataset("blobs", cfg_hps)
        print(dataset.x.shape, dataset.y.shape, np.unique(dataset.y))
        print(dataset.metadata)
        print("-" * 100)

    for cfg_name, cfg_hps in get_config("classification", 1000, 42):
        dataset = prepare_dataset("classification", cfg_hps)
        print(dataset.x.shape, dataset.y.shape, np.unique(dataset.y))
        print(dataset.metadata)
        print("-" * 100)
