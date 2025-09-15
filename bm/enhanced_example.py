"""
Sample code for the Enhanced Synthetic Distribution class.

Information and Decision Systems Group - FCFM - Universidad de Chile
"""

from typing import List, Set

import numpy as np

from bm.synthetic_distribution import FloatArray, ObjectArray, SyntheticDistribution

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
    np.asarray(a=cell_cond_probs, dtype=np.float64) for cell_cond_probs in demo_c_prob
]

distribution: SyntheticDistribution = SyntheticDistribution(
    symbol_probabilities=demo_s_prob,
    interval_bounds=demo_i_bound_array,
    cell_probabilities=demo_c_prob_array,
)
distribution.sample_data(num_samples=10_000, seed=1234)

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
