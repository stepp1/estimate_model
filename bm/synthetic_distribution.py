"""Enhanced synthetic distribution class.

Information and Decision Systems Group - FCFM - Universidad de Chile
"""

import functools
import itertools
import time
import warnings
from typing import Any

# Add numba import
# Removed import numba.random # Import numba.random for the RNG
import numpy as np
import numpy.typing as npt
from absl import logging
from tqdm import tqdm

NP_DTYPE = np.float64
NP_UINT_DTYPE = np.uint64
FloatArray = npt.NDArray[NP_DTYPE]
ObjectArray = npt.NDArray[np.object_]
IntArray = npt.NDArray[np.int64]
SIntArray = npt.NDArray[np.signedinteger]
BoolArray = npt.NDArray[np.bool_]

_INITIAL_NUM_VALUE: FloatArray = np.asarray([0.75], dtype=NP_DTYPE)

logging.set_verbosity(logging.INFO)


class SyntheticDistribution:
    """Represent a synthetic distribution for generating synthetic data.

    This class provides methods to:
    - Sample data from the synthetic distribution
    - Compute mutual information and entropy
    - Compute the analytical loss

    Attributes:
    ----------
    symbol_probabilities: FloatArray
        Array of probabilities for each symbol,
        shape (m, ), where m is the total number of symbols.
    interval_bounds: ObjectArray
        Array of interval bounds arrays for each dimension;
        shape (d, ).

        Each element (interval_bounds[i]) is of shape (n_(i + 1) + 1, ).
        d is the total number of dimensions and n_i is
        the number of intervals for the i-th dimension.
    cell_probabilities: ObjectArray
        Array of symbol-conditioned cell probabilities for each coordinate group;
        shape (G, ).

        Each element (cell_probabilities[i]) is of shape
        (m, n_{gamma(i,1)}, ..., n_{gamma(i,d_i)}).
        G is the total number of coordinate groups,
        m is the total number of symbols,
        d_i is the number of dimensions in the i-th coordinate group
        and gamma is the coordinate mapping.
        Defined in step A.8 (see PDF for more details).
    """

    def __init__(
        self,
        symbol_probabilities: FloatArray,
        interval_bounds: ObjectArray,
        cell_probabilities: ObjectArray,
    ) -> None:
        """Constructor for the Enhanced Synthetic Distribution class.

        Parameters
        ----------
        symbol_probabilities: FloatArray
            Array of probabilities for each symbol,
            shape (m, ), where m is the total number of symbols.
        interval_bounds: ObjectArray
            Array of interval bounds arrays for each dimension; shape (d, ). Each element
            (interval_bounds[i]) is of shape (n_(i + 1) + 1, ). d is the total number of dimensions and n_i is the
            number of intervals for the i-th dimension.
        cell_probabilities: ObjectArray
            Array of symbol-conditioned cell probabilities arrays for each coordinate group; this is an array of shape
            (G, ). Each element (cell_probabilities[i]) is of shape (m, n_{gamma(i,1)}, ..., n_{gamma(i,d_i)}). G is
            the total number of coordinate groups, m is the total number of symbols, d_i is the number of dimensions in
            the i-th coordinate group and gamma is the coordinate mapping defined in step A.8 (see PDF for more
            details).
        """
        self._check_parameters(
            symbol_probabilities=symbol_probabilities,
            interval_bounds=interval_bounds,
            cell_probabilities=cell_probabilities,
        )

        self.symbol_probabilities: FloatArray = symbol_probabilities
        self.interval_bounds: ObjectArray = interval_bounds
        self.cell_probabilities: ObjectArray = cell_probabilities

        self._check_cell_count()

        # Pre-compute frequently used values
        self._cumulative_dims = np.cumsum(
            [0] + [cp.ndim - 1 for cp in self.cell_probabilities],
        )
        self._group_dim_ranges = {}
        for group_idx in range(self.cell_probabilities.size):
            start = self._cumulative_dims[group_idx]
            end = self._cumulative_dims[group_idx + 1]
            self._group_dim_ranges[group_idx] = np.arange(start, end)

    @staticmethod
    def _check_non_empty_array(test_array: Any, object_name: str) -> None:
        """Checks if the given array is non-empty.

        Parameters
        ----------
        test_array: Any
            Array to test as non-empty.
        object_name: str
            Name of the array to test.

        Raises:
        ------
        ValueError
            If the given object is an empty array.
        """
        if test_array.size < 1:
            raise ValueError(f"Variable '{object_name}' must be non-empty")

    @staticmethod
    def _check_n_dim_array(test_object: Any, object_name: str, n_dims: int) -> None:
        """Checks if the given array is an n-dimensional array.

        Parameters
        ----------
        test_object: Any
            Object to test as an 1D array.
        object_name: str
            Name of the variable to test.
        n_dims: int
            Expected number of dimensions.

        Raises:
        ------
        TypeError
            If the given object is not a NumPy array.
        ValueError
            If the given object is not an n-dimensional array.
        """
        if not isinstance(test_object, np.ndarray):
            raise TypeError(f"Variable '{object_name}' must be a NumPy array")

        if test_object.ndim != n_dims:
            raise ValueError(f"Variable '{object_name}' must be a {n_dims}D array")

    @classmethod
    def _check_parameters(
        cls,
        symbol_probabilities: FloatArray,
        interval_bounds: ObjectArray,
        cell_probabilities: ObjectArray,
    ) -> None:
        """Check if given parameters are valid.

        Parameters
        ----------
        symbol_probabilities: FloatArray
            Array of probabilities for each symbol,
            shape (m, ), where m is the total number of symbols.
        interval_bounds: ObjectArray
            Array of interval bounds arrays for each dimension;
            shape (d, ).

            Each element (interval_bounds[i]) is of shape
            (n_(i + 1) + 1, ). d is the total number of dimensions and
            n_i is the number of intervals for the i-th dimension.
        cell_probabilities: ObjectArray
            Array of symbol-conditioned cell probabilities for each coordinate group;
            shape (G, ).

            Each element (cell_probabilities[i]) is of shape
            (m, n_{gamma(i,1)}, ..., n_{gamma(i,d_i)}).
            G is the total number of coordinate groups,
            m is the total number of symbols, d_i is the
            number of dimensions in the i-th coordinate group
            and gamma is the coordinate mapping.
            Defined in step A.8 (see PDF for more details).

        Raises:
        ------
        TypeError
            If the given parameters are not of the expected type.
        ValueError
            If the given parameters are invalid.
        """

        def _check_valid_float_array(test_array: npt.NDArray, array_name: str) -> None:
            """Checks if the given array is a valid float array.

            Parameters
            ----------
            test_array: npt.NDArray
                Array to test.
            array_name: str
                Name of the array to test.

            Raises:
            ------
            TypeError
                If the given array is not a valid float array.
            """
            if not np.issubdtype(arg1=test_array.dtype, arg2=np.floating):
                raise TypeError(f"Values in '{array_name}' must be of type float")

            if np.any(np.isnan(test_array)):
                raise ValueError(f"Values in '{array_name}' must not be NaN")

            if np.any(np.isinf(test_array)):
                raise ValueError(f"Values in '{array_name}' must not be infinite")

        def _check_non_negative_array(test_array: npt.NDArray, array_name: str) -> None:
            """Checks if the given array has non-negative values.

            Parameters
            ----------
            test_array: npt.NDArray
                Array to test.
            array_name: str
                Name of the array to test.

            Raises:
            ------
            ValueError
                If the given array has negative values.
            """
            if np.any(a=test_array < 0):
                raise ValueError(f"Values in '{array_name}' must be non-negative")

        # Checks for symbol_probabilities array
        cls._check_n_dim_array(
            test_object=symbol_probabilities,
            object_name="symbol_probabilities",
            n_dims=1,
        )
        cls._check_non_empty_array(
            test_array=symbol_probabilities,
            object_name="symbol_probabilities",
        )
        _check_valid_float_array(
            test_array=symbol_probabilities,
            array_name="symbol_probabilities",
        )
        _check_non_negative_array(
            test_array=symbol_probabilities,
            array_name="symbol_probabilities",
        )

        if not np.isclose(a=np.sum(a=symbol_probabilities), b=1.0):
            raise ValueError("Values in 'symbol_probabilities' must sum to 1")

        # Check for the interval_bounds object
        cls._check_n_dim_array(
            test_object=interval_bounds,
            object_name="interval_bounds",
            n_dims=1,
        )
        cls._check_non_empty_array(
            test_array=interval_bounds,
            object_name="interval_bounds",
        )

        # Checks for the arrays in the interval_bounds object
        bound_idx: int
        bound_array: npt.NDArray
        for bound_idx, bound_array in enumerate(interval_bounds):
            bound_array_name: str = f"interval_bounds[{bound_idx}]"
            cls._check_n_dim_array(
                test_object=bound_array,
                object_name=bound_array_name,
                n_dims=1,
            )
            _check_valid_float_array(
                test_array=bound_array,
                array_name=bound_array_name,
            )

            if bound_array.size < 2:
                raise ValueError(
                    f"Variable '{bound_array_name}' must have at least two elements",
                )

            if np.any(a=np.diff(a=bound_array) <= 0):
                raise ValueError(
                    f"Values in '{bound_array_name}' must be strictly increasing",
                )

        # Checks for the cell_probabilities object
        cls._check_n_dim_array(
            test_object=cell_probabilities,
            object_name="cell_probabilities",
            n_dims=1,
        )
        cls._check_non_empty_array(
            test_array=cell_probabilities,
            object_name="cell_probabilities",
        )

        # Checks for the arrays in the cell_probabilities object
        cell_prob_idx: int
        cell_prob_array: FloatArray
        for cell_prob_idx, cell_prob_array in enumerate(cell_probabilities):
            cell_prob_array_name: str = f"cell_probabilities[{cell_prob_idx}]"
            _check_valid_float_array(
                test_array=cell_prob_array,
                array_name=cell_prob_array_name,
            )
            _check_non_negative_array(
                test_array=cell_prob_array,
                array_name=cell_prob_array_name,
            )

            probs_close_to_one: BoolArray = np.isclose(
                a=np.sum(a=cell_prob_array, axis=tuple(range(1, cell_prob_array.ndim))),
                b=1.0,
            )
            if not np.all(a=probs_close_to_one):
                symbols_not_summing_to_one: str = np.array2string(
                    a=np.flatnonzero(a=~probs_close_to_one),
                    separator=", ",
                )
                raise ValueError(
                    f"Values in '{cell_prob_array_name}' must sum 1 for each symbol "
                    f"- Symbols not summing to 1: {symbols_not_summing_to_one}",
                )

        # Checks for dimension consistency
        num_symbols: int = symbol_probabilities.size
        for cell_prob_idx, cell_prob_array in enumerate(cell_probabilities):
            cell_prob_symbol_count: int = len(cell_prob_array)
            if cell_prob_symbol_count != num_symbols:
                raise ValueError(
                    f"Number of symbols in 'symbol_probabilities' ({num_symbols}) "
                    f"must match the number of symbols in "
                    f"'cell_probabilities[{cell_prob_idx}]' "
                    f"({cell_prob_array.shape[0]})",
                )

        if interval_bounds.size != sum(
            cell_probs.ndim - 1 for cell_probs in cell_probabilities
        ):
            raise ValueError(
                "Number of dimensions in 'interval_bounds' must match "
                "the sum of dimensions in 'cell_probabilities'",
            )

    def _check_cell_count(self) -> None:
        """Checks cell count.

        Checks the match between the amounts of intervals defined in
        each dimension of self.interval_bounds and the corresponding
        dimension value found in its respective coordinate group and
        inner-coordinate index in self.cell_probabilities.

        Raises:
        ------
        ValueError
            If there is a mismatch between the amounts of
            intervals and the corresponding dimension values.
        """
        full_coord_idx: int
        for full_coord_idx in range(self.interval_bounds.size):
            group_id: int
            inner_id: int
            group_id, inner_id = self.inverse_gamma(full_coord_id=full_coord_idx + 1)
            num_intervals: int = self.interval_bounds[full_coord_idx].size - 1
            prob_dim: int = self.cell_probabilities[group_id - 1].shape[inner_id]
            if num_intervals != prob_dim:
                raise ValueError(
                    f"Amount of intervals in dimension {full_coord_idx + 1} "
                    f"does not match the corresponding dimension value in "
                    f"coordinate group ID {group_id} and "
                    f"inner-coordinate ID {inner_id}",
                )

    def gamma(self, group_coord_id: int, coord_id: int) -> int:
        """Coordinate mapping function.

        Provides the full coordinate index for a given group and inner-coordinate index.

        Parameters
        ----------
            group_coord_id: int
                ID of the coordinate group (starts from 1).
            coord_id: int
                ID of the coordinate in the group (starts from 1).

        Raises:
        ------
            TypeError
                If the given indexes are not integers.
            ValueError
                If the given indexes are invalid.

        Returns:
        -------
            int
                Full coordinate index.
        """
        if not isinstance(group_coord_id, int):
            raise TypeError(
                f"Variable 'group_coord_idx' must be an integer, got {type(group_coord_id)}",
            )

        if not isinstance(coord_id, int):
            raise TypeError(
                f"Variable 'coord_idx' must be an integer, got {type(coord_id)}",
            )

        # g_value: int = self.cell_probabilities.size
        # if not 0 < coord_id <= g_value:
        #     raise ValueError(
        #         f"Coordinate index ('coord_idx') must be in {{1, ..., {g_value}}}, got {coord_id}"
        #     )

        d_i_value: int = self.cell_probabilities[group_coord_id - 1].ndim - 1
        if not 0 < coord_id <= d_i_value:
            raise ValueError(
                f"Inner coordinate index ('coord_id') must be in "
                f"{{1, ..., {d_i_value}}}, got {coord_id}",
            )

        return (
            sum(
                self.cell_probabilities[group_idx].ndim - 1
                for group_idx in range(group_coord_id - 1)
            )
            + coord_id
        )

    def inverse_gamma(self, full_coord_id: int) -> tuple[int, int]:
        """Inverse coordinate mapping function.

        Provides the group and inner-coordinate IDs for a given full
        coordinate ID. See step A.8 in the PDF for more details.

        Parameters
        ----------
            full_coord_id: int
                Full coordinate index.

        Raises:
        ------
            TypeError
                If the given index is not an integer.
            ValueError
                If the given index is invalid.

        Returns:
        -------
            Tuple[int, int]
                Tuple containing the group and inner-coordinate indexes.
        """
        if not isinstance(full_coord_id, int):
            raise TypeError(
                f"Variable 'full_coord_id' must be an integer, "
                f"got {type(full_coord_id)}",
            )

        num_dims: int = self.interval_bounds.size
        if not 0 < full_coord_id <= num_dims:
            raise ValueError(
                f"Full coordinate index ('full_coord_id') must be in "
                f"{{1, ..., {num_dims}}}, got {full_coord_id}",
            )

        cumulative_dims: IntArray = np.cumsum(
            a=[0] + [cell_probs.ndim - 1 for cell_probs in self.cell_probabilities],
        )
        group_id: int = int(
            np.searchsorted(a=cumulative_dims, v=full_coord_id, side="left"),
        )
        return group_id, int(full_coord_id - cumulative_dims[group_id - 1])

    def sample_data(
        self,
        num_samples: int,
        seed: int,
        *,
        report_time: bool = False,
        use_direct_sampling: bool = False,
    ) -> tuple[FloatArray, IntArray] | tuple[FloatArray, IntArray, dict[str, float]]:
        """Sample data from the distribution (performs steps B.1 to B.4).

        Parameters
        ----------
        num_samples: int
            Number of samples to generate.
        seed: int
            Seed for the random number generator.
        report_time: bool
            Flag to report the time taken for each step.
        use_direct_sampling: bool
            If True, uses a direct sampling method for cell indices (Strategy 3).
            If False, uses the original cartesian product method.
        use_numba: bool
            If True, attempts to use Numba compilation for accelerating the cell sampling loop.
            Requires use_direct_sampling=True for the Numba path.

        Returns:
        -------
        Tuple[FloatArray, IntArray] or Tuple[FloatArray, IntArray, Dict[str, float]]
            If report_time is False, returns a tuple containing the
            continuous samples and their corresponding symbols.
            If report_time is True, returns a tuple containing the
            continuous samples, their corresponding symbols, and a
            dictionary with the time taken for each step. The continuous
            sample array is of shape (N, d) and the symbols array is of
            shape (N, ), where N is the number of samples and d is the
            continuous part dimensionality.
            The dictionary contains the following keys:
            - "total": Total time taken for the sampling process.
            - "group_sizes": Time taken for computing group sizes.
            - "sampled_y_values": Time taken for sampling y values.
            - "x_array": Time taken for generating the x array.
        """
        times: dict[str, float] = {}
        if report_time:
            times = {
                "total": time.time(),
                "group_sizes": 0.0,
                "sampled_y_values": 0.0,
                "x_array": 0.0,
            }

        # Step B.1
        rng: np.random.Generator = np.random.default_rng(seed=seed)
        y_array: SIntArray = (
            rng.choice(
                a=self.symbol_probabilities.size,
                size=num_samples,
                p=self.symbol_probabilities,
            )
            + 1
        )

        # Step B.2. Samples in "cell_idx_array" are concatenated
        # realizations of "I_l" for l in {1, ..., G}
        num_dims: int = self.interval_bounds.size
        cell_idx_array: IntArray = np.empty(
            shape=(num_samples, num_dims),
            dtype=np.int64,
        )

        sampled_y_values: SIntArray
        sampled_y_values_inv_idx: SIntArray
        sampled_y_values, sampled_y_values_inv_idx = np.unique(
            ar=y_array,
            return_inverse=True,
        )
        # Pre-compute all indices in one go using advanced indexing
        sort_indices = np.argsort(sampled_y_values_inv_idx)
        split_points = np.searchsorted(
            sampled_y_values_inv_idx[sort_indices],
            np.arange(sampled_y_values.size),
        )
        y_idx_array = np.split(sort_indices, split_points[1:])

        num_groups: int = self.cell_probabilities.size
        group_idx_array: ObjectArray = np.empty(shape=num_groups, dtype=object)
        group_sizes: IntArray = np.empty(shape=num_groups, dtype=np.int64)
        group_idx: int

        if report_time:
            times["group_sizes"] = time.time()

        for group_idx in range(num_groups):
            num_coords = self.cell_probabilities[group_idx].ndim - 1
            coord_ids = np.arange(1, num_coords + 1)
            group_coord_ids = np.full(num_coords, group_idx + 1)
            group_idx_indexes = self.gamma_vectorized(group_coord_ids, coord_ids) - 1
            group_idx_array[group_idx] = group_idx_indexes
            group_sizes[group_idx] = group_idx_indexes.size

        if report_time:
            times["group_sizes"] = time.time() - times["group_sizes"]
            times["sampled_y_values"] = time.time()

        # Conditional logic to use Numba or original NumPy implementation

        # Use the original (or NumPy direct sampling) loops
        for y_idx, y_val in enumerate(sampled_y_values):
            y_indexes: IntArray = np.asarray(a=y_idx_array[y_idx], dtype=np.int64)
            y_count: int = y_indexes.size
            y_sub_array: IntArray = np.empty(
                shape=(y_count, num_dims),
                dtype=np.int64,
            )
            for group_idx in range(num_groups):
                group_indexes: IntArray = np.asarray(
                    a=group_idx_array[group_idx],
                    dtype=np.int64,
                )
                # Suppressing PyTypeChecker warning due to type hinting bug
                # noinspection PyTypeChecker
                p = self.cell_probabilities[group_idx][y_val - 1].flatten()

                if use_direct_sampling:
                    # Direct Sampling (Strategy 3) - NumPy implementation
                    # Calculate the shape of the current group's cell probabilities slice
                    group_prob_shape = self.cell_probabilities[group_idx].shape[1:]

                    # Sample flattened indices based on probabilities
                    sampled_flat_indices = rng.choice(
                        a=p.size,
                        size=y_count,
                        p=p,
                        replace=True,
                    )

                    # Unravel the flattened indices back to multi-dimensional indices
                    random_samples = np.unravel_index(
                        sampled_flat_indices,
                        group_prob_shape,
                    )
                    # Stack the resulting arrays of indices along a new axis
                    random_samples = np.stack(random_samples, axis=1)
                else:
                    # Original method (using itertools.product)
                    # Create array of dimension sizes
                    dim_sizes = [
                        self.cell_probabilities[group_idx].shape[dim_idx + 1]
                        for dim_idx in range(group_sizes[group_idx])
                    ]
                    # Generate ranges for each dimension
                    ranges = [np.arange(size) for size in dim_sizes]
                    # Create cartesian product of all ranges
                    a = np.asarray(
                        a=list(itertools.product(*ranges)),
                        dtype=np.int64,  # Added dtype for clarity
                    )
                    # Sample from the generated combinations
                    random_samples = rng.choice(
                        a=a,
                        size=y_count,
                        p=p,
                    )  # Use 'a' here

                # Assign the sampled indices to the correct dimensions in the sub-array
                y_sub_array[:, group_indexes] = (
                    random_samples  # Use 'random_samples' from either branch
                )
            cell_idx_array[y_indexes] = y_sub_array

        if report_time:
            times["sampled_y_values"] = time.time() - times["sampled_y_values"]

        # Steps B.3 and B.4. Sample X from the concatenated
        # realizations of "I_l" for l in {1, ..., G}
        x_array: FloatArray = rng.uniform(
            low=0.0,
            high=1.0,
            size=(num_samples, num_dims),
        )

        if report_time:
            times["x_array"] = time.time()

        for group_idx in range(num_groups):
            group_idx_indexes: IntArray = np.asarray(
                a=group_idx_array[group_idx],
                dtype=np.int64,
            )
            x_group_array: FloatArray = x_array[:, group_idx_indexes]

            sampled_cells, sampled_cells_inv_idx = np.unique(
                ar=cell_idx_array[:, group_idx_indexes],
                axis=0,
                return_inverse=True,
            )

            # Pre-allocate transformation arrays
            all_low_bounds = np.empty(
                (len(sampled_cells), group_sizes[group_idx]),
                dtype=NP_DTYPE,
            )
            all_high_bounds = np.empty(
                (len(sampled_cells), group_sizes[group_idx]),
                dtype=NP_DTYPE,
            )

            # Batch compute bounds for all unique cells
            gamma_indices = (
                self.gamma_vectorized(
                    np.full(group_sizes[group_idx], group_idx + 1),
                    np.arange(1, group_sizes[group_idx] + 1),
                )
                - 1
            )

            for sampled_idx, cell_indexes in enumerate(sampled_cells):
                for i, (gamma_idx, cell_idx) in enumerate(
                    zip(gamma_indices, cell_indexes, strict=False),
                ):
                    bounds = self.interval_bounds[gamma_idx]
                    all_low_bounds[sampled_idx, i] = bounds[cell_idx]
                    all_high_bounds[sampled_idx, i] = bounds[cell_idx + 1]

            # Apply transformations using broadcasting
            for sampled_idx in range(len(sampled_cells)):
                cell_mask = sampled_cells_inv_idx == sampled_idx
                x_group_array[cell_mask] = (
                    all_high_bounds[sampled_idx] - all_low_bounds[sampled_idx]
                ) * x_group_array[cell_mask] + all_low_bounds[sampled_idx]

            x_array[:, group_idx_indexes] = x_group_array

        if report_time:
            times["x_array"] = time.time() - times["x_array"]
            times["total"] = time.time() - times["total"]

        return (x_array, y_array, times) if report_time else (x_array, y_array)

    def get_mutual_information(
        self,
        coordinates: set[int],
        show_tqdm: bool | None = None,
        in_nats: bool | None = None,
    ) -> float:
        """Computes and returns the mutual information.

        Mutual information between the specified
        coordinates of X (as a joint vector) and Y.

        Parameters
        ----------
            coordinates: Set[int]
                Set of coordinates to compute the mutual information
                    (indexes start from 1).
            show_tqdm: bool
                Flag to show the progress bar.
            in_nats: bool
                Flag to return the mutual information in nats.

        Raises:
        ------
            TypeError
                If the given coordinates are not of the expected type.
            ValueError
                If the given coordinates are invalid.

        Returns:
        -------
            float
                Mutual information between the specified coordinates of X and Y.
        """
        if not isinstance(coordinates, set):
            raise TypeError(
                f"Variable 'coordinates' must be a set, got {type(coordinates)}",
            )

        if not all(isinstance(coord, int) for coord in coordinates):
            raise TypeError("All elements in 'coordinates' must be integers")

        if not all(
            0 < coord_id <= self.interval_bounds.size for coord_id in coordinates
        ):
            raise ValueError(
                f"All elements in 'coordinates' must be in the range "
                f"{{1, ..., {self.interval_bounds.size}}}",
            )

        logging.debug(
            f"Computing mutual information for coordinate set: {coordinates}",
        )

        def _unique_outer_product(
            unique_out_x: tuple[npt.NDArray, FloatArray],
            unique_out_y: tuple[npt.NDArray, FloatArray],
        ) -> tuple[npt.NDArray, FloatArray]:
            """Obtains point-wise outer product unique values and their counts.

            Auxiliary function.

            Parameters
            ----------
                unique_out_x: Tuple[npt.NDArray, FloatArray]
                    Tuple containing the unique values and their counts for x
                        Each element must be of shape (Nx, m).
                unique_out_y: Tuple[npt.NDArray, FloatArray]
                    Tuple containing the unique values and their counts for y.
                    Each element must be of shape (Ny, m).

            Returns:
            -------
                Tuple[npt.NDArray, FloatArray]
                    Tuple containing the unique outer product values and their counts.
                        Each element is of shape (N, m).
            """
            # TODO: Perform the checks for the input arrays
            outer_unique: npt.NDArray = np.expand_dims(
                a=unique_out_x[0],
                axis=1,
            ) * np.expand_dims(a=unique_out_y[0], axis=0)
            flattened_outer_unique: npt.NDArray = np.reshape(
                outer_unique,
                shape=(-1, outer_unique.shape[-1]),
            )

            del outer_unique

            outer_counts: FloatArray = np.outer(
                a=unique_out_x[1],
                b=unique_out_y[1],
            ).flatten()

            neo_uniques: FloatArray
            neo_counts: FloatArray
            neo_uniques, neo_counts = _unique_is_close_axis_0_optimized(
                ar=flattened_outer_unique,
                prior_counts=outer_counts,
            )

            del flattened_outer_unique, outer_counts

            current_unique_probs: int = len(neo_counts)
            increased_unique_probs: int = current_unique_probs - len(unique_out_x[1])
            if increased_unique_probs < 0:
                warnings.warn(
                    f"NEGATIVE INCREASE AT {pbar.n}:",
                    stacklevel=2,
                )

            description: str = (
                f"{current_unique_probs} unique probabilities "
                f"(diff. {increased_unique_probs})"
            )
            pbar.set_description(
                desc=description,
            )
            pbar.update()

            return neo_uniques, neo_counts

        def _unique_is_close_axis_0_optimized(
            ar: FloatArray,
            prior_counts: FloatArray | None = None,
            mantissa_bits: int = 17,
        ) -> tuple[FloatArray, FloatArray]:
            """Unique values and their counts with a tolerance margin.

            Replicates the behavior of np.unique(ar=ar, return_counts=True, axis=0)
            for a float array but considers a tolerance margin.
            Expressed by a certain number of mantissa bits.

            Parameters
            ----------
                ar: FloatArray
                    Array of float values.
                prior_counts: Optional[FloatArray]
                    Prior counts for the unique values.
                mantissa_bits: int
                    Number of mantissa bits relevant for the comparison.

            Raises:
            ------
                ValueError
                    If the given mantissa bits value is invalid.

            Returns:
            -------
                Tuple[FloatArray, FloatArray]
                    Tuple containing the unique values and their counts.
            """
            mantissa_size: int = np.finfo(dtype=NP_DTYPE).nmant
            if (
                not isinstance(mantissa_bits, int)
                or not 0 <= mantissa_bits <= mantissa_size
            ):
                raise ValueError(
                    f"Value of 'mantissa_bits' must be an integer in the range "
                    f"[0, {mantissa_size}], got {mantissa_bits}",
                )

            n_rows: int = ar.shape[0]
            if prior_counts is None:
                prior_counts: FloatArray = np.ones(shape=n_rows, dtype=NP_DTYPE)

            bit_shift: int = mantissa_size - mantissa_bits
            truncated_arr = ((ar.view(NP_UINT_DTYPE) >> bit_shift) << bit_shift).view(
                NP_DTYPE,
            )

            # Use lexsort for multi-column sorting
            if truncated_arr.ndim > 1:
                sort_idx = np.lexsort(truncated_arr.T)
                sorted_arr = truncated_arr[sort_idx]
                sorted_counts = (
                    prior_counts[sort_idx]
                    if prior_counts is not None
                    else np.ones(len(ar))
                )

                # Find unique row boundaries
                if len(sorted_arr) > 1:
                    row_diff = np.any(sorted_arr[1:] != sorted_arr[:-1], axis=1)
                    unique_idx = np.concatenate(([True], row_diff))
                    unique_positions = np.flatnonzero(unique_idx)

                    # Aggregate counts
                    counts = np.add.reduceat(sorted_counts, unique_positions)
                    return ar[sort_idx[unique_positions]], counts

            return ar, prior_counts if prior_counts is not None else np.ones(len(ar))

        n_groups: int = self.cell_probabilities.size
        group_coordinates: list[set[int]] = [set() for _ in range(n_groups)]
        coord_id: int
        for coord_id in coordinates:
            group_id: int
            inner_id: int
            group_id, inner_id = self.inverse_gamma(full_coord_id=coord_id)
            group_coordinates[group_id - 1].add(inner_id)

        projected_group_probs: list[npt.NDArray] = [
            np.sum(
                a=group_coords,
                axis=tuple(
                    set(range(1, self.cell_probabilities[group_idx].ndim)).difference(
                        group_coordinates[group_idx],
                    ),
                ),
            )
            for group_idx, group_coords in enumerate(self.cell_probabilities)
        ]
        reshaped_group_probs: list[npt.NDArray] = [
            np.reshape(group_coords, shape=(group_coords.shape[0], -1)).T
            for group_coords in projected_group_probs
        ]
        unique_group_probs: list[tuple[FloatArray, FloatArray]]
        unique_group_probs = [
            _unique_is_close_axis_0_optimized(ar=reshaped_g_prob)
            for reshaped_g_prob in reshaped_group_probs
        ]

        aggregated_iterable: list[tuple[npt.NDArray, npt.NDArray]] = [
            (unique_probs[0], np.asarray(a=unique_probs[1], dtype=NP_DTYPE))
            for unique_probs in unique_group_probs
        ]
        pbar: tqdm = tqdm(total=len(aggregated_iterable) - 1, disable=not show_tqdm)

        aggregated_probs: npt.NDArray
        aggregated_counts: FloatArray
        aggregated_probs, aggregated_counts = functools.reduce(
            _unique_outer_product,
            aggregated_iterable,
        )

        # Suppressing division by zero and nan warnings
        # due to the consideration of "0 * log2(0) = 0"
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = float(
                np.nansum(
                    aggregated_counts[:, np.newaxis]
                    * np.log2(
                        aggregated_probs
                        / np.sum(
                            a=aggregated_probs * self.symbol_probabilities,
                            axis=1,
                        )[:, np.newaxis],
                    )
                    * aggregated_probs
                    * self.symbol_probabilities,
                ),
            )
            logging.debug(
                f"Mutual information for coordinates {coordinates}: {mi}",
            )
            if in_nats:
                return mi / np.log2(np.e)
            return mi

    def get_symbol_entropy(self, in_nats: bool | None = None) -> float:
        """Computes and returns the entropy of the symbol Y.

        Parameters
        ----------
            in_nats: bool
                Flag to return the mutual information in nats.

        Returns:
        -------
            float
                Entropy of the symbol Y.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.nansum(
                self.symbol_probabilities * np.log2(self.symbol_probabilities),
            )
            if in_nats:
                return entropy / np.log2(np.e)
            return entropy

    def get_optimal_logits(self, x: np.ndarray) -> np.ndarray:
        """Computes the optimal logits for classification of Y given samples from X.

        Parameters
        ----------
        x:
            Samples of X of shape (N, d). "N" is the number of samples and "d" is the dimensionality of the continuous
            random variable X

        Returns:
        -------
            Array of logits of shape (N, m). "m" is the total number of symbols
        """
        raise NotImplementedError(
            "Not implemented yet. Needs to be ported from the old code."
        )

    def get_optimal_acc1(self, x: np.ndarray, y: np.ndarray) -> float:
        """Computes the optimal accuracy of the classifier.

        Parameters
        ----------
            y: np.ndarray
                Samples of Y of shape (N,). "N" is the number of samples.

        Returns:
        -------
            float
                Optimal accuracy of the classifier.
        """
        if np.min(y) == 1:
            y = y - 1

        return (
            np.sum(
                self.get_optimal_logits(x=x) == y,
            )
            / y.shape[0]
        )

    def gamma_vectorized(
        self,
        group_coord_ids: IntArray,
        coord_ids: IntArray,
    ) -> IntArray:
        """Vectorized version of gamma function for multiple coordinate pairs."""
        # Pre-compute cumulative dimensions once
        cumulative_dims = np.cumsum(
            [0] + [cp.ndim - 1 for cp in self.cell_probabilities],
        )
        # Compute offsets for each group
        offsets = cumulative_dims[group_coord_ids - 1]
        return offsets + coord_ids

    @staticmethod
    def compute_analytical_loss(
        mi: float,
        entropy_y: float,
    ) -> float:
        """Calculates the analytical loss.

        Based on the mutual information and entropy of the symbol Y.

        Parameters
        ----------
            mi: float
                Mutual information between the specified coordinates of X and Y.
            entropy_y: float
                Entropy of the symbol Y.
            in_nats: bool
                Flag to return the analytical loss in nats.
        """
        return entropy_y - mi

    @staticmethod
    def make_marginal_cond_probs(
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.int64],
        bounds: npt.NDArray[np.object_],
        grid_size: int,
    ) -> npt.NDArray[np.object_]:
        """Make marginal conditional probabilities."""
        n_features = x.shape[1]
        n_classes = len(np.unique(y))

        marginal_cond_probs: npt.NDArray[np.object_] = np.empty(
            shape=(n_features,),
            dtype=object,
        )
        marginal_cond_probs[:] = [
            np.empty(shape=(n_classes, grid_size)) for _ in range(n_features)
        ]

        for class_idx in range(n_classes):
            class_count = np.count_nonzero(y == class_idx)
            for feature_idx in range(n_features):
                t = (
                    np.histogramdd(
                        sample=x[y == class_idx, feature_idx : feature_idx + 1],
                        bins=(bounds[feature_idx],),
                        density=False,
                    )[0]
                    / class_count
                )
                marginal_cond_probs[feature_idx][class_idx] = t
        return marginal_cond_probs

    @staticmethod
    def make_joint_cond_probs(
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.int64],
        bounds: npt.NDArray[np.object_],
        grid_size: int,
        top_k_features: int | None = None,
    ) -> npt.NDArray[np.object_]:
        """Make joint conditional probabilities."""
        n_features = x.shape[1]
        n_classes = len(np.unique(y))

        # Use only top k features if specified
        if top_k_features is not None:
            top_k_features = min(top_k_features, n_features)
            selected_features = slice(0, top_k_features)
            x_selected = x[:, selected_features]
            bounds_selected = bounds[:top_k_features]
            n_features_to_use = top_k_features
        else:
            x_selected = x
            bounds_selected = bounds
            n_features_to_use = n_features

        joint_cond_probs: npt.NDArray[np.object_] = np.empty(
            shape=(1,),
            dtype=object,
        )

        joint_cond_probs[0] = np.empty(
            shape=(n_classes,) + (grid_size,) * n_features_to_use,
            dtype=np.float64,
        )

        for class_idx in range(n_classes):
            joint_cond_probs[0][class_idx] = np.histogramdd(
                sample=x_selected[y == class_idx, :],
                bins=tuple(bounds_selected[i] for i in range(n_features_to_use)),
                density=False,
            )[0] / np.count_nonzero(y == class_idx)

        return joint_cond_probs

    @staticmethod
    def make_bounds(
        x: npt.NDArray[np.float64],
        n_features: int,
        grid_size: int = 100,
    ) -> npt.NDArray[np.object_]:
        """Make bounds."""
        bounds: npt.NDArray[np.object_] = np.empty(shape=(n_features,), dtype=object)
        for i in range(n_features):
            bounds[i] = np.linspace(x[:, i].min(), x[:, i].max(), grid_size + 1)
        return bounds

    @staticmethod
    def compute_empirical_probs(
        y: npt.NDArray[np.int64],
        n_classes: int | None = None,
    ) -> npt.NDArray[np.object_]:
        """Compute empirical probabilities."""
        if n_classes is None:
            n_classes = np.max(y) + 1
        return np.count_nonzero(y == np.arange(n_classes)[:, None], axis=1) / y.size
