"""Synthetic distribution module for the bm package."""

from bm import estimate
from bm.estimate import estimate_synthetic_mi
from bm.synthetic_distribution import SyntheticDistribution

# Don't import run here to avoid circular imports when run.py is executed directly
__all__ = ["SyntheticDistribution", "estimate", "estimate_synthetic_mi"]

__version__ = "0.1.0"
