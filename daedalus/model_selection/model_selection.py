# daedalus/model_selection/model_selection.py

from __future__ import annotations
from ..daedalus_cpp import (
    train_test_split as _tts_cpp
)
from .._core import Matrix

def train_test_split(X: Matrix, y: Matrix, test_size: float = 0.2, seed: int = 42) -> tuple[Matrix, Matrix, Matrix, Matrix]:
    """
    Splits matrices into random train and test subsets.
    
    This function shuffles the dataset indices and partitions the features (X) 
        and targets (y) into two sets based on the @p test_size ratio.

    Args: 
        X: Feature matrix.
        y: Target matrix.
        test_size: The proportion of the dataset to include in the test split (0.0 to 1.0).
        seed: The seed for the random number generator to ensure reproducibility.

    Returns:
        A tuple of 4 matrices (train_X, test_X, train_y, test_y)
    """
    _tts_cpp(X, y, test_size, seed)