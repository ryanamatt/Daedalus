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
    X_tr_obj, X_te_obj, y_tr_obj, y_te_obj = _tts_cpp(X._obj, y._obj, test_size, seed)
    
    X_train = Matrix(X_tr_obj.rows, X_tr_obj.cols)
    X_test = Matrix(X_te_obj.rows, X_te_obj.cols)
    y_train = Matrix(y_tr_obj.rows, y_tr_obj.cols)
    y_test = Matrix(y_te_obj.rows, y_te_obj.cols)
    
    X_train._obj = X_tr_obj
    X_test._obj = X_te_obj
    y_train._obj = y_tr_obj
    y_test._obj = y_te_obj
    
    return (X_train, X_test, y_train, y_test)