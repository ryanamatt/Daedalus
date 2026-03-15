# daedalus/preprocessing/standard_scaler.py

from __future__ import annotations
from ..daedalus_cpp import StandardScaler as _StandardScalerCpp
from .._core import Matrix

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:
        z = (x - u) / s
    
    where `u` is the mean of the training samples, and `s` is the standard 
    deviation of the training samples.
    """

    def __init__(self) -> None:
        """
        Initializes the StandardScaler.
        """
        self._obj = _StandardScalerCpp()

    def fit(self, X: Matrix) -> StandardScaler:
        """
        Computes the mean and standard deviation for each feature in X 
        to be used for later scaling.

        Args:
            X: Feature matrix of shape (n_samples, n_features) to compute 
               scaling statistics from.

        Returns:
            self: The fitted scaler instance.
        """
        self._obj.fit(X)
        return self

    def transform(self, X: Matrix) -> Matrix:
        """
        Performs standardization by centering and scaling the features in X 
        using the statistics computed during fit.

        Args:
            X: Feature matrix of shape (n_samples, n_features) to be transformed.

        Returns:
            Matrix: A new Matrix containing the standardized features.
        """
        return self._obj.transform(X)

    def fit_transform(self, X: Matrix) -> Matrix:
        """
        Fits the scaler to the data X and returns a transformed version of X.
        This is a convenience method that combines fit() and transform().

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Matrix: A new Matrix containing the standardized features.
        """
        return self._obj.fit_transform(X)