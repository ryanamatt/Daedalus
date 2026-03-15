from __future__ import annotations
from .model import Model
from ..daedalus_cpp import KNN as _KNNCpp
from .._core import Matrix

class KNN(Model):
    """
    K-Nearest Neighbors implementation.
    Predicts values by finding the k-nearest neighbors in the training set
    based on Euclidean distance.
    """

    def __init__(self, k: int = 3) -> None:
        """
        Initializes the KNN model.

        Args:
            k: Number of neighbors to consider (default is 3).
        """
        self._obj = _KNNCpp(k)

    def fit(self, X: Matrix, y: Matrix) -> None:
        """
        Stores the training data for later prediction.

        Args:
            X: Training feature matrix.
            y: Training target matrix.
        """
        self._obj.fit(X, y)

    def predict(self, X: Matrix) -> Matrix:
        """
        Makes predictions by finding the k-nearest neighbors in the stored data.

        Args:
            X: Feature matrix to predict values for.

        Returns:
            A Matrix containing the predicted values.
        """
        return self._obj.predict(X)