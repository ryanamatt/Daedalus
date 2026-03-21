from __future__ import annotations
from .model import Model
from ..daedalus_cpp import LogisticRegression as _LogisticRegressionCpp
from .._core import Matrix

class LogisticRegression(Model):
    """
    A Binary Logistic Regression classifier using the Sigmoid function.
    Supports L1, L2, or no regularization.
    """

    def __init__(self, learning_rate: float = 0.01, reg_lambda: float = 0.01, 
            penalty: str = "none") -> None:
        """
        Initializes the Logistic Regression classifier.

        Args:
            learning_rate: Step size for gradient descent.
            reg_lambda: Regularization strength.
            penalty: Regularization type ("l1", "l2", or "none").
        """
        self._obj = _LogisticRegressionCpp(learning_rate, reg_lambda, penalty)

    def fit(self, X: Matrix, y: Matrix, epochs: int | None = None) -> None:
        """
        Trains the classifier using Log-Loss gradient descent.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target matrix of shape (n_samples, n_targets).
            epochs: Optional number of iterations. If None, uses default logic.
        """
        if epochs is not None:
            self._obj.fit(X._obj, y._obj, epochs)
        else:
            self._obj.fit(X._obj, y._obj)

    def predict(self, X: Matrix) -> Matrix:
        """
        Predicts binary labels (0.0 or 1.0) based on a 0.5 threshold.

        Args:
            X: Feature matrix to predict labels for.

        Returns:
            A Matrix of binary class predictions.
        """
        res_obj = self._obj.predict(X._obj)
        res = Matrix(res_obj.rows, res_obj.cols)
        res._obj = res_obj
        return res
    def predict_proba(self, X: Matrix) -> Matrix:
        """
        Returns the raw probability of the positive class (range [0, 1]).

        Args:
            X: Feature matrix.

        Returns:
            A Matrix of probabilities.
        """
        res_obj = self._obj.predict_proba(X._obj)
        res = Matrix(res_obj.rows, res_obj.cols)
        res._obj = res_obj
        return res

    def save_model(self, filename: str) -> None:
        """
        Saves model parameters to a file.

        Args:
            filename: Path to the destination file.
        """
        self._obj.save_model(filename)

    def load_model(self, filename: str) -> None:
        """
        Loads model parameters from a file.

        Args:
            filename: Path to the model file.
        """
        self._obj.load_model(filename)