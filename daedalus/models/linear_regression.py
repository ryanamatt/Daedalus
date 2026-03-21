from __future__ import annotations
from .model import Model
from ..daedalus_cpp import LinearRegression as _LinearRegressionCpp
from .._core import Matrix

class LinearRegression(Model):
    """
    Linear Regression model supporting OLS and Regularized Gradient Descent.
    The model predicts y = Xw + b.
    """

    def __init__(self, learning_rate: float = 0.01, reg_lambda: float = 0.01,
                 penalty: str = "none") -> None:
        """
        Initializes the Linear Regression model.

        Args:
            learning_rate: Step size for weight updates.
            reg_lambda: Regularization strength (ignored if penalty is "none").
            penalty: Type of regularization to apply ("l1", "l2", or "none").
        """
        self._obj = _LinearRegressionCpp(learning_rate, reg_lambda, penalty)

    def fit(self, X: Matrix, y: Matrix, epochs: int | None = None) -> None:
        """
        Trains the model on the provided dataset.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target matrix of shape (n_samples, n_targets).
            epochs: Optional number of gradient descent iterations. If None, 
                    uses the C++ default convergence logic.
        """
        if epochs is not None:
            self._obj.fit(X._obj, y._obj, epochs)
        else:
            self._obj.fit(X._obj, y._obj)

    def predict(self, X: Matrix) -> Matrix:
        """
        Makes continuous predictions using the trained model parameters.

        Args:
            X: Feature matrix to predict values for.

        Returns:
            A Matrix containing the predicted values.
        """
        res_obj = self._obj.predict(X._obj)
        res = Matrix(res_obj.rows, res_obj.cols)
        res._obj = res_obj
        return res
    
    def save_model(self, filename: str) -> None:
        """
        Serializes model weights and parameters to a file.

        Args:
            filename: Path to the destination file.
        """
        self._obj.save_model(filename)

    def load_model(self, filename: str) -> None:
        """
        Deserializes model weights and parameters from a file.

        Args:
            filename: Path to the model file to load.
        """
        self._obj.load_model(filename)