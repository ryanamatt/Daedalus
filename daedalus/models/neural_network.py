from __future__ import annotations
from .model import Model
from ..daedalus_cpp import NeuralNetwork as _NeuralNetworkCpp
from .._core import Matrix

class NeuralNetwork(Model):
    """
    A container model for stacking multiple layers to form a Deep Neural Network.
    Trains using forward and backward propagation.
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        """
        Initializes the Neural Network.

        Args:
            learning_rate: The step size applied during the backward pass.
        """
        self._obj = _NeuralNetworkCpp(learning_rate)

    def add(self, input_size: int, output_size: int) -> None:
        """
        Adds a new fully connected (Dense) layer to the network.

        Args:
            input_size: Number of input features to the layer.
            output_size: Number of neurons in the layer.
        """
        # The C++ binding handles the creation of the DenseLayer internally
        self._obj.add(input_size, output_size)

    def fit(self, X: Matrix, y: Matrix, epochs: int | None = None) -> None:
        """
        Trains the network using forward and backward propagation.

        Args:
            X: Training features.
            y: Target values.
            epochs: Optional number of iterations over the dataset.
        """
        if epochs is not None:
            self._obj.fit(X._obj, y._obj, epochs)
        else:
            self._obj.fit(X._obj, y._obj)

    def predict(self, X: Matrix) -> Matrix:
        """
        Performs a forward pass through all layers to get a prediction.

        Args:
            X: Feature matrix to predict values for.

        Returns:
            A Matrix containing the network's output activations.
        """
        res_obj = self._obj.predict(X._obj)
        res = Matrix(res_obj.rows, res_obj.cols)
        res._obj = res_obj
        return res