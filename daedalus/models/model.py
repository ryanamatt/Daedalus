from __future__ import annotations
from abc import ABC, abstractmethod
from ..daedalus_cpp import Model as _ModelCpp
from .._core import Matrix

class Model(ABC):
    """
    Abstract base class for all Daedalus models in Python.
    """

    def __init__(self, cpp_obj: _ModelCpp = None) -> None:
        """
        Insantiates the Model Class.

        Args:
            cpp_obj: A C++ Model class use to assing self._obj.
                If None defaults to the C++ model class.
        """
        if cpp_obj:
            self._obj = cpp_obj
        else:
            self._obj = _ModelCpp()

    @abstractmethod
    def fit(self, X: Matrix, y: Matrix) -> None:
        """
        Trains the model on the provided dataset.

        Args:
            X: The data to fit.
            Y: The result of the data.
        """
        raise NotImplementedError("This function is not implemented for this model.")
    
    def predict(self, X: Matrix) -> Matrix:
        """
        Makes predictions using the trained model parameters.

        Args:
            The data to predict on.

        Returns:
            A Matrix of the predictions.
        """
        raise NotImplementedError("This function is not implemented for this model.")