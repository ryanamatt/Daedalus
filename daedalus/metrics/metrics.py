# daedalus/metrics/metrics.py

from __future__ import annotations
from ..daedalus_cpp import (
    mean_squared_error as _mse_cpp,
    r2_score as _r2_cpp,
    confusion_matrix as _cm_cpp,
    accuracy_score as _acc_cpp, 
    precision_score as _prec_cpp,
    recall_score as _rec_cpp,
    f1_score as _f1_cpp,
    mcc_score as _mcc_cpp
)
from .._core import Matrix

def mean_squared_error(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Calculates the Mean Squared Error (MSE).

    Args:
        y_true: Column matrix of ground truth values.
        y_pred: Column matrix of predicted values.

    Returns:
        The calculated mean squared error as a float.
    """
    return _mse_cpp(y_true, y_pred)

def r2_score(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Calculates the Coefficient of Determination (R^2 Score).

    Args:
        y_true: Column matrix of ground truth values.
        y_pred: Column matrix of predicted values.

    Returns:
        The R-Squared score (typically between 0.0 and 1.0).
    """
    return _r2_cpp(y_true, y_pred)

def confusion_matrix(y_true: Matrix, y_pred: Matrix) -> Matrix:
    """
    Computes the confusion matrix to evaluate classification accuracy.

    Args:
        y_true: Column matrix of ground truth labels.
        y_pred: Column matrix of predicted labels.

    Returns:
        A 2x2 Matrix: [[True Positives, False Positives], 
                      [False Negatives, True Negatives]]
    """
    return _cm_cpp(y_true, y_pred)

def accuracy_score(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Calculates the Accuracy Score.

    Args:
        y_true: Column matrix of ground truth labels.
        y_pred: Column matrix of predicted labels.

    Returns:
        Accuracy ranging from 0.0 to 1.0.
    """
    return _acc_cpp(y_true, y_pred)

def precision_score(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Calculates the Precision Score.

    Args:
        y_true: Column matrix of ground truth labels.
        y_pred: Column matrix of predicted labels.

    Returns:
        Precision score (TP / (TP + FP)).
    """
    return _prec_cpp(y_true, y_pred)

def recall_score(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Calculates the Recall (Sensitivity) Score.

    Args:
        y_true: Column matrix of ground truth labels.
        y_pred: Column matrix of predicted labels.

    Returns:
        Recall score (TP / (TP + FN)).
    """
    return _rec_cpp(y_true, y_pred)

def f1_score(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Calculates the F1 Score (Harmonic mean of precision and recall).

    Args:
        y_true: Column matrix of ground truth labels.
        y_pred: Column matrix of predicted labels.

    Returns:
        F1 score ranging from 0.0 to 1.0.
    """
    return _f1_cpp(y_true, y_pred)

def mcc_score(y_true: Matrix, y_pred: Matrix) -> float:
    """
    Calculates the Matthews Correlation Coefficient (MCC).

    Args:
        y_true: Column matrix of ground truth labels.
        y_pred: Column matrix of predicted labels.

    Returns:
        MCC score between -1.0 and 1.0.
    """
    return _mcc_cpp(y_true, y_pred)

