"""
08_test_metrics.py
==================
Full-coverage test suite for daedalus/metrics/metrics.py.

Run:
    pytest tests/08_test_metrics.py

    or run all tests by

    pytest
"""

from __future__ import annotations
import pytest
import math
from daedalus import Matrix
from daedalus.metrics import (
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mcc_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def col(values: list[float]) -> Matrix:
    """Convenience: build a column Matrix from a flat list."""
    return Matrix([[v] for v in values])

# Perfect binary predictions: all 1s correctly predicted
Y_TRUE_ALL_POS  = col([1, 1, 1, 1])
Y_PRED_ALL_POS  = col([1, 1, 1, 1])

# Perfect binary predictions: all 0s correctly predicted
Y_TRUE_ALL_NEG  = col([0, 0, 0, 0])
Y_PRED_ALL_NEG  = col([0, 0, 0, 0])

# Mixed ground truth / predictions for thorough metric coverage
#   y_true: [1, 1, 0, 0]
#   y_pred: [1, 0, 1, 0]  -> TP=1, FN=1, FP=1, TN=1
Y_TRUE_MIXED = col([1, 1, 0, 0])
Y_PRED_MIXED = col([1, 0, 1, 0])

# Regression vectors
Y_TRUE_REG = col([3.0, -0.5, 2.0, 7.0])
Y_PRED_REG = col([2.5,  0.0, 2.0, 8.0])

# ===========================================================================
# 1. Regression Metrics
#   - Mean Square Error
#   - R^2 Error
# ===========================================================================

class TestRegressionMetrics:

    def test_mse(self):
        y = col([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == pytest.approx(0.0)
        # errors: [-0.5, 0.5, 0.0, 1.0]  -> squared: [0.25, 0.25, 0.0, 1.0]
        # mean = 1.5 / 4 = 0.375
        assert mean_squared_error(Y_TRUE_REG, Y_PRED_REG) == pytest.approx(0.375)
        assert mean_squared_error(col([3.0]), col([1.0])) == pytest.approx(4.0)
        # MSE(a, b) == MSE(b, a) because errors are squared.
        a = col([1.0, 2.0, 3.0])
        b = col([4.0, 5.0, 6.0])
        assert mean_squared_error(a, b) == pytest.approx(mean_squared_error(b, a))

        with pytest.raises(Exception):
            mean_squared_error(col([1.0, 2.0]), col([1.0]))

    def test_r2_score(self):
        y = col([1.0, 2.0, 3.0, 4.0])
        assert r2_score(y, y) == pytest.approx(1.0)
        # r2_score([3,-0.5,2,7],[2.5,0,2,8]) ≈ 0.9486
        result = r2_score(Y_TRUE_REG, Y_PRED_REG)
        assert result == pytest.approx(0.9486081370449679, rel=1e-5)

        # Predictions far from truth can yield negative R^2
        y_true = col([1.0, 2.0, 3.0])
        y_pred = col([10.0, 20.0, 30.0])
        assert r2_score(y_true, y_pred) < 0.0

        assert r2_score(col([5.0]), col([5.0])) == pytest.approx(1.0)

        with pytest.raises(Exception):
            r2_score(col([1.0, 2.0]), col([1.0]))


# ===========================================================================
# 3. Classification Metrics Test
#   - confusion_matrix
#   - confusion_matrix
#   - confusion_matrix
#   - confusion_matrix
#   - confusion_matrix
# ===========================================================================

class TestClassificationMetrics:

    def test_confusion_matrix(self):
        cm = confusion_matrix(Y_TRUE_MIXED, Y_PRED_MIXED)
        assert isinstance(cm, Matrix)
        assert cm.shape == (2, 2)
        # TP=1, FP=1, FN=1, TN=1
        assert cm(0, 0) == pytest.approx(1.0)  # TP
        assert cm(0, 1) == pytest.approx(1.0)  # FP
        assert cm(1, 0) == pytest.approx(1.0)  # FN
        assert cm(1, 1) == pytest.approx(1.0)  # TN

        cm = confusion_matrix(Y_TRUE_ALL_POS, Y_PRED_ALL_POS)
        assert cm(0, 0) == pytest.approx(4.0)  # TP
        assert cm(0, 1) == pytest.approx(0.0)  # FP
        assert cm(1, 0) == pytest.approx(0.0)  # FN
        assert cm(1, 1) == pytest.approx(0.0)  # TN

        cm = confusion_matrix(Y_TRUE_ALL_NEG, Y_PRED_ALL_NEG)
        assert cm(0, 0) == pytest.approx(0.0)  # TP
        assert cm(0, 1) == pytest.approx(0.0)  # FP
        assert cm(1, 0) == pytest.approx(0.0)  # FN
        assert cm(1, 1) == pytest.approx(4.0)  # TN

        # Predict all 1s when truth is all 0s
        y_true = col([0, 0, 0])
        y_pred = col([1, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        assert cm(0, 0) == pytest.approx(0.0)  # TP
        assert cm(0, 1) == pytest.approx(3.0)  # FP
        assert cm(1, 0) == pytest.approx(0.0)  # FN
        assert cm(1, 1) == pytest.approx(0.0)  # TN

        # Predict all 0s when truth is all 1s
        y_true = col([1, 1, 1])
        y_pred = col([0, 0, 0])
        cm = confusion_matrix(y_true, y_pred)
        assert cm(0, 0) == pytest.approx(0.0)  # TP
        assert cm(0, 1) == pytest.approx(0.0)  # FP
        assert cm(1, 0) == pytest.approx(3.0)  # FN
        assert cm(1, 1) == pytest.approx(0.0)  # TN

    def test_accuracy_score(self):
        assert accuracy_score(Y_TRUE_ALL_POS, Y_PRED_ALL_POS) == pytest.approx(1.0)

        # All predictions wrong
        y_true = col([1, 1, 0, 0])
        y_pred = col([0, 0, 1, 1])
        assert accuracy_score(y_true, y_pred) == pytest.approx(0.0)

        # Mixed: TP=1, TN=1, FP=1, FN=1 -> 2/4 = 0.5
        assert accuracy_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(0.5)

        with pytest.raises(Exception):
            accuracy_score(col([1.0, 0.0]), col([1.0]))

    def test_precision_score(self):
        assert precision_score(Y_TRUE_ALL_POS, Y_PRED_ALL_POS) == pytest.approx(1.0)

        # All predicted positive are false positives
        y_true = col([0, 0, 0])
        y_pred = col([1, 1, 1])
        assert precision_score(y_true, y_pred) == pytest.approx(0.0)

        # TP=1, FP=1 -> precision = 0.5
        assert precision_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(0.5)

        # Predict all negative: no positives predicted, so precision = 0.0
        y_true = col([1, 1, 0])
        y_pred = col([0, 0, 0])
        assert precision_score(y_true, y_pred) == pytest.approx(0.0)

        with pytest.raises(Exception):
            precision_score(col([1.0, 0.0]), col([1.0]))

    def test_recall_score(self):
        assert recall_score(Y_TRUE_ALL_POS, Y_PRED_ALL_POS) == pytest.approx(1.0)

        # All true positives are missed
        y_true = col([1, 1, 1])
        y_pred = col([0, 0, 0])
        assert recall_score(y_true, y_pred) == pytest.approx(0.0)

        # TP=1, FN=1 -> recall = 0.5
        assert recall_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(0.5)

        # No actual positives: recall denominator is 0 -> returns 0.0
        y_true = col([0, 0, 0])
        y_pred = col([0, 0, 0])
        assert recall_score(y_true, y_pred) == pytest.approx(0.0)

        with pytest.raises(Exception):
            recall_score(col([1.0, 0.0]), col([1.0]))

    def test_f1_score(self):
        assert f1_score(Y_TRUE_ALL_POS, Y_PRED_ALL_POS) == pytest.approx(1.0)

        # Predict all negative when truth is all positive -> p=0, r=0 -> f1=0
        y_true = col([1, 1, 1])
        y_pred = col([0, 0, 0])
        assert f1_score(y_true, y_pred) == pytest.approx(0.0)

        # p=0.5, r=0.5 -> f1 = 2*(0.5*0.5)/(0.5+0.5) = 0.5
        assert f1_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(0.5)

        # F1 when No positive preds
        y_true = col([1, 0, 1])
        y_pred = col([0, 0, 0])
        assert f1_score(y_true, y_pred) == pytest.approx(0.0)

        with pytest.raises(Exception):
            f1_score(col([1.0, 0.0]), col([1.0]))

    def test_mcc_score(self):
        assert mcc_score(Y_TRUE_ALL_POS, Y_PRED_ALL_POS) == pytest.approx(0.0)

        # TP=1, FP=1, FN=1, TN=1
        # numerator = (1*1) - (1*1) = 0  -> MCC = 0
        assert mcc_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(0.0)

        # 3 TP, 1 TN, 0 FP, 0 FN -> MCC = 1.0
        y_true = col([1, 1, 1, 0])
        y_pred = col([1, 1, 1, 0])
        assert mcc_score(y_true, y_pred) == pytest.approx(1.0)

        # All predictions inverted -> MCC = -1.0
        y_true = col([1, 1, 0, 0])
        y_pred = col([0, 0, 1, 1])
        assert mcc_score(y_true, y_pred) == pytest.approx(-1.0)

        # All predicted positive, all actually positive -> denominator factor
        # (TN+FP)=0 -> denominator=0 -> returns 0.0
        y_true = col([1, 1, 1])
        y_pred = col([1, 1, 1])
        assert mcc_score(y_true, y_pred) == pytest.approx(0.0)

        # MCC must always lie in [-1, 1].
        y_true = col([1, 0, 1, 0, 1])
        y_pred = col([0, 1, 1, 0, 1])
        result = mcc_score(y_true, y_pred)
        assert -1.0 <= result <= 1.0

        with pytest.raises(Exception):
            mcc_score(col([1.0, 0.0]), col([1.0]))
