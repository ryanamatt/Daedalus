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
#   y_pred: [1, 0, 1, 0]  → TP=1, FN=1, FP=1, TN=1
Y_TRUE_MIXED = col([1, 1, 0, 0])
Y_PRED_MIXED = col([1, 0, 1, 0])

# Regression vectors
Y_TRUE_REG = col([3.0, -0.5, 2.0, 7.0])
Y_PRED_REG = col([2.5,  0.0, 2.0, 8.0])


# ===========================================================================
# 1. mean_squared_error
# ===========================================================================

class TestMeanSquaredError:

    def test_perfect_predictions_give_zero(self):
        y = col([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        # errors: [-0.5, 0.5, 0.0, 1.0]  → squared: [0.25, 0.25, 0.0, 1.0]
        # mean = 1.5 / 4 = 0.375
        assert mean_squared_error(Y_TRUE_REG, Y_PRED_REG) == pytest.approx(0.375)

    def test_single_element(self):
        assert mean_squared_error(col([3.0]), col([1.0])) == pytest.approx(4.0)

    def test_symmetry_of_squared_error(self):
        """MSE(a, b) == MSE(b, a) because errors are squared."""
        a = col([1.0, 2.0, 3.0])
        b = col([4.0, 5.0, 6.0])
        assert mean_squared_error(a, b) == pytest.approx(mean_squared_error(b, a))

    def test_dimension_mismatch_raises(self):
        with pytest.raises(Exception):
            mean_squared_error(col([1.0, 2.0]), col([1.0]))


# ===========================================================================
# 2. r2_score
# ===========================================================================

class TestR2Score:

    def test_perfect_fit_returns_one(self):
        y = col([1.0, 2.0, 3.0, 4.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_known_value(self):
        # sklearn reference: r2_score([3,-0.5,2,7],[2.5,0,2,8]) ≈ 0.9486
        result = r2_score(Y_TRUE_REG, Y_PRED_REG)
        assert result == pytest.approx(0.9486081370449679, rel=1e-5)

    def test_worse_than_mean_is_negative(self):
        """Predictions far from truth can yield negative R²."""
        y_true = col([1.0, 2.0, 3.0])
        y_pred = col([10.0, 20.0, 30.0])
        assert r2_score(y_true, y_pred) < 0.0

    def test_single_element(self):
        assert r2_score(col([5.0]), col([5.0])) == pytest.approx(1.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(Exception):
            r2_score(col([1.0, 2.0]), col([1.0]))


# ===========================================================================
# 3. confusion_matrix
# ===========================================================================

class TestConfusionMatrix:

    def test_returns_matrix_type(self):
        cm = confusion_matrix(Y_TRUE_MIXED, Y_PRED_MIXED)
        assert isinstance(cm, Matrix)

    def test_shape_is_2x2(self):
        cm = confusion_matrix(Y_TRUE_MIXED, Y_PRED_MIXED)
        assert cm.shape == (2, 2)

    def test_known_values_mixed(self):
        # TP=1, FP=1, FN=1, TN=1
        cm = confusion_matrix(Y_TRUE_MIXED, Y_PRED_MIXED)
        assert cm(0, 0) == pytest.approx(1.0)  # TP
        assert cm(0, 1) == pytest.approx(1.0)  # FP
        assert cm(1, 0) == pytest.approx(1.0)  # FN
        assert cm(1, 1) == pytest.approx(1.0)  # TN

    def test_all_true_positives(self):
        cm = confusion_matrix(Y_TRUE_ALL_POS, Y_PRED_ALL_POS)
        assert cm(0, 0) == pytest.approx(4.0)  # TP
        assert cm(0, 1) == pytest.approx(0.0)  # FP
        assert cm(1, 0) == pytest.approx(0.0)  # FN
        assert cm(1, 1) == pytest.approx(0.0)  # TN

    def test_all_true_negatives(self):
        cm = confusion_matrix(Y_TRUE_ALL_NEG, Y_PRED_ALL_NEG)
        assert cm(0, 0) == pytest.approx(0.0)  # TP
        assert cm(0, 1) == pytest.approx(0.0)  # FP
        assert cm(1, 0) == pytest.approx(0.0)  # FN
        assert cm(1, 1) == pytest.approx(4.0)  # TN

    def test_all_false_positives(self):
        # Predict all 1s when truth is all 0s
        y_true = col([0, 0, 0])
        y_pred = col([1, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        assert cm(0, 0) == pytest.approx(0.0)  # TP
        assert cm(0, 1) == pytest.approx(3.0)  # FP
        assert cm(1, 0) == pytest.approx(0.0)  # FN
        assert cm(1, 1) == pytest.approx(0.0)  # TN

    def test_all_false_negatives(self):
        # Predict all 0s when truth is all 1s
        y_true = col([1, 1, 1])
        y_pred = col([0, 0, 0])
        cm = confusion_matrix(y_true, y_pred)
        assert cm(0, 0) == pytest.approx(0.0)  # TP
        assert cm(0, 1) == pytest.approx(0.0)  # FP
        assert cm(1, 0) == pytest.approx(3.0)  # FN
        assert cm(1, 1) == pytest.approx(0.0)  # TN


# ===========================================================================
# 4. accuracy_score
# ===========================================================================

class TestAccuracyScore:

    def test_perfect_accuracy(self):
        assert accuracy_score(Y_TRUE_ALL_POS, Y_PRED_ALL_POS) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        # All predictions wrong
        y_true = col([1, 1, 0, 0])
        y_pred = col([0, 0, 1, 1])
        assert accuracy_score(y_true, y_pred) == pytest.approx(0.0)

    def test_half_accuracy(self):
        # Mixed: TP=1, TN=1, FP=1, FN=1 → 2/4 = 0.5
        assert accuracy_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(0.5)

    def test_single_correct(self):
        assert accuracy_score(col([1.0]), col([1.0])) == pytest.approx(1.0)

    def test_single_incorrect(self):
        assert accuracy_score(col([1.0]), col([0.0])) == pytest.approx(0.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(Exception):
            accuracy_score(col([1.0, 0.0]), col([1.0]))


# ===========================================================================
# 5. precision_score
# ===========================================================================

class TestPrecisionScore:

    def test_perfect_precision(self):
        assert precision_score(Y_TRUE_ALL_POS, Y_PRED_ALL_POS) == pytest.approx(1.0)

    def test_zero_precision_all_fp(self):
        # All predicted positive are false positives
        y_true = col([0, 0, 0])
        y_pred = col([1, 1, 1])
        assert precision_score(y_true, y_pred) == pytest.approx(0.0)

    def test_known_value_mixed(self):
        # TP=1, FP=1 → precision = 0.5
        assert precision_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(0.5)

    def test_no_positive_predictions_returns_zero(self):
        # Predict all negative: no positives predicted, so precision = 0.0
        y_true = col([1, 1, 0])
        y_pred = col([0, 0, 0])
        assert precision_score(y_true, y_pred) == pytest.approx(0.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(Exception):
            precision_score(col([1.0, 0.0]), col([1.0]))


# ===========================================================================
# 6. recall_score
# ===========================================================================

class TestRecallScore:

    def test_perfect_recall(self):
        assert recall_score(Y_TRUE_ALL_POS, Y_PRED_ALL_POS) == pytest.approx(1.0)

    def test_zero_recall_all_fn(self):
        # All true positives are missed
        y_true = col([1, 1, 1])
        y_pred = col([0, 0, 0])
        assert recall_score(y_true, y_pred) == pytest.approx(0.0)

    def test_known_value_mixed(self):
        # TP=1, FN=1 → recall = 0.5
        assert recall_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(0.5)

    def test_no_positive_ground_truth_returns_zero(self):
        # No actual positives: recall denominator is 0 → returns 0.0
        y_true = col([0, 0, 0])
        y_pred = col([0, 0, 0])
        assert recall_score(y_true, y_pred) == pytest.approx(0.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(Exception):
            recall_score(col([1.0, 0.0]), col([1.0]))


# ===========================================================================
# 7. f1_score
# ===========================================================================

class TestF1Score:

    def test_perfect_f1(self):
        assert f1_score(Y_TRUE_ALL_POS, Y_PRED_ALL_POS) == pytest.approx(1.0)

    def test_zero_f1_when_precision_and_recall_zero(self):
        # Predict all negative when truth is all positive → p=0, r=0 → f1=0
        y_true = col([1, 1, 1])
        y_pred = col([0, 0, 0])
        assert f1_score(y_true, y_pred) == pytest.approx(0.0)

    def test_known_value_mixed(self):
        # p=0.5, r=0.5 → f1 = 2*(0.5*0.5)/(0.5+0.5) = 0.5
        assert f1_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(0.5)

    def test_f1_is_harmonic_mean_of_precision_and_recall(self):
        p = precision_score(Y_TRUE_MIXED, Y_PRED_MIXED)
        r = recall_score(Y_TRUE_MIXED, Y_PRED_MIXED)
        expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        assert f1_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(expected_f1)

    def test_f1_zero_when_no_positive_predictions(self):
        y_true = col([1, 0, 1])
        y_pred = col([0, 0, 0])
        assert f1_score(y_true, y_pred) == pytest.approx(0.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(Exception):
            f1_score(col([1.0, 0.0]), col([1.0]))


# ===========================================================================
# 8. mcc_score
# ===========================================================================

class TestMccScore:

    def test_perfect_mcc(self):
        assert mcc_score(Y_TRUE_ALL_POS, Y_PRED_ALL_POS) == pytest.approx(0.0)

    def test_known_value_mixed(self):
        # TP=1, FP=1, FN=1, TN=1
        # numerator = (1*1) - (1*1) = 0  → MCC = 0
        assert mcc_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(0.0)

    def test_strong_positive_correlation(self):
        # 3 TP, 1 TN, 0 FP, 0 FN → MCC = 1.0
        y_true = col([1, 1, 1, 0])
        y_pred = col([1, 1, 1, 0])
        assert mcc_score(y_true, y_pred) == pytest.approx(1.0)

    def test_strong_negative_correlation(self):
        # All predictions inverted → MCC = -1.0
        y_true = col([1, 1, 0, 0])
        y_pred = col([0, 0, 1, 1])
        assert mcc_score(y_true, y_pred) == pytest.approx(-1.0)

    def test_zero_denominator_returns_zero(self):
        # All predicted positive, all actually positive → denominator factor
        # (TN+FP)=0 → denominator=0 → returns 0.0
        y_true = col([1, 1, 1])
        y_pred = col([1, 1, 1])
        assert mcc_score(y_true, y_pred) == pytest.approx(0.0)

    def test_range_is_minus_one_to_one(self):
        """MCC must always lie in [-1, 1]."""
        y_true = col([1, 0, 1, 0, 1])
        y_pred = col([0, 1, 1, 0, 1])
        result = mcc_score(y_true, y_pred)
        assert -1.0 <= result <= 1.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(Exception):
            mcc_score(col([1.0, 0.0]), col([1.0]))


# ===========================================================================
# 9. Integration — metrics agree with each other
# ===========================================================================

class TestIntegration:

    def test_accuracy_consistent_with_confusion_matrix(self):
        cm = confusion_matrix(Y_TRUE_MIXED, Y_PRED_MIXED)
        tp = cm(0, 0)
        tn = cm(1, 1)
        n  = len([1, 1, 0, 0])
        expected_acc = (tp + tn) / n
        assert accuracy_score(Y_TRUE_MIXED, Y_PRED_MIXED) == pytest.approx(expected_acc)

    def test_f1_consistent_with_precision_and_recall(self):
        y_true = col([1, 1, 0, 1, 0])
        y_pred = col([1, 0, 0, 1, 1])
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        expected = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        assert f1_score(y_true, y_pred) == pytest.approx(expected)

    def test_mcc_consistent_with_confusion_matrix(self):
        y_true = col([1, 1, 0, 1, 0])
        y_pred = col([1, 0, 0, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        tp, fp, fn, tn = cm(0,0), cm(0,1), cm(1,0), cm(1,1)
        denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        expected = ((tp*tn) - (fp*fn)) / denom if denom != 0 else 0.0
        assert mcc_score(y_true, y_pred) == pytest.approx(expected)

    def test_regression_metrics_perfect_fit(self):
        y = col([1.0, 2.0, 3.0, 4.0])
        assert mean_squared_error(y, y) == pytest.approx(0.0)
        assert r2_score(y, y) == pytest.approx(1.0)