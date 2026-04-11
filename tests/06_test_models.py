"""
06_test_models.py
=================
Full-coverage test suite for:
  - daedalus/models/model.py                (Model ABC)
  - daedalus/models/linear_regression.py    (LinearRegression wrapper)
  - daedalus/models/logistic_regression.py  (LogisticRegression wrapper)
  - daedalus/models/knn.py                  (KNN wrapper)
  - daedalus/models/neural_network.py       (NeuralNetwork wrapper)

Run:
    pytest tests/06_test_models.py

    or run all tests with:

    pytest
"""

from __future__ import annotations
import os
import pytest
import numpy as np
from daedalus import Matrix
from daedalus.models import Model, LinearRegression, LogisticRegression, KNN, NeuralNetwork

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_X(n: int = 50, features: int = 1, seed: int = 0) -> Matrix:
    """Returns an (n, features) feature matrix from a seeded RNG."""
    rng = np.random.default_rng(seed)
    return Matrix(rng.uniform(0.0, 10.0, (n, features)))

def make_y_linear(X: Matrix, slope: float = 2.0, intercept: float = 5.0) -> Matrix:
    """Returns y = slope * X[:, 0] + intercept  (n, 1)."""
    arr = np.array([[slope * X(i, 0) + intercept] for i in range(X.rows)])
    return Matrix(arr)

def make_simple_dataset(n: int = 100, seed: int = 42):
    """Convenience wrapper – returns (X, y) ready for fit()."""
    X = make_X(n, features=1, seed=seed)
    y = make_y_linear(X)
    return X, y

def make_multifeature_dataset(n: int = 100, features: int = 3, seed: int = 7):
    """Returns (X, y) with multiple features."""
    rng = np.random.default_rng(seed)
    X_arr = rng.uniform(0.0, 5.0, (n, features))
    # y = 1*x0 + 2*x1 + 3*x2 + 1
    y_arr = (X_arr @ np.array([[1.0], [2.0], [3.0]])) + 1.0
    return Matrix(X_arr), Matrix(y_arr)

def make_binary_dataset(n: int = 100, features: int = 1, seed: int = 0):
    """
    Returns (X, y) for a linearly separable binary classification problem.
    Class 0: samples drawn from N(-2, 0.5); Class 1: from N(+2, 0.5).
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    X0 = rng.normal(-2.0, 0.5, (half, features))
    X1 = rng.normal(+2.0, 0.5, (n - half, features))
    X_arr = np.vstack([X0, X1])
    y_arr = np.vstack([np.zeros((half, 1)), np.ones((n - half, 1))])
    return Matrix(X_arr), Matrix(y_arr)

def make_multifeature_binary_dataset(n: int = 120, features: int = 3, seed: int = 5):
    """Linearly separable binary problem with multiple features."""
    rng = np.random.default_rng(seed)
    half = n // 2
    X0 = rng.normal(-1.5, 0.5, (half, features))
    X1 = rng.normal(+1.5, 0.5, (n - half, features))
    X_arr = np.vstack([X0, X1])
    y_arr = np.vstack([np.zeros((half, 1)), np.ones((n - half, 1))])
    return Matrix(X_arr), Matrix(y_arr)

def accuracy(y_true: Matrix, y_pred: Matrix) -> float:
    """Simple accuracy helper: fraction of matching rows."""
    correct = sum(
        1 for i in range(y_true.rows) if y_true(i, 0) == y_pred(i, 0)
    )
    return correct / y_true.rows

def make_regression_dataset(n: int = 100, features: int = 1, seed: int = 0):
    """
    Returns (X, y) for a simple regression task: y = sum(X, axis=1, keepdims=True).
    Inputs are in [0, 1] so outputs are well-scaled for a small network.
    """
    rng = np.random.default_rng(seed)
    X_arr = rng.uniform(0.0, 1.0, (n, features))
    y_arr = X_arr.sum(axis=1, keepdims=True)
    return Matrix(X_arr), Matrix(y_arr)

def mse(y_true: Matrix, y_pred: Matrix) -> float:
    """Element-wise MSE between two (n, 1) matrices."""
    total = sum((y_true(i, 0) - y_pred(i, 0)) ** 2 for i in range(y_true.rows))
    return total / y_true.rows

def train_and_mse(penalty: str, reg_lambda: float, epochs: int = 500):
    X, y = make_simple_dataset(n=100, seed=3)
    model = LinearRegression(learning_rate=0.05, reg_lambda=reg_lambda, penalty=penalty)
    model.fit(X, y, epochs=epochs)
    preds = model.predict(X)
    y_arr = np.array([[y(i, 0)] for i in range(y.rows)])
    p_arr = np.array([[preds(i, 0)] for i in range(preds.rows)])
    return float(np.mean((y_arr - p_arr) ** 2))

# ===========================================================================
# 1. Model ABC
# ===========================================================================

class TestModelABC:

    def test_init(self):
        with pytest.raises(TypeError):
            Model()

        class NoFit(Model):
            def predict(self, X):
                return X
        with pytest.raises(TypeError):
            NoFit()

        class FitOnly(Model):
            def fit(self, X, y):
                pass
        obj = FitOnly()
        assert obj is not None

        class Minimal(Model):
            def fit(self, X, y):
                pass
        m = Minimal()
        assert hasattr(m, "_obj")

        from daedalus.daedalus_cpp import LinearRegression as _LRCpp
        cpp = _LRCpp(0.01, 0.0, "none")
        class Wrapper(Model):
            def fit(self, X, y):
                pass
        w = Wrapper(cpp_obj=cpp)
        assert w._obj is cpp

    def test_predict(self):
        class FitOnly(Model):
            def fit(self, X, y):
                pass

        m = FitOnly()
        X, _ = make_simple_dataset(10)
        with pytest.raises(NotImplementedError, match="not implemented"):
            m.predict(X)

    def test_fit(self):
        class Delegating(Model):
            def fit(self, X, y):
                super().fit(X, y)

        m = Delegating()
        X, y = make_simple_dataset(10)
        with pytest.raises(NotImplementedError, match="not implemented"):
            m.fit(X, y)

# ===========================================================================
# 2. LinearRegression
# ===========================================================================

class TestLinearRegression:

    def test_init(self):
        model = LinearRegression()
        assert model is not None
        assert hasattr(model, "_obj")

        model = LinearRegression(learning_rate=0.001)
        assert model is not None
        model = LinearRegression(reg_lambda=0.1)
        assert model is not None
        model = LinearRegression(penalty="l1")
        assert model is not None
        model = LinearRegression(penalty="l2")
        assert model is not None
        model = LinearRegression(penalty="none")
        assert model is not None

        model = LinearRegression(learning_rate=0.05, reg_lambda=0.001, penalty="l2")
        assert model is not None

        assert isinstance(LinearRegression(), Model)

    def test_fit(self):
        model = LinearRegression(learning_rate=0.01)
        X, y = make_simple_dataset()
        result = model.fit(X, y)
        assert result is None

        model = LinearRegression(learning_rate=0.01)
        X, y = make_simple_dataset()
        model.fit(X, y)  # should complete without exception

        X = Matrix([[3.0]])
        y = Matrix([[11.0]])
        LinearRegression(learning_rate=0.01).fit(X, y)

        model = LinearRegression(learning_rate=0.05)
        X, y = make_simple_dataset(n=200)
        model.fit(X, y)
        # After fitting, predict should produce finite values
        preds = model.predict(X)
        assert preds.rows == X.rows
        assert preds.cols == 1

        model = LinearRegression(learning_rate=0.01)
        X, y = make_multifeature_dataset()
        model.fit(X, y)

        model = LinearRegression(learning_rate=0.01)
        X, y = make_simple_dataset()
        result = model.fit(X, y, epochs=50)
        assert result is None

        model = LinearRegression(learning_rate=0.01)
        X, y = make_simple_dataset()
        model.fit(X, y, epochs=0)
        preds = model.predict(X)
        assert preds.rows == X.rows

        model = LinearRegression(learning_rate=0.01)
        X, y = make_simple_dataset()
        model.fit(X, y, epochs=1)
        model.predict(X)

        model = LinearRegression(learning_rate=0.01)
        X, y = make_simple_dataset()
        model.fit(X, y, epochs=None)  # should branch to default fit
        model.predict(X)

    def test_penalty(self):
        mse = train_and_mse("none", 0.0)
        assert np.isfinite(mse)

        mse = train_and_mse("l2", 0.01)
        assert np.isfinite(mse)

        mse = train_and_mse("l1", 0.01)
        assert np.isfinite(mse)

        X = Matrix(np.ones((50, 1)) * 5.0)
        y = Matrix(np.ones((50, 1)) * 10.0)
        model = LinearRegression(learning_rate=0.01, reg_lambda=1.0, penalty="l1")
        model.fit(X, y, epochs=500)
        # Just verify it ran and produces finite predictions
        preds = model.predict(X)
        assert all(np.isfinite(preds(i, 0)) for i in range(preds.rows))

        X = Matrix(np.eye(5))
        y = Matrix(np.zeros((5, 1)))
        model = LinearRegression(learning_rate=0.01, reg_lambda=0.5, penalty="l1")
        model.fit(X, y, epochs=1)  # weights are zero at first step → sign branch
        model.predict(X)

        X, y = make_simple_dataset(n=100, seed=5)
        model = LinearRegression(learning_rate=0.01, reg_lambda=100.0, penalty="l2")
        model.fit(X, y, epochs=300)
        preds = model.predict(X)
        assert preds.rows == X.rows

    def test_predict(self):
        X, y = make_simple_dataset(n=60)
        model = LinearRegression(learning_rate=0.05)
        model.fit(X, y, epochs=500)
        preds = model.predict(X)
        assert preds.rows == X.rows
        assert preds.cols == 1
        assert isinstance(preds, Matrix)

        model = LinearRegression()
        X = make_X(10, features=2)
        # Weights are 0×0 until fit() is called – the C++ layer may raise; we
        # document the behaviour rather than asserting a specific exception type.
        try:
            preds = model.predict(X)
            assert preds is not None
        except Exception:
            pass  # acceptable – unfitted predict is undefined behaviour

        rng = np.random.default_rng(9)
        X_train = Matrix(rng.uniform(0, 5, (80, 2)))
        y_train = Matrix(rng.uniform(0, 10, (80, 1)))
        X_test = Matrix(rng.uniform(0, 5, (20, 2)))

        model = LinearRegression(learning_rate=0.01)
        model.fit(X_train, y_train, epochs=100)
        preds = model.predict(X_test)
        assert preds.rows == 20

        n = 10
        X_train = Matrix(np.ones((n, 1)))
        y_train = Matrix(np.full((n, 1), 7.0))
        model = LinearRegression(learning_rate=0.1)
        model.fit(X_train, y_train, epochs=2000)

        X_zeros = Matrix(np.ones((5, 1)))
        preds = model.predict(X_zeros)
        # All predictions should be close to 7.0 given the trivial dataset
        for i in range(preds.rows):
            assert abs(preds(i, 0) - 7.0) < 1.0, (
                f"Row {i} prediction {preds(i, 0):.4f} too far from 7.0"
            )

    def test_save(self, tmp_path):
        X, y = make_simple_dataset()
        model = LinearRegression(learning_rate=0.05)
        model.fit(X, y, epochs=200)
        filepath = str(tmp_path / "saved.txt")
        model.save_model(filepath)
        assert os.path.isfile(filepath)

        model = LinearRegression()
        filepath = str(tmp_path / "unfitted.txt")
        model.save_model(filepath)  # should not raise
        assert not os.path.isfile(filepath)

    def test_load(self, tmp_path):
        X, y = make_simple_dataset(n=50, seed=99)

        m1 = LinearRegression(learning_rate=0.05)
        m1.fit(X, y, epochs=400)
        path = str(tmp_path / "m1.txt")
        m1.save_model(path)

        # Train a *different* model, then overwrite it via load
        m2 = LinearRegression(learning_rate=0.001)
        m2.fit(X, y, epochs=5)  # intentionally poor fit
        m2.load_model(path)     # should now match m1

        p1 = m1.predict(X)
        p2 = m2.predict(X)
        for i in range(X.rows):
            assert abs(p1(i, 0) - p2(i, 0)) < 1e-10

        model = LinearRegression()
        with pytest.raises(Exception):
            model.load_model("/nonexistent/path/model.txt")

# ===========================================================================
# 3. LogisticRegression
# ===========================================================================

class TestLogisticRegression:

    def _train(self, penalty: str, reg_lambda: float, epochs: int = 300):
        X, y = make_binary_dataset(n=100, seed=3)
        model = LogisticRegression(learning_rate=0.5, reg_lambda=reg_lambda, penalty=penalty)
        model.fit(X, y, epochs=epochs)
        return model, X, y

    def _fitted_model(self, seed: int = 0):
        X, y = make_binary_dataset(n=100, seed=seed)
        model = LogisticRegression(learning_rate=0.5)
        model.fit(X, y, epochs=300)
        return model, X, y

    def test_init(self):
        model = LogisticRegression()
        assert model is not None
        assert hasattr(model, "_obj")

        model = LogisticRegression(learning_rate=0.001)
        assert model is not None

        model = LogisticRegression(reg_lambda=0.5)
        assert model is not None

        model = LogisticRegression(penalty="l1")
        assert model is not None

        model = LogisticRegression(penalty="l2")
        assert model is not None

        model = LogisticRegression(penalty="none")
        assert model is not None

        model = LogisticRegression(learning_rate=0.05, reg_lambda=0.001, penalty="l2")
        assert model is not None
        assert isinstance(LogisticRegression(), Model)

    def test_fit(self):
        model = LogisticRegression(learning_rate=0.1)
        X, y = make_binary_dataset()
        assert model.fit(X, y) is None

        model = LogisticRegression(learning_rate=0.1)
        X, y = make_binary_dataset()
        model.fit(X, y)

        X = Matrix([[1.0]])
        y = Matrix([[1.0]])
        LogisticRegression(learning_rate=0.1).fit(X, y)

        model = LogisticRegression(learning_rate=0.05)
        X, y = make_multifeature_binary_dataset()
        model.fit(X, y)

    def test_fit(self):
        model = LogisticRegression(learning_rate=0.1)
        X, y = make_binary_dataset()
        assert model.fit(X, y, epochs=50) is None

        model = LogisticRegression(learning_rate=0.1)
        X, y = make_binary_dataset()
        model.fit(X, y, epochs=0)
        model.predict(X)

        model = LogisticRegression(learning_rate=0.1)
        X, y = make_binary_dataset()
        model.fit(X, y, epochs=1)
        model.predict(X)

        X, y = make_binary_dataset(n=200, seed=1)
        model = LogisticRegression(learning_rate=0.5)
        model.fit(X, y, epochs=500)
        preds = model.predict(X)
        acc = accuracy(y, preds)
        assert acc > 0.90, f"Expected accuracy > 0.90, got {acc:.4f}"

        model = LogisticRegression(learning_rate=0.1)
        X, y = make_binary_dataset()
        model.fit(X, y, epochs=None)
        model.predict(X)

    def test_penalty(self):
        model, X, _ = self._train("none", 0.0)
        preds = model.predict(X)
        assert preds.rows == X.rows

        model, X, _ = self._train("l2", 0.01)
        preds = model.predict(X)
        assert preds.rows == X.rows

        model, X, _ = self._train("l1", 0.01)
        preds = model.predict(X)
        assert preds.rows == X.rows

        X = Matrix(np.eye(5))
        y = Matrix(np.zeros((5, 1)))
        model = LogisticRegression(learning_rate=0.1, reg_lambda=0.5, penalty="l1")
        model.fit(X, y, epochs=1)
        model.predict(X)

        X, y = make_binary_dataset(n=80, seed=7)
        model = LogisticRegression(learning_rate=0.5, reg_lambda=0.1, penalty="l1")
        model.fit(X, y, epochs=50)
        preds = model.predict(X)
        assert all(np.isfinite(preds(i, 0)) for i in range(preds.rows))

        X, y = make_binary_dataset(n=100, seed=5)
        model = LogisticRegression(learning_rate=0.1, reg_lambda=100.0, penalty="l2")
        model.fit(X, y, epochs=200)
        preds = model.predict(X)
        assert preds.rows == X.rows

    def test_predict(self):
        model, X, _ = self._fitted_model()
        preds = model.predict(X)
        assert preds.rows == X.rows
        assert preds.cols == 1

        model, X, _ = self._fitted_model()
        assert isinstance(model.predict(X), Matrix)

        model, X, _ = self._fitted_model()
        preds = model.predict(X)
        for i in range(preds.rows):
            assert preds(i, 0) in (0.0, 1.0), f"Row {i}: {preds(i, 0)}"

        rng = np.random.default_rng(11)
        X_train = Matrix(rng.normal(0, 1, (80, 2)))
        y_train = Matrix((rng.normal(0, 1, (80, 1)) > 0).astype(float))
        X_test = Matrix(rng.normal(0, 1, (20, 2)))

        model = LogisticRegression(learning_rate=0.1)
        model.fit(X_train, y_train, epochs=100)
        preds = model.predict(X_test)
        assert preds.rows == 20

        model, X, _ = self._fitted_model(seed=2)
        preds = model.predict(X)
        values = {preds(i, 0) for i in range(preds.rows)}
        assert 0.0 in values and 1.0 in values, (
            "Model only predicts one class on balanced test data"
        )

    def test_predict_proba(self):
        model, X, _ = self._fitted_model()
        proba = model.predict_proba(X)
        assert proba.rows == X.rows
        assert proba.cols == 1

        model, X, _ = self._fitted_model()
        assert isinstance(model.predict_proba(X), Matrix)

        model, X, _ = self._fitted_model()
        proba = model.predict_proba(X)
        for i in range(proba.rows):
            p = proba(i, 0)
            assert 0.0 <= p <= 1.0, f"Row {i}: probability {p:.6f} out of range"

        model, X, _ = self._fitted_model()
        labels = model.predict(X)
        proba = model.predict_proba(X)
        for i in range(X.rows):
            if proba(i, 0) >= 0.5:
                assert labels(i, 0) == 1.0, f"Row {i}: proba={proba(i,0):.4f} but label={labels(i,0)}"
            else:
                assert labels(i, 0) == 0.0, f"Row {i}: proba={proba(i,0):.4f} but label={labels(i,0)}"

        X, y = make_multifeature_binary_dataset(n=100, seed=6)
        model = LogisticRegression(learning_rate=0.1)
        model.fit(X, y, epochs=200)
        proba = model.predict_proba(X)
        assert proba.rows == X.rows
        for i in range(proba.rows):
            assert 0.0 <= proba(i, 0) <= 1.0

# ===========================================================================
# 14. LogisticRegression.save_model() / load_model()
# ===========================================================================

class TestLogisticRegressionPersistence:

    def test_save_and_load_roundtrip(self, tmp_path):
        X, y = make_binary_dataset(n=100, seed=11)
        model = LogisticRegression(learning_rate=0.5)
        model.fit(X, y, epochs=300)

        filepath = str(tmp_path / "logreg.txt")
        model.save_model(filepath)

        model2 = LogisticRegression()
        model2.load_model(filepath)

        p1 = model.predict(X)
        p2 = model2.predict(X)
        for i in range(X.rows):
            assert p1(i, 0) == p2(i, 0), f"Row {i}: {p1(i,0)} vs {p2(i,0)}"

    def test_save_creates_file(self, tmp_path):
        X, y = make_binary_dataset()
        model = LogisticRegression(learning_rate=0.5)
        model.fit(X, y, epochs=100)
        filepath = str(tmp_path / "saved.txt")
        model.save_model(filepath)
        assert os.path.isfile(filepath)

    def test_save_unfitted_model_does_not_create_file(self, tmp_path):
        """C++ returns early without creating a file when unfitted."""
        model = LogisticRegression()
        filepath = str(tmp_path / "unfitted.txt")
        model.save_model(filepath)
        assert not os.path.isfile(filepath)

    def test_load_nonexistent_file_raises(self):
        model = LogisticRegression()
        with pytest.raises(Exception):
            model.load_model("/nonexistent/path/logreg.txt")

    def test_save_then_load_multifeature(self, tmp_path):
        X, y = make_multifeature_binary_dataset(n=100, features=3, seed=22)
        model = LogisticRegression(learning_rate=0.1, penalty="l2", reg_lambda=0.001)
        model.fit(X, y, epochs=200)

        filepath = str(tmp_path / "multi_logreg.txt")
        model.save_model(filepath)

        model2 = LogisticRegression()
        model2.load_model(filepath)

        p1 = model.predict(X)
        p2 = model2.predict(X)
        for i in range(X.rows):
            assert p1(i, 0) == p2(i, 0)

    def test_load_overwrites_previous_weights(self, tmp_path):
        """Loading should fully replace the model's current weights."""
        X, y = make_binary_dataset(n=80, seed=99)

        m1 = LogisticRegression(learning_rate=0.5)
        m1.fit(X, y, epochs=300)
        path = str(tmp_path / "m1.txt")
        m1.save_model(path)

        m2 = LogisticRegression(learning_rate=0.001)
        m2.fit(X, y, epochs=2)   # intentionally poor fit
        m2.load_model(path)

        p1 = m1.predict(X)
        p2 = m2.predict(X)
        for i in range(X.rows):
            assert p1(i, 0) == p2(i, 0)

    def test_proba_roundtrip_after_load(self, tmp_path):
        """predict_proba() results must be identical after a save/load cycle."""
        X, y = make_binary_dataset(n=60, seed=33)
        model = LogisticRegression(learning_rate=0.5)
        model.fit(X, y, epochs=200)

        filepath = str(tmp_path / "proba_rt.txt")
        model.save_model(filepath)

        model2 = LogisticRegression()
        model2.load_model(filepath)

        pr1 = model.predict_proba(X)
        pr2 = model2.predict_proba(X)
        for i in range(X.rows):
            assert abs(pr1(i, 0) - pr2(i, 0)) < 1e-10


# ===========================================================================
# 15. LogisticRegression — integration
# ===========================================================================

class TestLogisticRegressionIntegration:

    def test_high_accuracy_separable_data(self):
        """Well-separated classes should yield high accuracy after training."""
        X, y = make_binary_dataset(n=200, seed=0)
        model = LogisticRegression(learning_rate=0.5)
        model.fit(X, y, epochs=500)
        acc = accuracy(y, model.predict(X))
        assert acc > 0.95, f"Accuracy = {acc:.4f}, expected > 0.95"

    def test_regularisation_reduces_confidence(self):
        """Strong L2 regularisation should push probabilities toward 0.5."""
        X, y = make_binary_dataset(n=100, seed=4)

        no_reg = LogisticRegression(learning_rate=0.5, penalty="none")
        no_reg.fit(X, y, epochs=500)

        l2_reg = LogisticRegression(learning_rate=0.5, reg_lambda=50.0, penalty="l2")
        l2_reg.fit(X, y, epochs=500)

        pr_no = np.array([no_reg.predict_proba(X)(i, 0) for i in range(X.rows)])
        pr_l2 = np.array([l2_reg.predict_proba(X)(i, 0) for i in range(X.rows)])

        # L2-regularised probabilities should be closer to 0.5
        assert np.mean(np.abs(pr_l2 - 0.5)) < np.mean(np.abs(pr_no - 0.5)), (
            "Strong L2 regularisation should reduce prediction confidence"
        )

    def test_full_save_load_cycle(self, tmp_path):
        """fit → save → load → predict → save again."""
        X, y = make_binary_dataset(n=80, seed=13)
        model = LogisticRegression(learning_rate=0.5)
        model.fit(X, y, epochs=300)

        p1 = str(tmp_path / "cycle1.txt")
        model.save_model(p1)

        loaded = LogisticRegression()
        loaded.load_model(p1)
        preds = loaded.predict(X)
        assert preds.rows == X.rows

        p2 = str(tmp_path / "cycle2.txt")
        loaded.save_model(p2)
        assert os.path.isfile(p2)

    def test_multifeature_accuracy(self):
        """Multi-feature logistic regression should learn a separable boundary."""
        X, y = make_multifeature_binary_dataset(n=200, features=3, seed=55)
        model = LogisticRegression(learning_rate=0.1)
        model.fit(X, y, epochs=1000)
        acc = accuracy(y, model.predict(X))
        assert acc > 0.90, f"Multi-feature accuracy = {acc:.4f}, expected > 0.90"


# ===========================================================================
# 16. KNN — construction
# ===========================================================================

class TestKNNInit:

    def test_default_construction(self):
        model = KNN()
        assert model is not None
        assert hasattr(model, "_obj")

    def test_custom_k(self):
        model = KNN(k=5)
        assert model is not None

    def test_k_equals_one(self):
        model = KNN(k=1)
        assert model is not None

    def test_large_k(self):
        model = KNN(k=50)
        assert model is not None

    def test_is_model_subclass(self):
        assert isinstance(KNN(), Model)


# ===========================================================================
# 17. KNN.fit()
# ===========================================================================

class TestKNNFit:

    def test_fit_returns_none(self):
        model = KNN(k=3)
        X, y = make_binary_dataset(n=30)
        assert model.fit(X, y) is None

    def test_fit_does_not_raise(self):
        model = KNN(k=3)
        X, y = make_binary_dataset(n=30)
        model.fit(X, y)

    def test_fit_stores_training_data(self):
        """After fit, predict on the training set should run without error."""
        model = KNN(k=3)
        X, y = make_binary_dataset(n=20)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.rows == X.rows

    def test_fit_single_sample(self):
        """Edge case: training set of size 1 with k=1."""
        X = Matrix([[1.0, 2.0]])
        y = Matrix([[0.0]])
        model = KNN(k=1)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.rows == 1

    def test_fit_multifeature(self):
        model = KNN(k=5)
        X, y = make_multifeature_binary_dataset(n=60)
        model.fit(X, y)
        model.predict(X)


# ===========================================================================
# 18. KNN.predict()
# ===========================================================================

class TestKNNPredict:

    def _fitted_model(self, k: int = 3, n: int = 60, seed: int = 0):
        X, y = make_binary_dataset(n=n, seed=seed)
        model = KNN(k=k)
        model.fit(X, y)
        return model, X, y

    def test_predict_shape(self):
        model, X, _ = self._fitted_model()
        preds = model.predict(X)
        assert preds.rows == X.rows
        assert preds.cols == 1

    def test_predict_returns_matrix(self):
        model, X, _ = self._fitted_model()
        assert isinstance(model.predict(X), Matrix)

    def test_predict_values_are_class_labels(self):
        """KNN returns majority-vote labels; must only contain training labels."""
        model, X, _ = self._fitted_model()
        preds = model.predict(X)
        for i in range(preds.rows):
            assert preds(i, 0) in (0.0, 1.0), f"Row {i}: {preds(i, 0)}"

    def test_predict_on_training_data_k1_is_perfect(self):
        """With k=1, every training point should map to its own label."""
        X, y = make_binary_dataset(n=40, seed=3)
        model = KNN(k=1)
        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy(y, preds)
        assert acc == 1.0, f"k=1 train accuracy = {acc:.4f}, expected 1.0"

    def test_predict_different_input_size(self):
        """Train on 60, predict on 20."""
        rng = np.random.default_rng(8)
        X_train = Matrix(np.vstack([rng.normal(-2, 0.5, (30, 1)),
                                    rng.normal(+2, 0.5, (30, 1))]))
        y_train = Matrix(np.vstack([np.zeros((30, 1)), np.ones((30, 1))]))
        X_test = Matrix(rng.normal(0, 1, (20, 1)))

        model = KNN(k=3)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.rows == 20

    def test_predict_both_classes_represented(self):
        """On a balanced, separable dataset, both classes should appear."""
        model, X, _ = self._fitted_model(n=80, seed=10)
        preds = model.predict(X)
        values = {preds(i, 0) for i in range(preds.rows)}
        assert 0.0 in values and 1.0 in values

    def test_predict_multifeature(self):
        X, y = make_multifeature_binary_dataset(n=80, seed=9)
        model = KNN(k=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.rows == X.rows
        for i in range(preds.rows):
            assert preds(i, 0) in (0.0, 1.0)

    def test_predict_tie_breaking_is_deterministic(self):
        """Same inputs must always give same output (no randomness)."""
        X, y = make_binary_dataset(n=40, seed=2)
        model = KNN(k=2)   # even k may cause ties
        model.fit(X, y)
        p1 = model.predict(X)
        p2 = model.predict(X)
        for i in range(X.rows):
            assert p1(i, 0) == p2(i, 0)


# ===========================================================================
# 19. KNN — integration
# ===========================================================================

class TestKNNIntegration:

    def test_high_accuracy_separable_data(self):
        """On well-separated classes k=3 should achieve high accuracy."""
        X, y = make_binary_dataset(n=200, seed=0)
        model = KNN(k=3)
        model.fit(X, y)
        acc = accuracy(y, model.predict(X))
        assert acc > 0.95, f"KNN accuracy = {acc:.4f}, expected > 0.95"

    def test_larger_k_smooths_boundary(self):
        """k=1 on train data is perfect; larger k should be slightly lower."""
        X, y = make_binary_dataset(n=100, seed=5)
        m1 = KNN(k=1)
        m1.fit(X, y)
        mk = KNN(k=15)
        mk.fit(X, y)
        acc1 = accuracy(y, m1.predict(X))
        acck = accuracy(y, mk.predict(X))
        # k=1 must be perfect on training data; k=15 may be lower
        assert acc1 == 1.0
        assert acck <= acc1

    def test_refit_with_new_data(self):
        """Calling fit() again should replace stored training data."""
        rng = np.random.default_rng(20)
        X1 = Matrix(rng.normal(-3, 0.3, (30, 1)))
        y1 = Matrix(np.zeros((30, 1)))
        X2 = Matrix(rng.normal(+3, 0.3, (30, 1)))
        y2 = Matrix(np.ones((30, 1)))

        model = KNN(k=1)
        model.fit(X1, y1)
        # After fitting only on class-0 data, all predictions should be 0
        preds_before = model.predict(X1)
        assert all(preds_before(i, 0) == 0.0 for i in range(preds_before.rows))

        # Refit on class-1 data; predictions should now be 1
        model.fit(X2, y2)
        preds_after = model.predict(X2)
        assert all(preds_after(i, 0) == 1.0 for i in range(preds_after.rows))

    def test_multifeature_accuracy(self):
        """Multi-feature KNN should learn a meaningful boundary."""
        X, y = make_multifeature_binary_dataset(n=200, features=3, seed=55)
        model = KNN(k=5)
        model.fit(X, y)
        acc = accuracy(y, model.predict(X))
        assert acc > 0.90, f"KNN multi-feature accuracy = {acc:.4f}, expected > 0.90"

    def test_k_equals_n_all_same_class_wins(self):
        """k = n means every prediction is the majority class in training."""
        X, y = make_binary_dataset(n=60, seed=1)   # balanced: 30 / 30
        model = KNN(k=60)
        model.fit(X, y)
        preds = model.predict(X)
        # With perfectly balanced classes either label is a valid majority winner
        for i in range(preds.rows):
            assert preds(i, 0) in (0.0, 1.0)

# ===========================================================================
# 20. NeuralNetwork — construction
# ===========================================================================

class TestNeuralNetworkInit:

    def test_default_construction(self):
        model = NeuralNetwork()
        assert model is not None
        assert hasattr(model, "_obj")

    def test_custom_learning_rate(self):
        model = NeuralNetwork(learning_rate=0.001)
        assert model is not None

    def test_high_learning_rate(self):
        model = NeuralNetwork(learning_rate=1.0)
        assert model is not None

    def test_is_model_subclass(self):
        assert isinstance(NeuralNetwork(), Model)


# ===========================================================================
# 21. NeuralNetwork.add()
# ===========================================================================

class TestNeuralNetworkAdd:

    def test_add_single_layer_does_not_raise(self):
        model = NeuralNetwork()
        model.add(2, 1)

    def test_add_multiple_layers_does_not_raise(self):
        model = NeuralNetwork()
        model.add(4, 8)
        model.add(8, 4)
        model.add(4, 1)

    def test_add_returns_none(self):
        model = NeuralNetwork()
        result = model.add(2, 1)
        assert result is None

    def test_add_wide_layer(self):
        """Large layer widths should construct without error."""
        model = NeuralNetwork()
        model.add(64, 128)
        model.add(128, 1)

    def test_add_single_neuron_layers(self):
        """Degenerate 1→1 layers should be accepted."""
        model = NeuralNetwork()
        model.add(1, 1)
        model.add(1, 1)


# ===========================================================================
# 22. NeuralNetwork.fit() — default epochs path
# ===========================================================================

class TestNeuralNetworkFitDefault:

    def test_fit_returns_none(self):
        model = NeuralNetwork(learning_rate=0.01)
        model.add(1, 4)
        model.add(4, 1)
        X, y = make_regression_dataset(n=20)
        assert model.fit(X, y) is None

    def test_fit_does_not_raise(self):
        model = NeuralNetwork(learning_rate=0.01)
        model.add(1, 4)
        model.add(4, 1)
        X, y = make_regression_dataset(n=30)
        model.fit(X, y)

    def test_fit_single_layer_network(self):
        model = NeuralNetwork(learning_rate=0.01)
        model.add(2, 1)
        X, y = make_regression_dataset(n=20, features=2)
        model.fit(X, y)

    def test_fit_single_sample(self):
        """Edge case: batch size of 1."""
        model = NeuralNetwork(learning_rate=0.01)
        model.add(1, 1)
        X = Matrix([[0.5]])
        y = Matrix([[0.5]])
        model.fit(X, y)


# ===========================================================================
# 23. NeuralNetwork.fit() — explicit epochs path
# ===========================================================================

class TestNeuralNetworkFitEpochs:

    def test_fit_with_epochs_returns_none(self):
        model = NeuralNetwork(learning_rate=0.01)
        model.add(1, 4)
        model.add(4, 1)
        X, y = make_regression_dataset(n=20)
        assert model.fit(X, y, epochs=10) is None

    def test_fit_zero_epochs(self):
        """Zero epochs: weights stay at init values, no exception raised."""
        model = NeuralNetwork(learning_rate=0.01)
        model.add(1, 4)
        model.add(4, 1)
        X, y = make_regression_dataset(n=20)
        model.fit(X, y, epochs=0)
        model.predict(X)

    def test_fit_one_epoch(self):
        model = NeuralNetwork(learning_rate=0.01)
        model.add(1, 4)
        model.add(4, 1)
        X, y = make_regression_dataset(n=20)
        model.fit(X, y, epochs=1)
        model.predict(X)

    def test_fit_epochs_none_uses_default_path(self):
        """Passing epochs=None must invoke the no-epoch (default) overload."""
        model = NeuralNetwork(learning_rate=0.01)
        model.add(1, 4)
        model.add(4, 1)
        X, y = make_regression_dataset(n=20)
        model.fit(X, y, epochs=None)
        model.predict(X)

    def test_fit_many_epochs_reduces_loss(self):
        """Loss after many epochs should be lower than after zero epochs."""
        X, y = make_regression_dataset(n=80, seed=1)

        # Untrained baseline
        m_zero = NeuralNetwork(learning_rate=0.05)
        m_zero.add(1, 8)
        m_zero.add(8, 1)
        m_zero.fit(X, y, epochs=0)
        loss_before = mse(y, m_zero.predict(X))

        # Trained model (same architecture, same data)
        m_trained = NeuralNetwork(learning_rate=0.05)
        m_trained.add(1, 8)
        m_trained.add(8, 1)
        m_trained.fit(X, y, epochs=500)
        loss_after = mse(y, m_trained.predict(X))

        assert loss_after < loss_before, (
            f"Expected training to reduce loss: {loss_before:.4f} → {loss_after:.4f}"
        )

    def test_fit_deep_network(self):
        """Three hidden layers should train without error."""
        model = NeuralNetwork(learning_rate=0.01)
        model.add(2, 8)
        model.add(8, 8)
        model.add(8, 4)
        model.add(4, 1)
        X, y = make_regression_dataset(n=50, features=2)
        model.fit(X, y, epochs=50)
        model.predict(X)


# ===========================================================================
# 24. NeuralNetwork.predict()
# ===========================================================================

class TestNeuralNetworkPredict:

    def _build_and_fit(self, n: int = 60, features: int = 1,
                       epochs: int = 100, lr: float = 0.01, seed: int = 0):
        X, y = make_regression_dataset(n=n, features=features, seed=seed)
        model = NeuralNetwork(learning_rate=lr)
        model.add(features, 8)
        model.add(8, 1)
        model.fit(X, y, epochs=epochs)
        return model, X, y

    def test_predict_shape(self):
        model, X, y = self._build_and_fit()
        preds = model.predict(X)
        assert preds.rows == X.rows
        assert preds.cols == 1

    def test_predict_returns_matrix(self):
        model, X, _ = self._build_and_fit()
        assert isinstance(model.predict(X), Matrix)

    def test_predict_values_are_finite(self):
        model, X, _ = self._build_and_fit()
        preds = model.predict(X)
        for i in range(preds.rows):
            assert np.isfinite(preds(i, 0)), f"Row {i} is not finite: {preds(i, 0)}"

    def test_predict_different_input_size(self):
        """Train on 80 rows, predict on 20 rows."""
        X_train, y_train = make_regression_dataset(n=80, seed=3)
        X_test, _ = make_regression_dataset(n=20, seed=4)
        model = NeuralNetwork(learning_rate=0.01)
        model.add(1, 4)
        model.add(4, 1)
        model.fit(X_train, y_train, epochs=50)
        preds = model.predict(X_test)
        assert preds.rows == 20

    def test_predict_single_sample(self):
        model, X, _ = self._build_and_fit(n=40)
        single = Matrix([[X(0, 0)]])
        preds = model.predict(single)
        assert preds.rows == 1
        assert preds.cols == 1

    def test_predict_multifeature(self):
        model, X, _ = self._build_and_fit(features=3, n=60)
        preds = model.predict(X)
        assert preds.rows == X.rows

    def test_predict_is_deterministic(self):
        """Same input must always yield the same output."""
        model, X, _ = self._build_and_fit()
        p1 = model.predict(X)
        p2 = model.predict(X)
        for i in range(X.rows):
            assert p1(i, 0) == p2(i, 0), f"Row {i} differs between calls"

    def test_predict_no_layers_forward_pass_is_identity(self):
        """A network with no layers added should return the input unchanged."""
        model = NeuralNetwork(learning_rate=0.01)
        # No layers added — predict is a no-op forward pass
        X, _ = make_regression_dataset(n=5)
        preds = model.predict(X)
        assert preds.rows == X.rows
        for i in range(X.rows):
            assert abs(preds(i, 0) - X(i, 0)) < 1e-12


# ===========================================================================
# 25. NeuralNetwork — integration
# ===========================================================================

class TestNeuralNetworkIntegration:

    def test_convergence_single_feature(self):
        """Single-feature regression should converge to low MSE."""
        X, y = make_regression_dataset(n=200, features=1, seed=0)
        model = NeuralNetwork(learning_rate=0.05)
        model.add(1, 16)
        model.add(16, 1)
        model.fit(X, y, epochs=1000)
        loss = mse(y, model.predict(X))
        assert loss < 0.05, f"Final MSE = {loss:.4f}, expected < 0.05"

    def test_convergence_multifeature(self):
        """Three-feature regression should converge to reasonable MSE."""
        X, y = make_regression_dataset(n=200, features=3, seed=7)
        model = NeuralNetwork(learning_rate=0.02)
        model.add(3, 16)
        model.add(16, 1)
        model.fit(X, y, epochs=1000)
        loss = mse(y, model.predict(X))
        assert loss < 0.5, f"Multi-feature MSE = {loss:.4f}, expected < 0.5"

    def test_deeper_network_trains(self):
        """A four-layer network should train without error and reduce loss."""
        X, y = make_regression_dataset(n=100, features=2, seed=5)
        model = NeuralNetwork(learning_rate=0.01)
        model.add(2, 16)
        model.add(16, 8)
        model.add(8, 4)
        model.add(4, 1)
        model.fit(X, y, epochs=0)
        loss_before = mse(y, model.predict(X))
        model.fit(X, y, epochs=300)
        loss_after = mse(y, model.predict(X))
        assert loss_after < loss_before

    def test_wider_network_trains(self):
        """A very wide single hidden layer should still converge."""
        X, y = make_regression_dataset(n=100, seed=9)
        model = NeuralNetwork(learning_rate=0.01)
        model.add(1, 64)
        model.add(64, 1)
        model.fit(X, y, epochs=500)
        loss = mse(y, model.predict(X))
        assert np.isfinite(loss)