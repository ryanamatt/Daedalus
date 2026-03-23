"""
test_model_selection.py
==============
Full-coverage test suite for the Model Seleciton Python package (daedalus/model_selection/*).

Run:
    pytest tests/05_test_model_selection.py

    or run all  tests by
    
    pytest
"""

from __future__ import annotations
import pytest
from daedalus import Matrix
from daedalus.model_selection import train_test_split

class TestTrainTestSplit():
    def test_return_parts_len(self):
        X = Matrix([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        y = Matrix(X.rows, 1)
        for i in range(X.rows): y[i, 0] = 1

        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y)

        assert len(X_tr) == 4
        assert len(X_ts) == 1
        assert len(y_tr) == 4
        assert len(y_ts) == 1
    
    def test_diff_test_size(self):
        X = Matrix([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        y = Matrix(X.rows, 1)
        for i in range(X.rows): y[i, 0] = 1
    
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = 0.4)

        assert len(X_tr) == 3
        assert len(X_ts) == 2
        assert len(y_tr) == 3
        assert len(y_ts) == 2