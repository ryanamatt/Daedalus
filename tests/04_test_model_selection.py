"""
04_test_model_selection.py
==============
Full-coverage test suite for the Model Seleciton Python package 
    (daedalus/model_selection/*).

Run:
    pytest tests/04_test_model_selection.py

    or run all  tests by
    
    pytest
"""

from __future__ import annotations
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

    def test_shuffling_and_alignment(self):
        """Verifies data is shuffled and X/y pairs remain synchronized."""
        # Unique values to track rows: X and y values match for each row
        X = Matrix([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        y = Matrix([[0], [1], [2], [3], [4]])
        
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.4, seed=42)
        
        # 1. Check Alignment: Ensure X[i] still matches y[i] after shuffle
        for i in range(X_tr.rows):
            assert X_tr[i, 0] == y_tr[i, 0]
        for i in range(X_ts.rows):
            assert X_ts[i, 0] == y_ts[i, 0]
            
        # 2. Check Shuffling: The resulting order should not be the original sequential order
        combined_y = [y_tr[i, 0] for i in range(y_tr.rows)] + \
                     [y_ts[i, 0] for i in range(y_ts.rows)]
        
        assert sorted(combined_y) == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert combined_y != [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_reproducibility(self):
        X = Matrix([[i, i] for i in range(10)])
        y = Matrix([[i] for i in range(10)])
        
        # Same seed results
        X_tr1, X_ts1, _, _ = train_test_split(X, y, seed=123)
        X_tr2, X_ts2, _, _ = train_test_split(X, y, seed=123)
        
        assert X_tr1 == X_tr2
        assert X_ts1 == X_ts2
        
        # Different seed results
        X_tr3, _, _, _ = train_test_split(X, y, seed=456)
        assert X_tr1 != X_tr3