"""
05_test_preprocessing.py
==============
Full-coverage test suite for
    (daedalus/preprocessing/standard_scaler.py).

Run:
    pytest tests/05_test_preprocessing.py

    or run all  tests by
    
    pytest
"""

from __future__ import annotations
import pytest
from daedalus import Matrix
from daedalus.preprocessing import StandardScaler

class TestStandardScaler:
    def test_init(self):
        scaler = StandardScaler()

        assert scaler
        assert scaler.means == []
        assert scaler.std_devs == []
        assert scaler.is_fitted == False

    def test_fit(self):
        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()
        scaler.fit(m)
        assert scaler.means != []
        assert scaler.std_devs != []
        assert scaler.is_fitted != False

        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()
        scaler.fit(m)
        assert scaler.means == [2.0, 3.0]
        assert scaler.std_devs == [1.0, 1.0]
        assert scaler.is_fitted == True

    def test_transform(self):
        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()
        scaler.fit(m)
        results: Matrix = scaler.transform(m)
        assert results == Matrix([[-1, -1], [1, 1]])

        m = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()
        with pytest.raises(RuntimeError, match="Scaler must be fitted first"):
            scaler.transform(m)

    def test_fit_transform(self):
        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()

        scaler.fit_transform(m)
        assert scaler.means != []
        assert scaler.std_devs != []
        assert scaler.is_fitted != False

        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()
        scaler.fit_transform(m)
        assert scaler.means == [2.0, 3.0]
        assert scaler.std_devs == [1.0, 1.0]
        assert scaler.is_fitted == True

        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()
        results: Matrix = scaler.fit_transform(m)
        assert results == Matrix([[-1, -1], [1, 1]])

    def test_zero_variance_column(self):
        # Col 0 has variance, Col 1 is constant 5.0
        m = Matrix([[1, 5], [3, 5]])
        scaler = StandardScaler()
        scaler.fit(m)
        
        # C++ logic should set std_dev to 1.0 for the constant column
        assert scaler.std_devs[1] == 1.0
        
        results = scaler.transform(m)
        # Standardized constant column should be all zeros (val - mean)/1.0 -> (5-5)/1.0
        assert results[0, 1] == 0.0
        assert results[1, 1] == 0.0