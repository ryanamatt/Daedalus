"""
test_standard_scaler.py
==============
Full-coverage test suite for the Matrix Python wrapper class (daedalus/preprocessing/standard_scaler.py).

Run:
    pytest tests/04_test_standard_scaler.py

    or run all  tests by
    
    pytest
"""

from __future__ import annotations
import pytest
import numpy as np
from daedalus import Matrix
from daedalus.preprocessing import StandardScaler

class TestInit:
    def test_init_normal(self):
        scaler = StandardScaler()

        assert scaler
        assert scaler.means == []
        assert scaler.std_devs == []
        assert scaler.is_fitted == False

class TestFit:
    def test_fit_normal_no_values(self):
        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()

        scaler.fit(m)
        assert scaler.means != []
        assert scaler.std_devs != []
        assert scaler.is_fitted != False

    def test_fit_property_values(self):
        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()

        scaler.fit(m)
        assert scaler.means == [2.0, 3.0]
        assert scaler.std_devs == [1.0, 1.0]
        assert scaler.is_fitted == True

class TestTransform:
    def test_transform(self):
        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()

        scaler.fit(m)
        results: Matrix = scaler.transform(m)

        assert results == Matrix([[-1, -1], [1, 1]])

    def test_transform_unfitted_raises(self):
        """Verifies that transform raises an error if fit hasn't been called."""
        m = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()
        with pytest.raises(RuntimeError, match="Scaler must be fitted first"):
            scaler.transform(m)

class TestFitTransform:
    def test_fit_normal_no_values(self):
        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()

        scaler.fit_transform(m)
        assert scaler.means != []
        assert scaler.std_devs != []
        assert scaler.is_fitted != False

    def test_fit_property_values(self):
        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()

        scaler.fit_transform(m)
        assert scaler.means == [2.0, 3.0]
        assert scaler.std_devs == [1.0, 1.0]
        assert scaler.is_fitted == True

    def test_transform(self):
        m  = Matrix([[1, 2], [3, 4]])
        scaler = StandardScaler()

        results: Matrix = scaler.fit_transform(m)

        assert results == Matrix([[-1, -1], [1, 1]])

class TestEdgeCases:
    def test_zero_variance_column(self):
        """
        Verifies that columns with zero variance (all values same) 
        are handled by scaling with 1.0 instead of dividing by zero.
        """
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