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