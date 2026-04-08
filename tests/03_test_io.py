"""
03_test_io.py
=================
Full-coverage test suite for the IO Python wrapper class (daedalus/_core/IO.py).

Run:
    pytest tests/03_test_io.py

    or run all tests by

    pytest
"""

from __future__ import annotations
from unittest.mock import patch
import pytest
from daedalus import read_csv, DataFrame

def test_read_csv():
    df: DataFrame = read_csv('tests/test.csv')

    assert df.get_column_names() == ['Test1', 'Test2', 'Test3']
    assert df.cols == 3
    assert df.rows == 2

    count = 1
    for i in range(df.rows):
        for j in range(df.cols):
            assert df.at(i, j) == count
            count += 1

    with pytest.raises(FileNotFoundError):
        df: DataFrame = read_csv('fakeName.csv')

    with patch('daedalus._core.io.read_csv_cpp') as mock_cpp_read:
        mock_cpp_read.side_effect = Exception("C++ Internal Error")

        with pytest.raises(RuntimeError, match="Failed to parse CSV via Daedalus engine"):
            read_csv('tests/test.csv')