"""
test_dataframe.py
=================
Full-coverage test suite for the DataFrame Python wrapper class (daedalus/_core/dataframe.py).

Run:
    pytest tests/02_test_dataframe.py

    or run all tests by

    pytest
"""

from __future__ import annotations
import pytest
from daedalus import DataFrame, Matrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_people_df() -> DataFrame:
    """Returns a 3-row DataFrame with columns: name (str), age (int), score (float)."""
    df = DataFrame()
    df.add_column("name",  ["Alice", "Bob", "Charlie"])
    df.add_column("age",   [25, 30, 22])
    df.add_column("score", [9.5, 8.0, 7.25])
    return df

def make_binary_df() -> DataFrame:
    """Returns a 4-row DataFrame with a binary string column 'label'."""
    df = DataFrame()
    df.add_column("label", ["yes", "no", "yes", "no"])
    df.add_column("value", [1.0, 2.0, 3.0, 4.0])
    return df

def make_single_col_df() -> DataFrame:
    """Returns a 3-row, single-column DataFrame."""
    return DataFrame("x", [10.0, 20.0, 30.0])

# ===========================================================================
# 1. __init__ — construction paths
# ===========================================================================

class TestInit:

    # --- no-arg path ---

    def test_init_empty_no_args(self):
        df = DataFrame()
        assert df.rows == 0
        assert df.cols == 0

    # --- (col_name, col_data) positional path ---

    def test_init(self):
        df = DataFrame("x", [1.0, 2.0, 3.0])
        assert df.rows == 3
        assert df.cols == 1
        assert df.columns == ["x"]

        df = DataFrame("names", ["Alice", "Bob"])
        assert df.rows == 2
        assert df.at(0, "names") == "Alice"

        df = DataFrame("counts", [10, 20, 30])
        assert df.rows == 3

        df = DataFrame("mixed", [1.0, "hello", 42])
        assert df.rows == 3

    # --- keyword path ---

    def test_init_with_kwargs(self):
        df = DataFrame(col_name="y", col_data=[7.0, 8.0])
        assert df.cols == 1
        assert df.rows == 2

        df = DataFrame(col_name="z", col_data=[0.0])
        assert df.columns == ["z"]

    # --- bad-signature path ---

    def test_init_excpetions(self):
        with pytest.raises(TypeError, match="DataFrame constructor expects"):
            DataFrame(123, 456)

        with pytest.raises(TypeError, match="DataFrame constructor expects"):
            DataFrame("a", [1.0], "extra")

        with pytest.raises(TypeError, match="DataFrame constructor expects"):
            DataFrame(99, [1.0, 2.0])

# ===========================================================================
# 2. Properties
# ===========================================================================

class TestProperties:

    def test_rows(self):
        assert DataFrame().rows == 0
        assert make_single_col_df().rows == 3

    def test_cols(self):
        assert DataFrame().cols == 0
        assert make_people_df().cols == 3

    def test_shape(self):
        assert DataFrame().shape == (0, 0)
        df = make_people_df()
        assert df.shape == (3, 3)

        df = make_people_df()
        assert df.shape == (df.rows, df.cols)

    def test_columns(self):
        assert DataFrame().columns == []
        df = make_people_df()
        assert df.columns == ["name", "age", "score"]

        df = make_people_df()
        assert list(df.get_column_names()) == df.columns

# ===========================================================================
# 3. Static Methods
# ===========================================================================


# ===========================================================================
# 4. Methods
# ===========================================================================

class TestMethods:

    def test_at(self):
        df = make_people_df()
        assert df.at(0, "name") == "Alice"
        assert df.at(1, "age") == 30

        df = make_people_df()
        # column 0 = "name", column 1 = "age", column 2 = "score"
        assert df.at(0, 0) == "Alice"
        assert df.at(2, 1) == 22

        df = make_people_df()
        assert df.at(2, "name") == "Charlie"
        df = make_people_df()
        assert df.at(0, "score") == pytest.approx(9.5)

        df = make_people_df()
        with pytest.raises(Exception):
            df.at(99, "name")
        df = make_people_df()
        with pytest.raises(Exception):
            df.at(0, "nonexistent")
        df = make_people_df()
        with pytest.raises(Exception):
            df.at(0, 999)

    def test_head(self):
        df = DataFrame()
        df.add_column("v", list(range(10)))
        h = df.head()
        assert h.rows == 5 and h.cols == 1

        h = df.head(2)
        assert h.rows == 2 and h.cols == 1

        df = make_people_df()
        h = df.head(1)
        assert h.at(0, "name") == "Alice"
        assert h.at(0, "age") == 25

    def test_tail(self):
        df = DataFrame()
        df.add_column("v", list(range(10)))
        t = df.tail()
        assert t.rows == 5 and t.cols == 1
        t = df.tail(2)
        assert t.rows == 2 and t.cols == 1
        df = make_people_df()
        t = df.tail(1)
        assert t.at(0, "name") == "Charlie"
        assert t.at(0, "age") == 22

    def test_get_column_names(self):
        df = make_people_df()
        col_names = df.get_column_names()
        assert len(col_names) == 3
        assert col_names == ["name", "age", "score"]

    def test_add_column(self):
        df = DataFrame()
        df.add_column("a", [1.0, 2.0])
        assert df.cols == 1
        assert df.rows == 2

        df = DataFrame()
        df.add_column("a", [1.0, 2.0])
        df.add_column("b", [3.0, 4.0])
        assert df.cols == 2

        df = DataFrame()
        df.add_column("a", [1.0, 2.0, 3.0])
        with pytest.raises(Exception):
            df.add_column("b", [1.0, 2.0])  # wrong length

        df = DataFrame()
        df.add_column("my_col", [True])
        assert "my_col" in df.columns

    def test_drop_column(self):
        df = make_people_df()
        df.drop_column("age")
        assert "age" not in df.columns
        assert df.cols == 2

        df = DataFrame("only", [1.0, 2.0])
        df.drop_column("only")
        assert df.rows == 0
        assert df.cols == 0

        df = make_people_df()
        with pytest.raises(Exception):
            df.drop_column("nonexistent")

        df = make_people_df()
        df.drop_column("score")
        assert df.columns == ["name", "age"]

    def test_filter(self):
        df = make_people_df()
        result = df.filter("age", lambda x: x > 24)
        assert result.rows == 2  # Alice (25) and Bob (30)

        df = make_people_df()
        result = df.filter("age", lambda x: x > 0)
        assert result.rows == df.rows
        df = make_people_df()
        result = df.filter("age", lambda x: x > 1000)
        assert result.rows == 0

        df = make_people_df()
        result = df.filter("name", lambda x: x == "Bob")
        assert result.rows == 1
        assert result.at(0, "name") == "Bob"

        df = make_people_df()
        result = df.filter("age", lambda x: x == 25)
        assert result.at(0, "score") == pytest.approx(9.5)

        df = make_people_df()
        with pytest.raises(Exception):
            df.filter("ghost", lambda x: True)

    def test_encode_binary(self):
        df = make_binary_df()
        df.encode_binary("label", val0="no", val1="yes")
        assert df.at(0, "label") == pytest.approx(1.0)  # "yes" → 1.0
        assert df.at(1, "label") == pytest.approx(0.0)  # "no"  → 0.0

        df = make_binary_df()
        df.encode_binary("label")  # auto-detect "no" and "yes"
        # After encoding, all values must be 0.0 or 1.0
        for r in range(df.rows):
            val = df.at(r, "label")
            assert val in (pytest.approx(0.0), pytest.approx(1.0))

        df = make_binary_df()
        before = [df.at(r, "value") for r in range(df.rows)]
        df.encode_binary("label")
        after = [df.at(r, "value") for r in range(df.rows)]
        assert before == after

        df = make_binary_df()
        with pytest.raises(Exception):
            df.encode_binary("ghost")

        df = DataFrame()
        df.add_column("cat", ["a", "b", "c"])
        with pytest.raises(Exception):
            df.encode_binary("cat")

        df = DataFrame()
        df.add_column("cat", ["yes", "yes", "yes"])
        with pytest.raises(Exception):
            df.encode_binary("cat")

    def test_to_matrix(self):
        df = make_people_df()
        m = df.to_matrix(["score"])
        assert isinstance(m, Matrix)
        assert m.rows == 3
        assert m.cols == 1
        assert m(0, 0) == pytest.approx(9.5)

        df = make_people_df()
        m = df.to_matrix(["age", "score"])
        assert m.shape == (3, 2)
        assert m(0, 0) == pytest.approx(25.0)

        df = make_people_df()
        m = df.to_matrix(["score", "age"])
        # score comes first → column 0
        assert m(0, 0) == pytest.approx(9.5)
        # age comes second → column 1
        assert m(0, 1) == pytest.approx(25.0)

# ===========================================================================
# 5. Dunder Methods - Acess Methods
# ===========================================================================

class TestDunderAccess:

    def test_getitem(self):
        df = make_people_df()
        col = df["name"]
        assert isinstance(col, list)
        assert col == ["Alice", "Bob", "Charlie"]

        df = make_people_df()
        col = df["score"]
        assert col[0] == pytest.approx(9.5)

        df = make_people_df()
        with pytest.raises(KeyError, match="not found"):
            _ = df["nonexistent"]

        df = make_people_df()
        assert len(df["age"]) == df.rows

    def test_setitem(self):
        df = make_people_df()
        df["height"] = [160.0, 175.0, 168.0]
        assert "height" in df
        assert df.cols == 4

        df = make_people_df()
        df["age"] = [0, 0, 0]
        assert df.at(0, "age") == 0
        assert df.at(1, "age") == 0

        df = make_people_df()
        df["flag"] = [True, False, True]
        assert df["flag"] == [True, False, True]

        df = make_people_df()
        df["score"] = [1.0, 2.0, 3.0]
        assert df["name"] == ["Alice", "Bob", "Charlie"]

# ===========================================================================
# 6. Dunder Methods - Other Methods
# ===========================================================================

class TestDunders:

    def test_repr(self):
        s = repr(DataFrame())
        assert "Empty" in s

        df = make_people_df()
        s = repr(df)
        assert "name" in s
        assert "age" in s
        assert "score" in s

    def test_len(self):
        assert len(DataFrame()) == 0
        assert len(make_people_df()) == 3
        df = make_people_df()
        result = df.filter("age", lambda x: x > 24)
        assert len(result) == 2

    def test_bool(self):
        assert bool(DataFrame()) is False
        assert bool(make_people_df()) is True

        df = make_people_df()
        result = df.filter("age", lambda x: False)
        assert bool(result) is False

    def test_contains(self):
        df = make_people_df()
        assert "age" in df
        df = make_people_df()
        assert "height" not in df

        df = make_people_df()
        df.drop_column("score")
        assert "score" not in df

    def test_iter(self):
        df = make_people_df()
        rows = list(df)
        assert len(rows) == 3
        assert all(isinstance(r, dict) for r in rows)
        df = make_people_df()
        for row in df:
            assert set(row.keys()) == {"name", "age", "score"}

        df = make_people_df()
        rows = list(df)
        assert rows[0]["name"] == "Alice"
        assert rows[1]["age"] == 30
        assert rows[2]["score"] == pytest.approx(7.25)

        rows = list(DataFrame())
        assert rows == []

# ===========================================================================
# 7. Integration / round-trip tests
# ===========================================================================

class TestIntegration:

    def test_add_then_drop_column(self):
        df = make_people_df()
        df.add_column("temp", [0.0, 0.0, 0.0])
        assert df.cols == 4
        df.drop_column("temp")
        assert df.cols == 3
        assert "temp" not in df.columns

    def test_filter_then_to_matrix(self):
        df = make_people_df()
        filtered = df.filter("age", lambda x: x >= 25)
        m = filtered.to_matrix(["age", "score"])
        assert m.rows == 2
        assert m.cols == 2

    def test_encode_binary_then_to_matrix(self):
        df = make_binary_df()
        df.encode_binary("label", val0="no", val1="yes")
        m = df.to_matrix(["label", "value"])
        assert m.rows == 4
        # Row 0 was "yes" → 1.0
        assert m(0, 0) == pytest.approx(1.0)
        # Row 1 was "no" → 0.0
        assert m(1, 0) == pytest.approx(0.0)

    def test_iter_then_reconstruct(self):
        """Iterating rows and re-inserting values gives the same column data."""
        df = make_people_df()
        ages = [row["age"] for row in df]
        assert ages == [25, 30, 22]

    def test_setitem_then_filter(self):
        df = make_people_df()
        df["doubled_age"] = [50, 60, 44]
        result = df.filter("doubled_age", lambda x: x > 50)
        assert result.rows == 1
        assert result.at(0, "name") == "Bob"

    def test_head_then_iter(self):
        df = make_people_df()
        rows = list(df.head(2))
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"
        assert rows[1]["name"] == "Bob"

    def test_chained_filters(self):
        df = make_people_df()
        result = df.filter("age", lambda x: x >= 22).filter("score", lambda x: x >= 9.0)
        assert result.rows == 1
        assert result.at(0, "name") == "Alice"

    def test_contains_after_setitem(self):
        df = make_people_df()
        assert "new_col" not in df
        df["new_col"] = [1.0, 2.0, 3.0]
        assert "new_col" in df

    def test_bool_after_all_rows_filtered_out(self):
        df = make_people_df()
        empty = df.filter("age", lambda x: x > 9999)
        assert bool(empty) is False

    def test_to_matrix_roundtrip_via_numpy(self):
        """to_matrix → to_numpy should preserve numeric values faithfully."""
        import numpy as np
        df = make_people_df()
        m = df.to_matrix(["age", "score"])
        arr = m.to_numpy()
        assert arr[0, 0] == pytest.approx(25.0)
        assert arr[0, 1] == pytest.approx(9.5)
        assert arr[1, 0] == pytest.approx(30.0)
        assert arr[2, 1] == pytest.approx(7.25)