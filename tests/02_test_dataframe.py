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

    def test_init_with_two_positional_args(self):
        df = DataFrame("x", [1.0, 2.0, 3.0])
        assert df.rows == 3
        assert df.cols == 1
        assert df.columns == ["x"]

    def test_init_with_string_column(self):
        df = DataFrame("names", ["Alice", "Bob"])
        assert df.rows == 2
        assert df.at(0, "names") == "Alice"

    def test_init_with_int_column(self):
        df = DataFrame("counts", [10, 20, 30])
        assert df.rows == 3

    def test_init_with_mixed_column(self):
        df = DataFrame("mixed", [1.0, "hello", 42])
        assert df.rows == 3

    # --- keyword path ---

    def test_init_with_kwargs(self):
        df = DataFrame(col_name="y", col_data=[7.0, 8.0])
        assert df.cols == 1
        assert df.rows == 2

    def test_init_with_partial_kwargs_col_name(self):
        df = DataFrame(col_name="z", col_data=[0.0])
        assert df.columns == ["z"]

    # --- bad-signature path ---

    def test_init_wrong_args_raises_type_error(self):
        with pytest.raises(TypeError, match="DataFrame constructor expects"):
            DataFrame(123, 456)

    def test_init_three_positional_args_raises(self):
        with pytest.raises(TypeError, match="DataFrame constructor expects"):
            DataFrame("a", [1.0], "extra")

    def test_init_non_string_col_name_raises(self):
        with pytest.raises(TypeError, match="DataFrame constructor expects"):
            DataFrame(99, [1.0, 2.0])


# ===========================================================================
# 2. Properties
# ===========================================================================

class TestProperties:

    def test_rows_empty(self):
        assert DataFrame().rows == 0

    def test_rows_nonempty(self):
        assert make_single_col_df().rows == 3

    def test_cols_empty(self):
        assert DataFrame().cols == 0

    def test_cols_nonempty(self):
        assert make_people_df().cols == 3

    def test_shape_empty(self):
        assert DataFrame().shape == (0, 0)

    def test_shape_nonempty(self):
        df = make_people_df()
        assert df.shape == (3, 3)

    def test_shape_matches_rows_and_cols(self):
        df = make_people_df()
        assert df.shape == (df.rows, df.cols)

    def test_columns_empty(self):
        assert DataFrame().columns == []

    def test_columns_order_preserved(self):
        df = make_people_df()
        assert df.columns == ["name", "age", "score"]

    def test_get_column_names_matches_columns(self):
        df = make_people_df()
        assert list(df.get_column_names()) == df.columns


# ===========================================================================
# 3. add_column
# ===========================================================================

class TestAddColumn:

    def test_add_first_column(self):
        df = DataFrame()
        df.add_column("a", [1.0, 2.0])
        assert df.cols == 1
        assert df.rows == 2

    def test_add_second_column_matching_length(self):
        df = DataFrame()
        df.add_column("a", [1.0, 2.0])
        df.add_column("b", [3.0, 4.0])
        assert df.cols == 2

    def test_add_column_length_mismatch_raises(self):
        df = DataFrame()
        df.add_column("a", [1.0, 2.0, 3.0])
        with pytest.raises(Exception):
            df.add_column("b", [1.0, 2.0])  # wrong length

    def test_add_column_name_appears_in_columns(self):
        df = DataFrame()
        df.add_column("my_col", [True])
        assert "my_col" in df.columns


# ===========================================================================
# 4. drop_column
# ===========================================================================

class TestDropColumn:

    def test_drop_existing_column(self):
        df = make_people_df()
        df.drop_column("age")
        assert "age" not in df.columns
        assert df.cols == 2

    def test_drop_last_column_resets_rows(self):
        df = DataFrame("only", [1.0, 2.0])
        df.drop_column("only")
        assert df.rows == 0
        assert df.cols == 0

    def test_drop_nonexistent_column_raises(self):
        df = make_people_df()
        with pytest.raises(Exception):
            df.drop_column("nonexistent")

    def test_drop_preserves_other_columns(self):
        df = make_people_df()
        df.drop_column("score")
        assert df.columns == ["name", "age"]


# ===========================================================================
# 5. head / tail
# ===========================================================================

class TestHeadAndTail:

    def test_head_default_returns_five_rows(self):
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

# ===========================================================================
# 6. at
# ===========================================================================

class TestAt:

    def test_at_by_string_column_name(self):
        df = make_people_df()
        assert df.at(0, "name") == "Alice"
        assert df.at(1, "age") == 30

    def test_at_by_int_column_index(self):
        df = make_people_df()
        # column 0 = "name", column 1 = "age", column 2 = "score"
        assert df.at(0, 0) == "Alice"
        assert df.at(2, 1) == 22

    def test_at_last_row(self):
        df = make_people_df()
        assert df.at(2, "name") == "Charlie"

    def test_at_float_value(self):
        df = make_people_df()
        assert df.at(0, "score") == pytest.approx(9.5)

    def test_at_row_out_of_range_raises(self):
        df = make_people_df()
        with pytest.raises(Exception):
            df.at(99, "name")

    def test_at_col_name_not_found_raises(self):
        df = make_people_df()
        with pytest.raises(Exception):
            df.at(0, "nonexistent")

    def test_at_col_index_out_of_range_raises(self):
        df = make_people_df()
        with pytest.raises(Exception):
            df.at(0, 999)


# ===========================================================================
# 7. filter
# ===========================================================================

class TestFilter:

    def test_filter_numeric_gt(self):
        df = make_people_df()
        result = df.filter("age", lambda x: x > 24)
        assert result.rows == 2  # Alice (25) and Bob (30)

    def test_filter_all_pass(self):
        df = make_people_df()
        result = df.filter("age", lambda x: x > 0)
        assert result.rows == df.rows

    def test_filter_none_pass(self):
        df = make_people_df()
        result = df.filter("age", lambda x: x > 1000)
        assert result.rows == 0

    def test_filter_string_equality(self):
        df = make_people_df()
        result = df.filter("name", lambda x: x == "Bob")
        assert result.rows == 1
        assert result.at(0, "name") == "Bob"

    def test_filter_preserves_all_columns(self):
        df = make_people_df()
        result = df.filter("score", lambda x: x >= 8.0)
        assert result.columns == df.columns

    def test_filter_values_are_correct(self):
        df = make_people_df()
        result = df.filter("age", lambda x: x == 25)
        assert result.at(0, "score") == pytest.approx(9.5)

    def test_filter_returns_dataframe_instance(self):
        df = make_people_df()
        assert isinstance(df.filter("age", lambda x: True), DataFrame)

    def test_filter_nonexistent_column_raises(self):
        df = make_people_df()
        with pytest.raises(Exception):
            df.filter("ghost", lambda x: True)


# ===========================================================================
# 8. encode_binary
# ===========================================================================

class TestEncodeBinary:

    def test_encode_binary_explicit_values(self):
        df = make_binary_df()
        df.encode_binary("label", val0="no", val1="yes")
        assert df.at(0, "label") == pytest.approx(1.0)  # "yes" → 1.0
        assert df.at(1, "label") == pytest.approx(0.0)  # "no"  → 0.0

    def test_encode_binary_auto_detect(self):
        df = make_binary_df()
        df.encode_binary("label")  # auto-detect "no" and "yes"
        # After encoding, all values must be 0.0 or 1.0
        for r in range(df.rows):
            val = df.at(r, "label")
            assert val in (pytest.approx(0.0), pytest.approx(1.0))

    def test_encode_binary_does_not_affect_other_columns(self):
        df = make_binary_df()
        before = [df.at(r, "value") for r in range(df.rows)]
        df.encode_binary("label")
        after = [df.at(r, "value") for r in range(df.rows)]
        assert before == after

    def test_encode_binary_nonexistent_column_raises(self):
        df = make_binary_df()
        with pytest.raises(Exception):
            df.encode_binary("ghost")

    def test_encode_binary_more_than_two_categories_raises(self):
        df = DataFrame()
        df.add_column("cat", ["a", "b", "c"])
        with pytest.raises(Exception):
            df.encode_binary("cat")

    def test_encode_binary_single_category_raises(self):
        df = DataFrame()
        df.add_column("cat", ["yes", "yes", "yes"])
        with pytest.raises(Exception):
            df.encode_binary("cat")


# ===========================================================================
# 9. to_matrix
# ===========================================================================

class TestToMatrix:

    def test_to_matrix_float_column(self):
        df = make_people_df()
        m = df.to_matrix(["score"])
        assert isinstance(m, Matrix)
        assert m.rows == 3
        assert m.cols == 1
        assert m(0, 0) == pytest.approx(9.5)

    def test_to_matrix_multiple_numeric_columns(self):
        df = make_people_df()
        m = df.to_matrix(["age", "score"])
        assert m.shape == (3, 2)
        assert m(0, 0) == pytest.approx(25.0)

    def test_to_matrix_string_column_maps_to_zero(self):
        df = make_people_df()
        m = df.to_matrix(["name"])
        for r in range(df.rows):
            assert m(r, 0) == pytest.approx(0.0)

    def test_to_matrix_int_column_cast_to_double(self):
        df = make_people_df()
        m = df.to_matrix(["age"])
        assert m(1, 0) == pytest.approx(30.0)

    def test_to_matrix_column_order_preserved(self):
        df = make_people_df()
        m = df.to_matrix(["score", "age"])
        # score comes first → column 0
        assert m(0, 0) == pytest.approx(9.5)
        # age comes second → column 1
        assert m(0, 1) == pytest.approx(25.0)

    def test_to_matrix_single_row(self):
        df = DataFrame()
        df.add_column("v", [42.0])
        m = df.to_matrix(["v"])
        assert m.shape == (1, 1)
        assert m(0, 0) == pytest.approx(42.0)

    def test_to_matrix_after_encode_binary(self):
        df = make_binary_df()
        df.encode_binary("label")
        m = df.to_matrix(["label", "value"])
        # All values must be numeric (float)
        for r in range(df.rows):
            assert isinstance(m(r, 0), float)
            assert isinstance(m(r, 1), float)


# ===========================================================================
# 10. Dunder Methods
# ===========================================================================

class TestDunders:

    # --- __repr__ ---

    def test_repr_empty(self):
        s = repr(DataFrame())
        assert "Empty" in s

    def test_repr_nonempty_contains_column_names(self):
        df = make_people_df()
        s = repr(df)
        assert "name" in s
        assert "age" in s
        assert "score" in s

    def test_repr_is_string(self):
        assert isinstance(repr(make_people_df()), str)

    # --- __len__ ---

    def test_len_empty(self):
        assert len(DataFrame()) == 0

    def test_len_nonempty(self):
        assert len(make_people_df()) == 3

    def test_len_after_filter(self):
        df = make_people_df()
        result = df.filter("age", lambda x: x > 24)
        assert len(result) == 2

    # --- __bool__ ---

    def test_bool_empty_is_false(self):
        assert bool(DataFrame()) is False

    def test_bool_nonempty_is_true(self):
        assert bool(make_people_df()) is True

    def test_bool_no_rows_is_false(self):
        df = make_people_df()
        result = df.filter("age", lambda x: False)
        assert bool(result) is False

    # --- __contains__ ---

    def test_contains_existing_column(self):
        df = make_people_df()
        assert "age" in df

    def test_contains_missing_column(self):
        df = make_people_df()
        assert "height" not in df

    def test_contains_after_drop(self):
        df = make_people_df()
        df.drop_column("score")
        assert "score" not in df

    # --- __iter__ ---

    def test_iter_yields_dicts(self):
        df = make_people_df()
        rows = list(df)
        assert len(rows) == 3
        assert all(isinstance(r, dict) for r in rows)

    def test_iter_dict_has_all_column_keys(self):
        df = make_people_df()
        for row in df:
            assert set(row.keys()) == {"name", "age", "score"}

    def test_iter_values_correct(self):
        df = make_people_df()
        rows = list(df)
        assert rows[0]["name"] == "Alice"
        assert rows[1]["age"] == 30
        assert rows[2]["score"] == pytest.approx(7.25)

    def test_iter_empty_dataframe(self):
        rows = list(DataFrame())
        assert rows == []

    # --- __getitem__ ---

    def test_getitem_returns_list(self):
        df = make_people_df()
        col = df["name"]
        assert isinstance(col, list)
        assert col == ["Alice", "Bob", "Charlie"]

    def test_getitem_float_column(self):
        df = make_people_df()
        col = df["score"]
        assert col[0] == pytest.approx(9.5)

    def test_getitem_missing_column_raises_key_error(self):
        df = make_people_df()
        with pytest.raises(KeyError, match="not found"):
            _ = df["nonexistent"]

    def test_getitem_length_matches_rows(self):
        df = make_people_df()
        assert len(df["age"]) == df.rows

    # --- __setitem__ ---

    def test_setitem_adds_new_column(self):
        df = make_people_df()
        df["height"] = [160.0, 175.0, 168.0]
        assert "height" in df
        assert df.cols == 4

    def test_setitem_replaces_existing_column(self):
        df = make_people_df()
        df["age"] = [0, 0, 0]
        assert df.at(0, "age") == 0
        assert df.at(1, "age") == 0

    def test_setitem_new_column_readable_via_getitem(self):
        df = make_people_df()
        df["flag"] = [True, False, True]
        assert df["flag"] == [True, False, True]

    def test_setitem_replace_preserves_other_columns(self):
        df = make_people_df()
        df["score"] = [1.0, 2.0, 3.0]
        assert df["name"] == ["Alice", "Bob", "Charlie"]


# ===========================================================================
# 11. _from_cpp class method (internal helper)
# ===========================================================================

class TestFromCpp:

    def test_from_cpp_wraps_head_result(self):
        """head() internally uses _from_cpp; verify the returned object is usable."""
        df = make_people_df()
        h = df.head(2)
        assert h.rows == 2
        assert h.columns == df.columns

    def test_from_cpp_wraps_filter_result(self):
        df = make_people_df()
        result = df.filter("age", lambda x: x == 25)
        assert isinstance(result, DataFrame)
        assert result.rows == 1

    def test_from_cpp_wraps_to_matrix_result(self):
        df = make_people_df()
        m = df.to_matrix(["age"])
        assert isinstance(m, Matrix)


# ===========================================================================
# 12. Integration / round-trip tests
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