"""
test_matrix.py
==============
Full-coverage test suite for the Matrix Python wrapper class (daedalus/_core/matrix.py).

Run:
    pytest tests/01_test_matrix.py

    or run all  tests by
    
    pytest
"""

from __future__ import annotations
import pytest
import numpy as np
from unittest.mock import patch
from daedalus import Matrix
import daedalus._core.matrix as matrix_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_2x3() -> Matrix:
    """Returns a 2x3 matrix [[1,2,3],[4,5,6]]."""
    return Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def make_3x2() -> Matrix:
    """Returns a 3x2 matrix [[1,2],[3,4],[5,6]]."""
    return Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def make_2x2() -> Matrix:
    """Returns a 2x2 identity-like matrix [[1,0],[0,1]]."""
    return Matrix([[1.0, 0.0], [0.0, 1.0]])


# ===========================================================================
# 1. __init__ — construction paths
# ===========================================================================

class TestInit:

    # --- (rows, cols) path ---

    def test_init_by_dimensions_basic(self):
        m = Matrix(3, 4)
        assert m.rows == 3
        assert m.cols == 4

    def test_init_by_dimensions_zero_rows(self):
        """0-row matrix is permitted (edge case used internally)."""
        m = Matrix(0, 5)
        assert m.rows == 0
        assert m.cols == 5

    def test_init_by_dimensions_zero_cols(self):
        m = Matrix(5, 0)
        assert m.rows == 5
        assert m.cols == 0

    def test_init_by_dimensions_zero_by_zero(self):
        m = Matrix(0, 0)
        assert m.rows == 0
        assert m.cols == 0

    def test_init_1d(self):
        m = Matrix([1, 2, 3])

        assert m.rows == 3
        assert m.cols == 1

    def test_init_empty_list(self):
        m = Matrix([])
        assert m.rows == 0
        assert m.cols == 0

    def test_init_by_dimensions_negative_rows(self):
        with pytest.raises(ValueError, match="dimensions must be positive"):
            Matrix(-1, 3)

    def test_init_by_dimensions_negative_cols(self):
        with pytest.raises(ValueError, match="dimensions must be positive"):
            Matrix(3, -1)

    # --- list-of-lists path ---

    def test_init_from_list_of_lists(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        assert m.rows == 2
        assert m.cols == 2
        assert m(0, 0) == pytest.approx(1.0)
        assert m(1, 1) == pytest.approx(4.0)

    # --- numpy path ---

    def test_init_from_numpy_2d(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = Matrix(arr)
        assert m.rows == 2
        assert m.cols == 3
        np.testing.assert_array_almost_equal(m.to_numpy(), arr)

    def test_init_from_numpy_non_contiguous(self):
        """Non-contiguous arrays (e.g. column-major views) are handled."""
        arr = np.asfortranarray(np.ones((3, 3)))
        m = Matrix(arr)
        assert m.rows == 3

    # --- bad-type path ---

    def test_init_wrong_single_arg_type(self):
        with pytest.raises(TypeError):
            Matrix("not a matrix")

    def test_init_wrong_single_arg_int(self):
        with pytest.raises(TypeError):
            Matrix(42)

    # --- bad-signature path ---

    def test_init_no_args(self):
        with pytest.raises(TypeError, match="Matrix constructor expects"):
            Matrix()

    def test_init_three_args(self):
        with pytest.raises(TypeError, match="Matrix constructor expects"):
            Matrix(2, 3, 4)

    def test_init_two_non_int_args(self):
        with pytest.raises(TypeError, match="Matrix constructor expects"):
            Matrix("a", "b")

# ===========================================================================
# 2. Properties
# ===========================================================================

class TestProperties:

    def test_rows(self):
        assert Matrix(5, 3).rows == 5

    def test_cols(self):
        assert Matrix(5, 3).cols == 3

    def test_shape(self):
        assert Matrix(4, 7).shape == (4, 7)

    def test_shape_matches_rows_and_cols(self):
        m = make_2x3()
        assert m.shape == (m.rows, m.cols)

    def test_size(self):
        assert Matrix(5, 3).size == 15
        assert Matrix(10, 10).size == 100

    def test_is_square(self):
        assert Matrix(4, 4).is_square
        assert Matrix(4, 3).is_square == False

    def test_is_vector(self):
        assert Matrix(10, 1).is_vector
        assert Matrix (1, 3).is_vector
        assert Matrix (1, 1).is_vector
        assert Matrix(2, 3).is_vector == False

    def test_is_symmetric(self):
        a = Matrix([[1, 2], [2, 1]])
        assert a.is_symmetric
        b = Matrix([[1, 2], [1, 2]])
        assert b.is_symmetric == False

    def test_is_orthogonal(self):
        a = Matrix([[1, 0], [0, 1]])
        assert a.is_orthogonal

        b = Matrix([[1, 2], [3, 4]])
        assert not b.is_orthogonal

    def test_is_invertible(self):
        a = Matrix([[1, 0], [0, 1]])
        assert a.is_invertible

        b = Matrix([[1, 1], [4, 4]])
        assert not b.is_invertible


# ===========================================================================
# 3. Static Methods
# ===========================================================================

class TestStaticMethods:

    def test_random_uniform_default(self):
        m = Matrix.random(4, 5)
        assert m.rows == 4
        assert m.cols == 5
        arr = m.to_numpy()
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_random_uniform_custom_range(self):
        m = Matrix.random(10, 10, distribution="uniform", low=2.0, high=3.0)
        arr = m.to_numpy()
        assert arr.min() >= 2.0
        assert arr.max() <= 3.0

    def test_random_normal(self):
        m = Matrix.random(100, 100, distribution="normal", loc=0.0, scale=1.0)
        assert m.rows == 100
        assert m.cols == 100

    def test_random_normal_default_params(self):
        """Normal distribution with no extra kwargs uses loc=0, scale=1."""
        m = Matrix.random(50, 50, distribution="normal")
        assert m.shape == (50, 50)

    def test_random_unsupported_distribution(self):
        with pytest.raises(ValueError, match="Unsupported distribution"):
            Matrix.random(3, 3, distribution="poisson")

    def test_random_import(self):
        with patch.object(matrix_module, 'HAS_NUMPY', False):
            with pytest.raises(ImportError):
                Matrix.random(5, 5)
    
    def test_identity(self):
        m = Matrix.Identity(3)

        assert m.rows == 3
        assert m.cols == 3
    
        for i in range(m.rows):
            for j in range(m.cols):
                assert m(i, j) == 1 if i == j else m(i, j) == 0

    def test_zeros(self):
        m = Matrix.Zeros(2, 2)
        assert m.rows == 2 and m.cols == 2
        for i in range(2):
            for j in range(2):
                assert m[i, j] == 0

    def test_ones(self):
        m = Matrix.Ones(2, 2)
        assert m.rows == 2
        assert m.cols == 2
        assert m(0, 0) == 1.0
        assert m(1, 1) == 1.0

    def test_fill(self):
        m = Matrix.Fill(3, 3, 4)
        assert m.rows == 3
        assert m.cols == 3
        for i in range(m.rows):
            for j in range(m.cols):
                assert m(i, j) == 4

    def test_diagonal_square_list(self):
        ls = [1, 2, 3]
        m = Matrix.Diagonal(ls)
        assert m.rows == 3
        assert m.cols == 3
        for i in range(m.rows):
            for j in range(m.cols):
                assert m(i, j) == ls[i] if i == j else m(i, j) == 0

    def test_diagonal_square_list_row_col(self):
        ls = [1, 2, 3]
        m = Matrix.Diagonal(3, 5, ls)
        assert m.rows == 3
        assert m.cols == 5
        for i in range(m.rows):
            for j in range(m.cols):
                assert m(i, j) == ls[i] if i == j else m(i, j) == 0

    def test_diagnoal_value_row_col(self):
        m = Matrix.Diagonal(3, 5, 2)
        assert m.rows == 3
        assert m.cols == 5
        for i in range(m.rows):
            for j in range(m.cols):
                assert m(i, j) == 2 if i == j else m(i, j) == 0

    def test_out_of_bounds_ls(self):
        m = Matrix.Diagonal(2, 2, [1, 2, 3])
        assert m.rows == 2
        assert m.cols == 2
        
        assert m(0, 0) == 1 and m(1, 1) == 2
        assert m(0, 1) == 0 and m(1, 0) == 0

    def test_diagonal_exceptions(self):
        with pytest.raises(TypeError):
            m = Matrix.Diagonal(1)

        with pytest.raises(TypeError):
            m = Matrix.Diagonal(1, 2)

        with pytest.raises(TypeError, match="Invalid Arguments"):
            Matrix.Diagonal(2, 2, "not-a-list-or-number")


# ===========================================================================
# 4. Instance Methods
# ===========================================================================

class TestInstanceMethods:

    def test_to_numpy(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        m = Matrix(data)
        arr = m.to_numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_almost_equal(arr, np.array(data))

    def test_to_numpy_import_error(self):
        m = Matrix(2, 2)
        # Patch the constant in the specific module where it's used
        with patch.object(matrix_module, 'HAS_NUMPY', False):
            with pytest.raises(ImportError, match="NumPy is required"):
                m.to_numpy()

    def test_norm(self):
        m = Matrix([[1, -2], [3, 4]])
        assert m.norm() == m.norm("fro")
        assert round(m.norm(), 2) == 5.48
        assert m.norm(1) == 6.0
        assert m.norm("inf") == 7.0

        with pytest.raises(ValueError):
            m.norm("2")

    def test_sum(self):
        a = Matrix([[1, 2], [3, 4]])
        a_s_0 = a.sum(axis=0)
        assert a_s_0[0, 0] == 4 and a_s_0[0, 1] == 6
        a_s_1 = a.sum(axis=1)
        assert a_s_1[0, 0] == 3 and a_s_1[1, 0] == 7
        assert a.sum() == 10.0

        with pytest.raises(TypeError):
            assert a.sum(3)

        with pytest.raises(TypeError):
            assert a.sum("axis=1")

    def test_mean(self):
        m = Matrix([[1, -2], [3, 4]])
        m_u0 = m.mean(axis=0)
        assert m_u0.rows == 1 and m_u0.cols == 2
        assert m_u0[0, 0] == 2 and m_u0[0, 1] == 1
        m_u1 = m.mean(axis=1)
        assert m_u1.rows == 2 and m_u1.cols == 1
        assert m_u1[0, 0] == -0.5 and m_u1[1, 0] == 3.5

        with pytest.raises(TypeError):
            m.mean(axis=2)

        with pytest.raises(TypeError):
            m.mean('2')

    def test_variance(self):
        m = Matrix([[1, -2], [3, 4]])
        m_s0 = m.variance(axis=0)
        assert m_s0.rows == 1 and m_s0.cols == 2
        assert m_s0[0, 0] == 1 and m_s0[0, 1] == 9
        m_s1 = m.variance(axis=1)
        assert m_s1.rows == 2 and m_s1.cols == 1
        assert m_s1[0, 0] == 2.25 and m_s1[1, 0] == 0.25

        with pytest.raises(TypeError):
            m.variance(axis=2)

        with pytest.raises(TypeError):
            m.variance('2')

    def test_std(self):
        m = Matrix([[1, -2], [3, 4]])
        m_s0 = m.std(axis=0)
        assert m_s0.rows == 1 and m_s0.cols == 2
        assert m_s0[0, 0] == 1 and m_s0[0, 1] == 3
        m_s1 = m.std(axis=1)
        assert m_s1.rows == 2 and m_s1.cols == 1
        assert m_s1[0, 0] == 1.5 and m_s1[1, 0] == 0.5

        with pytest.raises(TypeError):
            m.std(axis=2)

        with pytest.raises(TypeError):
            m.std('2')

    def test_reshape(self):
        m = Matrix([1, 2, 3 ,4])
        m_rs = m.reshape(2, 2)
        assert m_rs.rows == 2 and m_rs.cols == 2
        assert m_rs[0, 0] == 1 and m_rs[0, 1] == 2
        assert m_rs[1, 0] == 3 and m_rs[1, 1] == 4

        with pytest.raises(ValueError):
            m.reshape(3, 1)

        with pytest.raises(ValueError):
            m.reshape(2, 6)

    def test_flatten(self):
        m = Matrix([[1, 2], [3, 4]])
        m_flat = m.flatten()
        assert m_flat.rows == 1 and m_flat.cols == 4
        assert m_flat[0, 0] == 1 and m_flat[0, 3] == 4

    def test_transpose(self):
        m = make_2x3()
        t = m.transpose()
        assert t.rows == 3
        assert t.cols == 2
        assert t(0, 0) == pytest.approx(m(0, 0))
        assert t(2, 0) == pytest.approx(m(0, 2))
        assert t(0, 1) == pytest.approx(m(1, 0))

    def test_trace(self):
        a = Matrix.Diagonal([1, 2, 3])
        assert a.trace() == 6.0

        b = Matrix([[1, 2], [1, 2], [1, 2]])
        assert b.trace() == 3.0

    def test_rank(self):
        A = Matrix([[3, 2, 2], [2, 3, -2]])
        assert A.rank() == 2

    def test_det(self):
        a = Matrix([[1, 2], [3, 4]])
        assert a.det() == -2.0
        b = Matrix([[1, 2, 3], [4, 5, 6], [7, 7, 9]])
        assert b.det() == -6.0
        c = Matrix([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(ValueError):
            c.det()

    def test_inverse(self):
        a = Matrix([[1, 2], [3, 4]])
        assert a.rows == 2 and a.cols == 2
        a = a.inverse()
        assert round(a[0, 0]) == -2 and round(a[0, 1]) == 1
        assert round(a[1, 0], 2) == 1.5 and round(a[1, 1], 2) == -0.5

        c = Matrix([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(ValueError):
            c.inverse()

    def test_pinv(self):
        a = Matrix([[1, 0, 1], [0, 1, 1]])
        a_p = a.pinv()
        a_p_np = np.array([[2/3, -1/3], [-1/3, 2/3], [1/3, 1/3]])
        np.testing.assert_array_almost_equal(a_p.to_numpy(), a_p_np)
        np.testing.assert_array_almost_equal(a.to_numpy(), (a * a_p * a).to_numpy())
 
    def test_get_row(self):
        m = make_2x3()
        row = m.get_row(1)
        assert row.rows == 1
        assert row.cols == 3
        assert row(0, 0) == pytest.approx(4.0)
        assert row(0, 2) == pytest.approx(6.0)

    def test_set_row(self):
        m: Matrix = make_2x3()
        new_row = [-1, -2, -3]
        m.set_row(0, new_row)
        row = m.get_row(0)
        assert row[0, 0] == -1 and row[0, 1] == -2 and row[0, 2] == -3

        with pytest.raises(ValueError):
            m.set_row(3, [1, 2, 3])

        with pytest.raises(ValueError):
            m.set_row(0, [1, 2])

        with pytest.raises(ValueError):
            m.set_row(0, [1, 2, 3, 4])

    def test_get_col(self):
        m = make_2x3()
        col = m.get_col(0)
        assert col.rows == 2
        assert col.cols == 1
        assert col(0, 0) == pytest.approx(1.0)
        assert col(1, 0) == pytest.approx(4.0)

    def test_set_col(self):
        m: Matrix = make_2x3()
        new_col = [-1, -2]
        m.set_col(0, new_col)
        col = m.get_col(0)
        assert col[0, 0] == -1 and col[1, 0] == -2

        with pytest.raises(ValueError):
            m.set_col(3, [1, 2])

        with pytest.raises(ValueError):
            m.set_col(0, [1])

        with pytest.raises(ValueError):
            m.set_col(0, [1, 2, 3])

    def test_set(self):
        m = Matrix(2, 2)
        m.set(0, 1, 99.0)
        assert m(0, 1) == pytest.approx(99.0)

    def test_copy_independence(self):
        m = make_2x2()
        c = m.copy()
        c.set(0, 0, 999.0)
        assert m(0, 0) == pytest.approx(1.0)  # original unchanged

    def test_solve(self):
        A = Matrix([[-1, -11, -3], [1, 1, 0], [2, 5, 1]])
        b = Matrix([-37, -1, 10])
        x = A.solve(b)
        assert round(x[0, 0], 1) == -3.0 and round(x[1, 0], 1) == 2.0 and round(x[2, 0], 1) == 6.0

        d = Matrix([[1, 2], [1, 2]])
        with pytest.raises(ValueError):
            d.solve(b)

    def test_argmax(self):
        m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 5.0, 4.0]])
        assert m.argmax() == (1, 2)

        np.testing.assert_array_almost_equal(m.argmax(0).to_numpy(), np.array([[1, 1, 1]]))
        np.testing.assert_array_almost_equal(m.argmax(1).to_numpy(), np.array([[2], [2], [1]]))

        with pytest.raises(ValueError):
            m.argmax(axis=2)

        with pytest.raises(ValueError):
            m.argmax(axis="Hello")   

    def test_argmin(self):
        m = Matrix([[1.0, 2.0, 3.0], [-1.0, 5.0, 6.0], [1.0, 5.0, 4.0]])
        assert m.argmin() == (1, 0)

        np.testing.assert_array_almost_equal(m.argmin(0).to_numpy(), np.array([[1, 0, 0]]))
        np.testing.assert_array_almost_equal(m.argmin(1).to_numpy(), np.array([[0], [0], [0]]))

        with pytest.raises(ValueError):
            m.argmin(axis=2)

        with pytest.raises(ValueError):
            m.argmin(axis="Hello")  

    def test_log(self):
        m = Matrix([[1.0, 2], [3, 4]])
        m_log = m.log()
        assert m_log.shape == m.shape

        m_log_rd = round(m_log, 2)
        m_log_np = np.array([[0, 0.69], [1.10, 1.39]])
        np.testing.assert_array_almost_equal(m_log_rd.to_numpy(), m_log_np)

    def test_sqrt(self):
        m = Matrix([[1, 2], [3, 64]])
        m_sq = m.sqrt()
        assert m_sq.shape == m.shape

        m_sq_rd = round(m_sq, 2)
        m_sq_np = np.array([[1, 1.41], [1.73, 8]])
        np.testing.assert_array_almost_equal(m_sq_rd.to_numpy(), m_sq_np)

    def test_exp(self):
        m = Matrix([[1, 2], [3, 4]])
        m_exp = m.exp()
        assert m.shape == m_exp.shape

        m_exp_rd = round(m_exp, 2)
        m_exp_np = np.array([[2.72, 7.39], [20.09, 54.60]])
        np.testing.assert_almost_equal(m_exp_rd.to_numpy(), m_exp_np)

        m_re = m_exp.log()
        np.testing.assert_almost_equal(m_re.to_numpy(), m.to_numpy())

# ===========================================================================
# 5. Dunder — element access
# ===========================================================================

class TestDunderAccess:

    def test_call_operator(self):
        m = make_2x3()
        assert m(0, 2) == pytest.approx(3.0)
        assert m(1, 0) == pytest.approx(4.0)

    def test_getitem_single_int_returns_row(self):
        m = make_2x3()
        row = m[0]
        assert isinstance(row, Matrix)
        assert row.rows == 1
        assert row.cols == 3

    def test_getitem_tuple_single_element(self):
        m = make_2x3()
        val = m[1, 2]
        assert val == pytest.approx(6.0)

    def test_getitem_tuple_slice_row(self):
        m = make_2x3()
        sub = m[0:1, 0:3]
        assert isinstance(sub, Matrix)
        assert sub.rows == 1
        assert sub.cols == 3

    def test_getitem_tuple_slice_submatrix(self):
        m = Matrix([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]])
        sub = m[0:2, 1:3]
        assert sub.rows == 2
        assert sub.cols == 2
        assert sub(0, 0) == pytest.approx(2.0)
        assert sub(1, 1) == pytest.approx(6.0)

    def test_setitem(self):
        m = Matrix(3, 3)
        m[1, 2] = 7.5
        assert m(1, 2) == pytest.approx(7.5)

    def test_len(self):
        m = make_2x3()
        assert len(m) == 2

    def test_repr(self):
        m = Matrix(2, 2)
        r = repr(m)
        assert "Matrix" in r
        assert "2x2" in r

    def test_iter(self):
        m = make_2x3()
        rows = list(m)
        assert len(rows) == 2
        for row in rows:
            assert isinstance(row, Matrix)
            assert row.cols == 3

    def test_round(self):
        a = Matrix([2.12978, 2.14056])
        a_r_2 = round(a, 2)
        assert a_r_2[0, 0] == 2.13
        assert a_r_2[1, 0] == 2.14

        a_r_3 = round(a, 3)
        assert a_r_3[0, 0] == 2.13
        assert a_r_3[1, 0] == 2.141

        assert a[0, 0] == 2.12978
        assert a[1, 0] == 2.14056

    def test_contains(self):
        m = Matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert 2 in m
        assert not (11 in m)

# ===========================================================================
# 6. Test Decomposition Methods
# ===========================================================================

class TestDecomposition:

    def test_svd(self):
        A = Matrix([[3, 2, 2], [2, 3, -2]])
        U, sigma, V = A.svd()
        U_rd = round(U, 3)
        assert U_rd[0, 0] == 0.707 and U_rd[0, 1] == 0.707
        assert U_rd[1, 0] == 0.707 and U_rd[1, 1] == -0.707

        sigma_rd = round(sigma, 3)
        assert sigma_rd[0, 0] == 5.0 and sigma_rd[1, 1] == 3.0

        V_rd = round(V, 3)
        assert V_rd[0, 0] == 0.707 and V_rd[0, 1] == 0.236 and V_rd[0, 2] == 0
        assert V_rd[1, 0] == 0.707 and V_rd[1, 1] == -0.236 and V_rd[1, 2] == 0
        assert V_rd[2, 0] == 0 and V_rd[2, 1] == 0.943 and V_rd[2, 2] == 0

        A_reconstructed = round(U * sigma * V.T, 0)
        np.testing.assert_array_equal(A_reconstructed.to_numpy(), A.to_numpy())

    def test_cholesky(self):
        A = Matrix([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
        L = Matrix([[2, 0, 0], [6, 1, 0], [-8, 5, 3]])
        np.testing.assert_array_equal(A.cholesky().to_numpy(), L.to_numpy())
        A_rec = L * L.T
        np.testing.assert_array_equal(A_rec.to_numpy(), A.to_numpy())

        B = Matrix(3, 2)
        with pytest.raises(ValueError):
            B.cholesky()

        C = Matrix([[1, 2], [1, 2]])
        with pytest.raises(ValueError):
            C.cholesky()

        D = Matrix([[1, 2], [2, 1]])
        with pytest.raises(ValueError):
            C.cholesky()

    def test_LU(self):
        A = Matrix([[0, 2, 1], [3, -1, 1], [1, 1, -2]])
        P_act = Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        L_act = Matrix([[1, 0, 0], [0, 1, 0], [0.33, 0.67, 1]])
        U_act = Matrix([[3, -1, 1], [0, 2, 1], [0, 0, -3]])

        P, L, U = A.LU()
        np.testing.assert_array_almost_equal(round(P, 2).to_numpy(), P_act.to_numpy())
        np.testing.assert_array_almost_equal(round(L, 2).to_numpy(), L_act.to_numpy())
        np.testing.assert_array_almost_equal(round(U, 2).to_numpy(), U_act.to_numpy())
        np.testing.assert_array_almost_equal(round((P.inverse() * L * U), 2).to_numpy(), A.to_numpy())

    def test_QR(self):
        A = Matrix([[3, 1], [4, 2]])
        Q, R = A.QR()
        Q_np = np.array([[0.6, -0.8], [0.8, 0.6]])
        R_np = np.array([[5, 2.2], [0, 0.4]])
        np.testing.assert_almost_equal(Q.to_numpy(), Q_np)
        np.testing.assert_almost_equal(R.to_numpy(), R_np)
        np.testing.assert_array_almost_equal(A.to_numpy(), (Q * R).to_numpy())

# ===========================================================================
# 7. Arithmetic operators
# ===========================================================================

class TestArithmetic:

    # --- Addition ---

    def test_add_matrix(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[5.0, 6.0], [7.0, 8.0]])
        c = a + b
        np.testing.assert_array_almost_equal(c.to_numpy(), [[6, 8], [10, 12]])

    def test_add_scalar(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        c = a + 10.0
        np.testing.assert_array_almost_equal(c.to_numpy(), [[11, 12], [13, 14]])

    def test_add_scalar_int(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        c = a + 1
        assert c(0, 0) == pytest.approx(2.0)

    def test_radd_scalar(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        c = 10.0 + a
        np.testing.assert_array_almost_equal(c.to_numpy(), [[11, 12], [13, 14]])

    def test_iadd_matrix(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[1.0, 1.0], [1.0, 1.0]])
        a += b
        assert a(0, 0) == pytest.approx(2.0)

    def test_iadd_scalar(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        a += 5.0
        assert a(0, 0) == pytest.approx(6.0)

    def test_iadd_scalar_int(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        a += 2
        assert a(0, 0) == pytest.approx(3.0)

    # --- Subtraction ---

    def test_sub_matrix(self):
        a = Matrix([[5.0, 6.0], [7.0, 8.0]])
        b = Matrix([[1.0, 2.0], [3.0, 4.0]])
        c = a - b
        np.testing.assert_array_almost_equal(c.to_numpy(), [[4, 4], [4, 4]])

    def test_sub_scalar(self):
        a = Matrix([[5.0, 6.0], [7.0, 8.0]])
        c = a - 1.0
        np.testing.assert_array_almost_equal(c.to_numpy(), [[4, 5], [6, 7]])

    def test_sub_scalar_int(self):
        a = Matrix([[5.0, 6.0], [7.0, 8.0]])
        c = a - 1
        assert c(0, 0) == pytest.approx(4.0)

    def test_isub_matrix(self):
        a = Matrix([[5.0, 6.0], [7.0, 8.0]])
        b = Matrix([[1.0, 1.0], [1.0, 1.0]])
        a -= b
        assert a(0, 0) == pytest.approx(4.0)

    def test_isub_scalar(self):
        a = Matrix([[5.0, 6.0], [7.0, 8.0]])
        a -= 2.0
        assert a(0, 0) == pytest.approx(3.0)

    def test_isub_scalar_int(self):
        a = Matrix([[5.0, 6.0], [7.0, 8.0]])
        a -= 2
        assert a(0, 0) == pytest.approx(3.0)

    # --- Multiplication ---

    def test_mul_matrix(self):
        a = make_2x3()
        b = make_3x2()
        c = a * b
        assert c.rows == 2
        assert c.cols == 2

    def test_mul_scalar(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        c = a * 2.0
        np.testing.assert_array_almost_equal(c.to_numpy(), [[2, 4], [6, 8]])

    def test_mul_scalar_int(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        c = a * 3
        assert c(0, 0) == pytest.approx(3.0)

    def test_rmul_scalar(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        c = 3.0 * a
        np.testing.assert_array_almost_equal(c.to_numpy(), [[3, 6], [9, 12]])

    def test_imul_scalar(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        a *= 2.0
        assert a(0, 0) == pytest.approx(2.0)

    def test_imul_scalar_int(self):
        a = Matrix([[2.0, 4.0], [6.0, 8.0]])
        a *= 2
        assert a(0, 0) == pytest.approx(4.0)

    def test_mul_large_uses_tiled(self):
        """__mul__ switches to multiply_tiled when both dims >= 1024."""
        a = Matrix.random(1024, 1024)
        b = Matrix.random(1024, 1024)
        c = a * b
        assert c.rows == 1024
        assert c.cols == 1024

    def test_mul_medium_does_not_use_tiled(self):
        """Below 1024 threshold, standard path is used."""
        a = Matrix.random(512, 512)
        b = Matrix.random(512, 512)
        c = a * b
        assert c.rows == 512
        assert c.cols == 512

    def test_matmul(self):
        a = make_2x3()
        b = make_3x2()
        c = a @ b
        assert c.rows == 2
        assert c.cols == 2

        a = Matrix([[2, 2], [2, 2]])
        b = Matrix([[2, 2], [2, 2]])
        c = a @ b
        c_np = np.array([[8, 8], [8, 8]])
        np.testing.assert_array_almost_equal(c.to_numpy(), c_np)

    def test_mul_errors(self):
        a = make_2x3()
        b = make_2x3()
        with pytest.raises(ValueError):
            a * b

    # --- Division ---

    def test_true_division(self):
        m = Matrix([1, 2])
        m_2 = m / 2
        assert m_2.rows == m.rows and m_2.cols == m.cols
        assert m_2[0, 0] == 0.5 and m_2[1, 0] == 1

        with pytest.raises(TypeError):
            a = Matrix([1, 2])
            m / a

    # --- Power ---

    def test_power(self):
        a = Matrix.Diagonal([1, 2, 3]) ** 2
        b = np.diag(np.array([1, 4, 9]))
        np.testing.assert_array_almost_equal(a.to_numpy(), b)

    # --- abs ---

    def test_abs(self):
        a = Matrix([[2, -2], [-2, 2]])
        b = np.array([[2, 2], [2, 2]])
        np.testing.assert_array_almost_equal(abs(a).to_numpy(), b)

    # --- Unary ---

    def test_pos(self):
        a = Matrix([[1.0, -2.0], [3.0, -4.0]])
        b = +a
        np.testing.assert_array_almost_equal(a.to_numpy(), b.to_numpy())

    def test_neg(self):
        a = Matrix([[1.0, -2.0], [3.0, -4.0]])
        b = -a
        np.testing.assert_array_almost_equal(b.to_numpy(), [[-1, 2], [-3, 4]])

# ===========================================================================
# 8. External Dunders
# ===========================================================================

class ExternalDunders:

    def test_dunder_array(self):
        m = Matrix([[3, 2, 2], [2, 3, -2]])
        m_np = np.array(m)
        assert m_np == np.array([[3, 2, 2], [2, 3, -2]])

    def test_array_no_numpy(monkeypatch):
        import daedalus._core.matrix as matrix_module
        monkeypatch.setattr(matrix_module, 'HAS_NUMPY', False)
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ImportError, match="Must have NumPy imported."):
            m.__array__()

    def test_numpy_array_protocol_explicit(self):
        import numpy as np
        from daedalus import Matrix
        
        m = Matrix([[1, 2], [3, 4]])
        
        # 1. Trigger via explicit dunder call (Guaranteed coverage)
        # This directly hits line 842 with the 'copy' variable assigned
        arr_explicit = m.__array__(dtype=None, copy=True)
        assert isinstance(arr_explicit, np.ndarray)
        
        # 2. Trigger via np.array with explicit copy flags
        # This ensures the protocol works as intended for users
        arr_copy = np.array(m, copy=True)
        arr_no_copy = np.array(m, copy=False)
        
        assert arr_copy.shape == (2, 2)
        assert np.all(arr_copy == arr_no_copy)

# ===========================================================================
# 9. Comparison operators
# ===========================================================================

class TestComparisons:

    def test_bool_nonempty_true(self):
        assert bool(make_2x2()) is True

    def test_bool_zero_rows_false(self):
        assert bool(Matrix(0, 5)) is False

    def test_bool_zero_cols_false(self):
        assert bool(Matrix(5, 0)) is False

    def test_bool_zero_by_zero_false(self):
        assert bool(Matrix(0, 0)) is False

    def test_gt(self):
        a = Matrix([[1.0, 3.0], [2.0, 4.0]])
        b = a > 2.0
        np.testing.assert_array_almost_equal(b.to_numpy(), [[0, 1], [0, 1]])

    def test_gt_int_threshold(self):
        a = Matrix([[1.0, 3.0], [2.0, 4.0]])
        b = a > 2
        np.testing.assert_array_almost_equal(b.to_numpy(), [[0, 1], [0, 1]])

    def test_lt(self):
        a = Matrix([[1.0, 3.0], [2.0, 4.0]])
        b = a < 3.0
        np.testing.assert_array_almost_equal(b.to_numpy(), [[1, 0], [1, 0]])

    def test_lt_int_threshold(self):
        a = Matrix([[1.0, 3.0], [2.0, 4.0]])
        b = a < 3
        np.testing.assert_array_almost_equal(b.to_numpy(), [[1, 0], [1, 0]])

    def test_ge(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = a >= 2.0
        np.testing.assert_array_almost_equal(b.to_numpy(), [[0, 1], [1, 1]])

    def test_ge_int_threshold(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = a >= 2
        np.testing.assert_array_almost_equal(b.to_numpy(), [[0, 1], [1, 1]])

    def test_le(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = a <= 2.0
        np.testing.assert_array_almost_equal(b.to_numpy(), [[1, 1], [0, 0]])

    def test_le_int_threshold(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = a <= 2
        np.testing.assert_array_almost_equal(b.to_numpy(), [[1, 1], [0, 0]])

    def test_eq_equal(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[1.0, 2.0], [3.0, 4.0]])
        assert a == b

    def test_eq_not_equal(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[9.0, 2.0], [3.0, 4.0]])
        assert not (a == b)

    def test_ne_not_equal(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[9.0, 2.0], [3.0, 4.0]])
        assert a != b

    def test_ne_equal(self):
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[1.0, 2.0], [3.0, 4.0]])
        assert not (a != b)

    def test_comparisons_with_wrong_types(self):
        """Exercises the isinstance checks in __eq__ and __ne__."""
        m = Matrix(2, 2)
        assert (m == "not a matrix") is False
        assert (m != "not a matrix") is True

# ===========================================================================
# 10. Integration / round-trip tests
# ===========================================================================

class TestIntegration:

    def test_numpy_roundtrip(self):
        arr = np.random.rand(5, 7)
        m = Matrix(arr)
        np.testing.assert_array_almost_equal(m.to_numpy(), arr)

    def test_list_roundtrip(self):
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        m = Matrix(data)
        np.testing.assert_array_almost_equal(m.to_numpy(), np.array(data))

    def test_iter_gives_all_rows(self):
        m = make_2x3()
        rows = [row.to_numpy() for row in m]
        expected = m.to_numpy()
        for i, row in enumerate(rows):
            np.testing.assert_array_almost_equal(row, expected[i:i+1])

    def test_add_then_transpose(self):
        a = make_2x3()
        b = make_2x3()
        c = (a + b).transpose()
        assert c.rows == 3
        assert c.cols == 2

    def test_copy_is_independent(self):
        orig = make_2x2()
        clone = orig.copy()
        clone.set(0, 0, -999.0)
        assert orig(0, 0) == pytest.approx(1.0)

    def test_matmul_dimensions(self):
        a = Matrix.random(3, 2)
        b = Matrix.random(2, 4)
        c = a * b
        assert c.shape == (3, 4)

    def test_chained_arithmetic(self):
        a = Matrix([[2.0, 4.0], [6.0, 8.0]])
        result = (a * 2.0 - 1.0) + 1.0
        np.testing.assert_array_almost_equal(result.to_numpy(), [[4, 8], [12, 16]])

    def test_getitem_row_as_int_matches_get_row(self):
        m = make_2x3()
        via_bracket = m[0]
        via_method = m.get_row(0)
        np.testing.assert_array_almost_equal(via_bracket.to_numpy(), via_method.to_numpy())