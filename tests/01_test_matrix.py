import pytest
from daedalus import Matrix

def test_matrix_initialization():
    """Verify matrix initialization, default values, and boundary constraints."""
    rows, cols = 10, 5
    m1 = Matrix(rows, cols)
    assert m1.rows == rows
    assert m1.cols == cols
    
    # Verify all elements are 0.0 (Value-initialization check)
    for i in range(rows):
        for j in range(cols):
            assert m1(i, j) == 0.0 
            assert m1[i, j] == 0.0

    # Empty/Zero-sized matrix
    m2 = Matrix(0, 0)
    assert m2.rows == 0
    assert m2.cols == 0

    # Non-square "Vector-like" matrices
    m_tall = Matrix(100, 1)
    assert m_tall.rows == 100 and m_tall.cols == 1
    assert m_tall(99, 0) == 0.0 # Check last element

    m_wide = Matrix(1, 100)
    assert m_wide.rows == 1 and m_wide.cols == 100
    assert m_wide(0, 99) == 0.0 # Check last element

    # Invalid Dimensions (Negative)
    with pytest.raises(TypeError):
        Matrix(-1, 5)
    with pytest.raises(TypeError):
        Matrix(5, -1)

    # Large initialization (Sanity check for memory allocation)
    m_large = Matrix(1000, 1000)
    assert m_large.rows == 1000
    assert m_large[999, 999] == 0.0

def test_matrix_repr():
    """Verify that the string representation matches the expected C++ output format."""
    # Test a small 2x2 matrix
    m = Matrix(2, 2)
    m[0, 0] = 1.1
    m[0, 1] = 2.2
    m[1, 0] = 3.3
    m[1, 1] = 4.4
    
    expected_repr = (
        "Matrix(2x2) [\n"
        "  [1.1, 2.2],\n"
        "  [3.3, 4.4]\n"
        "]"
    )
    assert repr(m) == expected_repr

    # Test a single element matrix (1x1)
    m_single = Matrix(1, 1)
    m_single[0, 0] = 5.0
    expected_single = (
        "Matrix(1x1) [\n"
        "  [5]\n"
        "]"
    )
    assert expected_single == repr(m_single)

    # Test an empty matrix
    m_empty = Matrix(0, 0)
    assert "Matrix(0x0) [\n\n]" == repr(m_empty)

import pytest
from daedalus import Matrix

def test_matrix_dimensions():
    """Verify that rows and cols properties correctly report matrix dimensions."""
    # Test standard rectangular matrix
    m1 = Matrix(10, 5)
    assert m1.rows == 10
    assert m1.cols == 5

    # Test square matrix
    m2 = Matrix(100, 100)
    assert m2.rows == 100
    assert m2.cols == 100

    # Test vector shapes
    m_row = Matrix(1, 50)
    assert m_row.rows == 1
    assert m_row.cols == 50

    m_col = Matrix(50, 1)
    assert m_col.rows == 50
    assert m_col.cols == 1

    # Test empty matrix
    m_empty = Matrix(0, 0)
    assert m_empty.rows == 0
    assert m_empty.cols == 0

    # Verify dimensions remain unchanged after data modification
    m1[0, 0] = 99.9
    assert m1.rows == 10
    assert m1.cols == 5

    # Verify properties are read-only
    with pytest.raises(AttributeError):
        m1.rows = 20
    with pytest.raises(AttributeError):
        m1.cols = 10

def test_matrix_get_row():
    """Verify row extraction data integrity, shape, and error handling."""
    rows, cols = 3, 3
    m = Matrix(rows, cols)
    
    # Fill matrix: 
    # [0.0, 1.0, 2.0]
    # [3.0, 4.0, 5.0]
    # [6.0, 7.0, 8.0]
    counter = 0.0
    for i in range(rows):
        for j in range(cols):
            m[i, j] = counter
            counter += 1.0

    row1 = m.get_row(1)
    
    # Check dimensions: Should be (1, cols)
    assert row1.rows == 1
    assert row1.cols == 3
    
    # Check data integrity
    assert row1[0, 0] == 3.0
    assert row1[0, 1] == 4.0
    assert row1[0, 2] == 5.0

    # Verify extraction of the last row
    row2 = m.get_row(2)
    assert row2[0, 0] == 6.0
    assert row2[0, 2] == 8.0

    # Error Handling: Out of bounds
    # The C++ code throws std::out_of_range for idx < 0 or idx >= num_rows
    with pytest.raises(IndexError):
        m.get_row(3)
    
    with pytest.raises(IndexError):
        m.get_row(-1)

    # Verify the returned row is a copy, not a view
    row1[0, 0] = 99.0
    assert m[1, 0] == 3.0

def test_matrix_access_modification():
    """Verify element access and modification via set, __call__, __getitem__, and __setitem__."""
    m = Matrix(3, 3)

    # Test .set() and __call__()
    m.set(0, 0, 10.5)
    m.set(2, 2, 20.5)
    assert m(0, 0) == 10.5
    assert m(2, 2) == 20.5

    # Test __setitem__ and __getitem__ (Single Element)
    m[0, 1] = 15.0
    m[1, 1] = 25.0
    assert m[0, 1] == 15.0
    assert m[1, 1] == 25.0

    # Test Negative Indexing in __setitem__
    # bindings.cc handles negative r and c by adding self.rows() / self.cols()
    m[-1, -1] = 99.0
    assert m[2, 2] == 99.0
    
    m[-3, 0] = 55.0
    assert m[0, 0] == 55.0

    # Test Slicing via __getitem__
    # Setup values:
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    for i in range(3):
        for j in range(3):
            m[i, j] = float(i * 3 + j + 1)
            
    sub_matrix: Matrix = m[0:2, 1:3] # Rows 0,1 and Cols 1,2
    assert sub_matrix.rows == 2
    assert sub_matrix.cols == 2
    assert sub_matrix[0, 0] == 2.0  # m[0, 1]
    assert sub_matrix[0, 1] == 3.0  # m[0, 2]
    assert sub_matrix[1, 0] == 5.0  # m[1, 1]
    assert sub_matrix[1, 1] == 6.0  # m[1, 2]

    # Error Handling: Out of Bounds
    with pytest.raises(IndexError):
        val = m[3, 0]  # Row out of range
    
    with pytest.raises(IndexError):
        m[0, 3] = 1.0  # Col out of range

    # Error Handling: Invalid Index Types
    with pytest.raises(IndexError):
        val = m[0]     # Missing second index (must be 2-tuple)
        
    with pytest.raises(TypeError):
        val = m["0", 0] # Non-integer/slice index

def test_matrix_addition():
    """Verify element-wise addition and dimension mismatch handling."""
    # 1. Standard 2x2 addition
    m1 = Matrix(2, 2)
    m1[0, 0], m1[0, 1] = 1.0, 2.0
    m1[1, 0], m1[1, 1] = 3.0, 4.0

    m2 = Matrix(2, 2)
    m2[0, 0], m2[0, 1] = 10.0, 20.0
    m2[1, 0], m2[1, 1] = 30.0, 40.0

    result = m1 + m2
    
    # Check values
    assert result[0, 0] == 11.0
    assert result[0, 1] == 22.0
    assert result[1, 0] == 33.0
    assert result[1, 1] == 44.0

    # Verify original matrices are not modified (Immutability check)
    assert m1[0, 0] == 1.0
    assert m2[0, 0] == 10.0

    # Addition with a zero-initialized matrix
    m_zero = Matrix(2, 2)
    res_zero = m1 + m_zero
    assert res_zero[0, 0] == 1.0
    assert res_zero[1, 1] == 4.0

    # Dimension Mismatch Handling
    m_wrong_rows = Matrix(3, 2)
    with pytest.raises(ValueError):
        m1 + m_wrong_rows

    m_wrong_cols = Matrix(2, 3)
    with pytest.raises(ValueError):
        m1 + m_wrong_cols

    # Non-square addition (1x3)
    m_row1 = Matrix(1, 3)
    m_row2 = Matrix(1, 3)
    m_row1[0, 0], m_row1[0, 1], m_row1[0, 2] = 1.0, 2.0, 3.0
    m_row2[0, 0], m_row2[0, 1], m_row2[0, 2] = 4.0, 5.0, 6.0
    
    res_row = m_row1 + m_row2
    assert res_row.rows == 1 and res_row.cols == 3
    assert res_row[0, 2] == 9.0

def test_matrix_subtraction():
    """Verify element-wise subtraction and dimension mismatch handling."""
    m1 = Matrix(2, 2)
    m1[0, 0], m1[0, 1] = 10.0, 20.0
    m1[1, 0], m1[1, 1] = 30.0, 40.0

    m2 = Matrix(2, 2)
    m2[0, 0], m2[0, 1] = 1.0, 2.0
    m2[1, 0], m2[1, 1] = 3.0, 4.0

    result = m1 - m2
    
    assert result[0, 0] == 9.0
    assert result[0, 1] == 18.0
    assert result[1, 0] == 27.0
    assert result[1, 1] == 36.0

    # Verify original matrices are not modified (Immutability check)
    assert m1[0, 0] == 10.0
    assert m2[0, 0] == 1.0

    # Test subtraction resulting in negative values
    m3 = Matrix(2, 2)
    m3[0, 0] = 50.0
    res_neg = m2 - m3
    assert res_neg[0, 0] == -49.0

    # Dimension Mismatch Handling
    m_wrong_shape = Matrix(3, 2)
    with pytest.raises(ValueError):
        m1 - m_wrong_shape

    # Subtraction with self
    res_self = m1 - m1
    for i in range(m1.rows):
        for j in range(m1.cols):
            assert res_self[i, j] == 0.0

def test_matrix_scalar_multiplication():
    """Verify scalar multiplication (matrix * scalar and scalar * matrix)."""
    m = Matrix(2, 2)
    m[0, 0], m[0, 1] = 1.0, 2.0
    m[1, 0], m[1, 1] = 3.0, 4.0

    # Test Matrix * Scalar
    res1 = m * 2.5
    assert res1[0, 0] == 2.5
    assert res1[0, 1] == 5.0
    assert res1[1, 0] == 7.5
    assert res1[1, 1] == 10.0

    # Test Scalar * Matrix
    res2 = 3.0 * m
    assert res2[0, 0] == 3.0
    assert res2[0, 1] == 6.0
    assert res2[1, 0] == 9.0
    assert res2[1, 1] == 12.0

    # Multiply by Negative Scalar
    res_neg = m * -1.0
    assert res_neg[0, 0] == -1.0
    assert res_neg[1, 1] == -4.0

    # Multiply by Zero
    res_zero = m * 0.0
    for i in range(m.rows):
        for j in range(m.cols):
            assert res_zero[i, j] == 0.0

    # Immutability: Verify original matrix remains unchanged
    assert m[0, 0] == 1.0
    assert m[1, 1] == 4.0

def test_matrix_multiplication():
    """Verify matrix-matrix multiplication (dot product) and dimension validation."""
    # Rectangular multiplication: (2x3) * (3x2) -> (2x2)
    # [1, 2, 3]   [ 7,  8]    [ 58,  64]
    # [4, 5, 6] * [ 9, 10] =  [139, 154]
    #             [11, 12]
    m1 = Matrix(2, 3)
    m1[0, 0], m1[0, 1], m1[0, 2] = 1.0, 2.0, 3.0
    m1[1, 0], m1[1, 1], m1[1, 2] = 4.0, 5.0, 6.0

    m2 = Matrix(3, 2)
    m2[0, 0], m2[0, 1] = 7.0, 8.0
    m2[1, 0], m2[1, 1] = 9.0, 10.0
    m2[2, 0], m2[2, 1] = 11.0, 12.0

    result = m1 * m2

    assert result.rows == 2
    assert result.cols == 2
    assert result[0, 0] == 58.0
    assert result[0, 1] == 64.0
    assert result[1, 0] == 139.0
    assert result[1, 1] == 154.0

    # Square matrix multiplication (Identity property check)
    # [1, 2] * [1, 0] = [1, 2]
    # [3, 4]   [0, 1]   [3, 4]
    m_orig = Matrix(2, 2)
    m_orig[0, 0], m_orig[0, 1] = 1.0, 2.0
    m_orig[1, 0], m_orig[1, 1] = 3.0, 4.0

    m_identity = Matrix(2, 2)
    m_identity[0, 0], m_identity[1, 1] = 1.0, 1.0

    res_id = m_orig * m_identity
    assert res_id[0, 0] == 1.0
    assert res_id[0, 1] == 2.0
    assert res_id[1, 0] == 3.0
    assert res_id[1, 1] == 4.0

    # Dimension Mismatch Handling
    # Matrix A (2x3) cannot be multiplied by Matrix C (2x2)
    m_invalid = Matrix(2, 2)
    with pytest.raises(ValueError): # std::invalid_argument maps to ValueError
        m1 * m_invalid

    # Vector Dot Product (1xN * Nx1 -> 1x1)
    v1 = Matrix(1, 3)
    v2 = Matrix(3, 1)
    for i in range(3):
        v1[0, i] = i + 1.0 # [1, 2, 3]
        v2[i, 0] = i + 1.0 # [1, 2, 3]^T
    
    res_vec = v1 * v2
    assert res_vec.rows == 1 and res_vec.cols == 1
    assert res_vec[0, 0] == 14.0 # 1*1 + 2*2 + 3*3

def test_matrix_transpose():
    """Verify matrix transpose dimensions and data mapping."""
    # Test 2x3 -> 3x2 transpose
    # Original (2x3):
    # [1.0, 2.0, 3.0]
    # [4.0, 5.0, 6.0]
    m = Matrix(2, 3)
    m[0, 0], m[0, 1], m[0, 2] = 1.0, 2.0, 3.0
    m[1, 0], m[1, 1], m[1, 2] = 4.0, 5.0, 6.0

    m_t = m.transpose()

    # Verify swapped dimensions
    assert m_t.rows == 3
    assert m_t.cols == 2
    
    # Verify values: Transposed (3x2):
    # [1.0, 4.0]
    # [2.0, 5.0]
    # [3.0, 6.0]
    assert m_t[0, 0] == 1.0
    assert m_t[0, 1] == 4.0
    assert m_t[1, 0] == 2.0
    assert m_t[1, 1] == 5.0
    assert m_t[2, 0] == 3.0
    assert m_t[2, 1] == 6.0

    # Verify original matrix is unchanged (Immutability check)
    assert m.rows == 2
    assert m.cols == 3
    assert m[0, 1] == 2.0

    # Test Row Vector (1x3) -> Column Vector (3x1)
    v_row = Matrix(1, 3)
    v_row[0, 0], v_row[0, 1], v_row[0, 2] = 10.0, 20.0, 30.0
    v_col = v_row.transpose()
    
    assert v_col.rows == 3
    assert v_col.cols == 1
    assert v_col[0, 0] == 10.0
    assert v_col[1, 0] == 20.0
    assert v_col[2, 0] == 30.0

    # Double transpose returns to original dimensions
    m_back = m_t.transpose()
    assert m_back.rows == 2
    assert m_back.cols == 3
    assert m_back[1, 2] == 6.0

    # Test Square Matrix (2x2)
    m_sq = Matrix(2, 2)
    m_sq[0, 0], m_sq[0, 1] = 1.1, 2.2
    m_sq[1, 0], m_sq[1, 1] = 3.3, 4.4
    
    m_sq_t = m_sq.transpose()
    assert m_sq_t[0, 1] == 3.3
    assert m_sq_t[1, 0] == 2.2