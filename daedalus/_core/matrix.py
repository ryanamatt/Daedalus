from __future__ import annotations
import typing
from ..daedalus_cpp import Matrix as _MatrixCpp

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class Matrix:
    """
    A high-performance 2D Matrix class backed by C++.
    
    Supports standard arithmetic operations, slicing, and basic linear algebra.
    """

    @typing.overload
    def __init__(self, rows: int, cols: int): ...

    @typing.overload
    def __init__(self, data: list[list[float]] | np.ndarray): ...

    def __init__(self, *args, **kwargs):
        """
        Initialize the Matrix.

        This methods supports 2 ways to create a Matrix.
        1. Enter Rows, Cols to create a Matrix filled with all Zeros with dimensions.
        2. Create Matrix with filled values from 1D list, 2D list, 1D Numpy Array, or
            2D Numpy Array.
        
        Args:
            rows (int): Number of rows (if using dimensions). 
            cols (int): Number of columns (if using dimensions). 
            data (list[list[int | float]] | list[int | float] | np.ndarray): A nested list to populate the matrix.
            
        Raises:
            ValueError: If dimensions are invalid or input data is inconsistent.
            TypeError: If input types do not match expected signatures.
        """
        # Case 1: Initialize by dimensions (rows, cols)
        if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
            rows, cols = args
            if rows < 0 or cols < 0:
                raise ValueError(f"Matrix dimensions must be positive. Got {rows}x{cols}")
            self._obj = _MatrixCpp(rows, cols)

        # Case 2: Initialize from Data
        elif len(args) == 1:
            data = args[0]

            if isinstance(data, list):
                # Check if list is 1D or 2D
                if len(data) > 0 and not isinstance(data[0], list):
                    # 1D List -> Column Vector (N x 1)
                    self._obj = _MatrixCpp(len(data), 1, data)
                else:
                    # 2D List or empty list
                    self._obj = _MatrixCpp(data)
                
            # Optional: Keep NumPy support if the user happens to have it
            elif HAS_NUMPY and isinstance(data, np.ndarray):
                data = np.ascontiguousarray(data, dtype=np.float64)
                rows, cols = data.shape
                self._obj = _MatrixCpp(rows, cols)
                np.copyto(np.asarray(self._obj), data)
            else:
                raise TypeError("Input data must be a list of lists (or numpy.ndarray if installed).")

        else:
            raise TypeError(
                "Matrix constructor expects (rows, cols), list, or np.ndarray"
            )
            
    # --------------------------------
    # Propertys
    # --------------------------------

    @property
    def rows(self) -> int:
        """Returns the number of rows in the matrix."""
        return self._obj.rows

    @property
    def cols(self) -> int:
        """Returns the number of columns in the matrix."""
        return self._obj.cols
    
    @property
    def shape(self) -> tuple[int, int]:
        """Returns the dimentions of the matrix (rows, cols)"""
        return (self.rows, self.cols)
    
    # --------------------------------
    # Static Methods
    # --------------------------------

    @staticmethod
    def random(rows: int, cols: int, distribution: str ="uniform", **kwargs) -> Matrix:
        """
        Creates a new Matrix with random values.
        
        Args:
            rows (int): Number of rows.
            cols (int): Number of columns.
            distribution (str): 'uniform' or 'normal'.
            **kwargs: parameters for distribution (low, high for uniform; loc, scale for normal).
        """
        if distribution == "uniform":
            low = kwargs.get('low', 0.0)
            high = kwargs.get('high', 1.0)
            data = np.random.uniform(low, high, (rows, cols))
        elif distribution == 'normal':
            loc = kwargs.get('loc', 0.0)
            scale = kwargs.get('scale', 1.0)
            data=np.random.normal(loc, scale, (rows, cols))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        return Matrix(data)
    
    @staticmethod
    def Ones(rows: int, cols: int) -> Matrix:
        """
        Creates a Matrix filled with ones with the given rows and cols.
        
        Args:
            rows (int): Number of rows.
            cols (int): Number of cols.

        Returns:
            An Matrix with given dimensions filled with ones.
        """
        m = Matrix(rows, cols)
        m._obj = _MatrixCpp.create_filled_matrix(rows, cols, 1)
        return m
    
    @staticmethod
    def Fill(rows: int, cols: int, fill_value: int | float) -> Matrix:
        """
        Creates a Matrix filled with the fill value with the given rows and cols.
        
        Args:
            rows (int): Number of rows.
            cols (int): Number of cols.
            fill_value (int | float): The value to fill the Matrix with.

        Returns:
            An Matrix with given dimensions filled with fill value.
        """
        m = Matrix(rows, cols)
        m._obj = _MatrixCpp.create_filled_matrix(rows, cols, float(fill_value))
        return m

    @staticmethod
    def Identity(size: int) -> Matrix:
        """
        Creates an Identity Matrix with the given rows and cols.
        
        Args:
            size (int): Number of rows & cols.

        Returns:
            An Identity Matrix with given dimensions.
        """
        m = Matrix(size, size)
        m._obj = _MatrixCpp.created_diagonal_scaler(size, size, 1)
        return m
    
    @staticmethod
    @typing.overload
    def Diagonal(rows: int, cols: int, value: int | float) -> Matrix: ...

    @staticmethod
    @typing.overload
    def Diagonal(rows: int, cols: int, values: list[int | float]) -> Matrix: ...

    @staticmethod
    @typing.overload
    def Diagonal(values: list[int | float]) -> Matrix: ...

    @staticmethod
    def Diagonal(*args, **kwargs) -> Matrix:
        """
        Creates a Diagonal Matrix.

        This method supports three different call signatures:
        1. Diagonal(rows, cols, value) -> Scaled diagonal matrix.
        2. Diagonal(rows, cols, values_list) -> Diagonal matrix from list.
        3. Diagonal(values_list) -> Square diagonal matrix from list.

        Args:
            rows (int, optional): Number of rows.
            cols (int, optional): Number of columns.
            value (int | float, optional): A single value to fill the diagonal.
            values (list, optional): A list of values to place on the diagonal.

        Returns:
            Matrix: A new Matrix instance with the specified diagonal.
        """
        if len(args) == 1 and isinstance(args[0], list):
            value_or_list = args[0]
            rows = len(args[0])
            cols = len(args[0])

        elif len(args) == 3:
            rows, cols, value_or_list = args

        else:
            raise TypeError("Invalid Arguments. list[int | float] for Square Matrix or" \
            "(rows: int, cols: int, value_or_list = list[int | float]) "
            "or (rows: int, cols: int, value_or_list: int | float")

        m = Matrix(rows, cols)
        if isinstance(value_or_list, list):
            m._obj = _MatrixCpp.created_diagonal_vector(rows, cols, value_or_list)

        elif isinstance(value_or_list, int) or isinstance(value_or_list, float):
            m._obj = _MatrixCpp.created_diagonal_scaler(rows, cols, value_or_list)

        else:
            raise TypeError("Invalid Arguments. list[int | float] for Square Matrix or" \
            "(rows: int, cols: int, value_or_list = list[int | float]) "
            "or (rows: int, cols: int, value_or_list: int | float")
        
        return m
    
    # --------------------------------
    # Callable Functions
    # --------------------------------

    def to_numpy(self) -> np.ndarray:
        """
        Convert the Daedalus Matrix back to a NumPy array.
        Useful for visualization or using other Python ML tools.

        Raises:
            ImportError numpy is not installed.

        Returns:
            np.ndarray: Numpy Array
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for to_numpy(). Install it via 'pip install numpy'")
        return np.asarray(self._obj)

    def transpose(self) -> Matrix:
        """Returns a new Matrix that is the transpose of the current matrix."""
        res = Matrix(0, 0)
        res._obj = self._obj.transpose()
        return res

    def trace(self) -> float:
        """Returns a new Matrix that is the Trace of the current Matrix."""
        return self._obj.trace()

    def det(self) -> float:
        """Returns the Determinant of a Matrix."""
        if self.rows != self.cols:
            raise ValueError("Must be a Square Matrix to take Determininat (rows == cols).")
        return self._obj.det()

    def inverse(self) -> Matrix:
        """Returns the inverse of the matrix."""
        if self.rows != self.cols:
            raise ValueError("Must be a Square Matrix to take inverse (rows == cols).")
        res = Matrix(0, 0)
        res._obj = self._obj.inverse()
        return res

    def get_row(self, idx: int) -> Matrix:
        """Returns a specific row as a new Matrix."""
        res = Matrix(0, 0)
        res._obj = self._obj.get_row(idx)
        return res
    
    def set(self, r: int, c: int, val: float) -> None:
        """Explicit setter used by tests."""
        self._obj.set(r, c, val)

    def copy(self) -> Matrix:
        """Returns a deep copy of the matrix."""
        res = Matrix(0, 0)
        res._obj = self._obj.copy()
        return res

    # -------------------------------- 
    # Decomposition Methods 
    # --------------------------------

    def svd(self) -> tuple[Matrix, list[float], Matrix]:
        """
        Computes the Singular Value Decomposition.

        Returns (U, singular_values, V) such that A = U * diag(S) * V^T
        """
        u_raw, s_list, v_raw = self._obj.svd()
        
        # Wrap the C++ objects back into the Python Matrix class
        U = Matrix(0, 0)
        U._obj = u_raw
        
        V = Matrix(0, 0)
        V._obj = v_raw
        
        return U, s_list, V
    
    # -------------------------------- 
    # Dunder Methods 
    # --------------------------------

    def __call__(self, r: int, c: int) -> float:
        """Allows m(r, c) access as seen in test_matrix_initialization."""
        return self._obj(r, c)

    def __getitem__(self, index: int | tuple[int | slice, int | slice]) -> float | Matrix:
        """
        Access elements, rows, or sub-matrices.
        Supports m[r, c], m[r], and slicing m[0:2, 1:3].
        
        Supports single element access (returns float) and slicing (returns a new Matrix).
        """
        if isinstance(index, int):
            res = Matrix(0, 0)
            res._obj = self._obj.get_row(index)
            return res

        result = self._obj[index]
        if isinstance(result, _MatrixCpp):
            # Wrap the returned C++ sub-matrix back into Python class
            new_mat = Matrix(0, 0)
            new_mat._obj = result
            return new_mat
        
        return result
    
    def __setitem__(self, index: tuple[int, int], value: float) -> None:
        """
        Set the value of a specific element at (row, col).
        
        Note: Currently supports single-element assignment.
        """
        self._obj[index] = value

    def __len__(self) -> int:
        """Returns the number of rows"""
        return self.rows
    
    def __repr__(self) -> str:
        """Returns a string representation of the matrix."""
        return self._obj.to_string()
    
    def __iter__(self) -> typing.Iterator[Matrix]:
        """Allows iteration over the rows of the matrix."""
        for i in range(self.rows):
            yield self.get_row(i)

    def __round__(self, places: int) -> Matrix:
        result = Matrix(self.rows, self.cols)
        result._obj = self._obj.round(places)
        return result

    # --- Basic Dunder Mathematical Operations ---

    def __add__(self, other: Matrix) -> Matrix:
        """Element-wise addition."""
        res = Matrix(0, 0)
        if isinstance(other, (int, float)):
            res._obj = self._obj + float(other)
        else:
            res._obj = self._obj + other._obj
        return res
    
    def __radd__(self, scalar: float) -> Matrix:
        """Scalar addition from the left (e.g., 5.0 + matrix)."""
        return self.__add__(scalar)
    
    def __iadd__(self, other: float | Matrix) -> Matrix:
        if isinstance(other, (int, float)):
            self._obj += float(other)
        else:
            self._obj += other._obj
        return self

    def __sub__(self, other: float | Matrix) -> Matrix:
        """Element-wise subtraction."""
        res = Matrix(0, 0)
        if isinstance(other, (int, float)):
            res._obj = self._obj - float(other)
        else:
            res._obj = self._obj - other._obj
        return res
    
    def __isub__(self, other: float | Matrix) -> Matrix:
        """In-place element-wise subtraction."""
        if isinstance(other, (int, float)):
            self._obj -= float(other)
        else:
            self._obj -= other._obj
        return self

    def __mul__(self, other: Matrix | float) -> Matrix:
        """Matrix multiplication or scalar multiplication."""
        res = Matrix(0, 0)
        if isinstance(other, Matrix):
            if self.rows >= 1024 and other.cols >= 1024:
                res._obj = self._obj.multiply_tiled(other._obj)
            else:
                res._obj = self._obj * other._obj
        else:
            res._obj = self._obj * float(other)
        return res

    def __rmul__(self, scalar: float) -> Matrix:
        """Scalar multiplication from the left (e.g., 2.0 * matrix)."""
        res = Matrix(0, 0)
        res._obj = float(scalar) * self._obj
        return res
    
    def __imul__(self, other: float) -> Matrix:
        self._obj *= float(other)
        return self

    def __matmul__(self, other: Matrix | float) -> Matrix:
        return self.__mul__(other)

    def __pow__(self, power_value: int | float):
        self._obj = self._obj.power_to(power_value)
        return self

    def __abs__(self):
        self._obj = self._obj.abs()
        return self
    
    def __pos__(self) -> Matrix:
        """Returns a copy of the Matrix (+m)."""
        return self.copy()
    
    def __neg__(self) -> Matrix:
        """Returns a new Matrix with all signs flipped (-m)."""
        res = Matrix(0, 0)
        res._obj = self._obj * -1.0
        return res

    # --- Dunder Comparisons --- 

    def __bool__(self) -> bool:
        """Returns True if the matrix has a non-zero number of elements."""
        return self.rows > 0 and self.cols > 0

    def __gt__(self, threshold: int | float) -> Matrix:
        """
        Performs element-wise greater-than comparison against a scalar.
        Returns a Matrix of 1s and 0s.
        """
        res = Matrix(0, 0)
        res._obj = self._obj > float(threshold)
        return res

    def __lt__(self, threshold: int | float) -> Matrix:
        """
        Performs element-wise less-than comparison against a scalar.
        Returns a Matrix of 1s and 0s.
        """
        res = Matrix(0, 0)
        res._obj = self._obj < float(threshold)
        return res
    
    def __ge__(self, threshold: int | float) -> Matrix:
        """
        Performs element-wise greater-than-or-equal-to comparison against a scalar.
        Returns a Matrix of 1s and 0s.
        """
        res = Matrix(0, 0)
        res._obj = self._obj >= float(threshold)
        return res

    def __le__(self, threshold: int | float) -> Matrix:
        """
        Performs element-wise less-than-or-equal-to comparison against a scalar.
        Returns a Matrix of 1s and 0s.
        """
        res = Matrix(0, 0)
        res._obj = self._obj <= float(threshold)
        return res

    def __eq__(self, other: Matrix) -> bool:
        """Checks if two matrices are equal."""
        if not isinstance(other, Matrix):
            return False
        return self._obj == other._obj
    
    def __ne__(self, other: Matrix) -> bool:
        """Checks if two matrices are not equal."""
        if not isinstance(other, Matrix):
            return True
        return self._obj != other._obj