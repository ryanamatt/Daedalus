from __future__ import annotations
import typing
import numpy as np
from ..daedalus_cpp import Matrix as _MatrixCpp

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
        
        Args:
            rows (int): Number of rows (if using dimensions). 
            cols (int): Number of columns (if using dimensions). 
            data (list[list[float]]): A nested list to populate the matrix.
            
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

        # Case 2: Initialize from NumPy array or Python List
        elif len(args) == 1:
            data = args[0]
            
            # Convert lists to numpy for unified processing
            if isinstance(data, list):
                data = np.array(data, dtype=np.float64)
            
            if isinstance(data, np.ndarray):
                if data.ndim != 2:
                    raise ValueError(f"Expected 2D array, but got {data.ndim}D.")
                
                data = np.ascontiguousarray(data, dtype=np.float64)
                
                rows, cols = data.shape
                self._obj = _MatrixCpp(rows, cols)
                
                np.copyto(np.asarray(self._obj), data)
            else:
                raise TypeError("Input data must be a list of lists or a numpy.ndarray.")

        else:
            raise TypeError(
                "Matrix constructor expects (rows, cols), list[list], or np.ndarray"
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
    
    # --------------------------------
    # Callable Functions
    # --------------------------------

    def to_numpy(self) -> np.ndarray:
        """
        Convert the Daedalus Matrix back to a NumPy array.
        Useful for visualization or using other Python ML tools.
        """
        return np.asarray(self._obj)

    def transpose(self) -> Matrix:
        """Returns a new Matrix that is the transpose of the current matrix."""
        res = Matrix(0, 0)
        res._obj = self._obj.transpose()
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
    
    def __rsub__(self, scalar: float) -> Matrix:
        """Scalar subtraction from the left (e.g., 10.0 - matrix)."""
        res = Matrix(0, 0)
        res._obj = float(scalar) - self._obj
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