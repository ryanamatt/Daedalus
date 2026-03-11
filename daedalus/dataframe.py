from __future__ import annotations
import typing
from .daedalus_cpp import DataFrame as _DataFrameCpp

class DataFrame:
    """
    A container for storing and manipulating heterogeneous tabular data, backed by C++.

    Stores data in a column-major format. Supported cell types are float, int, and str.
    Column order is preserved. Provides Pythonic access to rows, columns, filtering,
    encoding, and conversion to Matrix.
    """

    @typing.overload
    def __init__(self): ...

    @typing.overload
    def __init__(self, col_name: str, col_data: list[float | int | str]): ...

    def __init__(self, *args, **kwargs):
        """
        Initialize the DataFrame.

        Args:
            (no args): Creates an empty DataFrame.
            col_name (str): Name of the first column.
            col_data (list): A list of float, int, or str values for the column.

        Raises:
            TypeError: If arguments do not match an expected signature.
        """
        if len(args) == 0 and len(kwargs) == 0:
            self._obj = _DataFrameCpp()

        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], list):
            col_name, col_data = args
            self._obj = _DataFrameCpp(col_name, col_data)

        elif "col_name" in kwargs or "col_data" in kwargs:
            col_name = kwargs["col_name"]
            col_data = kwargs["col_data"]
            self._obj = _DataFrameCpp(col_name, col_data)

        else:
            raise TypeError(
                "DataFrame constructor expects no arguments, or (col_name: str, col_data: list)."
            )
        
    # --------------------------------
    # Internal Helpers
    # --------------------------------

    @classmethod
    def _from_cpp(cls, cpp_obj: _DataFrameCpp) -> DataFrame:
        """Wraps an existing C++ DataFrame object in the Python class."""
        instance = cls.__new__(cls)
        instance._obj = cpp_obj
        return instance
        
    # --------------------------------
    # Propertys
    # --------------------------------

    @property
    def rows(self) -> int:
        """Returns the number of rows in the DataFrame."""
        return self._obj.rows

    @property
    def cols(self) -> int:
        """Returns the number of columns in the DataFrame."""
        return self._obj.cols

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the dimensions of the DataFrame as (rows, cols)."""
        return (self.rows, self.cols)

    @property
    def columns(self) -> list[str]:
        """Returns the ordered list of column names."""
        return list(self._obj.get_column_names())
    
    # --------------------------------
    # Callable Functions
    # --------------------------------

    def head(self, n: int = 5) -> DataFrame:
        """
        Returns the first n rows as a new DataFrame.

        Args:
            n (int): Number of rows to return. Defaults to 5.

        Returns:
            A new DataFrame containing the first n rows.
        """
        return DataFrame._from_cpp(self._obj.head(n))
    
    def get_column_names(self) -> list[str]:
        """
        """
        return self._obj.get_column_names()

    def add_column(self, name: str, col_data: list[float | int | str]) -> None:
        """
        Appends a new column to the DataFrame.

        Args:
            name (str): The name of the new column.
            col_data (list): A list of float, int, or str values.

        Raises:
            ValueError: If the column length does not match the existing row count.
        """
        self._obj.add_column(name, col_data)

    def drop_column(self, name: str) -> None:
        """
        Removes a column from the DataFrame in-place.

        Args:
            name (str): The name of the column to remove.

        Raises:
            KeyError: If the column name does not exist.
        """
        self._obj.drop_column(name)

    def filter(self, col_name: str, predicate: typing.Callable[[float | int | str], bool]) -> DataFrame:
        """
        Returns a new DataFrame containing only rows where predicate returns True.

        Args:
            col_name (str): The column to evaluate the predicate against.
            predicate (callable): A function that accepts a cell value and returns bool.

        Returns:
            A new filtered DataFrame.

        Raises:
            KeyError: If col_name does not exist.

        Example:
            >>> df.filter("age", lambda x: x > 25)
        """
        return DataFrame._from_cpp(self._obj.filter(col_name, predicate))

    def encode_binary(self, column_name: str, val0: str = "", val1: str = "") -> None:
        """
        Performs binary encoding on a categorical string column in-place.

        Maps val0 → 0.0 and val1 → 1.0. If val0 and val1 are not provided,
        the two unique categories in the column are auto-detected.

        Args:
            column_name (str): The column to encode.
            val0 (str): The string value to map to 0.0.
            val1 (str): The string value to map to 1.0.

        Raises:
            KeyError: If the column does not exist.
            RuntimeError: If the column does not contain exactly two unique categories
                          (when auto-detecting).
        """
        self._obj.encode_binary(column_name, val0, val1)

    def to_matrix(self, target_columns: list[str]):
        """
        Extracts numeric columns into a Matrix object.

        Non-numeric values (strings) are converted to 0.0. Integers are cast to float.

        Args:
            target_columns (list[str]): Ordered list of column names to include.

        Returns:
            A Matrix of shape (rows, len(target_columns)).

        Raises:
            KeyError: If any column name in target_columns does not exist.
        """
        # Import here to avoid circular imports at module level
        from .matrix import Matrix
        result = Matrix(0, 0)
        result._obj = self._obj.to_matrix(target_columns)
        return result

    def at(self, row: int, col: int | str) -> float | int | str:
        """
        Returns the value at a specific row and column.

        Args:
            row (int): The row index.
            col (int | str): Column index or column name.

        Returns:
            The value at the given position (float, int, or str).

        Raises:
            IndexError: If row or column index is out of range.
            KeyError: If col is a string and the column name does not exist.
        """
        return self._obj.at(row, col)
    

    # --------------------------------
    # Dunder Methods
    # --------------------------------

    def __repr__(self) -> str:
        """Returns a formatted string representation of the DataFrame."""
        return self._obj.to_string()

    def __len__(self) -> int:
        """Returns the number of rows in the DataFrame."""
        return self.rows

    def __bool__(self) -> bool:
        """Returns True if the DataFrame contains at least one cell."""
        return self.rows > 0 and self.cols > 0

    def __contains__(self, col_name: str) -> bool:
        """
        Checks whether a column name exists in the DataFrame.

        Example:
            >>> "age" in df
            True
        """
        return col_name in self.columns

    def __iter__(self) -> typing.Iterator[dict[str, float | int | str]]:
        """
        Iterates over the rows of the DataFrame, yielding each as a dict.

        Example:
            >>> for row in df:
            ...     print(row["age"])
        """
        col_names = self.columns
        for r in range(self.rows):
            yield {name: self._obj.at(r, name) for name in col_names}

    def __getitem__(self, col_name: str) -> list[float | int | str]:
        """
        Returns all values for a given column as a Python list.

        Args:
            col_name (str): The name of the column to retrieve.

        Returns:
            A list of cell values (float, int, or str).

        Raises:
            KeyError: If the column name does not exist.

        Example:
            >>> df["age"]
            [25, 30, 22]
        """
        if col_name not in self.columns:
            raise KeyError(f"Column not found: '{col_name}'")
        
        return [self._obj.at(r, col_name) for r in range(self.rows)]

    def __setitem__(self, col_name: str, col_data: list[float | int| str]) -> None:
        """
        Adds or replaces a column by name.

        If the column already exists, it is dropped first, then re-added.
        Note: Replacing a column does not preserve its original position.

        Args:
            col_name (str): The column name.
            col_data (list): A list of values for the column.

        Example:
            >>> df["score"] = [0.9, 0.75, 0.85]
        """
        if col_name in self.columns:
            self._obj.drop_column(col_name)

        self._obj.add_column(col_name, col_data)
