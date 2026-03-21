from __future__ import annotations
import os
from ..daedalus_cpp import read_csv as read_csv_cpp
from .._core import DataFrame

def read_csv(filename: str, has_header: bool = True) -> DataFrame:
    """
    Loads a CSV file into a Daedalus DataFrame.

    Args:
        filename (str): The path to the .csv file.
        has_header (bool): Whether the first row should be treated as column names. 
                           Defaults to True.

    Returns:
        DataFrame: A Daedalus DataFrame object populated with the file data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        RuntimeError: If the C++ engine encounters a parsing error.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' could not be found.")
    
    try:
        df = DataFrame()
        df._obj = read_csv_cpp(filename, has_header)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to parse CSV via Daedalus engine: {e}")
    
