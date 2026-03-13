# daedalus/__init__.py
import sys

from ._core.matrix import Matrix
from ._core.dataframe import DataFrame
from ._core.io import read_csv

__all__ = ['Matrix', 'DataFrame', 'read_csv']

if f"{__name__}._core" in sys.modules:
    del sys.modules[f"{__name__}._core"]

del sys.modules("_core")
