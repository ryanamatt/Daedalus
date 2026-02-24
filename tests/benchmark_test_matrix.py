import pytest
import numpy as np
from daedalus import Matrix

pytestmark = pytest.mark.benchmark_test

def from_numpy(arr: np.ndarray):
    """Helper to create a Daedalus Matrix from a numpy array."""
    m = Matrix(arr.shape[0], arr.shape[1])
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            m.set(r, c, float(arr[r, c]))
    return m

# --- Benchmarks ---

@pytest.mark.parametrize("size", [64, 128, 256])
def test_benchmark_multiplication(benchmark, size):
    """Benchmarks O(n^3) Matrix Multiplication."""
    # Create two random matrices
    a_np = np.random.rand(size, size)
    b_np = np.random.rand(size, size)
    
    a = from_numpy(a_np)
    b = from_numpy(b_np)
    
    # The benchmark fixture runs the lambda multiple times to get an average
    result = benchmark(lambda: a * b)
    
    assert result.rows == size

@pytest.mark.parametrize("size", [128, 512, 1024])
def test_benchmark_transpose(benchmark, size):
    """Benchmarks the Tiled Transpose implementation."""
    data = np.random.rand(size, size)
    m = from_numpy(data)
    
    # Benchmarking the tiled approach (block_size=32 in your C++)
    result = benchmark(m.transpose)
    
    assert result.rows == size

@pytest.mark.parametrize("size", [500, 1000])
def test_benchmark_addition(benchmark, size):
    """Benchmarks simple element-wise addition."""
    m1 = from_numpy(np.random.rand(size, size))
    m2 = from_numpy(np.random.rand(size, size))
    
    result = benchmark(lambda: m1 + m2)
    
    assert result.rows == size

def test_benchmark_slicing(benchmark):
    """Benchmarks the get_slice (copying) operation."""
    m = from_numpy(np.random.rand(1000, 1000))
    
    # Slicing a 500x500 chunk out of the center
    result = benchmark(lambda: m[250:750, 250:750])
    
    assert result.rows == 500