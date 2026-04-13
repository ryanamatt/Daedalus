"""
07_test_optimization.py
=======================
Full-coverage test suite for:
  - daedalus/optimization/optimization.py   (SolutionStatus, OptimizationResult)
  - daedalus/optimization/simplex_solver.py (SimplexSolver wrapper)

Run:
    pytest tests/07_test_optimization.py

    or run all tests with:

    pytest
"""

from __future__ import annotations
import pytest
import numpy as np
from daedalus import Matrix
from daedalus.optimization import SolutionStatus, OptimizationResult, SimplexSolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_solver() -> SimplexSolver:
    """Returns a fresh SimplexSolver instance."""
    return SimplexSolver()

def solve_basic_2var() -> OptimizationResult:
    """
    Solves the canonical 2-variable LP:
        Maximize   z = 5x1 + 4x2
        Subject to:
            6x1 + 4x2 <= 24
            x1  + 2x2 <= 6
            x1, x2 >= 0
    Optimal solution: x1=3, x2=3/2  →  z = 21
    """
    A = Matrix([[6.0, 4.0], [1.0, 2.0]])
    b = Matrix([[24.0], [6.0]])
    c = Matrix([[5.0, 4.0]])
    return make_solver().solve(A, b, c)

def solve_single_var() -> OptimizationResult:
    """
    Solves the single-variable LP:
        Maximize   z = 3x
        Subject to:
            x <= 5
            x >= 0
    Optimal solution: x=5, z=15
    """
    A = Matrix([[1.0]])
    b = Matrix([[5.0]])
    c = Matrix([[3.0]])
    return make_solver().solve(A, b, c)

def solve_three_var() -> OptimizationResult:
    """
    Solves a 3-variable LP:
        Maximize   z = 2x1 + 3x2 + x3
        Subject to:
            x1 + x2 + x3 <= 40
            2x1 + x2      <= 60
            x1, x2, x3 >= 0
    Optimal solution: x1=0, x2=40, x3=0 → z=120
    """
    A = Matrix([[1.0, 1.0, 1.0], [2.0, 1.0, 0.0]])
    b = Matrix([[40.0], [60.0]])
    c = Matrix([[2.0, 3.0, 1.0]])
    return make_solver().solve(A, b, c)

# ===========================================================================
# 1. SolutionStatus enum
# ===========================================================================

class TestSolutionStatus:

    def test_member(self):
        assert hasattr(SolutionStatus, "OPTIMAL")
        assert hasattr(SolutionStatus, "INFEASIBLE")
        assert hasattr(SolutionStatus, "UNBOUNDED")
        assert hasattr(SolutionStatus, "ERROR")

        assert isinstance(SolutionStatus.OPTIMAL, SolutionStatus)
        assert isinstance(SolutionStatus.INFEASIBLE, SolutionStatus)
        assert isinstance(SolutionStatus.UNBOUNDED, SolutionStatus)
        assert isinstance(SolutionStatus.ERROR, SolutionStatus)

    def test_str(self):
        assert str(SolutionStatus.OPTIMAL) == "OPTIMAL"
        assert str(SolutionStatus.INFEASIBLE) == "INFEASIBLE"
        assert str(SolutionStatus.UNBOUNDED) == "UNBOUNDED"
        assert str(SolutionStatus.ERROR) == "ERROR"

    def test_eq(self):
        assert SolutionStatus.OPTIMAL == "OPTIMAL"
        assert SolutionStatus.INFEASIBLE == "INFEASIBLE"
        assert SolutionStatus.UNBOUNDED == "UNBOUNDED"
        assert SolutionStatus.ERROR == "ERROR"

    def test_ne(self):
        assert not (SolutionStatus.OPTIMAL != "OPTIMAL")
        assert not (SolutionStatus.INFEASIBLE != "INFEASIBLE")
        assert not (SolutionStatus.UNBOUNDED != "UNBOUNDED")
        assert not (SolutionStatus.ERROR != "ERROR")

        assert SolutionStatus.OPTIMAL != "INFEASIBLE"
        assert SolutionStatus.INFEASIBLE != "UNBOUNDED"
        assert SolutionStatus.UNBOUNDED != "ERROR"
        assert SolutionStatus.ERROR != "OPTIMAL"

# ===========================================================================
# 2. OptimizationResult dataclass
# ===========================================================================

class TestOptimizationResult:

    def _make_result(self, x_data=None, obj=0.0, status=SolutionStatus.OPTIMAL):
        x = Matrix(x_data if x_data is not None else [[1.0], [2.0]])
        return OptimizationResult(x=x, objective_value=obj, status=status)

    def test_stores(self):
        result = self._make_result([[3.0], [1.5]])
        assert result.x(0, 0) == pytest.approx(3.0)
        assert result.x(1, 0) == pytest.approx(1.5)
        assert isinstance(result.x, Matrix)

        result = self._make_result(obj=42.5)
        assert result.objective_value == pytest.approx(42.5)

        result = self._make_result(status=SolutionStatus.OPTIMAL)
        assert result.status == SolutionStatus.OPTIMAL

        result = self._make_result(status=SolutionStatus.INFEASIBLE)
        assert result.status == SolutionStatus.INFEASIBLE

        result = self._make_result(status=SolutionStatus.UNBOUNDED)
        assert result.status == SolutionStatus.UNBOUNDED

        result = self._make_result(status=SolutionStatus.ERROR)
        assert result.status == SolutionStatus.ERROR

    def test_repr(self):
        result = self._make_result(obj=7.0)
        r = repr(result)
        assert "7.0" in r

        result = self._make_result(status=SolutionStatus.OPTIMAL)
        r = repr(result)
        assert "OPTIMAL" in r

# ===========================================================================
# 3. SimplexSolver
# ===========================================================================

class TestSimplexSolver:

    def test_init(self):
        solver = SimplexSolver()
        assert solver is not None

    def test_solve(self):
        result = solve_basic_2var()
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.x, Matrix)
        assert isinstance(result.objective_value, float)
        assert isinstance(result.status, SolutionStatus)

        result = solve_basic_2var()
        assert result.status == SolutionStatus.OPTIMAL
        assert result.objective_value == pytest.approx(21.0, abs=1e-6)
        assert result.x(0, 0) == pytest.approx(3.0, abs=1e-6)
        assert result.x(1, 0) == pytest.approx(1.5, abs=1e-6)
        for i in range(result.x.rows):
            assert result.x(i, 0) >= -1e-9
        x1, x2 = result.x(0, 0), result.x(1, 0)
        assert 6 * x1 + 4 * x2 <= 24 + 1e-6
        assert x1 + 2 * x2 <= 6 + 1e-6

        # The returned objective_value must equal c · x for the basic 2-var problem.
        c = [5.0, 4.0]
        computed = sum(c[i] * result.x(i, 0) for i in range(2))
        assert result.objective_value == pytest.approx(computed, abs=1e-6)

        # --- Single-variable LP ---

        result = solve_single_var()
        assert result.status == SolutionStatus.OPTIMAL
        assert result.objective_value == pytest.approx(15.0, abs=1e-6)
        assert result.x(0, 0) == pytest.approx(5.0, abs=1e-6)

        # Maximize 0x s.t. x<=10  →  z=0
        A = Matrix([[1.0]])
        b = Matrix([[10.0]])
        c = Matrix([[0.0]])
        result = make_solver().solve(A, b, c)
        assert result.objective_value == pytest.approx(0.0, abs=1e-9)
        assert result.status == SolutionStatus.OPTIMAL

        # RHS values in the thousands — checks numerical stability.
        A = Matrix([[1.0, 0.0], [0.0, 1.0]])
        b = Matrix([[1000.0], [2000.0]])
        c = Matrix([[1.0, 1.0]])
        result = make_solver().solve(A, b, c)
        assert result.status == SolutionStatus.OPTIMAL
        assert result.objective_value == pytest.approx(3000.0, abs=1e-4)

        # Fractional RHS — solution should also be fractional.
        A = Matrix([[2.0]])
        b = Matrix([[3.0]])
        c = Matrix([[1.0]])
        result = make_solver().solve(A, b, c)
        assert result.status == SolutionStatus.OPTIMAL
        assert result.x(0, 0) == pytest.approx(1.5, abs=1e-6)
        assert result.objective_value == pytest.approx(1.5, abs=1e-6)
