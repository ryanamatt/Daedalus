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

    def test_optimal_member_exists(self):
        assert hasattr(SolutionStatus, "OPTIMAL")

    def test_infeasible_member_exists(self):
        assert hasattr(SolutionStatus, "INFEASIBLE")

    def test_unbounded_member_exists(self):
        assert hasattr(SolutionStatus, "UNBOUNDED")

    def test_error_member_exists(self):
        assert hasattr(SolutionStatus, "ERROR")

    def test_optimal_is_solution_status(self):
        assert isinstance(SolutionStatus.OPTIMAL, SolutionStatus)

    def test_infeasible_is_solution_status(self):
        assert isinstance(SolutionStatus.INFEASIBLE, SolutionStatus)

    def test_unbounded_is_solution_status(self):
        assert isinstance(SolutionStatus.UNBOUNDED, SolutionStatus)

    def test_error_is_solution_status(self):
        assert isinstance(SolutionStatus.ERROR, SolutionStatus)


# ===========================================================================
# 2. OptimizationResult dataclass
# ===========================================================================

class TestOptimizationResult:

    def _make_result(self, x_data=None, obj=0.0, status=SolutionStatus.OPTIMAL):
        x = Matrix(x_data if x_data is not None else [[1.0], [2.0]])
        return OptimizationResult(x=x, objective_value=obj, status=status)

    def test_stores_x(self):
        result = self._make_result([[3.0], [1.5]])
        assert result.x(0, 0) == pytest.approx(3.0)
        assert result.x(1, 0) == pytest.approx(1.5)

    def test_stores_objective_value(self):
        result = self._make_result(obj=42.5)
        assert result.objective_value == pytest.approx(42.5)

    def test_stores_status_optimal(self):
        result = self._make_result(status=SolutionStatus.OPTIMAL)
        assert result.status == SolutionStatus.OPTIMAL

    def test_stores_status_infeasible(self):
        result = self._make_result(status=SolutionStatus.INFEASIBLE)
        assert result.status == SolutionStatus.INFEASIBLE

    def test_stores_status_unbounded(self):
        result = self._make_result(status=SolutionStatus.UNBOUNDED)
        assert result.status == SolutionStatus.UNBOUNDED

    def test_stores_status_error(self):
        result = self._make_result(status=SolutionStatus.ERROR)
        assert result.status == SolutionStatus.ERROR

    def test_repr_contains_objective_value(self):
        result = self._make_result(obj=7.0)
        r = repr(result)
        assert "7.0" in r

    def test_repr_contains_status(self):
        result = self._make_result(status=SolutionStatus.OPTIMAL)
        r = repr(result)
        assert "OPTIMAL" in r

    def test_x_is_matrix(self):
        result = self._make_result()
        assert isinstance(result.x, Matrix)

    def test_objective_value_zero(self):
        result = self._make_result(obj=0.0)
        assert result.objective_value == pytest.approx(0.0)

    def test_objective_value_negative(self):
        """Objective value can be negative (e.g. all-zero solution)."""
        result = self._make_result(obj=-5.5)
        assert result.objective_value == pytest.approx(-5.5)


# ===========================================================================
# 3. SimplexSolver — construction
# ===========================================================================

class TestSimplexSolverInit:

    def test_instantiation(self):
        solver = SimplexSolver()
        assert solver is not None

    def test_has_solve_method(self):
        solver = SimplexSolver()
        assert callable(getattr(solver, "solve", None))

    def test_multiple_instances_are_independent(self):
        s1 = SimplexSolver()
        s2 = SimplexSolver()
        assert s1 is not s2


# ===========================================================================
# 4. SimplexSolver.solve — result types
# ===========================================================================

class TestSolveReturnTypes:

    def test_returns_optimization_result(self):
        result = solve_basic_2var()
        assert isinstance(result, OptimizationResult)

    def test_x_is_matrix(self):
        result = solve_basic_2var()
        assert isinstance(result.x, Matrix)

    def test_objective_value_is_float(self):
        result = solve_basic_2var()
        assert isinstance(result.objective_value, float)

    def test_status_is_solution_status(self):
        result = solve_basic_2var()
        assert isinstance(result.status, SolutionStatus)

    def test_x_shape_matches_num_vars(self):
        """x should have one row per decision variable."""
        result = solve_basic_2var()
        assert result.x.rows == 2
        assert result.x.cols == 1

    def test_x_shape_single_var(self):
        result = solve_single_var()
        assert result.x.rows == 1
        assert result.x.cols == 1

    def test_x_shape_three_vars(self):
        result = solve_three_var()
        assert result.x.rows == 3
        assert result.x.cols == 1


# ===========================================================================
# 5. SimplexSolver.solve — correctness (optimal cases)
# ===========================================================================

class TestSolveCorrectness:

    # --- 2-variable LP ---

    def test_basic_2var_status_optimal(self):
        result = solve_basic_2var()
        assert result.status == SolutionStatus.OPTIMAL

    def test_basic_2var_objective_value(self):
        """
        Maximize 5x1 + 4x2 s.t. 6x1+4x2<=24, x1+2x2<=6
        Known optimum: z = 21
        """
        result = solve_basic_2var()
        assert result.objective_value == pytest.approx(21.0, abs=1e-6)

    def test_basic_2var_x1_value(self):
        result = solve_basic_2var()
        assert result.x(0, 0) == pytest.approx(3.0, abs=1e-6)

    def test_basic_2var_x2_value(self):
        result = solve_basic_2var()
        assert result.x(1, 0) == pytest.approx(1.5, abs=1e-6)

    def test_basic_2var_solution_nonnegative(self):
        result = solve_basic_2var()
        for i in range(result.x.rows):
            assert result.x(i, 0) >= -1e-9

    def test_basic_2var_constraints_satisfied(self):
        result = solve_basic_2var()
        x1, x2 = result.x(0, 0), result.x(1, 0)
        assert 6 * x1 + 4 * x2 <= 24 + 1e-6
        assert x1 + 2 * x2 <= 6 + 1e-6

    # --- Single-variable LP ---

    def test_single_var_status_optimal(self):
        assert solve_single_var().status == SolutionStatus.OPTIMAL

    def test_single_var_objective_value(self):
        """Maximize 3x s.t. x<=5  →  z=15"""
        assert solve_single_var().objective_value == pytest.approx(15.0, abs=1e-6)

    def test_single_var_x_value(self):
        result = solve_single_var()
        assert result.x(0, 0) == pytest.approx(5.0, abs=1e-6)

    # --- Three-variable LP ---

    def test_three_var_status_optimal(self):
        assert solve_three_var().status == SolutionStatus.OPTIMAL

    def test_three_var_objective_value(self):
        """Maximize 2x1+3x2+x3 s.t. x1+x2+x3<=40, 2x1+x2<=60  →  z=120"""
        assert solve_three_var().objective_value == pytest.approx(120.0, abs=1e-6)

    def test_three_var_solution_nonnegative(self):
        result = solve_three_var()
        for i in range(result.x.rows):
            assert result.x(i, 0) >= -1e-9

    def test_three_var_constraints_satisfied(self):
        result = solve_three_var()
        x = [result.x(i, 0) for i in range(3)]
        assert x[0] + x[1] + x[2] <= 40 + 1e-6
        assert 2 * x[0] + x[1] <= 60 + 1e-6

    # --- Zero-objective (trivial) LP ---

    def test_zero_objective_value(self):
        """Maximize 0x s.t. x<=10  →  z=0"""
        A = Matrix([[1.0]])
        b = Matrix([[10.0]])
        c = Matrix([[0.0]])
        result = make_solver().solve(A, b, c)
        assert result.objective_value == pytest.approx(0.0, abs=1e-9)
        assert result.status == SolutionStatus.OPTIMAL

    # --- Tight single constraint ---

    def test_tight_constraint(self):
        """Maximize x1+x2 s.t. x1+x2<=7  →  z=7"""
        A = Matrix([[1.0, 1.0]])
        b = Matrix([[7.0]])
        c = Matrix([[1.0, 1.0]])
        result = make_solver().solve(A, b, c)
        assert result.objective_value == pytest.approx(7.0, abs=1e-6)
        assert result.status == SolutionStatus.OPTIMAL

    # --- Column vector c ---

    def test_c_as_column_vector(self):
        """c supplied as a column vector (n×1) should still work."""
        A = Matrix([[1.0, 0.0], [0.0, 1.0]])
        b = Matrix([[4.0], [6.0]])
        c = Matrix([[2.0], [3.0]])   # column vector
        result = make_solver().solve(A, b, c)
        # Maximize 2x1+3x2 s.t. x1<=4, x2<=6  →  x1=4, x2=6, z=26
        assert result.objective_value == pytest.approx(26.0, abs=1e-6)
        assert result.status == SolutionStatus.OPTIMAL

    # --- Row vector c ---

    def test_c_as_row_vector(self):
        """c supplied as a row vector (1×n) should still work."""
        A = Matrix([[1.0, 0.0], [0.0, 1.0]])
        b = Matrix([[4.0], [6.0]])
        c = Matrix([[2.0, 3.0]])     # row vector
        result = make_solver().solve(A, b, c)
        assert result.objective_value == pytest.approx(26.0, abs=1e-6)
        assert result.status == SolutionStatus.OPTIMAL

    # --- Verify objective equals c^T x ---

    def test_objective_equals_dot_product(self):
        """The returned objective_value must equal c · x for the basic 2-var problem."""
        result = solve_basic_2var()
        c = [5.0, 4.0]
        computed = sum(c[i] * result.x(i, 0) for i in range(2))
        assert result.objective_value == pytest.approx(computed, abs=1e-6)

    def test_objective_equals_dot_product_three_var(self):
        result = solve_three_var()
        c = [2.0, 3.0, 1.0]
        computed = sum(c[i] * result.x(i, 0) for i in range(3))
        assert result.objective_value == pytest.approx(computed, abs=1e-6)

    # --- Multiple constraints, known solution ---

    def test_four_constraints(self):
        """
        Maximize z = x1 + x2
        Subject to:
            x1       <= 10
            x2       <= 10
            x1 + x2  <= 15
            x1 + 2x2 <= 25
        Optimal: x1=5, x2=10  →  z=15
        """
        A = Matrix([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ])
        b = Matrix([[10.0], [10.0], [15.0], [25.0]])
        c = Matrix([[1.0, 1.0]])
        result = make_solver().solve(A, b, c)
        assert result.status == SolutionStatus.OPTIMAL
        assert result.objective_value == pytest.approx(15.0, abs=1e-6)

    # --- Solver can be reused ---

    def test_solver_can_be_reused(self):
        """Calling solve twice on the same instance should give consistent results."""
        solver = SimplexSolver()
        A = Matrix([[1.0]])
        b = Matrix([[3.0]])
        c = Matrix([[1.0]])
        r1 = solver.solve(A, b, c)
        r2 = solver.solve(A, b, c)
        assert r1.objective_value == pytest.approx(r2.objective_value, abs=1e-9)

    def test_different_problems_different_solvers_match(self):
        """Two separate solver instances should agree on the same problem."""
        A = Matrix([[2.0, 1.0], [1.0, 2.0]])
        b = Matrix([[10.0], [10.0]])
        c = Matrix([[3.0, 3.0]])
        r1 = SimplexSolver().solve(A, b, c)
        r2 = SimplexSolver().solve(A, b, c)
        assert r1.objective_value == pytest.approx(r2.objective_value, abs=1e-9)


# ===========================================================================
# 6. SimplexSolver.solve — edge cases
# ===========================================================================

class TestSolveEdgeCases:

    def test_single_constraint_single_var(self):
        """1×1 problem — smallest possible LP."""
        A = Matrix([[1.0]])
        b = Matrix([[100.0]])
        c = Matrix([[7.0]])
        result = make_solver().solve(A, b, c)
        assert result.status == SolutionStatus.OPTIMAL
        assert result.objective_value == pytest.approx(700.0, abs=1e-6)
        assert result.x(0, 0) == pytest.approx(100.0, abs=1e-6)

    def test_large_rhs_values(self):
        """RHS values in the thousands — checks numerical stability."""
        A = Matrix([[1.0, 0.0], [0.0, 1.0]])
        b = Matrix([[1000.0], [2000.0]])
        c = Matrix([[1.0, 1.0]])
        result = make_solver().solve(A, b, c)
        assert result.status == SolutionStatus.OPTIMAL
        assert result.objective_value == pytest.approx(3000.0, abs=1e-4)

    def test_fractional_rhs(self):
        """Fractional RHS — solution should also be fractional."""
        A = Matrix([[2.0]])
        b = Matrix([[3.0]])
        c = Matrix([[1.0]])
        result = make_solver().solve(A, b, c)
        assert result.status == SolutionStatus.OPTIMAL
        assert result.x(0, 0) == pytest.approx(1.5, abs=1e-6)
        assert result.objective_value == pytest.approx(1.5, abs=1e-6)

    def test_solution_x_nonnegative_always(self):
        """x >= 0 is a hard constraint of the LP; every component must be >= 0."""
        result = solve_basic_2var()
        arr = result.x.to_numpy()
        assert np.all(arr >= -1e-9)

    def test_b_zero_rhs(self):
        """If all RHS are 0 the only feasible point is the origin."""
        A = Matrix([[1.0, 0.0], [0.0, 1.0]])
        b = Matrix([[0.0], [0.0]])
        c = Matrix([[1.0, 1.0]])
        result = make_solver().solve(A, b, c)
        assert result.status == SolutionStatus.OPTIMAL
        assert result.objective_value == pytest.approx(0.0, abs=1e-9)

    def test_x_values_accessible_via_numpy(self):
        """x can be converted to a numpy array for downstream processing."""
        result = solve_basic_2var()
        arr = result.x.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 1)


# ===========================================================================
# 7. Integration tests
# ===========================================================================

class TestIntegration:

    def test_repr_round_trip(self):
        """repr() should not raise and should mention key fields."""
        result = solve_basic_2var()
        r = repr(result)
        assert "Objective Value" in r
        assert "Status" in r

    def test_result_fields_consistent(self):
        """objective_value must be consistent with x and c."""
        result = solve_basic_2var()
        c = [5.0, 4.0]
        manual_z = sum(c[i] * result.x(i, 0) for i in range(2))
        assert result.objective_value == pytest.approx(manual_z, abs=1e-6)

    def test_full_workflow_three_steps(self):
        """End-to-end: build inputs → solve → inspect result."""
        A = Matrix([[1.0, 2.0], [3.0, 1.0]])
        b = Matrix([[14.0], [14.0]])
        c = Matrix([[2.0, 3.0]])

        solver = SimplexSolver()
        result = solver.solve(A, b, c)

        assert result.status == SolutionStatus.OPTIMAL
        assert result.objective_value > 0
        for i in range(result.x.rows):
            assert result.x(i, 0) >= -1e-9

    def test_numpy_input_via_matrix_wrapper(self):
        """Inputs constructed from numpy arrays produce the same result."""
        A_np = np.array([[6.0, 4.0], [1.0, 2.0]])
        b_np = np.array([[24.0], [6.0]])
        c_np = np.array([[5.0, 4.0]])

        result = make_solver().solve(Matrix(A_np), Matrix(b_np), Matrix(c_np))
        assert result.status == SolutionStatus.OPTIMAL
        assert result.objective_value == pytest.approx(21.0, abs=1e-6)

    def test_status_is_always_set(self):
        """Every solve call must set a status (never None)."""
        result = solve_single_var()
        assert result.status is not None

    def test_multiple_solves_independent(self):
        """Results from sequential solves must not bleed into each other."""
        r1 = solve_single_var()   # z=15
        r2 = solve_basic_2var()   # z=21
        assert r1.objective_value == pytest.approx(15.0, abs=1e-6)
        assert r2.objective_value == pytest.approx(21.0, abs=1e-6)