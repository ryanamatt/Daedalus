from .optimization import OptimizationResult, SolutionStatus
from ..daedalus_cpp import SimplexSolver as _SimplexSolverCpp
from .._core import Matrix

class SimplexSolver:
    """
    SimplexSolver Optimization solver.
    """
    def __init__(self) -> None:
        """
        Initializes the SimplexSolver class.
        """
        self._obj = _SimplexSolverCpp()

    def solve(self, A: Matrix, b: Matrix, c: Matrix) -> OptimizationResult:
        """
        Solves LP: Max c*x subject to Ax <= b, x >= 0
        """
        # Call C++ solver
        cpp_res = self._obj.solve(A._obj, b._obj, c._obj)
        
        cpp_x = cpp_res.x
        # Wrap C++ Matrix in Python Matrix wrapper
        x_matrix = Matrix(cpp_x.rows, cpp_x.cols)
        x_matrix._obj = cpp_x
        
        # Return the custom Python result object
        return OptimizationResult(
            x=x_matrix,
            objective_value=cpp_res.objective_value,
            status=SolutionStatus(cpp_res.status)
        )
    

