from daedalus import Matrix
from daedalus.optimization import SimplexSolver

# Define the constraints Matrix A (Coefficients of inequalities)
# Row 1: 5x1 + 2x2 <= 20
# Row 2: 2x1 + 4x2 <= 12
A = Matrix([
    [5.0, 2.0],
    [2.0, 4.0]
])

#  Define the resource limits Matrix b
b = Matrix([
    [20.0],
    [12.0]
])

# Define the objective coefficients Matrix c (Profit)
# We want to maximize 50x1 + 30x2
c = Matrix([
    [50.0, 30.0]
])

# Initialize the solver and solve
solver = SimplexSolver()
result = solver.solve(A, b, c)

# Interpret the results using your new OptimizationResult class
if result.status.name == "OPTIMAL":
    print(f"Success! Status: {result.status.name}")
    print(f"Maximum Profit: ${result.objective_value}")
    print("Optimal Production Plan:")
    print("Result.x.shape:", result.x.shape)
    print(f" - Tables: {result.x(0, 0)}")
    print(f" - Chairs: {result.x(1, 0)}")
else:
    print(f"Solver failed with status: {result.status.name}")
