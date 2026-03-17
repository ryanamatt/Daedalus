// src/optimization/SimplexSolver.cc
#include "daedalus/optimization/SimplexSolver.h"
#include <cmath>
#include <limits>

namespace daedalus {
namespace optimization {

void SimplexSolver::build_tableau(const Matrix<double>& A, const Matrix<double>& b, const Matrix<double>& c) {
    num_constraints = A.rows();
    num_vars = c.rows(); // c is usually a column vector (n x 1)

    // Tableau size: (constraints + 1) rows x (vars + slack + 1) columns
    tableau = Matrix<double>(num_constraints + 1, num_vars + num_constraints + 1);

    // 1. Fill A (constraints)
    for (size_t i = 0; i < num_constraints; ++i) {
        for (size_t j = 0; j < num_vars; ++j) {
            tableau(i, j) = A(i, j);
        }
        // 2. Add Slack variables (identity matrix)
        tableau(i, num_vars + i) = 1.0;
        // 3. Add b (right hand side)
        tableau(i, tableau.cols() - 1) = b(i, 0);
    }

    // 4. Fill Objective Function Row (Bottom row)
    // We use -c because we move variables to the left side: z - cx = 0
    for (size_t j = 0; j < num_vars; ++j) {
        tableau(num_constraints, j) = -c(j, 0);
    }
}

bool SimplexSolver::find_pivot(size_t& row, size_t& col) {
    // Column: Most negative value in the objective row (bottom)
    double min_val = 0;
    bool found_col = false;
    for (size_t j = 0; j < tableau.cols() - 1; ++j) {
        if (tableau(num_constraints, j) < min_val) {
            min_val = tableau(num_constraints, j);
            col = j;
            found_col = true;
        }
    }

    if (!found_col) return false; // Optimal!

    // Row: Minimum ratio test (b_i / A_ij) for positive A_ij
    double min_ratio = std::numeric_limits<double>::max();
    bool found_row = false;
    for (size_t i = 0; i < num_constraints; ++i) {
        if (tableau(i, col) > 0) {
            double ratio = tableau(i, tableau.cols() - 1) / tableau(i, col);
            if (ratio < min_ratio) {
                min_ratio = ratio;
                row = i;
                found_row = true;
            }
        }
    }
    return found_row;
}

void SimplexSolver::pivot(size_t pivot_row, size_t pivot_col) {
    // Scale pivot row so pivot element is 1.0
    double divisor = tableau(pivot_row, pivot_col);
    tableau.scale_row(pivot_row, 1.0 / divisor);

    // Eliminate other entries in pivot column
    for (size_t i = 0; i < tableau.rows(); ++i) {
        if (i != pivot_row) {
            double factor = -tableau(i, pivot_col);
            tableau.add_scaled_row(pivot_row, i, factor);
        }
    }
}

OptimizationResult SimplexSolver::solve(const Matrix<double>& A, const Matrix<double>& b, const Matrix<double>& c) {
    build_tableau(A, b, c);

    size_t p_row, p_col;
    while (find_pivot(p_row, p_col)) {
        pivot(p_row, p_col);
    }

    // Extract Results
    OptimizationResult result{
        Matrix<double>(num_vars, 1), 
        tableau(num_constraints, tableau.cols() - 1),
        SolutionStatus::OPTIMAL,
        ""
    };

    // Basic variables extraction (check for columns that are unit vectors)
    for (size_t j = 0; j < num_vars; ++j) {
        int row_idx = -1;
        bool is_unit = true;
        int ones_count = 0;

        for (size_t i = 0; i < tableau.rows(); ++i) {
            if (std::abs(tableau(i, j) - 1.0) < 1e-9) {
                row_idx = i;
                ones_count++;
            } else if (std::abs(tableau(i, j)) > 1e-9) {
                is_unit = false;
                break;
            }
        }
        result.x(j, 0) = (is_unit && ones_count == 1 && row_idx < (int)num_constraints) 
                         ? tableau(row_idx, tableau.cols() - 1) : 0.0;
    }

    return result;
}

} // namespace optimization
} // namespace daedalus