// include/daedalus/optimization/SimplexSolver.h
#ifndef SIMPLEX_SOLVER_H
#define SIMPLEX_SOLVER_H

#include "Optimization.h"
#include "../core/Matrix.h"

namespace daedalus {
namespace optimization {

class SimplexSolver {
public:
    SimplexSolver() = default;

    /**
     * @brief Solves LP: Max c*x subject to Ax <= b, x >= 0
     */
    OptimizationResult solve(const Matrix<double>& A, 
                            const Matrix<double>& b, 
                            const Matrix<double>& c);

private:
    // The Simplex Tableau
    Matrix<double> tableau{0, 0};
    size_t num_constraints{0};
    size_t num_vars{0};

    bool find_pivot(size_t& row, size_t& col);
    void pivot(size_t pivot_row, size_t pivot_col);
    void build_tableau(const Matrix<double>& A, const Matrix<double>& b, const Matrix<double>& c);
};

}
}


#endif