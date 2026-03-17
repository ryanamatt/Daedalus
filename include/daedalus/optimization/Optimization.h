// include/daedalus/optimization/Optimization.h
#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include "../core/Matrix.h"
#include <vector>
#include <string>

namespace daedalus {
namespace optimization {

    enum class SolutionStatus {
        OPTIMAL,
        INFEASIBLE,
        UNBOUNDED,
        ERROR
    };

    struct OptimizationResult {
        Matrix<double> x;            // Optimal variable values
        double objective_value;      // Max/Min value of z
        SolutionStatus status;
        std::string message;
    };

} // namespace optimization
} // namespace daedalus

#endif