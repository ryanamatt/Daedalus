/**
 * @file Preprocessing.h
 * @brief Tools for data scaling and feature transformation.
 */

// include/daedalus/core/Preprocessing.h

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "Matrix.h"
#include <cmath>
#include <vector>

/**
 * @class StandardScaler
 * @brief Standardizes features by removing the mean and scaling to unit variance.
 * * The standard score of a sample @f$ x @f$ is calculated as:
 * $$z = \frac{(x - u)}{s}$$
 * where @f$ u @f$ is the mean of the training samples and @f$ s @f$ is the standard deviation.
 */
class StandardScaler {
private:
    std::vector<double> means;
    std::vector<double> std_devs;
    bool is_fitted = false;

public:
    /**
     * @brief Computes the mean and standard deviation for each column in the matrix.
     * * @param X The input Matrix used to compute scaling parameters.
     * @note If a column has zero variance (standard deviation of 0), it is scaled by 1.0 to avoid division by zero.
     */
    void fit(const Matrix<double>& X) {
        size_t rows = X.rows(); //
        size_t cols = X.cols(); //
        means.assign(cols, 0.0);
        std_devs.assign(cols, 0.0);

        // Calculate Mean
        for (size_t c = 0; c < cols; ++c) {
            double sum = 0.0;
            for (size_t r = 0; r < rows; ++r) {
                sum += X(r, c); //
            }
            means[c] = sum / rows;
        }

        // Calculate Standard Deviation
        for (size_t c = 0; c < cols; ++c) {
            double variance_sum = 0.0;
            for (size_t r = 0; r < rows; ++r) {
                double diff = X(r, c) - means[c];
                variance_sum += diff * diff;
            }
            std_devs[c] = std::sqrt(variance_sum / rows);
            if (std_devs[c] == 0) std_devs[c] = 1.0; // Prevent division by zero
        }
        is_fitted = true;
    }

    /**
     * @brief Performs standardization by centering and scaling the input matrix.
     * * Uses the parameters computed during the @c fit() call.
     * @param X The Matrix to be transformed.
     * @return Matrix<double> The transformed (scaled) matrix.
     * @throws std::runtime_error If transform is called before the scaler is fitted.
     */
    Matrix<double> transform(const Matrix<double>& X) const {
        if (!is_fitted) throw std::runtime_error("Scaler must be fitted first.");
        
        Matrix<double> result(X.rows(), X.cols()); //
        for (size_t c = 0; c < X.cols(); ++c) {
            for (size_t r = 0; r < X.rows(); ++r) {
                result(r, c) = (X(r, c) - means[c]) / std_devs[c]; //
            }
        }
        return result;
    }
};

#endif // PREPROCESSING_H