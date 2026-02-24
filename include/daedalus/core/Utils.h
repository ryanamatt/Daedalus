/**
 * @file Utils.h
 * @brief General utility functions for data manipulation and dataset preparation.
 */

// include/daedalus/core/Utils.h

#ifndef UTILS_H
#define UTILS_H

#include "Matrix.h"
#include <algorithm>
#include <random>
#include <tuple>
#include <vector>
#include <numeric>

/**
 * @brief Splits matrices into random train and test subsets.
 * * This function shuffles the dataset indices and partitions the features (X) 
 * and targets (y) into two sets based on the @p test_size ratio.
 * @tparam T The numeric type stored in the matrices.
 * @param X Feature matrix.
 * @param y Target matrix.
 * @param test_size The proportion of the dataset to include in the test split (0.0 to 1.0).
 * @param seed The seed for the random number generator to ensure reproducibility.
 * @return std::tuple<Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>> 
 * A tuple containing {X_train, X_test, y_train, y_test}.
 */
template <typename T>
std::tuple<Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>> train_test_split(
    const Matrix<T>& X, const Matrix<T>& y, double test_size = 0.2, int seed = 42) {
    
    size_t total_rows = X.rows();
    size_t x_cols = X.cols();
    size_t y_cols = y.cols();
    size_t test_rows = static_cast<size_t>(total_rows * test_size);
    size_t train_rows = total_rows - test_rows;

    std::vector<size_t> indices(total_rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

    Matrix<T> X_train(train_rows, x_cols);
    Matrix<T> X_test(test_rows, x_cols);
    Matrix<T> y_train(train_rows, y_cols);
    Matrix<T> y_test(test_rows, y_cols);

    for (size_t i = 0; i < train_rows; ++i) {
        for (size_t j = 0; j < x_cols; ++j) X_train(i, j) = X(indices[i], j);
        for (size_t j = 0; j < y_cols; ++j) y_train(i, j) = y(indices[i], j);
    }

    for (size_t i = 0; i < test_rows; ++i) {
        for (size_t j = 0; j < x_cols; ++j) X_test(i, j) = X(indices[train_rows + i], j);
        for (size_t j = 0; j < y_cols; ++j) y_test(i, j) = y(indices[train_rows + i], j);
    }

    return {X_train, X_test, y_train, y_test};
}

#endif // UTILS_H