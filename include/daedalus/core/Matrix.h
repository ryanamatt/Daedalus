/**
 * @file Matrix.h
 * @brief Template Matrix class for linear algebra operations.
 * * This file provides a high-performance Matrix container supporting standard 
 * arithmetic, scalar operations, and optimized multiplication algorithms.
 */

// include/daedalus/core/Matrix.h

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <tuple>
#include <stdexcept>
#include <sstream>
#include <functional>
#include <cmath>
#include <algorithm>
#include <iterator>

template <typename T>

/**
 * @class Matrix
 * @brief A generic matrix class stored in row-major order.
 * * @tparam T The numeric type of the matrix elements (e.g., float, double, int).
 */
class Matrix {
    size_t num_rows, num_cols;
    std::vector<T> data;

public:
    /**
     * @brief Constructs a new Matrix with specified dimensions.
     * @param r Number of rows.
     * @param c Number of columns.
     * @throws std::invalid_argument if dimensions are less than zero.
     */
    Matrix(size_t r, size_t c) : num_rows(r), num_cols(c), data(r * c) {}

    /**
     * @brief Constructs a Matrix from a nested vector (list of lists).
     * @param nested_data The input data.
     * @throws std::invalid_argument if the input is empty or rows have inconsistent lengths.
     */
    Matrix(const std::vector<std::vector<T>>& nested_data) {
        num_rows = nested_data.size();
        if (num_rows == 0) {
            num_cols = 0;
            return;
        }
        
        num_cols = nested_data[0].size();
        data.reserve(num_rows * num_cols);

        for (const auto& row : nested_data) {
            if (row.size() != num_cols) {
                throw std::invalid_argument("All rows must have the same number of columns.");
            }
            data.insert(data.end(), row.begin(), row.end());
        }
    }

    /**
     * @brief Constructs a Matrix from a flat vector and dimensions.
     * @param r Number of rows.
     * @param c Number of columns.
     * @param flat_data 1D vector of elements.
     * @throws std::invalid_argument if size doesn't match r * c.
     */
    Matrix(size_t r, size_t c, const std::vector<T>& flat_data) 
        : num_rows(r), num_cols(c), data(flat_data) {
        if (flat_data.size() != r * c) {
            throw std::invalid_argument("Data size does not match matrix dimensions.");
        }
    }

    /** @brief Accesses an element at (r, c) for modification. */
    T& operator()(size_t r, size_t c) { 
        if (r >= num_rows || c >= num_cols) {
            throw std::out_of_range("Matrix index out of bounds.");
        }
        return data[r * num_cols + c]; 
    }

    /** @brief Accesses an element at (r, c) for reading. */
    const T& operator()(size_t r, size_t c) const { 
        if (r >= num_rows || c >= num_cols) {
            throw std::out_of_range("Matrix index out of bounds.");
        }
        return data[r * num_cols + c]; 
    }

    /**
     * @brief Creates a submatrix (slice) from the current matrix.
     * @param start_row The start row of the slice.
     * @param end_row The end row of the slice.
     * @param start_col The start col of the slice
     * @param end_col The end col of the slice
     * @throws std::out_of_range When row or col are out of Matrix index.
     */
    Matrix get_slice(size_t start_row, size_t end_row, size_t start_col, size_t end_col) const {
        if (start_row >= end_row || start_col >= end_col || end_row > num_rows || end_col > num_cols) {
            throw std::out_of_range("Slice indices out of bounds or invalid range.");
        }
        Matrix result(end_row - start_row, end_col - start_col);
        for (size_t i = 0; i < result.rows(); ++i) {
            for (size_t j = 0; j < result.cols(); ++j) {
                result(i, j) = (*this)(start_row + i, start_col + j);
            }
        }
        return result;
    }

    /**
     * @brief Converts the matrix to a human-readable string.
     * @return std::string Formatted representation of the matrix.
     */
    std::string to_string() const {
        std::stringstream ss;
        ss << "Matrix(" << num_rows << "x" << num_cols << ") [\n";
        for (size_t i = 0; i < num_rows; ++i) {
            ss << "  [";
            for (size_t j = 0; j < num_cols; ++j) {
                ss << (*this)(i, j) << (j == num_cols - 1 ? "" : ", ");
            }
            ss << "]" << (i == num_rows - 1 ? "" : ",\n");
        }
        ss << "\n]";
        return ss.str();
    }

    // --- Getters ---

    /** @return Number of rows in the matrix. */
    size_t rows() const { return num_rows; }

    /** @return Number of columns in the matrix. */
    size_t cols() const { return num_cols; }

    /** @return Pointer to the underlying raw data. */
    T* data_ptr() { return data.data(); }
    const T* data_ptr() const { return data.data(); }

    /**
     * @brief Creates a Matrix and fills it with desired value for all elements.
     * @param rows The rows of the Matrix
     * @param cols The cols of the Matrix
     * @param fill_value The value to fill the matrix with.
     * @return static Matrix<T> filled with desired value.
     */
    static Matrix<T> create_filled_matrix(size_t rows, size_t cols, double fill_value) {
        Matrix<T> mat(rows, cols);
        T* ptr = mat.data_ptr();

        for (size_t i = 0; i < rows * cols; ++i) {
            ptr[i] = static_cast<T>(fill_value);
        }
        return mat;
    }

    /**
     * @brief Creates a rectangular diagonal Matrix.
     * @param r Number of rows.
     * @param c Number of columns.
     * @param values Vector of values to place on the diagonal.
     * @return Matrix<T> A rectangular matrix where mat(i, i) = values[i].
     */
    static Matrix<T> create_diagonal_vector(size_t rows, size_t cols, const std::vector<T>& values) {
        Matrix<T> mat(rows, cols);
            T* ptr = mat.data_ptr();
            
            size_t diag_len = std::min({rows, cols, values.size()});
            
            for (size_t i = 0; i < diag_len; ++i) {
                ptr[i * cols + i] = values[i];
            }
            return mat;
        }

    /**
     * @brief Creates a square diagonal Matrix filled with a single scalar value.
     * @param rows The rows of the matrix.
     * @param cols The cols of the matrix.
     * @param value The value to place on the diagonal.
     */
    static Matrix<T> create_diagonal_scaler(size_t rows, size_t cols, double value) {
        Matrix<T> mat(rows, cols);
        T* ptr = mat.data_ptr();

        size_t diag_len = std::min({rows, cols});
        
        for (size_t i = 0; i < diag_len; ++i) {
            ptr[i * cols + i] = value;
        }
        return mat;
    }
    
    /**
     * @brief Extracts a single row from the matrix.
     * @param row_idx The index of the row to extract (0-indexed).
     * @return Matrix<T> A new matrix of shape (1, cols).
     * @throws std::out_of_range If row index > than number of rows.
     */
    Matrix<T> get_row(int idx) const {
        if (idx < 0 || idx >= num_rows) {
            throw std::out_of_range("Row index out of bounds");
        }
        Matrix<T> row_matrix(1, num_cols);
        for (int j = 0; j < num_cols; ++j) {
            row_matrix(0, j) = (*this)(idx, j);
        }

        return row_matrix;
    }

    /**
     * @brief Sets the row at the desired index.
     * @param idx The index to add the row to.
     * @param new_row The new row to insert.
     */
    void set_row(int idx, std::vector<T> new_row) {
        int start_pos = idx * num_cols;
        std::copy(new_row.begin(), new_row.end(), data.begin() + start_pos);
    }

    /**
     * @brief Extracts a single col from the matrix.
     * @param idx The index of the col to extract (0-indexed).
     * @return Matrix<T> A new matrix of shape (rows, 1).
     * @throws std::out_of_range If col index > than number of col.
     */
    Matrix<T> get_col(int idx) const {
        if (idx < 0 || idx >= num_cols) {
            throw std::out_of_range("Col index out of bounds");
        }
        Matrix<T> col_matrix(num_rows, 1);
        for (size_t i = 0; i < num_rows; ++i) {
            col_matrix(i, 0) = (*this)(i, idx);
        }

        return col_matrix;
    }

    /**
     * @brief Sets the col at the desired index.
     * @param idx The index to add the col to.
     * @param new_col The new col to insert.
     */
    void set_col(int idx, std::vector<T> new_col) {
        for (size_t i = 0; i < num_rows; ++i) {
        (*this)(i, idx) = new_col[i]; // Correctly updates element at (row i, col idx)
    }
    }

    /** @brief Creates a deepcopy of the matrix */
    Matrix copy() const {
        Matrix result(num_rows, num_cols);
        result.data = this->data;
        return result;
    }

    /** @brief Multiples an entire row by a scalar constant. */
    void scale_row(size_t row_idx, T scalar) {
        if (row_idx >= num_rows) throw std::out_of_range("Row index out of bounds.");
        for (size_t j = 0; j < num_cols; ++j) {
            (*this)(row_idx, j) *= scalar;
        }
    }

    /** * @brief Adds a multiple of one row to another row.
     * Equation: row_dest = row_dest + (row_src * scalar)
     */
    void add_scaled_row(size_t src_idx, size_t dest_idx, T scalar) {
        if (src_idx >= num_rows || dest_idx >= num_rows) {
            throw std::out_of_range("Row index out of bounds.");
        }
        for (size_t j = 0; j < num_cols; ++j) {
            (*this)(dest_idx, j) += (*this)(src_idx, j) * scalar;
        }
    }

    /** @brief Swaps two rows in the matrix. */
    void swap_rows(size_t r1, size_t r2) {
        if (r1 >= num_rows || r2 >= num_rows) throw std::out_of_range("Row index out of bounds.");
        if (r1 == r2) return;
        for (size_t j = 0; j < num_cols; ++j) {
            std::swap((*this)(r1, j), (*this)(r2, j));
        }
    }

    // --- Standard Operators ---

    // --- Addition ---

    /** @brief In-place scalar addition (Broadcasting). */
     Matrix& operator+=(const T& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += scalar;
        }
        return *this;
    }

    /** @brief Scalar addition (Matrix + scalar). */
    friend Matrix operator+(Matrix lhs, const T& scalar) {
        lhs += scalar;
        return lhs;
    }

    /** @brief Scalar addition (scalar + Matrix). */
    friend Matrix operator+(const T& scalar, Matrix rhs) {
        rhs += scalar;
        return rhs;
    }

    /** @brief In-place matrix addition. 
     * @throws std::invalid_argument on dimension mismatch. 
    */
    Matrix& operator+=(const Matrix& other) {
        if (num_rows != other.num_rows || num_cols != other.num_cols) {
            throw std::invalid_argument("Matrix dimensions must match for addition.");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    /** @brief Matrix addition. */
    friend Matrix operator+(Matrix lhs, const Matrix& rhs) {
        lhs += rhs;
        return lhs;
    }

    // --- Subtraction ---

    /** @brief In-place scalar subtraction (Broadcasting). */
     Matrix& operator-=(const T& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= scalar;
        }
        return *this;
    }

    /** @brief Scalar subtraction (Matrix - scalar). */
    friend Matrix operator-(Matrix lhs, const T& scalar) {
        lhs -= scalar;
        return lhs;
    }

    /** @brief In-place matrix subtraction. 
     * @throws std::invalid_argument on dimension mismatch. 
    */
    Matrix& operator-=(const Matrix& other) {
        if (num_rows != other.num_rows || num_cols != other.num_cols) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction.");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    /** @brief Matrix subtraction. */
    friend Matrix operator-(Matrix lhs, const Matrix& rhs) {
        lhs -= rhs;
        return lhs;
    }

    // --- Multiplication ---

    /** @brief In-place scalar multiplication. */
    Matrix& operator*=(const T& scalar) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] *= scalar;
    }
    return *this;
    }

    /** @brief Scalar multiplication (Matrix * scalar). */
    Matrix operator*(const T& scalar) const {
        Matrix result = *this;
        result *= scalar;
        return result;
    }

    /** @brief Scalar multiplication (scalar * Matrix). */
    friend Matrix operator*(const T& scalar, const Matrix& m) {
        return m * scalar;
    }

    /**
     * @brief Performs matrix multiplication using the most efficient algorithm available.
     * * @section Multiplication Heuristics:
     * - Uses Cache-friendly standard multiplication ($O(n^3)$) otherwise.
     * @param other The right-hand side matrix.
     * @return Matrix Result of (this * other).
     * @throws std::invalid_argument If inner dimensions do not match.
     */
    Matrix operator*(const Matrix& other) const {
        if (num_cols != other.num_rows) {
            throw std::invalid_argument("Cols of Matrix A do not Match Rows of Matrix B");
        }

        Matrix result(num_rows, other.num_cols);
        for (size_t i = 0; i < num_rows; ++i) {
            for (size_t k = 0; k < num_cols; ++k) {
                T temp = (*this)(i, k);
                for (size_t j = 0; j < other.num_cols; ++j) {
                    result(i, j) += temp * other(k, j);
                }
            }
        }
        return result;
    }

    /**
     * @brief Performs matrix multiplication using the most efficient algorithm available.
     * * @section Multiplication Heuristics:
     * - Uses Cache-friendly standard multiplication ($O(n^3)$) otherwise. Uses a tiled approach
     * and is for large matrices N > 1024.
     * @param other The right-hand side matrix.
     * @return Matrix Result of (this * other).
     */
    Matrix multiply_tiled(const Matrix& other) const {
        Matrix result(num_rows, other.num_cols);
        const size_t block_size = 32; // Optimized for L1 cache size

        for (size_t i0 = 0; i0 < num_rows; i0 += block_size) {
            for (size_t k0 = 0; k0 < num_cols; k0 += block_size) {
                for (size_t j0 = 0; j0 < other.num_cols; j0 += block_size) {

                    // Inner mini-matrix multiplication ('Tile')
                    for (size_t i = i0; i < std::min(i0 + block_size, num_rows); ++i) {
                        for (size_t k = k0; k < std::min(k0 + block_size, num_cols); ++k) {
                            T temp = (*this)(i, k);
                            for (size_t j = j0; j < std::min(j0 + block_size, other.num_cols); ++j) {
                                result(i, j) += temp * other(k, j);
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    /** @brief In-place scalar division. */
    Matrix& operator/=(const T& scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] /= scalar;
        }
        return *this;
    }

    /** @brief Scalar multiplication (Matrix / scalar). */
    Matrix operator/(const T& scalar) const {
        Matrix result = *this;
        result /= scalar;
        return result;
    }

    /**
     * @brief Takes each element in the Matrix and raises it to the power of the power_value
     * @param power_value The value of the power to raise each element to
     * @return Matrix Result of same Matrix but taken to desired power.
     */
    Matrix<T> power_to(const T& power_value) {
        Matrix<T> result(num_rows, num_cols);
        for (size_t i = 0; i < data.size(); ++i) {
            double current_val = static_cast<double>(data.at(i));
            result.data[i] = static_cast<T>(std::pow(current_val, static_cast<double>(power_value)));
        }
        return result;
    }

    Matrix<T> operator>(const T& threshold) const {
        Matrix<T> res(num_rows, num_cols);
        const T* src = this->data_ptr();
        T* dst = res.data_ptr();
        size_t total = num_rows * num_cols;

        for (size_t i = 0; i < total; ++i) {
            dst[i] = (src[i] > threshold) ? static_cast<T>(1) : static_cast<T>(0);
        }
        return res;
    }

    Matrix<T> operator<(const T& threshold) const {
        Matrix<T> res(num_rows, num_cols);
        const T* src = this->data_ptr();
        T* dst = res.data_ptr();
        size_t total = num_rows * num_cols;

        for (size_t i = 0; i < total; ++i) {
            dst[i] = (src[i] < threshold) ? static_cast<T>(1) : static_cast<T>(0);
        }
        return res;
    }

    Matrix<T> operator>=(const T& threshold) const {
        Matrix<T> res(num_rows, num_cols);
        const T* src = this->data_ptr();
        T* dst = res.data_ptr();
        size_t total = num_rows * num_cols;

        for (size_t i = 0; i < total; ++i) {
            dst[i] = (src[i] >= threshold) ? static_cast<T>(1) : static_cast<T>(0);
        }
        return res;
    }

    Matrix<T> operator<=(const T& threshold) const {
        Matrix<T> res(num_rows, num_cols);
        const T* src = this->data_ptr();
        T* dst = res.data_ptr();
        size_t total = num_rows * num_cols;

        for (size_t i = 0; i < total; ++i) {
            dst[i] = (src[i] <= threshold) ? static_cast<T>(1) : static_cast<T>(0);
        }
        return res;
    }

    /**
     * @brief Checks if two matrices are equal.
     * @return true if dimensions and all elements match, false otherwise.
     */
    bool operator==(const Matrix<T>& other) const {
        if (data.size() != other.data.size()) return false;

        return data == other.data;
    }

    /** @brief Checks if two matrices are not equal. */
    bool operator!=(const Matrix<T>& other) const {
        return !(*this == other);
    }

    Matrix round(int places) {
        Matrix result(num_rows, num_cols);
        std::transform(data.begin(), data.end(), result.data.begin(),
                [places](T i){ 
                    double multiplier = std::pow(10.0, places);
                    return std::round(i * multiplier) / multiplier;
                 });
        return result;
    }

    /**
     * @brief Takes the absolute value of all elements in the Matrix
     * @return A Matrix that is the abs value of all elements.
     */
    Matrix abs() {
        Matrix<double> result(num_rows, num_cols);

        std::transform(data.begin(), data.end(), 
            result.data.begin(), [](T i) { return std::abs(i); });
        return result;
    }

    /**
     * @brief Returns whether or not said item exists in the Matrix.
     * @returns True if value exists in Matrix, otherwise False.
     */
    bool contains(const T value) {
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] == value) return true;
        }
        return false;
    }

    /**
     * @brief Takes the noramlization of the Matrix.
     * @param type The string of the type of normalization.
     * @return The norm of the Matrix
     */
    T norm(const std::string& type = "fro") const {
        if (type == "fro") { // Frobenius Norm: Square root of sum of squares
            T sum = 0;
            for (const auto& val : data) {
                sum += val * val;
            }
            return std::sqrt(sum);
        }

        else if (type == "1") { // 1-Norm: Maximum absolute column sum
            T max_col_sum = 0;
            for (size_t j = 0; j < num_cols; ++j) {
                T current_col_sum = 0;
                for (size_t i = 0; i < num_rows; ++i) {
                    current_col_sum += std::abs((*this)(i, j));
                }
                max_col_sum = std::max(max_col_sum, current_col_sum);
            } 
            return max_col_sum;
        }

        else if (type == "inf") { // Infinity Norm: Maximum absolute row sum
            T max_row_sum = 0;
            for (size_t i = 0; i < num_rows; ++i) {
                T current_row_sum = 0;
                for (size_t j = 0; j < num_cols; ++j) {
                    current_row_sum += std::abs((*this)(i, j));
                }
                max_row_sum = std::max(max_row_sum, current_row_sum);
            }
            return max_row_sum;
        }
    }

    /**
     * @brief Computes the sum of elements along a specified axis.
     * @param axis 0 to sum down columns (result is 1 x cols), 
     * 1 to sum across rows (result is rows x 1).
     * @return Matrix<double> containing the sums.
     * @throws std::invalid_argument if axis is not 0 or 1.
     */
    Matrix<double> sum(int axis) const {
        if (axis == 0) {
            // Collapses rows: Result is a 1 x num_cols matrix

            Matrix<double> result(1, num_cols);

            for (size_t j = 0; j < num_cols; ++j) {
                double col_sum = 0;
                for (size_t i = 0; i < num_rows; ++i) {
                    col_sum += static_cast<double>((*this)(i, j));
                }
                result(0, j) = col_sum;
            }
            return result;
        }
        else if (axis == 1) {
        // Collapses columns: Result is a num_rows x 1 matrix
        Matrix<double> result(num_rows, 1);
        for (size_t i = 0; i < num_rows; ++i) {
            double row_sum = 0;
            for (size_t j = 0; j < num_cols; ++j) {
                row_sum += static_cast<double>((*this)(i, j));
            }
            result(i, 0) = row_sum;
        }
        return result;
        }  else {
            throw std::invalid_argument("Axis must be 0 (columns) or 1 (rows).");
        }
    }

    double sum_all_elements() const {
        double sum = 0.0;
        for (size_t i = 0; i < data.size(); ++i) {
            sum += static_cast<double>(data[i]);
        }
        return sum;
    }

    /**
     * @brief Takes the means along the desired axis.
     * @param axis The axis to take the mean along.
     * @return A Matrix of the resulting mean along the desired axis.
     */
    Matrix mean(int axis) {
        Matrix<double> result = sum(axis);

        if (axis == 0) {
            for (size_t j = 0; j < num_cols; ++j) {
                result(0, j) = result(0, j) / num_cols;
            }
        }

        else if (axis == 1) {
            for (size_t i = 0; i < num_rows; ++i) {
                result(i, 0) = result(i, 0) / num_rows;
            }
        }

        return result;
    }

    /**
     * @brief Calculates the variance along an axis.
     * @param axis The desired axis.
     * @return The variance Matrix along desired axis.
     */
    Matrix variance(int axis) {
        Matrix<double> means = mean(axis);
        Matrix<double> result(means.rows(), means.cols());

        if (axis == 0) {
            double sum_deviation = 0.0;
            for (size_t j = 0; j < num_cols; ++j) {
                double variance_sum = 0.0;
                for (size_t i = 0; i < num_rows; ++i) {
                    double diff = static_cast<double>((*this)(i, j)) - means(0, j);
                    variance_sum += diff * diff;
                }
                result(0, j) = (variance_sum / num_rows);
            }
        }

        else if (axis == 1) {
            for (size_t i = 0; i < num_rows; ++i) {
                double variance_sum = 0.0;
                for (size_t j = 0; j < num_cols; ++j) {
                    double diff = static_cast<double>((*this)(i, j)) - means(i, 0);
                    variance_sum += diff * diff;
                }
                result(i, 0) = (variance_sum / num_cols);
            }
        }

        return result;
    }

    /**
     * @brief Calculates the standard deviation along an axis.
     * @param axis The desired axis.
     * @return The standard deviation Matrix along desired axis.
     */
    Matrix standard_deviation(int axis) {
        Matrix<double> means = mean(axis);
        Matrix<double> result(means.rows(), means.cols());

        if (axis == 0) {
            double sum_deviation = 0.0;
            for (size_t j = 0; j < num_cols; ++j) {
                double variance_sum = 0.0;
                for (size_t i = 0; i < num_rows; ++i) {
                    double diff = static_cast<double>((*this)(i, j)) - means(0, j);
                    variance_sum += diff * diff;
                }
                result(0, j) = std::sqrt(variance_sum / num_rows);
            }
        }

        else if (axis == 1) {
            for (size_t i = 0; i < num_rows; ++i) {
                double variance_sum = 0.0;
                for (size_t j = 0; j < num_cols; ++j) {
                    double diff = static_cast<double>((*this)(i, j)) - means(i, 0);
                    variance_sum += diff * diff;
                }
                result(i, 0) = std::sqrt(variance_sum / num_cols);
            }
        }

        return result;
    }

    /**
     * @brief Reshapes a Matrix to the new dimensions. 
     * Assumes new_rows * new_cols == num_rows * num_cols
     * @param new_rows Int value for new_rows.
     * @param new_cols Int value for new_cols
     * @return A Matrix with the new shape.
     */
    Matrix reshape(int new_rows, int new_cols) {
        Matrix<T> result(new_rows, new_cols, this->data);
        return result;
    }

    /**
     * @brief Finds the index of the maximum value.
     * @return The index (row, col) of the maximum value.
     */
    std::tuple<int, int> argmax_global() {
        auto max_it = std::max_element(data.begin(), data.end());
        int max_index_1d = std::distance(data.begin(), max_it);
        // Convert to 2D coordinates
        int row = max_index_1d / num_cols;
        int col = max_index_1d % num_cols;

        return {static_cast<int>(row), static_cast<int>(col)};
    }

    /**
    * @brief Finds the index of the minimum value.
    * @return The index (row, col) of the minimum value.
    */
    std::tuple<int, int> argmin_global() {
        auto min_it = std::min_element(data.begin(), data.end());
        int min_index_1d = std::distance(data.begin(), min_it);
        // Convert to 2D coordinates
        int row = min_index_1d / num_cols;
        int col = min_index_1d % num_cols;

        return {static_cast<int>(row), static_cast<int>(col)};
    }

    /**
     * @brief Computes the transpose of the matrix.
     * * Uses a blocked (tiled) approach to optimize L1/L2 cache hits, 
     * significantly improving performance for large matrices.
     * @return Matrix The transposed matrix.
     */
    Matrix transpose() const {
        Matrix result(num_cols, num_rows);
        const size_t block_size = 32; // Optimized for L1 cache size

        for (size_t i = 0; i < num_rows; i += block_size) {
            for (size_t j = 0; j < num_cols; j += block_size) {
                for (size_t ii = i; ii < std::min(i + block_size, num_rows); ++ii) {
                    for (size_t jj = j; jj < std::min(j + block_size, num_cols); ++jj) {
                        result(jj, ii) = (*this)(ii, jj);
                    }
                }
            }
        }
        return result;
    }

    /**
     * @brief Computes the determinant of the matrix using LU Decomposition. Assumes Matrix is Square.
     * @return double The determinant value.
     */
    double det() const {
        if (num_rows == 0) return 1.0;
        if (num_rows == 1) return static_cast<double>(data[0]);

        Matrix<T> temp = this->copy();
        double determinant = 1.0;
        size_t n = num_rows;

        for (size_t i = 0; i < n; ++ i) {
            size_t pivot = i;
            for (size_t j = i + 1; j < n; ++j) {
                if (std::abs(temp(i, j)) > std::abs(temp(pivot, i))) {
                    pivot = j;
                }
            }

            if (std::abs(temp(pivot, i)) < 1e-12) { return 0.0; } // Singular Matrix

            if (pivot != i) {
                temp.swap_rows(i, pivot);
                determinant *= -1.0;
            }

            determinant *= temp(i, i);

            for (size_t j = i + 1; j < n; ++j) {
                T factor = temp(j, i) / temp(i, i);
                for (size_t k = i + 1; k < n; ++k) {
                    temp(j, k) -= factor * temp(i, k);
                }
            }
        }
        return determinant;
    }

    /**
     * @brief Computes the inverse of the matrix using Gauss-Jordan elimination.
     * @return Matrix<T> The inverse matrix.
     */
    Matrix<T> inverse() const {
        double determinant = this->det();
        if (std::abs(determinant) < 1e-12) {
            throw std::logic_error("Matrix is singular and cannot be inverted.");
        }

        size_t n = num_rows;
        Matrix<T> aug = this->copy();
        Matrix<T> inv = Matrix<T>::create_diagonal_scaler(n, n, 1.0); // Start with Identity

        for (size_t i = 0; i < n; ++i) {
            // Pivot selection
            size_t pivot = i;
            for (size_t j = i + 1; j < n; ++j) {
                if (std::abs(aug(j, i)) > std::abs(aug(pivot, i))) pivot = j;
            }

            aug.swap_rows(i, pivot);
            inv.swap_rows(i, pivot);

            // Scale pivot row to 1
            T div = aug(i, i);
            aug.scale_row(i, 1.0 / div);
            inv.scale_row(i, 1.0 / div);

            // Eliminate other entries in column
            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    T factor = aug(j, i);
                    aug.add_scaled_row(i, j, -factor);
                    inv.add_scaled_row(i, j, -factor);
                }
            }
        }
        return inv;
    }

    /**
     * @brief Returs the trace of the Matrix. Sum of Diagonals.
     * @return the Trace of the Matrix
     */
    double trace() const {
        size_t diag_len = std::min({num_rows, num_cols});
        double trace_val = 0.0;
        for (size_t i = 0; i < diag_len; ++i) {
            trace_val += data[i * (diag_len + 1)];
        }
        return trace_val;
    }

    /**
     * @brief Finds the rank of the Matrix
     * @returns Returns the Rank of the Matrix.
     */
    int rank() {
        auto [U, sigma, V] = svd();
        int count = 0;
        int len = std::min(sigma.rows(), sigma.cols());
        for (size_t i = 0; i < len; ++i) {
            if (sigma(i, i) != 0.0) count++;
        }
        return count;
    }

    // --- Decomposition ---
    
    /**
     * @brief Computes SVD using a basic Jacobi rotation algorithm.
     * All singular values and corresponding vector components below the 
     * specified tolerance are strictly set to 0.
     * @return std::tuple containing 3 Matrices U, Sigma, and V_transpose.
     */
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> svd(double tol = 1e-12, int max_sweeps = 100) const {  
        size_t m = num_rows;
        size_t n = num_cols;
        
        Matrix<T> U_mat = this->copy(); 
        Matrix<T> V_mat = Matrix<T>::create_diagonal_scaler(n, n, 1.0);
        std::vector<T> sigma_vec(n);

        // --- Phase 1: Jacobi Rotations ---
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            double max_off_diag = 0.0;
            for (size_t j = 0; j < n - 1; ++j) {
                for (size_t k = j + 1; k < n; ++k) {
                    double a_jj = 0.0, a_kk = 0.0, a_jk = 0.0;
                    
                    for (size_t i = 0; i < m; ++i) {
                        a_jj += U_mat(i, j) * U_mat(i, j);
                        a_kk += U_mat(i, k) * U_mat(i, k);
                        a_jk += U_mat(i, j) * U_mat(i, k);
                    }

                    max_off_diag = std::max(max_off_diag, std::abs(a_jk));

                    if (std::abs(a_jk) > tol) {
                        double tau = (a_kk - a_jj) / (2.0 * a_jk);
                        double t = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                        double c = 1.0 / std::sqrt(1.0 + t * t);
                        double s = t * c;

                        for (size_t i = 0; i < m; ++i) {
                            T tmp_j = U_mat(i, j);
                            U_mat(i, j) = c * tmp_j - s * U_mat(i, k);
                            U_mat(i, k) = s * tmp_j + c * U_mat(i, k);
                        }
                        for (size_t i = 0; i < n; ++i) {
                            T tmp_vj = V_mat(i, j);
                            V_mat(i, j) = c * tmp_vj - s * V_mat(i, k);
                            V_mat(i, k) = s * tmp_vj + c * V_mat(i, k);
                        }
                    }
                }
            }
            if (max_off_diag < tol) break;
        }

        // --- Phase 2: Normalization & Below-Tolerance Zeroing ---
        for (size_t j = 0; j < n; ++j) {
            double norm_sq = 0.0;
            for (size_t i = 0; i < m; ++i) norm_sq += U_mat(i, j) * U_mat(i, j);
            double norm = std::sqrt(norm_sq);
            
            // If the norm is below tolerance, the singular value is 0
            if (norm <= tol) {
                sigma_vec[j] = static_cast<T>(0);
                for (size_t i = 0; i < m; ++i) U_mat(i, j) = static_cast<T>(0);
            } else {
                sigma_vec[j] = static_cast<T>(norm);
                for (size_t i = 0; i < m; ++i) U_mat(i, j) /= norm;
            }
        }

        // --- Phase 3: Sorting ---
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;

        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return sigma_vec[a] > sigma_vec[b];
        });

        size_t k = std::min(m, n);
        Matrix<T> sorted_U(m, k);
        Matrix<T> sorted_sigma(k, n);
        Matrix<T> sorted_V(n, n);

        for (size_t j = 0; j < k; ++j) {
            size_t old_idx = indices[j];
                        
            for (size_t i = 0; i < m; ++i) {
                sorted_U(i, j) = U_mat(i, old_idx);
            }

            if (j < m) {
                sorted_sigma(j, j) = (sigma_vec[old_idx] <= tol) ? static_cast<T>(0) : sigma_vec[old_idx];
            }

            for (size_t i = 0; i < n; ++i) {
                sorted_V(i, j) = V_mat(i, old_idx);
            }
        }

        return {sorted_U, sorted_sigma, sorted_V};
    }

};

#endif // MATRIX_H
