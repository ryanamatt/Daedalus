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
#include <stdexcept>
#include <sstream>
#include <functional>

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

    /** @brief Creates a deepcopy of the matrix */
    Matrix copy() const {
        Matrix result(num_rows, num_cols);
        result.data = this->data;
        return result;
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

    /** @brief Scalar subtraction (scalar - Matrix). */
    friend Matrix operator-(const T& scalar, Matrix rhs) {
        rhs -= scalar;
        return rhs;
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

};

#endif // MATRIX_H