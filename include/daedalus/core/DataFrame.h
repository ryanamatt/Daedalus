/**
 * @file DataFrame.h
 * @brief Definition of the DataFrame class for tabular data manipulation.
 */

 // include/daedalus/core/DataFrame.h

#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <vector>
#include <string>
#include <map>
#include <variant>
#include <unordered_map>
#include "Matrix.h"

/**
 * @class DataFrame
 * @brief A container for storing and manipulating heterogenous tabular data.
 * * The DataFrame stores data in a column-major format using a map of column names 
 * to vectors of variants. Supported types include double, int, and std::string.
 */
class DataFrame {
    /** @brief Type alias for the data held in a single column. */
    using ColumnData = std::vector<std::variant<double, int, std::string>>;

    std::vector<std::string> column_names;
    std::unordered_map<std::string, ColumnData> data;
    size_t num_rows = 0;

public:
    /** @brief Default constructor creating an empty DataFrame. */
    DataFrame() = default;

    /**
     * @brief Constructor creating a DataFrame with an initial column.
     * @param col_name The name of the first column.
     * @param col_data The vector of variant data for the column.
     */
    DataFrame(const std::string& col_name, const ColumnData& col_data) {
        column_names.push_back(col_name);
        data[col_name] = col_data;
        num_rows = col_data.size();
    }

    /** @brief Returns the number of rows. */
    size_t rows() const { return num_rows; }

    /** @brief Returns the number of columns. */
    size_t cols() const { return column_names.size(); }

    /** @brief Returns a reference to the list of column names. */
    const std::vector<std::string>& get_column_names() const { return column_names; }

    /**
     * @brief Returns the value at a specific row and column.
     * @param row The row index.
     * @param col_name The name of the column.
     * @return The variant value at that coordinate.
     * @throws std::out_of_range If Row index is out of range.
     * @throws std::invalid_argument IF col_name is not found.
     */
    std::variant<double, int, std::string> at(size_t row, const std::string& col_name) const {
        if (row >= num_rows) {
            throw std::out_of_range("Row index out of bounds.");
        }
        auto it = data.find(col_name);
        if (it == data.end()) {
            throw std::invalid_argument("Column not found: " + col_name);
        }
        return it->second[row]; // it->second is the ColumnData vector
    }

        /**
     * @brief Returns the value at a specific row and column.
     * @param row The row index.
     * @param col The col index.
     * @return The variant value at that coordinate.
     * @throws std::out_of_range If Row or Col index is out of range.
     */
    std::variant<double, int, std::string> at(size_t row, size_t col) const {
        if (row >= num_rows) {
            throw std::out_of_range("Row index out of bounds.");
        }
        if (col >= column_names.size()) {
            throw std::out_of_range("Column index out of bounds.");
        }
        const std::string& col_name = column_names[col];
        return data.at(col_name)[row];
    }

    /**
     * @brief Returns a string representation of the DataFrame.
     * @return A formatted string displaying dimensions, headers, and the first 10 rows.
     */
    std::string to_string() const {
        if (column_names.empty()) return "Empty DataFrame";
        
        std::stringstream ss;
        ss << "DataFrame (" << num_rows << " rows x " << column_names.size() << " cols)\n";
        
        // Header
        for (size_t i = 0; i < column_names.size(); ++i) {
            ss << column_names[i] << (i == column_names.size() - 1 ? "" : "\t");
        }
        ss << "\n" << std::string(column_names.size() * 8, '-') << "\n";

        // Data (limiting to first 10 rows for brevity)
        size_t display_rows = std::min(num_rows, (size_t)10);
        for (size_t r = 0; r < display_rows; ++r) {
            for (size_t c = 0; c < column_names.size(); ++c) {
                const auto& val = data.at(column_names[c])[r];
                std::visit([&ss](auto&& arg) { ss << arg << "\t"; }, val);
            }
            ss << "\n";
        }
        if (num_rows > 10) ss << "...\n";
        return ss.str();
    }

    /**
     * @brief Creates a new DataFrame containing the first n rows.
     * @param n The number of rows to retrieve (default is 5).
     * @return A new DataFrame instance containing the subset of data.
     */
    DataFrame head(size_t n = 5) const {
        DataFrame result;
        size_t rows_to_copy = std::min(n, num_rows); // Ensure we don't exceed actual row count

        for (const auto& name : column_names) {
            const auto& full_col = data.at(name);
            // Create a sub-vector containing only the first 'n' elements
            ColumnData head_col(full_col.begin(), full_col.begin() + rows_to_copy);
            result.add_column(name, head_col);
        }

        return result;
    }

    /**
     * @brief Adds a new column to the DataFrame.
     * @param name The unique name of the column.
     * @param col_data A vector of variants containing the column data.
     * @throws std::invalid_argument If the length of col_data does not match existing rows.
     */
    void add_column(const std::string& name, const ColumnData& col_data) {
        if (num_rows != 0 && col_data.size() != num_rows) {
            throw std::invalid_argument("Column length mismatch.");
        }
        if (num_rows == 0) num_rows = col_data.size();
        
        column_names.push_back(name);
        data[name] = col_data;
    }

    /**
     * @brief Removes a column from the DataFrame.
     * @param name The name of the column to be removed.
     * @throws std::invalid_argument If the column name does not exist.
     */
    void drop_column(const std::string& name) {
        auto it = data.find(name);
        if (it == data.end()) {
            throw std::invalid_argument("Column not found: " + name);
        }

        data.erase(it);

        column_names.erase(
            std::remove(column_names.begin(), column_names.end(), name), 
            column_names.end()
        );

        // If no columns remain, reset num_rows
        if (column_names.empty()) {
            num_rows = 0;
        }
    }

    /**
     * @brief Creates a new DataFrame containing only rows that meet a condition.
     * @param col_name The column to evaluate.
     * @param predicate A lambda function that returns true for rows to keep.
     * @throws std::invalid_argument When a column is not found.
     */
    DataFrame filter(const std::string& col_name, 
                    std::function<bool(const std::variant<double, int, std::string>&)> predicate) const {
        if (data.find(col_name) == data.end()) {
            throw std::invalid_argument("Column not found: " + col_name);
        }

        std::vector<size_t> keep_indices;
        const auto& target_col = data.at(col_name);
        for (size_t i = 0; i < num_rows; ++i) {
            if (predicate(target_col[i])) {
                keep_indices.push_back(i);
            }
        }

        DataFrame filtered_df;
        for (const auto& name : column_names) {
            ColumnData new_col;
            new_col.reserve(keep_indices.size());
            const auto& original_col = data.at(name);

            for (size_t idx : keep_indices) {
                new_col.push_back(original_col[idx]);
            }
            filtered_df.add_column(name, new_col);
        }

        return filtered_df;
    }

    /**
     * @brief Performs binary encoding (0.0/1.0) on a categorical string column.
     * * If val0 and val1 are not provided, the method attempts to auto-detect the 
     * two unique string categories in the column.
     * @param column_name Name of the column to encode.
     * @param val0 The string value to be mapped to 0.0.
     * @param val1 The string value to be mapped to 1.0.
     * @throws std::invalid_argument If the column does not exist.
     * @throws std::runtime_error If the column does not contain exactly two unique categories.
     */
    void encode_binary(const std::string& column_name, std::string val0 = "", std::string val1 = "") {
        if (data.find(column_name) == data.end()) {
            throw std::invalid_argument("Column not found: " + column_name);
        }

        auto& col = data[column_name];

        // Auto-detect values if not provided
        if (val0.empty() || val1.empty()) {
            std::set<std::string> unique_values;
            for (const auto& v : col) {
                if (std::holds_alternative<std::string>(v)) {
                    unique_values.insert(std::get<std::string>(v));
                }
            }
            if (unique_values.size() != 2) {
                throw std::runtime_error("encode_binary requires exactly 2 unique categories in the column.");
            }
            auto it = unique_values.begin();
            val0 = *it;
            val1 = *(++it);
        }

        // Perform the transformation
        for (size_t i = 0; i < num_rows; ++i) {
            if (std::holds_alternative<std::string>(col[i])) {
                std::string current_val = std::get<std::string>(col[i]);
                if (current_val == val0) col[i] = 0.0;
                else if (current_val == val1) col[i] = 1.0;
                else throw std::runtime_error("Unexpected value in binary encoding: " + current_val);
            }
        }
    }

    /**
     * @brief Extracts specified numeric columns into a Matrix object.
     * * Non-numeric types (strings) are converted to 0.0. Integers are cast to doubles.
     * @param target_columns Vector of column names to include in the Matrix.
     * @return A Matrix<double> of size [num_rows x target_columns.size()].
     */
    Matrix<double> to_matrix(const std::vector<std::string>& target_columns) const {
        Matrix<double> result(num_rows, target_columns.size());
        
        for (size_t c = 0; c < target_columns.size(); ++c) {
            const auto& col = data.at(target_columns[c]);
            for (size_t r = 0; r < num_rows; ++r) {
                // Visit the variant to extract a double, even if it's stored as int
                result(r, c) = std::visit([](auto&& arg) -> double {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_arithmetic_v<T>) return static_cast<double>(arg);
                    return 0.0; // Default for non-numeric
                }, col[r]);
            }
        }
        return result;
    }
};

#endif // DATAFRAME_H