/**
 * @file IO.h
 * @brief Provides Input/Output utilities for the Daedalus data library.
 * * This file contains functions to interface with external file formats, 
 * primarily focusing on loading data into DataFrame objects.
 */

// include/daedalus/core/IO.h

#ifndef IO_H
#define IO_H

#include "DataFrame.h"
#include <fstream>
#include <sstream>

/**
 * @brief Reads a CSV file and populates a DataFrame.
 * * This function parses a Comma-Separated Values (CSV) file. It automatically 
 * detects headers and attempts to infer data types for each cell.
 * * @section parsing_logic Parsing Logic:
 * The function attempts to parse each value as a @c double first. If 
 * conversion fails (e.g., if the value is text), it falls back to storing 
 * the value as a @c std::string.
 * @param filename The path to the CSV file to be read.
 * @param has_header If true (default), the first line is treated as column names.
 * @return DataFrame A populated DataFrame containing the CSV data.
 * @throws std::runtime_error If the file cannot be opened.
 */
inline DataFrame read_csv(const std::string& filename, bool has_header = true) {
    DataFrame df;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open CSV file.");

    std::string line, word;
    std::vector<std::string> headers;

    // Handle Headers
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        while (std::getline(ss, word, ',')) {
            headers.push_back(word);
        }
    }

    // Prepare storage for columns
    std::vector<std::vector<std::variant<double, int, std::string>>> temp_data(headers.size());

    // Parse Rows
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        size_t col_idx = 0;
        while (std::getline(ss, word, ',')) {
            try {
                // Try to store as double first
                temp_data[col_idx].push_back(std::stod(word));
            } catch (...) {
                // Fallback to string
                temp_data[col_idx].push_back(word);
            }
            col_idx++;
        }
    }

    // Populate DataFrame
    for (size_t i = 0; i < headers.size(); ++i) {
        df.add_column(headers[i], temp_data[i]);
    }

    return df;
}

#endif // IO_H