// src/models/linearRegression.cc

#include <iostream>
#include <iomanip>
#include "daedalus/models/linearRegression.h"

Matrix<double> LinearRegression::predict(const Matrix<double>& X) const {
    Matrix<double> projection = X * weights;
    for (size_t i = 0; i < projection.rows(); ++i) {
        projection(i, 0) += bias(0, 0); // Add the scalar bias to each prediction
    }
    return projection;
}

void LinearRegression::fit(const Matrix<double>& X, const Matrix<double>& y) {
    this->fit(X, y, 100); 
}

void LinearRegression::fit(const Matrix<double>& X, const Matrix<double>& y, int epochs) {
    int m = X.rows();
    int n = X.cols();

    weights = Matrix<double>(n, 1);
    bias = Matrix<double>(1, 1);
    bias(0, 0) = 0.0;

    for (int i = 0; i < epochs; ++i) {
        Matrix<double> predictions = predict(X);
        Matrix<double> error = predictions - y;

        // Weight Gradient: (X^T * error) / m
        Matrix<double> gradientW = X.transpose() * error;

        // Apply Regularization to the Gradient
        for (size_t j = 0; j < weights.rows(); ++j) {
            double reg_term = 0.0;
            if (penalty == "l2") {
                // Derivative of λ * w^2 is λ * w
                reg_term = reg_lambda * weights(j, 0);
            } 
            else if (penalty == "l1") {
                // Derivative of λ * |w| is λ * sign(w)
                reg_term = reg_lambda * (weights(j, 0) > 0 ? 1.0 : (weights(j, 0) < 0 ? -1.0 : 0.0));
            }

            // Update weights: w = w - alpha * (gradient + reg_term)
            weights(j, 0) -= (alpha / static_cast<double>(m)) * (gradientW(j, 0) + reg_term);
        }

        // Bias Gradient: (Sum of errors) / m
        double bias_gradient = 0.0;
        for (size_t r = 0; r < error.rows(); ++r) {
            bias_gradient += error(r, 0);
        }
        bias(0, 0) -= (alpha / static_cast<double>(m)) * bias_gradient;
    }
}

void LinearRegression::saveModel(const std::string& filename) const {
    if (weights.rows() == 0 || weights.cols() == 0) {
        std::cerr << "Error: Model has not been fited yet." << std::endl;
        return;
    }

    std::cout << "Attempting to save model to: " << filename << "..." << std::endl;
    std::ofstream outFile(filename, std::ios::out | std::ios::trunc); // Ensure fresh file
    
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file! Path might be invalid or locked." << std::endl;
        return;
    }

    outFile << std::setprecision(17);

    // Write data with explicit flushing
    outFile << alpha << std::endl;
    outFile << weights.rows() << " " << weights.cols() << std::endl;
    
    for (size_t r = 0; r < weights.rows(); ++r) {
        for (size_t c = 0; c < weights.cols(); ++c) {
            outFile << weights(r, c) << " ";
        }
    }
    outFile << std::endl;

    outFile << bias.rows() << " " << bias.cols() << std::endl;
    outFile << bias(0, 0) << std::endl;

    outFile.flush();
    if (outFile.fail()) {
        std::cerr << "Error: Write operation failed!" << std::endl;
        outFile.close();
        return;
    }
    outFile.close();
    
    std::cout << "Model saved successfully to disk." << std::endl;
}

void LinearRegression::loadModel(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile) throw std::runtime_error("Could not open file for loading.");

    size_t rows, cols;

    // Load Alpha
    inFile >> alpha;

    // Load Weights
    inFile >> rows >> cols;
    weights = Matrix<double>(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            inFile >> weights(r, c);
        }
    }

    // Load Bias
    inFile >> rows >> cols;
    bias = Matrix<double>(rows, cols);
    inFile >> bias(0, 0);

    inFile.close();
}