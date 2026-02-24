// src//models/logisticRegression.cc

#include <iostream>
#include <iomanip>
#include "daedalus/models/logisticRegression.h"

Matrix<double> LogisticRegression::predict_proba(const Matrix<double>& X) const {
    Matrix<double> z = X * weights;
    for (size_t i = 0; i < z.rows(); ++i) {
        double val = z(i, 0) + bias(0, 0);
        z(i, 0) = sigmoid(val);
    }
    return z;
}

Matrix<double> LogisticRegression::predict(const Matrix<double>& X) const {
    Matrix<double> proba = predict_proba(X);
    for (size_t i = 0; i < proba.rows(); ++i) {
        // Convert probability to hard class 0 or 1
        proba(i, 0) = (proba(i, 0) >= 0.5) ? 1.0 : 0.0;
    }
    return proba;
}

void LogisticRegression::fit(const Matrix<double>& X, const Matrix<double>& y) {
    this->fit(X, y, 100);
}

void LogisticRegression::fit(const Matrix<double>& X, const Matrix<double>& y, int epochs) {
    int m = X.rows();
    weights = Matrix<double>(X.cols(), 1);
    bias = Matrix<double>(1, 1);
    bias(0, 0) = 0.0;

    for (int i = 0; i < epochs; ++i) {
        Matrix<double> predictions = predict_proba(X);
        Matrix<double> error = predictions - y;

        Matrix<double> gradientW = X.transpose() * error;

        for (size_t j = 0; j < weights.rows(); ++j) {
            double reg_term = 0.0;
            if (penalty == "l2") {
                reg_term = reg_lambda * weights(j, 0);
            } else if (penalty == "l1") {
                reg_term = reg_lambda * (weights(j , 0) > 0 ? 1.0 : (weights(j, 0) < 0 ? -1.0 : 0.0));
            }
            weights(j, 0) -= (alpha / static_cast<double>(m)) * (gradientW(j, 0) + reg_term);
        }

        double bias_gradient = 0.0;
        for (size_t r = 0; r < error.rows(); ++r) {
            bias_gradient += error(r, 0);
        }
        bias(0, 0) -= (alpha / static_cast<double>(m)) * bias_gradient;
    }
}

void LogisticRegression::saveModel(const std::string& filename) const {
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

void LogisticRegression::loadModel(const std::string& filename) {
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