// src/models/Layer.h

#include <random>
#include "../../include/daedalus/models/DenseLayer.h"

DenseLayer::DenseLayer(int input_size, int output_size)
    : weights(input_size, output_size), bias(1, output_size), last_input(0, 0) {

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 / input_size));

    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            weights(i, j) = distribution(generator);
        }
    }

    for (int j = 0; j < output_size; ++j) {
        bias(0, j) = 0.0;
    }
}

Matrix<double> DenseLayer::forward(const Matrix<double>& input) {
    last_input = input;
    
    // Perform Matrix Multiplication: Result is (BatchSize x OutputSize)
    Matrix<double> output = input * weights;
    
    // Manual Broadcasting: Add the (1 x OutputSize) bias to every row of the output
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            output(i, j) += bias(0, j);
        }
    }
    
    return output;
}

Matrix<double> DenseLayer::backward(const Matrix<double>& gradient, double learning_rate) {
    // Calculate gradient with respect to weights: dW = X^T * dY
    Matrix<double> input_T = last_input.transpose();
    Matrix<double> d_weights = input_T * gradient;

    // Calculate gradient with respect to bias: dB = sum(dY) over batch
    // (Assuming row-vector bias applied to each row of the input batch)
    Matrix<double> d_bias(1, bias.cols());
    for (int j = 0; j < gradient.cols(); ++j) {
        double col_sum = 0;
        for (int i = 0; i < gradient.rows(); ++i) {
            col_sum += gradient(i, j);
        }
        d_bias(0, j) = col_sum;
    }

    // Calculate gradient with respect to input to pass to previous layer: dX = dY * W^T
    Matrix<double> weights_T = weights.transpose();
    Matrix<double> d_input = gradient * weights_T;

    // Update weights and bias using Gradient Descent
    weights = weights - (d_weights * learning_rate);
    bias = bias - (d_bias * learning_rate);

    return d_input;
}