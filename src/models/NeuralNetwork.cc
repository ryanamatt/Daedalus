// src/models/NeuralNetwork.cc

#include "../../include/daedalus/models/NeuralNetwork.h"

void NeuralNetwork::add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

Matrix<double> NeuralNetwork::predict(const Matrix<double>& x) const {
    Matrix<double> output = x;
    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetwork::fit(const Matrix<double>& X, const Matrix<double>& y, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Matrix<double> output = predict(X);

        // Compute Loss Gradient (MSE Derivative: 2 * (output - y) / n)
        // This acts as the initial 'gradient' for the backward pass
        Matrix<double> error_gradient = (output - y) * (2.0 / X.rows());

        // Backward Pass (Iterate in reverse)
        Matrix<double> current_gradient = error_gradient;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            current_gradient = (*it)->backward(current_gradient, learning_rate);
        }
    }
}

void NeuralNetwork::fit(const Matrix<double>& X, const Matrix<double>& y) {
    fit(X, y, 100);
}