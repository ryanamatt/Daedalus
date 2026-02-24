/**
 * @file Establishes the Neural Network
 */

// include/daedalus/models/NeuralNetwork.h

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include "Model.h"
#include "Layer.h"

/**
 * @class NeuralNetwork
 * @brief A container model for stacking multiple layers.
 */
class NeuralNetwork : public Model<double> {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    double learning_rate;

public:
    /**
     * @brief Constructs a Neural Network.
     * @param lr The learning rate applied during the backward pass.
     */
    NeuralNetwork(double lr = 0.01) : learning_rate(lr) {}

    /** @brief Adds a new layer to the end of the network. */
    void add(std::unique_ptr<Layer> layer);

    /** @brief Forward pass through all layers to get a prediction. */
    Matrix<double> predict(const Matrix<double>& x) const override;

    /** * @brief Trains the network using forward and backward propagation.
     * @param X Training features.
     * @param y Target values.
     * @param epochs Number of iterations over the dataset.
     */
    void fit(const Matrix<double>& X, const Matrix<double>& y) override;
    void fit(const Matrix<double>& X, const Matrix<double>& y, int epochs);
};

#endif // NEURAL_NETWORK_H