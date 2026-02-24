/**
 * @file DenseLayer.h
 * This file establishs the DenseLayer abstract class for Neutral Networks.
 */

// include/daedalus/models/Layer.h

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "Layer.h"
#include <random>

/**
 * @class DenseLayer
 * @brief A fully connected layer that performs Y = XW + B.
 */
class DenseLayer : public Layer {
private:
    Matrix<double> weights;
    Matrix<double> bias;
    Matrix<double> last_input; 

public:
    /**
     * @brief Constructs a Dense Layer with randomized weights.
     * @param input_size Number of input features.
     * @param output_size Number of neurons in this layer.
     */
    DenseLayer(int input_size, int output_size);

    /**
     * @brief Forward pass: computes the linear transformation.
     * @cite The logic mirrors the prediction formula in linearRegression.h.
     */
    Matrix<double> forward(const Matrix<double>& input) override;

    /**
     * @brief Backward pass: updates weights/bias and propagates the gradient.
     */
    Matrix<double> backward(const Matrix<double>& gradient, double learning_rate) override;

    // Getters for serialization (useful for saveModel/loadModel)
    /**
     * @brief Gets the weights.
     */
    const Matrix<double>& getWeights() const { return weights; }

    /**
     * @brief Gets the biases.
     */
    const Matrix<double>& getBias() const { return bias; }
};

#endif // DENSE_LAYER_H