/**
 * @file Layer.h
 * This file establishs the Layer abstract class for Neutral Networks.
 */

// include/daedalus/models/Layer.h

#ifndef LAYER_H
#define LAYER_H

#include "../core/Matrix.h"

/**
 * @class Layer
 * @brief Abstract base class for all neural network layers.
 */
class Layer {
public:
    virtual ~Layer() = default;

    /**
     * @brief Computes the output of the layer.
     * @param input Matrix of input features.
     * @return Matrix representing the layer's activations.
     */
    virtual Matrix<double> forward(const Matrix<double>& input) = 0;

    /**
     * @brief Computes gradients and updates weights.
     * @param gradient The gradient of the loss with respect to the output.
     * @param learning_rate Step size for weight updates.
     * @return Matrix representing the gradient with respect to the input.
     */
    virtual Matrix<double> backward(const Matrix<double>& gradient, double learning_rate) = 0;
};

 #endif // LAYER_H