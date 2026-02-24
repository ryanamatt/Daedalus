/**
 * @file linearRegression.h
 * @brief Implementation of Linear Regression with regularization.
 */

// include/daedalus/models/linearRegression.h

#ifndef LINREN_H
#define LINREN_H

#include <string>
#include <fstream>
#include "Model.h"

/**
 * @class LinearRegression
 * @brief A Linear Regression model supporting OLS and Regularized Gradient Descent.
 * * The model predicts @f$ \hat{y} = Xw + b @f$.
 */
class LinearRegression : public Model<double> {
private:
    Matrix<double> weights;
    Matrix<double> bias;
    double alpha;
    double reg_lambda;
    std::string penalty; // "l1", "l2", or "none"

public:
    /**
     * @brief Constructs a Linear Regression object.
     * @param learning_rate Step size for weight updates.
     * @param lambda Regularization strength (ignored if penalty is "none").
     * @param penalty Type of regularization to apply.
     */
    LinearRegression(double learning_rate = 0.01, double lambda = 0.01, std::string penalty = "none")
        : weights(0, 0), bias(0, 0), alpha(learning_rate), reg_lambda(lambda), penalty(penalty) {}

    /** @brief Standard fit using the default number of iterations or convergence logic. */
    void fit(const Matrix<double>& X, const Matrix<double>& y) override;

    /**
     * @brief Fits the model using a specific number of gradient descent epochs.
     * @param X Training features.
     * @param y Training targets.
     * @param epochs Number of times to iterate over the training set.
     */
    void fit(const Matrix<double>& X, const Matrix<double>& y, int epochs);

    /** @brief Predicts continuous values for the input matrix X. */
    Matrix<double> predict(const Matrix<double>& x) const override;

    /** @brief Serializes model weights and parameters to a file. */
    void saveModel(const std::string& filename) const;

    /** @brief Deserializes model weights and parameters from a file. */
    void loadModel(const std::string& filename);
};

#endif // LINREN_H