/**
 * @file logisticRegression.h
 * @brief Implementation of Logistic Regression for binary classification.
 */

// include/daedalus/models/logisticRegression.h

#ifndef LOGREG_H
#define LOGREG_H

#include <string>
#include <fstream>
#include "Model.h"
#include "cmath"

/**
 * @class LogisticRegression
 * @brief A Binary Logistic Regression classifier.
 * * Uses the Logistic Sigmoid function: @f$ \sigma(z) = \frac{1}{1 + e^{-z}} @f$.
 */
class LogisticRegression : public Model<double> {
private:
    Matrix<double> weights;
    Matrix<double> bias;
    double alpha;
    double reg_lambda;      // Regularization strength
    std::string penalty;    // "l1", "l2", or "none"

    /** @brief Internal helper to compute the sigmoid mapping. */
    double sigmoid(double z) const {
        return 1.0 / (1.0 + std::exp(-z));
    }

public:
/**
     * @brief Constructs a Logistic Regression classifier.
     * @param learning_rate Step size for gradient descent.
     * @param lambda Regularization strength.
     * @param penalty Regularization type ("l1", "l2", or "none").
     */
    LogisticRegression(double learning_rate = 0.01, double lambda = 0.01, std::string penalty = "none")
        : weights(0, 0), bias(0, 0), alpha(learning_rate), reg_lambda(lambda), penalty(penalty) {}

    /** @brief Trains the classifier using Log-Loss gradient descent. */
    void fit(const Matrix<double>& X, const Matrix<double>& y) override;

    /** @brief Trains the classifier for a fixed number of epochs. */
    void fit(const Matrix<double>& X, const Matrix<double>& y, int epochs);

    /** @brief Predicts binary labels (0.0 or 1.0) based on a 0.5 probability threshold. */
    Matrix<double> predict(const Matrix<double>& X) const override;
    
    /**
     * @brief Returns the raw probability of the positive class.
     * @return Matrix<double> Column matrix of probabilities in range [0, 1].
     */
    Matrix<double> predict_proba(const Matrix<double>& X) const;

    /** @brief Saves model parameters to a file. */
    void LogisticRegression::saveModel(const std::string& filename) const;

    /** @brief Loads model parameters from a file. */
    void LogisticRegression::loadModel(const std::string& filename);
};

#endif // LOGREG_H