/**
 * @file Model.h
 * @brief Abstract base class for all machine learning models.
 * * Defines the standard interface for fitting models to data and making predictions.
 */

// include/daedalus/models/Model.h

#ifndef MODEL_H
#define MODEL_H

#include "../core/Matrix.h"

/**
 * @class Model
 * @brief Template interface for machine learning algorithms.
 * @tparam T The numeric type for data (defaults to double).
 */
template <typename T = double>
class Model {
public:
    /** @brief Virtual destructor to ensure proper cleanup of derived classes. */
    virtual ~Model() = default;

    /**
     * @brief Trains the model on the provided dataset.
     * @param X Feature matrix of shape (n_samples, n_features).
     * @param y Target matrix of shape (n_samples, n_targets).
     */
    virtual void fit(const Matrix<T>& X, const Matrix<T>& y) = 0;

    /**
     * @brief Makes predictions using the trained model parameters.
     * @param X Feature matrix to predict values for.
     * @return Matrix<T> A matrix containing the predicted values.
     */
    virtual Matrix<T> predict(const Matrix<T>& X) const = 0;
};

#endif // MODEL_H