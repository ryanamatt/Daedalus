/**
 * @file knn.h
 * @brief Implementation of the K-Nearest Neighbors algorithm.
 */

// include/daedalus/models/knn.h

#ifndef KNN_H
#define KNN_H

#include <vector>
#include <algorithm>
#include "Model.h"

/**
 * @class KNN
 * @brief A K-Nearest Neighbors implementation.
 */
class KNN : public Model<double> {
private:
    Matrix<double> train_X;
    Matrix<double> train_y;
    int k;

    /** @brief Computes Euclidean distance between two row vectors. */
    double compute_distance(const Matrix<double>& row1, const Matrix<double>& row2) const;

public:
    /**
     * @param k Number of neighbors to consider.
     */
    KNN(int k = 3) : train_X(0, 0), train_y(0, 0), k(k) {}

    /** @brief KNN "fit" simply stores the training data. */
    void fit(const Matrix<double>& X, const Matrix<double>& y) override {
        this->train_X = X;
        this->train_y = y;
    }

    /** @brief Predicts by finding the k-nearest neighbors in train_X. */
    Matrix<double> predict(const Matrix<double>& X) const override;
};

#endif // KNN_H