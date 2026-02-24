// src/models/knn.cc

#include <cmath>
#include <map>
#include "daedalus/models/knn.h"

double KNN::compute_distance(const Matrix<double>& row1, const Matrix<double>& row2) const {
    double distance = 0.0;
    // Assuming row1 and row2 are single-row Matrices or vectors
    for (int i = 0; i < row1.cols(); ++i) {
        double diff = row1(0, i) - row2(0, i);
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

Matrix<double> KNN::predict(const Matrix<double>& X) const {
    // Result matrix: same number of rows as input X, 1 column for prediction
    Matrix<double> predictions(X.rows(), 1);

    for (int i = 0; i < X.rows(); ++i) {
        // Store pairs of {distance, index_in_training_set}
        std::vector<std::pair<double, int>> distances;

        // Calculate distance from current test point to ALL training points
        for (int j = 0; j < train_X.rows(); ++j) {
            double dist = compute_distance(X.get_row(i), train_X.get_row(j));
            distances.push_back({dist, j});
        }

        // Sort by distance (ascending) and pick top K
        std::sort(distances.begin(), distances.end());

        // Classification
        std::map<double, int> class_counts;
        for (int k_idx = 0; k_idx < this->k; ++k_idx) {
            int train_index = distances[k_idx].second;
            double label = train_y(train_index, 0); 
            class_counts[label]++;
        }

        // Find the majority class
        double best_class = -1;
        int max_votes = -1;
        for (auto const& [label, count] : class_counts) {
            if (count > max_votes) {
                max_votes = count;
                best_class = label;
            }
        }
        predictions(i, 0) = best_class;
    }

    return predictions;
}