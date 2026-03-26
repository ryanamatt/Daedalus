/**
 * @file Metrics.h
 * @brief Evaluation metrics for regression and classification tasks.
 * * This namespace provides standard performance measures to evaluate 
 * the accuracy and error of predictive models.
 */

// include/daedalus/core/Metrics.h

#ifndef METRICS_H
#define METRICS_H

#include "Matrix.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <iostream>

/**
 * @namespace Metrics
 * @brief Mathematical evaluation functions for model performance.
 */
namespace Metrics {
    /**
     * @brief Calculates the Mean Squared Error (MSE).
     * * MSE measures the average of the squares of the errors—that is, the 
     * average squared difference between the estimated values and the actual value.
     * * The formula used is:
     * $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{true,i} - y_{pred,i})^2$$
     * @param y_true Column matrix of ground truth values.
     * @param y_pred Column matrix of predicted values.
     * @return double The calculated mean squared error.
     * @throws std::invalid_argument if the number of rows in input matrices do not match.
     */
    inline double mean_squared_error(const Matrix<double>& y_true, const Matrix<double> y_pred) {
        if (y_true.rows() != y_pred.rows()) throw std::invalid_argument("Dimensions must match.");

        double mse = 0.0;
        size_t n = y_true.rows();
        for (size_t i = 0; i < n; ++i) {
            double error = y_true(i, 0) - y_pred(i, 0);
            mse += error * error;
        }
        return mse / static_cast<double>(n);
    }

    /**
     * @brief Calculates the Coefficient of Determination ($R^2$ Score).
     * * $R^2$ provides an indication of goodness of fit and therefore a measure 
     * of how well unseen samples are likely to be predicted by the model.
     * * The formula is:
     * $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
     * where $SS_{res} = \sum (y_{true} - y_{pred})^2$ and $SS_{tot} = \sum (y_{true} - \bar{y})^2$.
     * @param y_true Column matrix of ground truth values.
     * @param y_pred Column matrix of predicted values.
     * @return double The R-Squared score (typically between 0 and 1).
     * @throws std::invalid_argument if dimensions mismatch.
     */
    inline double r2_score(const Matrix<double>& y_true, const Matrix<double>& y_pred) {
        if (y_true.rows() != y_pred.rows()) throw std::invalid_argument("Dimensions must match.");

        size_t n = y_true.rows();
        double sum_y = 0.0;
        for (size_t i = 0; i < n; ++i) sum_y += y_true(i, 0);
        double mean_y = sum_y / n;

        double ss_res = 0.0;
        double ss_tot = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double res = y_true(i, 0) - y_pred(i, 0);
            double tot = y_true(i, 0) - mean_y;
            ss_res += res * res;
            ss_tot += tot * tot;
        }

        if (ss_tot == 0.0) return (ss_res == 0.0) ? 1.0 : 0.0;

        return 1.0 - (ss_res / ss_tot);
    }

    /**
     * @brief Finds the True Positives of the Confusion Matrix.
     * @param y_true Column matrix of ground truth values.
     * @param y_pred Column matrix of predicted values.
     * @return Matrix The confusion matrix.
     * @throws std::invalid_argument if the number of rows in input matrices do not match.
     */
    inline Matrix<double> confusion_matrix(const Matrix<double>& y_true, const Matrix<double>& y_pred) {
        double tp = 0, fp = 0, fn = 0, tn = 0;
        for (size_t i = 0; i < y_true.rows(); ++i) {
            if (y_pred(i, 0) == 1.0) {
                if (y_true(i, 0) == 1.0) tp++;
                else fp++;
            }
            else {
                if (y_true(i, 0) == 1.0) fn++;
                else tn++;
            }
        }

        Matrix<double> conf_mat(2, 2);
        conf_mat(0, 0) = tp;
        conf_mat(0, 1) = fp;
        conf_mat(1, 0) = fn;
        conf_mat(1, 1) = tn;
        return conf_mat;
    }

    /**
     * @brief Calculates the Accuracy Score for classification.
     * * Ratio of correct predictions to total number of input samples.
     * @param y_true Column matrix of ground truth labels.
     * @param y_pred Column matrix of predicted labels.
     * @return double Accuracy ranging from 0.0 to 1.0.
     * @throws std::invalid_argument if the number of rows in input matrices do not match.
     */
    inline double accuracy_score(const Matrix<double>& y_true, const Matrix<double>& y_pred) {
        if (y_true.rows() != y_pred.rows()) throw std::invalid_argument("Dimensions mismatch");
        size_t correct = 0;
        for (size_t i = 0; i < y_true.rows(); ++i) {
            if (y_true(i, 0) == y_pred(i, 0)) correct++;
        }
        return static_cast<double>(correct) / y_true.rows();
    }

    /**
     * @brief Calculates Precision.
     * * Precision is the ability of the classifier not to label as positive 
     * a sample that is negative. 
     * * Formula: $$\frac{tp}{tp + fp}$$
     * @param y_true Column matrix of ground truth labels.
     * @param y_pred Column matrix of predicted labels.
     * @return double Precision score.
     * @throws std::invalid_argument if the number of rows in input matrices do not match.
     */
    inline double precision_score(const Matrix<double>& y_true, const Matrix<double>& y_pred) {
        if (y_true.rows() != y_pred.rows()) throw std::invalid_argument("Dimensions must match.");
        double tp = 0, fp = 0;
        for (size_t i = 0; i < y_true.rows(); ++i) {
            if (y_pred(i, 0) == 1.0) {
                if (y_true(i, 0) == 1.0) tp++;
                else fp++;
            }
        }
        return (tp + fp > 0) ? tp / (tp + fp) : 0.0;
    }

    /**
     * @brief Calculates Recall (Sensitivity).
     * * Recall is the ability of the classifier to find all the positive samples.
     * * Formula: $$\frac{tp}{tp + fn}$$
     * @param y_true Column matrix of ground truth labels.
     * @param y_pred Column matrix of predicted labels.
     * @return double Recall score.
     * @throws std::invalid_argument if the number of rows in input matrices do not match.
     */
    inline double recall_score(const Matrix<double>& y_true, const Matrix<double>& y_pred) {
        if (y_true.rows() != y_pred.rows()) throw std::invalid_argument("Dimensions must match.");
        double tp = 0, fn = 0;
        for (size_t i = 0; i < y_true.rows(); ++i) {
            if (y_true(i, 0) == 1.0) {
                if (y_pred(i, 0) == 1.0) tp++;
                else fn++;
            }
        }
        return (tp + fn > 0) ? tp / (tp + fn) : 0.0;
    }

    /**
     * @brief Calculates the F1 Score.
     * * The F1 score is the harmonic mean of precision and recall.
     * It reaches its best value at 1 and worst at 0.
     * * Formula: $$2 \cdot \frac{precision \cdot recall}{precision + recall}$$
     * @param y_true Column matrix of ground truth labels.
     * @param y_pred Column matrix of predicted labels.
     * @return double F1 score.
     * @throws std::invalid_argument if the number of rows in input matrices do not match.
     */
    inline double f1_score(const Matrix<double>& y_true, const Matrix<double>& y_pred) {
        if (y_true.rows() != y_pred.rows()) throw std::invalid_argument("Dimensions must match.");
        double p = precision_score(y_true, y_pred);
        double r = recall_score(y_true, y_pred);
        return (p + r > 0) ? 2 * (p * r) / (p + r) : 0.0;
    }

    inline double mcc_score(const Matrix<double>& y_true, const Matrix<double>& y_pred) {
        if (y_true.rows() != y_pred.rows()) throw std::invalid_argument("Dimensions must match.");
        Matrix conf_mat = confusion_matrix(y_true, y_pred);
        double TP = conf_mat(0, 0);
        double FP = conf_mat(0, 1);
        double FN = conf_mat(1, 0);
        double TN = conf_mat(1, 1);

        double numerator = (TP * TN) - (FP * FN);
        double denominator = std::sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
        if (denominator == 0) { return 0.0; }

        return numerator / denominator;
    }
}

#endif // METRICS_H