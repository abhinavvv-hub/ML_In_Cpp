/*
    GUIDE
    import this as #include "EvaluationMetric.hpp"
    Every metric is inside evalMetric namespace.
    To use any metric write evalMetric::[Metric]
    for e.g., to use Max Abs Error write "evalMetric::MaximumAbsoluteError(Y_Original, Y_Pred)"
*/

#pragma once

#ifndef EvaluationMetric_HPP
#define EvaluationMetric_HPP

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "matrix.hpp"

namespace evalMetric {
    inline float MeanSquaredError(const Matrix &Y_Original, const Matrix &Y_Pred) {
        auto size_Pred = Y_Pred.size();
        auto size_Original = Y_Original.size();
        if (size_Pred.first != size_Original.first || size_Pred.second != size_Original.second) {
            throw std::invalid_argument("Size Mismatch!");
        }
        if (size_Pred.second != 1) {
            throw std::invalid_argument("Number of Columns in Labels is not equal to 1!");
        }
        float ans = 0.0f;
        for (size_t i = 0; i < static_cast<size_t>(size_Pred.first); i++) {
            float diff = Y_Pred.peek(i, 0) - Y_Original.peek(i, 0);
            ans += diff * diff; 
        }
        ans /= static_cast<float>(size_Pred.first);
        return ans;
    }

    inline float RootMeanSquaredError(const Matrix &Y_Original, const Matrix &Y_Pred) {
        auto size_Pred = Y_Pred.size();
        auto size_Original = Y_Original.size();
        if (size_Pred.first != size_Original.first || size_Pred.second != size_Original.second) {
            throw std::invalid_argument("Size Mismatch!");
        }
        if (size_Pred.second != 1) {
            throw std::invalid_argument("Number of Columns in Labels is not equal to 1!");
        }
        float ans = 0.0f;
        for (size_t i = 0; i < static_cast<size_t>(size_Pred.first); i++) {
            float diff = Y_Pred.peek(i, 0) - Y_Original.peek(i, 0);
            ans += diff * diff; 
        }
        ans /= static_cast<float>(size_Pred.first);
        return std::sqrt(ans);
    }

    inline float MaximumAbsoluteError(const Matrix &Y_Original, const Matrix &Y_Pred) {
        auto size_Pred = Y_Pred.size();
        auto size_Original = Y_Original.size();
        if (size_Pred.first != size_Original.first || size_Pred.second != size_Original.second) {
            throw std::invalid_argument("Size Mismatch!");
        }
        if (size_Pred.second != 1) {
            throw std::invalid_argument("Number of Columns in Labels is not equal to 1!");
        }
        float ans = 0.0f;
        for (size_t i = 0; i < static_cast<size_t>(size_Pred.first); i++) {
            ans = std::max(ans, std::fabs(Y_Pred.peek(i, 0) - Y_Original.peek(i, 0)));
        }
        return ans;
    }

    inline float BinaryCrossEntropy(const Matrix &Y_Pred, const Matrix &Y_Original) {
        auto size_Pred = Y_Pred.size();
        auto size_Original = Y_Original.size();
        if (size_Pred.first != size_Original.first || size_Pred.second != size_Original.second) {
            throw std::invalid_argument("Size Mismatch!");
        }
        if (size_Pred.second != 1) {
            throw std::invalid_argument("Labels must be a single column vector!");
        }
        float total_loss = 0.0f;
        size_t n = static_cast<size_t>(size_Pred.first);
        const float epsilon = 1e-7f;

        for (size_t i = 0; i < n; i++) {
            float y_true = Y_Original.peek(i, 0);
            float y_pred = Y_Pred.peek(i, 0);
            y_pred = std::max(epsilon, std::min(1.0f - epsilon, y_pred));
            total_loss += (y_true * std::log(y_pred) + (1.0f - y_true) * std::log(1.0f - y_pred));
        }
        return -total_loss / static_cast<float>(n);
    }

    inline float AccuracyScore(const Matrix &Y_Pred, const Matrix &Y_Original, float threshold = 0.5f) {
        auto size_Pred = Y_Pred.size();
        auto size_Original = Y_Original.size();
        if (size_Pred.first != size_Original.first || size_Pred.second != size_Original.second) {
            throw std::invalid_argument("Size Mismatch!");
        }
        size_t correct_predictions = 0;
        size_t n = static_cast<size_t>(size_Pred.first);
        for (size_t i = 0; i < n; i++) {
            int predicted_class = (Y_Pred.peek(i, 0) >= threshold) ? 1 : 0;
            int actual_class = (Y_Original.peek(i, 0) >= threshold) ? 1 : 0;

            if (predicted_class == actual_class) {
                correct_predictions++;
            }
        }
        return static_cast<float>(correct_predictions) / static_cast<float>(n);
    }

    inline Matrix GenerateConfusionMatrix(const Matrix &Y_Pred, const Matrix &Y_Original, float threshold = 0.5f) {
        auto size_Pred = Y_Pred.size();
        auto size_Original = Y_Original.size();
        if (size_Pred.first != size_Original.first || size_Pred.second != size_Original.second) {
            throw std::invalid_argument("Size Mismatch!");
        }
        Matrix cm(2, 2);
        size_t n = static_cast<size_t>(size_Pred.first);
        for (size_t i = 0; i < n; i++) {
            bool is_pred_positive = (Y_Pred.peek(i, 0) >= threshold);
            bool is_actual_positive = (Y_Original.peek(i, 0) >= 0.5f);

            if (is_pred_positive && is_actual_positive)  cm.update(0, 0, 1 + cm.peek(0, 0));
            if (is_pred_positive && !is_actual_positive) cm.update(0, 1, 1 + cm.peek(0, 1));
            if (!is_pred_positive && !is_actual_positive) cm.update(1, 0, 1 + cm.peek(1, 0));
            if (!is_pred_positive && is_actual_positive)  cm.update(1, 1, 1 + cm.peek(1, 1));
        }
        return cm;
    }
}

#endif
