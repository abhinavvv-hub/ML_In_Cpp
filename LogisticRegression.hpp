/*
    GUIDE
    import this as #include "LogisticRegression.hpp"
    if there are 'n' number of data and 'k' number of features then size of each Matrix:
        X_Train             : n x k
        Y_Train             : n x 1
        X_Test              : 1 x k
        Weights_and_Biases  : (k + 1) x 1
    No support for non numeric data. Use map to bind a non numeric data to a numeric value.
*/

#pragma once

#ifndef LogisticRegression_HPP
#define LogisticRegression_HPP

#include <stdexcept>
#include "matrix.hpp"
#include "ActivationFunctions.hpp"

class LogisticRegression {
    Matrix wnb{0,1};
    uint16_t parameters{};

    void train(const Matrix &X_Train, const Matrix &Y_Train, float learning_rate, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            Matrix Z = X_Train * wnb;
            Matrix A = Z.Sigmoid();
            Matrix Error = A - Y_Train;
            Matrix Gradient = X_Train.transpose() * Error;
            Gradient.scale(learning_rate / static_cast<float>(X_Train.size().first));
            wnb -= Gradient;
        }
    }

public:
    explicit LogisticRegression(const Matrix &X_Train, const Matrix &Y_Train, float learning_rate=0.05f, int epochs=10000) {
        std::pair<uint16_t, uint16_t> size = X_Train.size();
        parameters = size.second + 1;
        wnb = Matrix(parameters, 1);
        Matrix X_Train_Augmented(size.first, parameters);
        for (size_t i = 0; i < size.first; i++) {
            for (size_t j = 0; j < static_cast<size_t>(parameters - 1); j++) {
                X_Train_Augmented.update(i, j, X_Train.peek(i, j));
            }
        }
        for (size_t i = 0; i < size.first; i++) {
            X_Train_Augmented.update(i, parameters - 1, 1);
        }
        train(X_Train_Augmented, Y_Train, learning_rate, epochs);
    }

    Matrix predict(const Matrix &X_Test) const {
        if (X_Test.size().second != parameters - 1) {
            throw std::invalid_argument("X_Test features count mismatch!");
        }
        Matrix ans(X_Test.size().first, 1);
        for (size_t i = 0; i < (size_t)X_Test.size().first; i++) {
            float res = wnb.peek(parameters - 1, 0);
            for (size_t j = 0; j < parameters - 1; j++) {
                res += wnb.peek(j, 0) * X_Test.peek(i, j);
            }
            ans.update(i, 0, actFn::Sigmoid(res));
        }
        return ans;
    }
};

#endif
