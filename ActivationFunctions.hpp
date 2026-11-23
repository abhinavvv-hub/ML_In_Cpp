/*
    GUIDE
    import this as #include "ActivationFunctions.hpp"
    All the activation functions are defined under the namespace "actFn" so to use any function write it as actFn::[The activation function]
    e.g., if you intend to use Sigmoid Function just write "actFn::Sigmoid(z)"
*/

#pragma once

#ifndef ActivationFunctions_HPP
#define ActivationFunctions_HPP

#include <cmath>
#include <algorithm>

namespace actFn {
    inline double ELU(float z, float alpha) {
        return z <= 0 ? alpha * (exp(z) - 1) : z;
    }

    inline double LeakyReLU(float z, float alpha) {
        return z <= 0 ? alpha * z : z;
    }

    inline double ReLU(float z) {
        return std::max(0.0f, z);
    }

    inline double Sigmoid(float z) {
        return 1.0 / (1 + exp(-z));
    }

    inline double Tanh(float z) {
        return (exp(z) - exp(-z)) / (exp(z) + exp(-z));
    }
}


#endif
