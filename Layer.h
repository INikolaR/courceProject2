#pragma once

#include <cstdio>
#include <Eigen/Dense>
#include <random>

#include "ActivationFunction.h"
#include "CustomTypes.h"

namespace neural_network {
class Layer {
public:
    static std::mt19937 engine;

    Layer(int input_dimension, int output_dimension, ActivationFunction f);
    Matrix evaluate(const Matrix& input) const;
    Matrix getGradA(const Matrix& u, const Matrix& x);
    Matrix getGradB(const Matrix& u, const Matrix& x);
    Matrix getNextU(const Matrix& u, const Matrix& x);
    void updA(const Matrix& gradient_correction);
    void updB(const Vector& gradient_correction);
    Index getInputSize() const;
    Index getOutputSize() const;

private:
    Matrix a_;
    Vector b_;
    ActivationFunction f_;
};
}  // namespace neural_network
