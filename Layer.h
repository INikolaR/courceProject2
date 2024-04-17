#pragma once

#include <Eigen/Dense>
#include <random>

#include "ActivationFunction.h"
#include "CustomTypes.h"
#include "Random.h"

namespace neural_network {
class Layer {
public:
    Layer(Index input_dimension, Index output_dimension, ActivationFunction f);
    Matrix evaluate(const Matrix& input) const;
    Matrix getGradA(const Matrix& u, const Matrix& x);
    Matrix getGradB(const Matrix& u, const Matrix& x);
    Matrix getNextU(const Matrix& u, const Matrix& x);
    void updA(const Matrix& gradient_correction);
    void updB(const Vector& gradient_correction);
    Index getInputSize() const;
    Index getOutputSize() const;

private:
    ActivationFunction f_;
    Random rnd_;
    Matrix a_;
    Vector b_;
};
}  // namespace neural_network
