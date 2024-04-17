#include "Layer.h"

#include <iostream>

namespace neural_network {
Layer::Layer(Index input_dimension, Index output_dimension,
             ActivationFunction f)
    : f_(std::move(f)),
      rnd_(Random()),
      a_(rnd_.normalMatrix(output_dimension, input_dimension)),
      b_(rnd_.normalMatrix(output_dimension, 1)) {
    assert(input_dimension > 0);
    assert(output_dimension > 0);
}

Matrix neural_network::Layer::evaluate(const Matrix &input) const {
    return f_.evaluate0((a_ * input).colwise() + b_);
}

Matrix neural_network::Layer::getGradA(const Matrix &u, const Matrix &x) {
    Matrix result_grad_a = Matrix::Zero(a_.rows(), a_.cols());
    for (int i = 0; i < u.rows(); ++i) {
        result_grad_a += (f_.evaluate1(a_ * x.col(i) + b_).asDiagonal() *
                          u.row(i).transpose()) *
                         x.col(i).transpose();
    }
    return result_grad_a / u.rows();
}

Matrix neural_network::Layer::getGradB(const Matrix &u, const Matrix &x) {
    Matrix result_grad_b = Vector::Zero(b_.size());
    for (int i = 0; i < u.rows(); ++i) {
        result_grad_b += (f_.evaluate1(a_ * x.col(i) + b_).asDiagonal() *
                          u.row(i).transpose());
    }
    return result_grad_b / u.rows();
}

Matrix neural_network::Layer::getNextU(const Matrix &u, const Matrix &x) {
    Matrix next_u(x.cols(), x.rows());
    for (int i = 0; i < x.cols(); ++i) {
        next_u.row(i) =
            (u.row(i) * f_.evaluate1(a_ * x.col(i) + b_).asDiagonal()) * a_;
    }
    return next_u;
}

void neural_network::Layer::updA(const Matrix &gradient_correction) {
    a_ -= gradient_correction;
}

void neural_network::Layer::updB(const Vector &gradient_correction) {
    b_ -= gradient_correction;
}

Index Layer::getInputSize() const {
    return a_.cols();
}

Index Layer::getOutputSize() const {
    return a_.rows();
}
}  // namespace neural_network
