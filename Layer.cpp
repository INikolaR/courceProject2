#include "Layer.h"

#include <EigenRand/EigenRand>

#include <iostream>

namespace neural_network {
std::mt19937 Layer::engine = std::mt19937(42345);

Layer::Layer(int input_dimension, int output_dimension, ActivationFunction f)
    : f_(f),
      a_(Eigen::Rand::normal<Matrix>(output_dimension, input_dimension,
                                     engine)),
      b_(Eigen::Rand::normal<Matrix>(output_dimension, 1, engine)) {
}

Matrix neural_network::Layer::evaluate(const Matrix &input) const {
    Matrix multicolumn_b(a_.rows(), input.cols());
    for (int i = 0; i < input.cols(); ++i) {
        multicolumn_b.col(i) = b_;
    }
    return (a_ * input + multicolumn_b)
        .unaryExpr(
            std::bind(&ActivationFunction::evaluate0, f_, std::placeholders::_1));
}

Matrix neural_network::Layer::getGradA(const Matrix &u, const Matrix &x) {
    Matrix result_grad_a = Matrix::Zero(a_.rows(), a_.cols());
    for (int i = 0; i < u.rows(); ++i) {
        result_grad_a += ((a_ * x.col(i) + b_)
                              .unaryExpr(std::bind(&ActivationFunction::evaluate1,
                                                   f_, std::placeholders::_1))
                              .asDiagonal() *
                          u.row(i).transpose()) *
                         x.col(i).transpose();
    }
//    std::cout << "A:\n";
//    std::cout << result_grad_a << std::endl;
//    std::cout << "A/n:\n";
//    std::cout << result_grad_a / u.rows() << std::endl;
    return result_grad_a / u.rows();
}

Matrix neural_network::Layer::getGradB(const Matrix &u, const Matrix &x) {
    Matrix result_grad_b = Vector::Zero(b_.size());
    for (int i = 0; i < u.rows(); ++i) {
        result_grad_b += ((a_ * x.col(i) + b_)
                              .unaryExpr(std::bind(&ActivationFunction::evaluate1,
                                                   f_, std::placeholders::_1))
                              .asDiagonal() *
                          u.row(i).transpose());
    }
    return result_grad_b / u.rows();
}

Matrix neural_network::Layer::getNextU(const Matrix &u, const Matrix &x) {
    Matrix next_u(x.cols(), x.rows());
    for (int i = 0; i < x.cols(); ++i) {
        next_u.row(i) =
            (u.row(i) * (a_ * x.col(i) + b_)
                            .unaryExpr(std::bind(&ActivationFunction::evaluate1, f_,
                                                 std::placeholders::_1))
                            .asDiagonal()) *
            a_;
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
