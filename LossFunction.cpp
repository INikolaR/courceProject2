#include "LossFunction.h"

namespace neural_network {
    LossFunction::LossFunction(std::function<double(const Matrix &, const Matrix &)> &&f0,
                                               std::function<Matrix(const Matrix &, const Matrix &)> &&f1)
        : f0_(f0), f1_(f1) {
    }
    double LossFunction::dist(const neural_network::Matrix &x, const neural_network::Matrix &y) const {
        return f0_(x, y);
    }
    Matrix LossFunction::derivativeDist(const Matrix &x, const Matrix &y) const {
        return f1_(x, y);
    }
    }
