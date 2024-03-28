#pragma once

#include <Eigen/Dense>
#include <functional>

namespace neural_network {
using Matrix = Eigen::MatrixXd;

class LossFunction {
public:
    LossFunction(std::function<double(const Matrix&, const Matrix&)>&& f0,
                 std::function<Matrix(const Matrix&, const Matrix&)>&& f1);
    double dist(const Matrix& x, const Matrix& y) const;
    Matrix derivativeDist(const Matrix& x, const Matrix& y) const;

private:
    const std::function<double(const Matrix&, const Matrix&)> f0_;
    const std::function<Matrix(const Matrix&, const Matrix&)> f1_;
};
}  // namespace neural_network
