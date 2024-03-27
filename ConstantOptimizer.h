#pragma once

#include <Eigen/Dense>

namespace neural_network {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class ConstantOptimizer {
public:
    explicit ConstantOptimizer(double step);
    Matrix getNextGradientCorrection(const Matrix& g);
    void updateToNextIteration();
private:
    double step_;
};
}
