#pragma once
#include <Eigen/Dense>

namespace neural_network {
using Index = Eigen::Index;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
struct TrainUnit {
    Vector x;
    Vector y;
};
}  // namespace neural_network
