#pragma once
#include <functional>

#include "CustomTypes.h"

namespace neural_network {
class ActivationFunction {
public:
    static ActivationFunction ReLU();
    static ActivationFunction LeakyReLU();
    static ActivationFunction Sigmoid();
    ActivationFunction(std::function<double(double)>&& f0, std::function<double(double)>&& f1);
    double evaluate0(double x) const;
    double evaluate1(double x) const;
    Matrix evaluate0(const Matrix& x) const;
    Matrix evaluate1(const Matrix& x) const;

private:
    static constexpr double LeakyReLUCoefficient = 0.01;

    const std::function<double(double)> f0_;
    const std::function<double(double)> f1_;
};
}  // namespace neural_network
