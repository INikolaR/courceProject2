#pragma once

namespace neural_network {
class ConstantOptimizer {
public:
    explicit ConstantOptimizer(double step);
    double getNextStep();
private:
    double step_;
};
}
