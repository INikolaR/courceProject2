#include "ConstantOptimizer.h"

namespace neural_network {
    ConstantOptimizer::ConstantOptimizer(double step) : step_(step) {}
    double ConstantOptimizer::getNextStep() {
        return step_;
    }
}