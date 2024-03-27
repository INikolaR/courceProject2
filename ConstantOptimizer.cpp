#include "ConstantOptimizer.h"

namespace neural_network {
    ConstantStepOptimizer::ConstantStepOptimizer(double step) : step_(step) {}
    double ConstantStepOptimizer::getNextStep() {
        return step_;
    }
}