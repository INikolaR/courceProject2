#include "ConstantOptimizer.h"

namespace neural_network {
ConstantOptimizer::ConstantOptimizer(double step) : step_(step) {}

Matrix ConstantOptimizer::getNextGradientCorrection(const Matrix &g) {
    return step_ * g;
}

void ConstantOptimizer::updateToNextIteration() {}
}