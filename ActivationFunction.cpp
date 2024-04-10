#include "ActivationFunction.h"

#include <cassert>
#include <functional>

namespace neural_network {
double ActivationFunction::evaluate0(double value) const {
    assert(f0_);
    return f0_(value);
}

double ActivationFunction::evaluate1(double value) const {
    assert(f1_);
    return f1_(value);
}
ActivationFunction::ActivationFunction(std::function<double(double)>&& f0, std::function<double(double)>&& f1)
    : f0_(std::move(f0)), f1_(std::move(f1)) {
}
}  // namespace neural_network
