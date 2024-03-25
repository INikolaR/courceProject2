#include "ActivationFunction.h"
#include <functional>

namespace neural_network {
    double ActivationFunction::evaluate0(double value) const {
        return f0_(value);
    }

    double ActivationFunction::evaluate1(double value) const {
        return f1_(value);
    }
    ActivationFunction::ActivationFunction(std::function<double(double)>&& f0,
                                           std::function<double(double)>&& f1) : f0_(f0), f1_(f1) {
    }
    }
