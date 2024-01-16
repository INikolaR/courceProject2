#include "ActivationFunction.h"

namespace neural_network {
    double ActivationFunction::evaluate0(double value) {
        return (value > 0) * ((1 - A) * value) + A * value;
    }

    double ActivationFunction::evaluate1(double value) {
        return (value > 0) * (1 - A) + A;
    }
}
