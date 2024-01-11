#include "ActivateFunction.h"
namespace neural_network {
    double ActivateFunction::evaluate0(double value) {
        return (value > 0) * ((1 - A) * value) + A * value;
    }

    double ActivateFunction::evaluate1(double value) {
        return (value > 0) * (1 - A) + A;
    }
}
