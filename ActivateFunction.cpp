#include "ActivateFunction.h"

double ActivateFunction::evaluate0(double value) {
    return (value > 0 ? value : 0);
}

double ActivateFunction::evaluate1(double value) {
    return (value > 0 ? 1 : 0);
}