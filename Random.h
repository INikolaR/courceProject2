#pragma once

#include <random>

#include "CustomTypes.h"

namespace neural_network {
class Random {
public:
    Random();
    Matrix normalMatrix(Index rows, Index cols);
private:
    std::mt19937 engine_;
};
}  // namespace neural_network
