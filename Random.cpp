#include "Random.h"

#include <EigenRand/EigenRand>

namespace neural_network {
Random::Random() : engine_(std::mt19937(42345)) {}
Matrix Random::normalMatrix(Index rows, Index cols) {
    return Eigen::Rand::normal<Matrix>(rows, cols, engine_);
}
}  // namespace neural_network
