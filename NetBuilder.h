#pragma once

#include "Layer.h"
#include "Net.h"

namespace neural_network {
class NetBuilder {
public:
    NetBuilder(Index input_dimension, Index output_dimension,
               const ActivationFunction& f);
    NetBuilder& addLayer(int new_result_size, const ActivationFunction& f);
    Net build();

private:
    Net net_;
};
}  // namespace neural_network
