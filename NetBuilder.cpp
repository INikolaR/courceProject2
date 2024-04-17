#include "NetBuilder.h"
namespace neural_network {
NetBuilder::NetBuilder(Index input_dimension, Index output_dimension,
                       const ActivationFunction& f)
    : net_(Net(Layer(input_dimension, output_dimension, f))) {
}
NetBuilder& NetBuilder::addLayer(int new_result_size,
                                 const ActivationFunction& f) {
    net_.addLayer(new_result_size, f);
    return *this;
}
Net NetBuilder::build() {
    return net_;
}
}  // namespace neural_network
