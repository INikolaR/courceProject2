#include "ConstantOptimizer.h"

namespace neural_network {
ConstantOptimizer::ConstantOptimizer(double step) : step_(step) {
}

void ConstantOptimizer::reset(std::list<Layer>& layers) {
}

void ConstantOptimizer::trainOneBatch(std::list<Layer>& layers, const LossFunction& l,
                                      const std::vector<Element>& dataset, ConstElemIterator start,
                                      ConstElemIterator end) {
    std::list<Matrix> mid_values;
    Matrix u;
    Matrix x(layers.front().getInputSize(), end - start);
    Matrix y(layers.back().getOutputSize(), end - start);
    for (auto i = start; i != end; ++i) {
        x.col(i - start) = i->x;
        y.col(i - start) = i->y;
    }

    for (const auto& layer : layers) {
        mid_values.emplace_back(x);
        x = layer.evaluate(x);
    }
    u = l.derivativeDist(x, y);
    std::list<Matrix>::reverse_iterator it_x = mid_values.rbegin();
    for (std::list<Layer>::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); ++layer, ++it_x) {
        Matrix grad_a = layer->getGradA(u, *it_x);
        Vector grad_b = layer->getGradB(u, *it_x);
        u = layer->getNextU(u, *it_x);
        layer->updA(step_ * grad_a);
        layer->updB(step_ * grad_b);
    }
}
}  // namespace neural_network