#pragma once

#include <Eigen/Dense>
#include <list>

#include "Net.h"

namespace neural_network {

class ConstantOptimizer {
public:
    explicit ConstantOptimizer(double step);
    void reset(std::list<Layer>& layers);
    void trainOneBatch(std::list<Layer>& layers, const LossFunction& l, const std::vector<Element>& dataset,
                       ConstElemIterator start, ConstElemIterator end);

private:
    double step_;
};
}  // namespace neural_network
