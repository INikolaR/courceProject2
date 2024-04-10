#pragma once
#include <functional>
#include <list>

#include "CustomTypes.h"
#include "Layer.h"
#include "LossFunction.h"

namespace neural_network {
class Optimizer {
public:
    Optimizer(
        std::function<void(std::list<Layer> *, const LossFunction &,
                           const std::vector<TrainUnit> &, int, int)> &&f);
    void fit(std::list<Layer> *layers, const LossFunction &l,
             const std::vector<TrainUnit> &dataset, int size_of_batch,
             int n_of_epochs);

private:
    std::function<void(std::list<Layer> *, const LossFunction &,
                       const std::vector<TrainUnit> &, Index, Index)>
        f_;

};
}  // namespace neural_network
