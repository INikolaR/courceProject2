#pragma once

#include <list>

#include "CustomTypes.h"
#include "Layer.h"
#include "LossFunction.h"
namespace neural_network {

class ConstantOptimizer {
    using ConstTrainUnitIterator = std::vector<TrainUnit>::const_iterator;

public:
    explicit ConstantOptimizer(double alpha);
    void operator()(std::list<Layer> *layers, const LossFunction &l,
                    const std::vector<TrainUnit> &dataset, int size_of_batch,
                    int n_of_epochs);

private:
    void train_one_epoch(std::list<Layer> *layers, const LossFunction &l,
                         const std::vector<TrainUnit> &dataset,
                         std::vector<ConstTrainUnitIterator> &borders);
    void train_one_batch(std::list<Layer> *layers, const LossFunction &l,
                         ConstTrainUnitIterator start,
                         ConstTrainUnitIterator end);

    double alpha_;
};
}  // namespace neural_network
