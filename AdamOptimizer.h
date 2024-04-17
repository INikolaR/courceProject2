#pragma once

#include <list>

#include "CustomTypes.h"
#include "Layer.h"
#include "LossFunction.h"

namespace neural_network {
class AdamOptimizer {
    using ConstTrainUnitIterator = std::vector<TrainUnit>::const_iterator;
    struct Momentum {
        Matrix a;
        Vector b;
    };

public:
    AdamOptimizer(double start_alpha, double beta1, double beta2,
                  double epsilon);
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

private:
    double start_alpha_;
    double beta1_;
    double beta2_;
    double epsilon_;
    double beta1_t_;
    double beta2_t_;
    std::list<Momentum> m_;
    std::list<Momentum> v_;
};
}  // namespace neural_network
