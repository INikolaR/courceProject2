#pragma once

#include <array>
#include <cmath>
#include <cstdio>
#include <Eigen/Dense>
#include <iostream>
#include <list>
#include <vector>

#include "ActivationFunction.h"
#include "CustomTypes.h"
#include "Layer.h"
#include "LossFunction.h"
#include "Optimizer.h"

namespace neural_network {
class Net {
    using ConstElemIterator = std::vector<TrainUnit>::const_iterator;

public:
    const static ActivationFunction ReLU;
    const static ActivationFunction LeakyReLU;
    const static ActivationFunction Sigmoid;

    const static LossFunction Euclid;
    Net(std::initializer_list<int> k,
        std::initializer_list<ActivationFunction> f);
    Matrix predict(const Matrix &x) const;
    void fit(const std::vector<TrainUnit> &dataset, const LossFunction &l,
             int size_of_batch, int n_of_epochs, Optimizer optimizer);
    double getLoss(const std::vector<TrainUnit> &dataset,
                   const LossFunction &l) const;
    double accuracy(const std::vector<TrainUnit> &dataset) const;
    Index getInputSize() const;
    Index getOutputSize() const;

private:
    int get_index_max(Vector &v) const;

    constexpr static double StartStep = 0.001;
    constexpr static double EpsilonMSE = 0.0001;
    constexpr static double LeakyReluA = 0.01;

    std::list<Layer> layers_;
};
}  // namespace neural_network
