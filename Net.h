#pragma once

#include <array>
#include <cmath>
#include <cstdio>
#include <Eigen/Dense>
#include <iostream>
#include <list>
#include <vector>

#include "ActivationFunction.h"
#include "Layer.h"
#include "LossFunction.h"

namespace neural_network {
using Index = Eigen::Index;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
struct Element {
    Vector x;
    Vector y;
};
using ConstElemIterator = std::vector<Element>::const_iterator;

class Net {
public:
    const static ActivationFunction ReLU;
    const static ActivationFunction LeakyReLU;
    const static ActivationFunction Sigmoid;

    const static LossFunction Euclid;
    Net(std::initializer_list<int> k, std::initializer_list<ActivationFunction> f, LossFunction l);
    Matrix predict(const Matrix &x) const;
    double MSE(const std::vector<Element> &dataset) const;
    double accuracy(const std::vector<Element> &dataset) const;
    Index getInputSize() const;
    Index getOutputSize() const;

    template <class T>
    void fit(const std::vector<Element> &dataset, int n_of_batches, int n_of_epochs, T optimizer) {
        optimizer.reset(layers_);
        int64_t size_of_batch = dataset.size() / n_of_batches;
        std::vector<ConstElemIterator> borders(0);
        for (int64_t i = 0; i < n_of_batches; ++i) {
            borders.emplace_back(dataset.begin() + i * size_of_batch);
        }
        borders.push_back(dataset.end());
        for (int epoch = 0; epoch < n_of_epochs; epoch++) {
            train_one_epoch(dataset, borders, optimizer);
        }
    }

private:
    template <class T>
    void train_one_epoch(const std::vector<Element> &dataset, const std::vector<ConstElemIterator> &borders,
                         T &optimizer) {
        for (size_t i = 0; i < borders.size() - 1; ++i) {
            optimizer.trainOneBatch(layers_, l_, dataset, borders[i], borders[i + 1]);
        }
    }
    int get_index_max(Vector &v) const;

    constexpr static double StartStep = 0.001;
    constexpr static double EpsilonMSE = 0.0001;
    constexpr static double LeakyReluA = 0.01;

    std::list<Layer> layers_;
    LossFunction l_;
};
}  // namespace neural_network
