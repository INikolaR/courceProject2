#include <cstdio>
#include <iostream>
#include <vector>
#include <array>
#include <list>
#include <Eigen/Dense>
#include <cmath>
#include "Layer.h"
#include "ActivationFunction.h"
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

        const static LossFunction Euclid;

        Net(std::initializer_list<int> k, std::initializer_list<ActivationFunction> f, LossFunction l);
        Matrix predict(const Matrix &x) const;
        void fit(const std::vector<Element>& dataset, int n_of_batches, int n_of_epochs, double start_step);
        double MSE(const std::vector<Element>& dataset) const;
        Index getInputSize() const;
        Index getOutputSize() const;
    private:
        void train_one_epoch(const std::vector<Element> &dataset, const std::vector<ConstElemIterator>& borders, double start_step);
        void train_one_batch(const std::vector<Element> &dataset, ConstElemIterator start, ConstElemIterator end, double start_step);

        constexpr static double StartStep = 0.001;
        constexpr static double EpsilonMSE = 0.0001;
        constexpr static double LeakyReluA = 0.01;

        std::list<Layer> layers_;
        LossFunction l_;
    };
}
