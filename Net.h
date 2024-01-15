#include <cstdio>
#include <iostream>
#include <vector>
#include <array>
#include <list>
#include <Eigen/Dense>
#include <cmath>
#include "Layer.h"

namespace neural_network {
    using Matrix = Eigen::MatrixXd;
    using Batch = std::vector<std::pair<Matrix, Matrix>>;

    class Net {
    public:
        Net(const std::initializer_list<int>& k);

        Matrix predict(const Matrix &x) const;

        void fit(const std::vector<Batch>& dataset);

        double countMSE(const std::vector<Batch>& dataset) const;

    private:
        Matrix dLdz(const Matrix &z, const Matrix &y) const;

        std::list<Layer> layers_;
        std::list<std::list<Matrix>> x_;
        constexpr static double StartStep = 0.0001;
        constexpr static double EpsilonMSE = 0.001;
    };
}
