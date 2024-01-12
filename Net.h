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
        Net(const std::initializer_list<int>& k) : layers_(std::list<Layer>()), x_(std::list<std::list<Matrix>>()) {
            auto curr = k.begin();
            auto prev = curr++;
            for (;curr != k.end();++prev, ++curr) {
                layers_.push_back(Layer(*prev, *curr, ActivateFunction()));
            }
        }
        Matrix predict(const Matrix &x) const {
            Matrix z = x;
            for (const auto& l : layers_) {
                z = l.evaluate(z);
            }
            return z;
        }

        void fit(const std::vector<Batch>& dataset) {
            double prev_mse = -INT_MAX;
            double curr_mse = countMSE(dataset);
            while (abs(prev_mse - curr_mse) > EpsilonMSE) {
                double step = StartStep; // <- tried `step = StartStep / (i + 1)`, but that did not work.
                for (const auto &batch: dataset) {
                    x_.clear();
                    Matrix l = Matrix::Zero(1, layers_.back().getOutputSize());
                    for (const auto &xy: batch) {
                        x_.push_back(std::list<Matrix>());
                        Matrix z = xy.first;
                        for (const auto &layer: layers_) {
                            x_.back().push_back(z);
                            z = layer.evaluate(z);
                        }
                        l += dLdz(z, xy.second);
                    }
                    l = (1 / static_cast<double>(batch.size())) * l;
                    for (const auto &xy: batch) {
                        Matrix u = l;
                        for (auto layer = layers_.end(); layer != layers_.begin();) {
                            --layer;
                            layer->addToUpdA(step, -layer->getGradA(u, x_.back().back()));
                            layer->addToUpdB(step, -layer->getGradB(u, x_.back().back()));
                            u = layer->getNextU(u, x_.back().back());
                            x_.back().pop_back();
                        }
                        x_.pop_back();
                    }
                    for (auto &layer: layers_) {
                        layer.updateAB();
                    }
                }
                prev_mse = curr_mse;
                curr_mse = countMSE(dataset);
            }
        }

        double countMSE(const std::vector<Batch>& dataset) const {
            int counter = 0;
            double mse = 0;
            for (const auto& batch : dataset) {
                for (const auto& xy : batch) {
                    Matrix diff = predict(xy.first) - xy.second;
                    mse += (diff.transpose() * diff)(0, 0);
                    ++counter;
                }
            }
            return (counter == 0 ? 0 : mse / counter);
        }

    private:
        Matrix dLdz(const Matrix &z, const Matrix &y) {
            return 2 * (z - y).transpose();
        }

        std::list<Layer> layers_;
        std::list<std::list<Matrix>> x_;
        constexpr static double StartStep = 0.0001;
        constexpr static double EpsilonMSE = 0.001;
    };
}
