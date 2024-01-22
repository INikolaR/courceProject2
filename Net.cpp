#include "Net.h"
#include "ActivationFunction.h"

#include <limits>

namespace neural_network {
    Net::Net(std::initializer_list<int> k, ActivationFunction f) {
        auto curr = k.begin();
        auto prev = curr++;
        for (; curr != k.end(); ++prev, ++curr) {
            layers_.emplace_back(Layer(*prev, *curr, f));
        }
    }

    Matrix Net::predict(const neural_network::Matrix &x) const {
        Matrix z = x;
        for (const auto &l: layers_) {
            z = l.evaluate(z);
        }
        return z;
    }

    void Net::fit(const std::vector<Batch> &dataset) {
        double prev_mse = std::numeric_limits<double>::min();
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
                l = (1. / batch.size()) * l;
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
                    layer.updateAndResetWeights();
                }
            }
            prev_mse = curr_mse;
            curr_mse = countMSE(dataset);
        }
    }

    double Net::countMSE(const std::vector<Batch> &dataset) const {
        int counter = 0;
        double mse = 0;
        for (const auto &batch: dataset) {
            for (const auto &xy: batch) {
                Matrix diff = predict(xy.first) - xy.second;
                mse += (diff.transpose() * diff)(0, 0);
                ++counter;
            }
        }
        return (counter == 0 ? 0 : mse / counter);
    }

    Matrix Net::dLdz(const Matrix &z, const Matrix &y) const {
        return 2 * (z - y).transpose();
    }
}
