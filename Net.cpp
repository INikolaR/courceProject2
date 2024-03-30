#include "Net.h"

#include <cmath>
#include <iostream>

#include "SmoothFunction.h"

namespace neural_network {
const SmoothFunction Net::ReLU = SmoothFunction([](double x) { return (x > 0) * x; }, [](double x) { return (x > 0); });
const SmoothFunction Net::LeakyReLU =
    SmoothFunction([](double x) { return (x > 0) * (1 - LeakyReluA) * x + LeakyReluA * x; },
                       [](double x) { return (x > 0) * (1 - LeakyReluA) + LeakyReluA; });
const SmoothFunction Net::Sigmoid = SmoothFunction([](double x) { return 1 / (1 + exp(-x)); },
                                                           [](double x) {
                                                               double s = 1 / (1 + exp(-x));
                                                               return s * (1 - s);
                                                           });

const LossFunction Net::Euclid = LossFunction(
    [](const Matrix &x, const Matrix &y) {
        Matrix d = x - y;
        return (d.transpose() * d)(0, 0);
    },
    [](const Matrix &x, const Matrix &y) { return 2 * (x - y).transpose(); });

Net::Net(std::initializer_list<int> k, std::initializer_list<SmoothFunction> f, LossFunction l) : l_(l) {
    auto curr_size = k.begin();
    auto prev_size = curr_size++;
    auto activation_function = f.begin();
    for (; curr_size != k.end(); ++prev_size, ++curr_size, ++activation_function) {
        layers_.emplace_back(Layer(*prev_size, *curr_size, *activation_function));
    }
}

Matrix Net::predict(const neural_network::Matrix &x) const {
    Matrix z = x;
    for (const auto &l : layers_) {
        z = l.evaluate(z);
    }
    return z;
}

Index Net::getInputSize() const {
    return layers_.front().getInputSize();
}
Index Net::getOutputSize() const {
    return layers_.back().getOutputSize();
}
double Net::MSE(const std::vector<Element> &dataset) const {
    double mse = 0;
    for (int i = 0; i < dataset.size(); ++i) {
        mse += l_.dist(predict(dataset[i].x), dataset[i].y);
    }
    return mse;
}

double Net::accuracy(const std::vector<Element> &dataset) const {
    double successful_guess_count = 0;
    for (int i = 0; i < dataset.size(); ++i) {
        Vector z = predict(dataset[i].x);
        int ans = get_index_max(z);
        successful_guess_count += dataset[i].y(ans);
    }
    return successful_guess_count / dataset.size();
}
int Net::get_index_max(Vector &v) const {
    double max = v(0);
    int max_index = 0;
    for (int i = 0; i < v.rows(); ++i) {
        if (v(i) > max) {
            max = v(i);
            max_index = i;
        }
    }
    return max_index;
}
}  // namespace neural_network
