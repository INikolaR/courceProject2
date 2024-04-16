#include "Net.h"

#include <cmath>
#include <iostream>

#include "ActivationFunction.h"

namespace neural_network {
const ActivationFunction Net::ReLU = ActivationFunction(
    [](double x) { return (x > 0) * x; }, [](double x) { return (x > 0); });
const ActivationFunction Net::LeakyReLU = ActivationFunction(
    [](double x) { return (x > 0) * (1 - LeakyReluA) * x + LeakyReluA * x; },
    [](double x) { return (x > 0) * (1 - LeakyReluA) + LeakyReluA; });
const ActivationFunction Net::Sigmoid =
    ActivationFunction([](double x) { return 1 / (1 + exp(-x)); },
                       [](double x) {
                           double s = 1 / (1 + exp(-x));
                           return s * (1 - s);
                       });

const LossFunction Net::Euclid = LossFunction(
    [](const Matrix &x, const Matrix &y) {
        Matrix d = x - y;
        double dist = 0;
        for (int i = 0; i < d.cols(); ++i) {
            dist += (d.col(i).transpose() * d.col(i))(0, 0);
        }
        return dist;
    },
    [](const Matrix &x, const Matrix &y) { return 2 * (x - y).transpose(); });

Net::Net(std::initializer_list<int> k,
         std::initializer_list<ActivationFunction> f) {
    assert(k.size() == f.size() - 1);
    auto curr_size = k.begin();
    auto prev_size = curr_size++;
    auto activation_function = f.begin();
    for (; curr_size != k.end();
         ++prev_size, ++curr_size, ++activation_function) {
        layers_.emplace_back(
            Layer(*prev_size, *curr_size, *activation_function));
    }
}

Net::Net(const Layer &layer) {
    layers_.emplace_back(layer);
}

void Net::addLayer(int new_result_size, const ActivationFunction &f) {
    layers_.emplace_back(Layer(getOutputSize(), new_result_size, f));
}

Matrix Net::predict(const neural_network::Matrix &x) const {
    Matrix z = x;
    for (const auto &l : layers_) {
        z = l.evaluate(z);
    }
    return z;
}

void Net::fit(const std::vector<TrainUnit> &dataset, const LossFunction &l,
              int size_of_batch, int n_of_epochs, Optimizer optimizer) {
    optimizer.fit(&layers_, l, dataset, size_of_batch, n_of_epochs);
}

double Net::getLoss(const std::vector<TrainUnit> &dataset,
                    const LossFunction &l) const {
    assert(dataset.size() > 0);
    double loss = 0;
    for (int i = 0; i < dataset.size(); ++i) {
        loss += l.dist(predict(dataset[i].x), dataset[i].y);
    }
    return loss / dataset.size();
}

Index Net::getInputSize() const {
    return layers_.front().getInputSize();
}
Index Net::getOutputSize() const {
    return layers_.back().getOutputSize();
}

double Net::accuracy(const std::vector<TrainUnit> &dataset) const {
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
